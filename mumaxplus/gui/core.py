"""Simulation controller of the mumax+ GUI, modelled on the mumax3 GUI (engine/gui.go).

Everything that touches the mumax+ world happens on ONE thread: the thread that calls
:meth:`Controller.loop`. It advances the solver in short chunks and executes the
commands that the web page queued in between (mumax3 does the same with its ``Inject``
channel).
HTTP threads only queue commands and read published snapshots, so a GUI edit never races
with a solver step, and nothing GPU-related is called from another thread. (The mumax+
bindings do not release the GIL, so one long ``run()`` would also freeze the web
server.)
"""

from __future__ import annotations

import ast
import collections
import contextlib
import inspect
import io
import math
import os
from pathlib import Path
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback

import numpy as np

import mumaxplus
import mumaxplus.util as U

from .plots import movie_frame, vector_rgba, write_movie

SOLVER_METHODS = ["Heun", "BogackiShampine", "CashKarp", "Fehlberg", "DormandPrince"]
DEFAULT_METHOD = "Fehlberg"  # documented default; mumax+ cannot report the current one
# the time solver's settings under "more solver settings" (fixdt and maxerr are above)
SOLVER_SETTINGS = (
    "headroom",
    "lower_bound",
    "upper_bound",
    "sensible_factor",
    "sensible_timestep_default",
)
CONFIGS = [
    "twodomain",
    "vortex",
    "antivortex",
    "neelskyrmion",
    "blochskyrmion",
    "gaussian_spherical_OoP",
    "gaussian_spherical_IP",
    "gaussian_uniform_IP",
]

CHUNK_SECONDS = 0.05  # wall time of one simulation chunk
RENDER_INTERVAL = 0.10  # minimum wall time between two display frames while running
TORQUE_INTERVAL = 0.5
LIVE_SAMPLE_INTERVAL = 0.15  # live table sampling (wall time) when TableAutosave is 0
TABLE_MAX_ROWS = 20000
# excitation waveforms: factor(t) multiplying the amplitude, from the panel's fields
WAVEFORMS = {
    "sine": "np.sin(2 * np.pi * {f} * (t - {t0}) + {phi}) * (t >= {t0})",
    "gauss": "np.exp(-0.5 * ((t - {t0}) / {sigma}) ** 2)",
    "rect": "((t >= {t0}) & (t < {t0} + {tau}))",
    "sinc": "np.sinc(2 * {fc} * (t - {t0}))",
}
# numpy functions an expression may use without "np."
NP_NAMES = (
    "sin",
    "cos",
    "tan",
    "exp",
    "sqrt",
    "tanh",
    "sinc",
    "pi",
    "abs",
    "heaviside",
)
# what the numbers typed in the page may use besides the script's namespace
MATH = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
PERF_WINDOW = 2.0  # wall time over which steps/s and simulated time/s are measured
NO_RESET = {
    "break",
    "render",
    "refresh",
    "table_using",
    "region",
    "log_only",
    "paint",
    "excite_preview",
    "movie_start",
    "movie_stop",
    "view_add",
    "view_remove",
    "plot_add",
    "plot_remove",
    "plot_quantity",
    "column_save",
    "table_custom",
    "call",
    "script",
    "script_clear",
    "save_ovf",
    "view_autosave",
    "checkpoint_remove",
}


# ----------------------------------------------------------------- small helpers
# a display's plane: the world axes (0 = x, 1 = y, 2 = z) of the image's horizontal and
# vertical, and of the slice (the axis normal to it)
PLANES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}


def _check_plane(cfg):
    """Check a display's plane; a layer of None is its middle slice, xy's bottom one."""
    if cfg.get("plane", "xy") not in PLANES:
        raise ValueError(f"plane must be xy, xz or yz, not {cfg['plane']!r}")
    if cfg.get("layer") is None and cfg.get("plane", "xy") == "xy":
        cfg["layer"] = 0


def _mro(obj):
    return {c.__name__ for c in type(obj).__mro__}


def _param_json(x):
    """Make a parameter value JSON friendly, keeping non-finite floats visible.

    NaN/inf become "nan"/"inf"/"-inf" (valid input for ``_eval``) instead of None,
    so the page does not mistake them for a non-uniform parameter.
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_param_json(v) for v in x]
    x = float(x)
    return x if math.isfinite(x) else str(x)


def _clean(x):
    """Make an object JSON friendly (numpy -> python, NaN/inf -> None)."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _doc(obj):
    """First paragraph of a docstring on one line (mumax3 shows these as tooltips)."""
    text = inspect.getdoc(obj) or ""
    return re.sub(r"\s+", " ", text.split("\n\n")[0]).strip()


def _num(v):
    """Shortest text that reads back as the float v (5e+09, not 5000000000.0)."""
    v = float(v)
    return min(
        (f"{v:.{p}g}" for p in range(1, 18) if float(f"{v:.{p}g}") == v),
        key=len,
        default=repr(v),
    )


def _col_name(base, ncomp, i):
    """Table column of component i of a quantity: m, or mx, my, mz, or m0, m1, ..."""
    return base if ncomp == 1 else base + ("xyz"[i] if ncomp == 3 else str(i))


class Table:
    """The mumax3 table: t and the columns the user added, sampled during the run.

    Columns added with TableAdd are saved in the table file. The plots can also show
    any quantity of a magnet without adding it to the file: those columns are
    "plot-only", sampled like the others but only kept in memory, and dropped when no
    plot shows them any more.
    """

    def __init__(self, path="table.txt"):
        self.path = path
        # {"name", "unit", "fn", "saved", "key"}; fn() -> float, key identifies the
        # quantity ("magnet:quantity:component") so a plot reuses its column
        self.cols = []
        self.rows = []  # one value per column, saved or not
        self.period = 0.0  # TableAutosave: 0 = live preview only (not saved)
        self.next_save = None
        self._header_written = False
        # the page fetches new rows only; rows that change otherwise (clear, thinning,
        # rewind, a column added or removed) make a new version, which it fetches whole
        self.version = 0
        self.lock = threading.Lock()

    def names(self):
        """Return the column names, starting with t."""
        return ["t"] + [c["name"] for c in self.cols]

    def clear(self):
        """Remove all rows; the header is written again on the next save."""
        with self.lock:
            self.rows.clear()
            self.version += 1
        self._header_written = False

    def add(self, name, unit, fn, saved=True, key=None, t=None):
        """Add a column; return False if the name is already taken.

        The rows so far get NaN in it, except a last row at time ``t`` (the current
        time), which gets its value, so a paused simulation shows it at once.
        """
        if any(c["name"] == name for c in self.cols):
            return False
        col = {"name": name, "unit": unit, "fn": fn, "saved": saved, "key": key}
        with self.lock:
            self.cols.append(col)
            for r in self.rows:
                r.append(float("nan"))
            if self.rows and t is not None and self.rows[-1][0] == t:
                self.rows[-1][-1] = self._value(col)
            self.version += 1
        if saved:
            self._header_written = False
        return True

    def remove(self, name):
        """Remove a plot-only column."""
        i = next(i for i, c in enumerate(self.cols) if c["name"] == name)
        if self.cols[i]["saved"]:
            raise ValueError(f"column {name!r} is saved in {self.path}")
        with self.lock:
            del self.cols[i]
            for r in self.rows:
                del r[i + 1]
            self.version += 1

    def save(self, name):
        """Save a plot-only column in the table file from now on."""
        col = next(c for c in self.cols if c["name"] == name)
        if not col["saved"]:
            col["saved"] = True
            self._header_written = False
            with self.lock:
                self.version += 1

    def find(self, key):
        """Return the name of the column of quantity ``key``, or None."""
        return next((c["name"] for c in self.cols if c["key"] == key), None)

    @staticmethod
    def _value(col):
        try:
            return float(col["fn"]())
        except Exception:
            return float("nan")

    def sample(self, t, to_file):
        """Append a row at time t, and write it to the table file if to_file."""
        row = [float(t)] + [self._value(c) for c in self.cols]
        with self.lock:
            self.rows.append(row)
            if len(self.rows) > TABLE_MAX_ROWS:
                self.rows[:] = self.rows[::2]
                self.version += 1
        if to_file:
            saved = self._saved()
            with open(self.path, "a") as f:
                if not self._header_written:
                    f.write(self._header(saved))
                    self._header_written = True
                f.write(self._line(row, saved))

    def _saved(self):
        """Return the indices (into a row) of t and the saved columns."""
        return [0] + [i + 1 for i, c in enumerate(self.cols) if c["saved"]]

    def _header(self, idx):
        head = ["t (s)"] + [f"{c['name']} ({c['unit']})" for c in self.cols]
        return "# " + "\t".join(head[i] for i in idx) + "\n"

    @staticmethod
    def _line(row, idx):
        return "\t".join(f"{row[i]:.8g}" for i in idx) + "\n"

    def rewind(self, t):
        """Go back to time t (a checkpoint was restored).

        Rows after t are dropped from memory; the table file only gets a comment,
        as its rows were saved already.
        """
        with self.lock:
            self.rows[:] = [r for r in self.rows if r[0] <= t * (1 + 1e-12)]
            self.version += 1
        self.next_save = None
        if self._header_written:
            with open(self.path, "a") as f:
                f.write(f"# checkpoint restored: back to t = {t:.8g} s\n")

    def text(self):
        """Return the rows in memory, plot-only columns too, as the file has them."""
        with self.lock:
            rows = list(self.rows)
        idx = list(range(len(self.cols) + 1))
        return self._header(idx) + "".join(self._line(r, idx) for r in rows)

    def since(self, version, start):
        """Return the rows the page does not have yet.

        The page has rows [0, start) of ``version``; when the version changed, it gets
        all rows again.
        """
        with self.lock:
            if version != self.version or not 0 <= start <= len(self.rows):
                start = 0
            return {
                "version": self.version,
                "start": start,
                "names": self.names(),
                "units": ["s"] + [c["unit"] for c in self.cols],
                "saved": [True] + [c["saved"] for c in self.cols],
                "rows": [r[:] for r in self.rows[start:]],
            }

    def column(self, spec):
        """1-based index or name -> index into a row."""
        names = self.names()
        spec = str(spec).strip()
        if spec.isdigit() and 1 <= int(spec) <= len(names):
            return int(spec) - 1
        if spec in names:
            return names.index(spec)
        raise ValueError(
            f"unknown column {spec!r}; columns are: "
            + ", ".join(f"{i + 1}={n}" for i, n in enumerate(names))
        )


class Controller:
    def __init__(self, world, namespace=None, title=None, table_file="table.txt"):
        self.world = world
        self.lock = threading.RLock()
        self.cmds: "queue.Queue" = queue.Queue()
        self._stop = False
        self.title = title or (
            Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else "mumax+"
        )

        self.ns = namespace if namespace is not None else {}
        self.ns.setdefault("world", world)

        # run state (mumax3: pause / Run / Steps / Break)
        self.running = False
        self.run_until = None  # stop when time >= this
        self.steps_left = None
        self._run_t0 = 0.0
        self._run_n0 = 0
        self.progress = 0
        self.busy = ""
        self.steps_done = 0
        self.method = DEFAULT_METHOD
        self.method_known = False
        self._nchunk = 1
        # TimeSolver.steps resets the adaptive dt; only wanted after a change
        self._fresh = True
        self.maxtorque = None
        self._last_torque = 0.0
        self.last_error = ""

        # what the page shows
        self.targets = {}
        self.sel = None  # selected magnet (display, parameters, mesh, initial m)
        self.region = -1
        # one dict per display panel: {"id", "quant", "comp", "plane", "layer", ...}
        self.views = []
        self._next_view = 0
        self.table = Table(table_file)
        # one dict per plot panel: table columns x against y1,y2,... (names)
        self.plots = [{"id": 0, "x": "t", "y": "", "logx": False, "logy": False}]
        self._next_plot = 1
        self.desc = {}
        self.desc_version = 0
        self.frame_id = 0
        self.frames = {}  # view id -> (meta, bytes)
        self.snap = {}
        self.snap_id = 0  # goes up when the snapshot changes
        self.changed = threading.Condition(self.lock)  # notified then
        self.log = []
        self.log_id = 0
        self._dirty = True
        self._last_render = self._last_live = 0.0

        # "Copy as Python": what was done in the page, as lines to add to the script
        self.script = []
        self._script_imports = []
        self._rec_t = self._rec_n = None  # time / step of the unrecorded part of a run

        # files the page may download (movies made here)
        self.outputs = set()

        # excitations added in the page (time terms of parameters), see _excite
        self.excitations = []
        self._next_excitation = 1

        # painting on a display: the stroke being painted, and how to undo the last one
        self._stroke = None
        self._paint_undo = None

        # checkpoints: copies of the state (m, elastic fields, time) in host memory
        self.checkpoints = []
        self._next_checkpoint = 1

        # performance: (wall time, simulation time, step) at the end of recent chunks
        self._perf = collections.deque()
        self.perf = {"steps_s": None, "sim_s": None}

        self._install_helpers()
        self.refresh_targets()
        self._init_selection()
        self.describe()

    # ------------------------------------------------------------------- logging
    def say(self, text, kind="out"):
        """kind: 'in' (command echo), 'out' (// message), 'err' (// error)."""
        with self.lock:
            self.log_id += 1
            self.log.append({"id": self.log_id, "kind": kind, "text": str(text)})
            del self.log[:-500]
            if kind == "err":
                self.last_error = str(text)

    def _ts(self):
        return self.world.timesolver

    def _now(self):
        return float(self.world.timesolver.time)

    # ------------------------------------------------------------- Copy as Python
    def record(self, code, imports=()):
        """Add ``code`` to the recorded script, after the run done so far."""
        self._record_run()
        self.script.append(code)
        self._script_imports += [i for i in imports if i not in self._script_imports]

    def _did(self, code, imports=()):
        """Echo ``code`` in the console and record it."""
        self.say(code, "in")
        self.record(code, imports)

    def _record_run(self):
        """Record the part of the current run that was not recorded yet."""
        if not self.running or self._rec_t is None:
            return
        now = self._now()
        if self.steps_left is not None:
            n = self.steps_done - self._rec_n
            if n > 0:
                self.script.append(f"world.timesolver.steps({n})")
        else:
            # a finished Run(t) records t, not t plus the overshoot of the last step
            end = self.run_until
            if end is None or now < end * (1 - 1e-12):
                end = now
            if end > self._rec_t:
                self.script.append(f"world.timesolver.run({end - self._rec_t:.10g})")
        self._rec_t, self._rec_n = now, self.steps_done

    def script_text(self):
        """Return the recorded script, including the run so far."""
        self._record_run()
        head = [
            f"# What was done in the mumax+ GUI of {self.title}, as Python. Add these",
            "# lines to the end of the script, in place of gui.run(), to repeat it.",
        ]
        if not self.script:
            return "\n".join(head + ["# (nothing recorded yet)"]) + "\n"
        imports = self._script_imports and ["", *self._script_imports]
        return "\n".join(head + imports + [""] + self.script) + "\n"

    # -------------------------------------------------------------------- targets
    def refresh_targets(self):
        """Update the selectable magnets from the world."""
        targets = {}
        for name, magnet in dict(self.world.magnets).items():
            targets[name] = magnet
            subs = getattr(type(magnet), "sublattices", None)
            if isinstance(subs, property):
                try:
                    for i, sub in enumerate(magnet.sublattices, start=1):
                        targets[f"{name}:sub{i}"] = sub
                except Exception:
                    pass
        self.targets = targets
        self.ns.setdefault("magnets", targets)

    def _primary(self):
        return [n for n in self.targets if ":sub" not in n]

    def _ref(self, name):
        """Python expression for a target, as the user would type it in a script."""
        if ":sub" in name:
            parent, sub = name.split(":sub")
            return f"{self._ref(parent)}.sub{sub}"
        obj = self.targets[name]
        impl = getattr(obj, "_impl", None)  # world.magnets makes new Python objects
        for k, v in self.ns.items():
            if k.startswith("_"):
                continue
            if v is obj or (impl is not None and getattr(v, "_impl", None) is impl):
                return k
        return f"world.magnets[{name!r}]"

    def _init_selection(self):
        names = list(self.targets)
        self.sel = names[0] if names else None
        self.add_view()

    def _default_quantity(self):
        if self.sel is None:
            return None
        cls = type(self.targets[self.sel])
        if isinstance(getattr(cls, "magnetization", None), property):
            return "magnetization"
        return next(
            (n for n in dir(cls) if isinstance(getattr(cls, n, None), property)), None
        )

    # ---------------------------------------------------------------------- views
    def add_view(self, **cfg):
        """Add a display panel.

        It starts as a copy of the last one (or the default quantity).
        """
        keys = ("quant", "comp", "plane", "layer", "all")
        base = (
            {k: self.views[-1][k] for k in keys}
            if self.views
            else {
                "quant": self._default_quantity(),
                "comp": "",
                "plane": "xy",
                "layer": 0,
                "all": False,
            }
        )
        base.update({k: v for k, v in cfg.items() if k in keys})
        _check_plane(base)
        base["autosave"] = 0.0  # mumax3 AutoSave period of the shown quantity (OVF)
        base["_next_save"] = None
        base["id"] = self._next_view
        self._next_view += 1
        self.views.append(base)
        self._dirty = True
        return base["id"]

    def add_plot(self, x=None, y=None, logx=None, logy=None):
        """Add a plot panel of table columns x:y (numbers or names).

        It starts as a copy of the last one. ``y`` may name several columns.
        """
        p = dict(self.plots[-1], id=self._next_plot)
        self._next_plot += 1
        self._plot_set(p, x, y, logx, logy)
        self.plots.append(p)
        return p["id"]

    def _plot(self, pid):
        p = next((p for p in self.plots if p["id"] == pid), None)
        if p is None:
            raise ValueError(f"no plot {pid!r}")
        return p

    def _plot_set(self, p, x=None, y=None, logx=None, logy=None):
        """Set a plot's columns.

        Column numbers become names, as plot-only columns come and go (and the
        numbers after them change).
        """

        def name(spec):
            try:
                return self.table.names()[self.table.column(spec)]
            except ValueError:
                return str(spec).strip()  # not (yet) a column: the page says so

        if x is not None:
            p["x"] = name(x)
        if y is not None:
            ys = y if isinstance(y, (list, tuple)) else str(y).split(",")
            p["y"] = ",".join(name(v) for v in ys if str(v).strip())
        if logx is not None:
            p["logx"] = bool(logx)
        if logy is not None:
            p["logy"] = bool(logy)

    def _series(self, target, quant, comp=None):
        """Return the table column of a quantity of a magnet.

        That is a ScalarQuantity, or component ``comp`` of the average of a
        FieldQuantity. Without one yet, a plot-only column is added (not saved in
        the table file).
        """
        obj = getattr(self.targets[target], quant)
        field = hasattr(obj, "average")
        n = int(obj.ncomp) if field else 1
        i = int(comp or 0) if field else None
        if field and not 0 <= i < n:
            raise ValueError(f"{quant} has no component {comp!r}")
        key = f"{target}:{quant}:{'' if i is None else i}"
        name = self.table.find(key)
        if name:
            return name
        base = quant if len(self.targets) == 1 else f"{target}.{quant}"
        base = _col_name(base, n, i)
        taken, name, k = self.table.names(), base, 1
        while name in taken:
            k += 1
            name = f"{base}_{k}"
        fn = (lambda: np.atleast_1d(obj.average())[i]) if field else obj.eval
        unit = getattr(obj, "unit", "") or ""
        self.table.add(name, unit, fn, saved=False, key=key, t=self._now())
        return name

    def _prune_columns(self):
        """Drop the plot-only columns that no plot shows."""
        used = {s for p in self.plots for s in [p["x"], *p["y"].split(",")]}
        for c in list(self.table.cols):
            if not c["saved"] and c["name"] not in used:
                self.table.remove(c["name"])

    def _table_custom(self, c):
        """Add a column (saved in the table file) computed by a Python expression."""
        name, expr, unit = (
            str(c.get(k) or "").strip() for k in ("name", "expr", "unit")
        )
        if not re.fullmatch(r"[A-Za-z_][\w.]*", name):
            raise ValueError("give the column a name: letters, digits, _ and .")
        code = compile(expr, "<table column>", "eval")

        def fn():
            return eval(code, {**MATH, "np": np, **self.ns})

        float(np.asarray(fn()))  # an error now, rather than NaN in every row
        self.table_add(fn, name, unit)
        if c.get("plot") is not None:
            p = self._plot(c["plot"])
            self._plot_set(p, y=[*p["y"].split(","), name])

    def _view(self, vid):
        v = next((v for v in self.views if v["id"] == vid), None)
        if v is None:
            raise ValueError(f"no display {vid!r}")
        return v

    def _view_state(self, cfg):
        """Return what the page needs of a view: its settings and latest frame id."""
        frame = self.frames.get(cfg["id"])
        state = {k: v for k, v in cfg.items() if not k.startswith("_")}
        state["frame_id"] = frame[0]["id"] if frame else -1
        mv = cfg.get("_movie")
        state["movie"] = mv and {"frames": mv["n"], "period": mv["period"]}
        return state

    # --------------------------------------------------------------- introspection
    def _inspect(self, obj):
        info = {
            "quantities": [],
            "params": [],
            "switches": [],
            "paintable": [],
            "scalars": [],  # ScalarQuantity: can be plotted, not displayed
        }
        cls = type(obj)
        for pname in sorted(dir(cls), key=str.lower):
            prop = getattr(cls, pname, None)
            if pname.startswith("_") or not isinstance(prop, property):
                continue
            try:
                val = getattr(obj, pname)
                m = _mro(val)
                item = {
                    "name": pname,
                    "unit": getattr(val, "unit", ""),
                    "doc": _doc(prop),
                }
                settable = prop.fset is not None
                if "Parameter" in m and "InterParameter" not in m:
                    item["ncomp"] = int(val.ncomp)
                    info["params"].append(item)
                    info["quantities"].append(item)
                    info["paintable"].append(item)
                elif "FieldQuantity" in m or "Variable" in m:
                    item["ncomp"] = int(val.ncomp)
                    info["quantities"].append(item)
                    if "Variable" in m and settable:  # m, elastic fields
                        info["paintable"].append(item)
                elif "ScalarQuantity" in m:
                    info["scalars"].append(item)
                elif isinstance(val, (bool, int, float)) and settable:
                    kind = "bool" if isinstance(val, bool) else "num"
                    info["switches"].append({**item, "kind": kind, "value": val})
            except Exception:
                continue
        try:
            info["mesh"] = {
                "n": [int(v) for v in obj.grid.size],
                "c": [float(v) for v in obj.cellsize],
                "inside": float(np.asarray(obj.geometry).mean()),
            }
        except Exception:
            info["mesh"] = None
        return info

    def _param_values(self, obj, params):
        out = {}
        for p in params:
            try:
                q = getattr(obj, p["name"])
                if self.region < 0:
                    out[p["name"]] = (
                        _param_json(q.uniform_value) if q.is_uniform else None
                    )
                else:
                    inside = np.asarray(obj.regions) == self.region
                    geom = np.asarray(obj.geometry, dtype=bool)
                    # a cell that is really part of the magnet
                    idx = np.argwhere(inside & geom)
                    if not len(idx):
                        idx = np.argwhere(inside)
                    if not len(idx):
                        out[p["name"]] = None
                        continue
                    iz, iy, ix = idx[0]
                    v = np.asarray(q.eval())[:, iz, iy, ix]
                    out[p["name"]] = _param_json(v[0] if p["ncomp"] == 1 else v)
            except Exception:
                out[p["name"]] = None
        return out

    def _config_info(self):
        out = [{"name": "Uniform", "doc": "Uniform magnetization", "args": "(1, 0, 0)"}]
        c, cell = self._center(), self._cell()
        defaults = {
            "position": f"({c[0]:.4g}, {c[1]:.4g}, {c[2]:.4g})",
            "diameter": f"{5 * cell:.4g}",
            "radius": f"{5 * cell:.4g}",
            "circulation": "1",
            "polarization": "1",
            "charge": "1",
            "m1": "(1, 0, 0)",
            "mw": "(0, 1, 0)",
            "m2": "(-1, 0, 0)",
            "wallposition": f"{c[0]:.4g}",
            "wallthickness": f"{4 * cell:.4g}",
        }
        for n in CONFIGS:
            fn = getattr(U, n, None)
            if fn is None:
                continue
            params = list(inspect.signature(fn).parameters)
            args = [defaults.get(p, "1") for p in params]
            info = {"name": n, "doc": _doc(fn), "args": f"({', '.join(args)})"}
            if params[:1] == ["position"]:  # can be placed with the paint tool
                info["place"] = ", ".join(args[1:])
            out.append(info)
        return out

    def _center(self):
        try:
            return tuple(float(v) for v in self.targets[self.sel].center)
        except Exception:
            return (0.0, 0.0, 0.0)

    def _cell(self):
        try:
            return float(min(self.targets[self.sel].cellsize))
        except Exception:
            return 1e-9

    def describe(self):
        """Rebuild everything the page needs besides the per-tick numbers."""
        targets = {n: self._inspect(o) for n, o in self.targets.items()}
        for n, info in targets.items():
            info["values"] = self._param_values(self.targets[n], info["params"])
        ts = self._ts()
        solver = {k: getattr(ts, k) for k in SOLVER_SETTINGS if hasattr(ts, k)}
        world_info = {"vectors": [], "numbers": [], "pbc": None}
        wcls = type(self.world)
        for pname in sorted(dir(wcls), key=str.lower):
            prop = getattr(wcls, pname, None)
            if pname.startswith("_") or not isinstance(prop, property):
                continue
            if prop.fset is None:
                continue
            try:
                v = getattr(self.world, pname)
            except Exception:
                continue
            item = {"name": pname, "doc": _doc(prop)}
            if pname == "pbc_repetitions":
                world_info["pbc"] = [int(x) for x in v]
            elif isinstance(v, (tuple, list)) and all(
                isinstance(x, (int, float)) for x in v
            ):
                world_info["vectors"].append({**item, "value": [float(x) for x in v]})
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                world_info["numbers"].append({**item, "value": float(v)})
        version = (
            f"mumax⁺ {getattr(mumaxplus, '__version__', '')}  ·  "
            f"{getattr(mumaxplus, 'FP_PRECISION', '')} precision"
        )
        desc = {
            "title": self.title,
            "version": version,
            "targets": targets,
            "world": world_info,
            "solver": solver,  # the settings under "more solver settings"
            "methods": SOLVER_METHODS,
            "configs": self._config_info(),
            "ffmpeg": shutil.which("ffmpeg") is not None,  # MP4 movies
        }
        with self.lock:
            self.desc = _clean(desc)
            self.desc_version += 1

    # --------------------------------------------------------------------- display
    def _eval_quant(self, name, quant):
        arr = np.asarray(getattr(self.targets[name], quant).eval())
        return arr[None] if arr.ndim == 3 else arr

    def _compose_all(self, quant, ncomp):
        """Paste all magnets that have ``quant`` on one array of the world grid.

        They share the world's cellsize; grid origins are cell indices. Empty cells
        are NaN.
        """
        parts = []
        for n in self._primary():
            try:
                arr = self._eval_quant(n, quant)
                origin = [int(v) for v in self.targets[n].grid.origin]
            except Exception:
                continue
            if arr.shape[0] == ncomp:
                parts.append((n, origin, arr))
        lo = [min(o[i] for _, o, _ in parts) for i in range(3)]
        hi = [max(o[i] + a.shape[3 - i] for _, o, a in parts) for i in range(3)]
        out = np.full(
            (ncomp, hi[2] - lo[2], hi[1] - lo[1], hi[0] - lo[0]),
            np.nan,
            dtype=np.float32,
        )
        boxes = []
        for n, o, a in parts:
            x0, y0, z0 = (o[i] - lo[i] for i in range(3))
            _, nz, ny, nx = a.shape
            out[:, z0 : z0 + nz, y0 : y0 + ny, x0 : x0 + nx] = a
            boxes.append(
                {"name": n, "x0": x0, "y0": y0, "z0": z0, "nx": nx, "ny": ny, "nz": nz}
            )
        return out, boxes, lo

    def _render(self):
        for cfg in self.views:
            self._render_view(cfg)
        self._last_render = time.perf_counter()
        self._dirty = False

    def _render_view(self, cfg):
        if self.sel not in self.targets or not cfg["quant"]:
            return
        boxes, lo = None, [0, 0, 0]
        try:
            fq = getattr(self.targets[self.sel], cfg["quant"])
            arr = self._eval_quant(self.sel, cfg["quant"])
            if cfg.get("all") and len(self._primary()) > 1 and ":sub" not in self.sel:
                arr, boxes, lo = self._compose_all(cfg["quant"], arr.shape[0])
        except Exception as e:
            msg = f"display: cannot evaluate {cfg['quant']}: {e}"
            if cfg.get("_err") != msg:  # say it once, not on every frame while running
                self.say(msg, "err")
            cfg["_err"] = msg
            return
        cfg.pop("_err", None)
        ncomp = arr.shape[0]
        axes = PLANES[cfg.get("plane", "xy")]
        n = arr.shape[3 - axes[2]]  # slices; world axis a is array axis 3 - a
        if cfg["layer"] is None:
            cfg["layer"] = n // 2
        layer = min(max(int(cfg["layer"]), 0), n - 1)
        # (ncomp, vertical, horizontal): the vertical axis comes first in (z, y, x)
        sl = np.take(arr, layer, axis=3 - axes[2])
        if boxes:  # in the image's cells: x right, y up, z through the slices
            k = "xyz"
            boxes = [
                {"name": b["name"]}
                | {f"{d}0": b[f"{k[a]}0"] for d, a in zip(k, axes)}
                | {f"n{d}": b[f"n{k[a]}"] for d, a in zip(k, axes)}
                for b in boxes
            ]
        cell = self._cell_size()
        comp = cfg["comp"]
        vector = ncomp == 3 and comp == ""
        rgba = b""
        if vector:
            data = sl
            empty = np.isnan(sl[0])
            img = vector_rgba(np.nan_to_num(sl))
            img[empty, 3] = 0  # space between magnets: transparent
            rgba = img.tobytes()
        else:
            c = {"x": 0, "y": 1, "z": 2}.get(
                comp, int(comp) if str(comp).isdigit() else 0
            )
            c = min(c, ncomp - 1)
            data = sl[c : c + 1]
        data = np.ascontiguousarray(data, dtype="<f4")
        finite = data[np.isfinite(data)]
        meta = {
            "vector": vector,
            "ncomp": int(data.shape[0]),
            "w": int(data.shape[2]),
            "h": int(data.shape[1]),
            "layer": layer,
            "nslices": n,
            "axes": axes,
            "cell": [cell[axes[0]], cell[axes[1]]],  # m, of the image's cells
            "quant": cfg["quant"],
            "comp": comp,
            "vmin": float(finite.min()) if finite.size else 0.0,
            "vmax": float(finite.max()) if finite.size else 0.0,
            "unit": getattr(fq, "unit", ""),
            "time": self._now(),
            "boxes": boxes,
            "origin": lo,  # "show all": world cell (x, y, z) of the composed cell 0
        }
        with self.lock:
            self.frame_id += 1
            meta["id"] = self.frame_id
            meta["view"] = cfg["id"]
            self.frames[cfg["id"]] = (_clean(meta), data.tobytes() + rgba)

    # ----------------------------------------------------------------------- table
    def table_add(self, quantity, name=None, unit=None):
        """TableAdd(quantity).

        quantity is a name (of the selected magnet), a FieldQuantity (its average is
        added, per component), a ScalarQuantity, or any function without arguments
        that returns a number (e.g. a domain wall position).
        """
        by_name = isinstance(quantity, str)
        obj = getattr(self.targets[self.sel], quantity) if by_name else quantity
        m = _mro(obj)
        field = "FieldQuantity" in m or "Variable" in m
        scalar = "ScalarQuantity" in m
        if not (field or scalar or callable(obj)):
            raise TypeError(
                f"cannot add {obj!r} to the table: "
                "give a quantity or a function returning a number"
            )
        if name is None:
            if by_name:
                name = quantity
            elif field or scalar:
                name = getattr(obj, "name", "quantity").replace(":", "_")
            else:
                name = getattr(obj, "__name__", "quantity")
        unit = getattr(obj, "unit", "") if unit is None else unit
        n = int(obj.ncomp) if field else 1
        keys = [None] * n
        if by_name:
            keys = [f"{self.sel}:{quantity}:{i if field else ''}" for i in range(n)]
            shown = [self.table.find(k) for k in keys]
            if all(shown):  # plotted already: save those columns from now on
                for col in shown:
                    self.table.save(col)
                self.say(f"saving {', '.join(shown)} in {self.table.path}")
                return
        cols = [_col_name(name, n, i) for i in range(n)]
        taken = [col for col in cols if col in self.table.names()]
        if taken:
            raise ValueError(
                f"table already has a column {taken[0]!r}; "
                "choose another one with name=..."
            )
        t = self._now()  # the last row gets the new column's value
        for i, col in enumerate(cols):
            if field:
                fn = lambda o=obj, i=i: np.atleast_1d(o.average())[i]  # noqa: E731
            else:
                fn = obj.eval if scalar else obj
            self.table.add(col, unit, fn, key=keys[i], t=t)
        cols = ", ".join(f"{i + 1}={n}" for i, n in enumerate(self.table.names()))
        self.say(f"table columns: {cols}")

    def _live_sample(self):
        """Take a live-preview row (only when TableAutosave is 0).

        Never writes table.txt.
        """
        if self.table.period == 0:
            self._table_tick(force=True)

    def _table_tick(self, force=False):
        t = self._now()
        tb = self.table
        if tb.period > 0:
            if tb.next_save is None:
                tb.next_save = t + tb.period if tb.rows else t
            if t >= tb.next_save * (1 - 1e-12) or force:
                tb.sample(t, to_file=True)
                while tb.next_save <= t * (1 + 1e-12):
                    tb.next_save += tb.period
        else:
            # live preview: sample every LIVE_SAMPLE_INTERVAL of wall time
            # (and at the end of a run)
            now = time.perf_counter()
            if force or now - self._last_live > LIVE_SAMPLE_INTERVAL:
                self._last_live = now
                if not tb.rows or t != tb.rows[-1][0]:
                    tb.sample(t, to_file=False)

    # ------------------------------------------------------------------ snapshots
    def snapshot(self):
        """Return a copy of the state shown in the page."""
        with self.lock:
            return dict(self.snap)

    def wait_snapshot(self, last, timeout):
        """Wait up to ``timeout`` s for a snapshot newer than the one with id ``last``.

        Return (id, snapshot), the same id when there is none yet, or None once the
        loop stopped.
        """
        with self.changed:
            self.changed.wait_for(lambda: self.snap_id != last or self._stop, timeout)
            return None if self._stop else (self.snap_id, dict(self.snap))

    def _publish(self):
        ts = self._ts()
        snap = _clean(
            {
                "nsteps": self.steps_done,
                "time": ts.time,
                "dt": ts.timestep,
                "busy": self.busy or ("Running" if self.running else "Paused"),
                "running": self.running,
                "progress": self.progress,
                "maxtorque": self.maxtorque,
                "fixdt": 0.0 if ts.adaptive_timestep else ts.timestep,
                "maxerr": ts.max_error,
                "method": self.method,
                "method_known": self.method_known,
                "autosave": self.table.period,
                "plots": [dict(p) for p in self.plots],
                "sel": self.sel,
                "region": self.region,
                "views": [self._view_state(cfg) for cfg in self.views],
                "frame_id": self.frame_id,
                "desc_version": self.desc_version,
                "log_id": self.log_id,
                "script_lines": len(self.script),
                "paint_undo": self._paint_undo is not None,
                "excitations": [
                    {k: v for k, v in e.items() if k in ("id", "target", "label")}
                    for e in self.excitations
                ],
                "perf": dict(self.perf, eta=self._eta()),
                "checkpoints": [
                    {k: v for k, v in cp.items() if k != "state"}
                    for cp in self.checkpoints
                ],
                "error": self.last_error,
                "targets": list(self.targets),
                # the page fetches the table when this changes
                "table": [self.table.version, len(self.table.rows)],
            }
        )
        with self.changed:
            if snap != self.snap:  # wake the pages waiting for it (/api/events)
                self.snap = snap
                self.snap_id += 1
                self.changed.notify_all()

    def _update_torque(self):
        best = None
        for n in self._primary():
            try:
                v = float(self.targets[n].max_torque.eval())
                best = v if best is None else max(best, v)
            except Exception:
                continue
        self.maxtorque = best
        self._last_torque = time.perf_counter()

    # --------------------------------------------------------------------- commands
    def submit(self, cmd, wait=5.0):
        """Queue a command for the simulation thread and wait up to ``wait`` s."""
        done, box = threading.Event(), {}
        self.cmds.put((cmd, box, done))
        if done.wait(wait):
            return box.get("result", {"ok": True})
        return {"ok": True, "pending": True}

    def _drain(self, timeout=0.0):
        try:
            item = self.cmds.get(timeout=timeout) if timeout else self.cmds.get_nowait()
        except queue.Empty:
            return False
        while True:
            cmd, box, done = item
            try:
                box["result"] = self._execute(cmd) or {"ok": True}
            except Exception as e:  # a bad command must never kill the loop
                self.say(f"{type(e).__name__}: {e}", "err")
                box["result"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            self._publish()
            done.set()
            try:
                item = self.cmds.get_nowait()
            except queue.Empty:
                return True

    # numbers typed in the page are expressions, like in mumax3
    # (e.g. 1e-9, 5e-9*2, (0,0,1))
    def _eval(self, text):
        return eval(str(text), {**MATH, **self.ns})

    def _vec(self, text, ncomp):
        v = self._eval(text)
        if ncomp == 1:
            return float(v)
        v = tuple(float(x) for x in v)
        if len(v) != ncomp:
            raise ValueError(f"expected {ncomp} components, got {len(v)}")
        return v

    def _start_run(self, t=None, n=None):
        """Run for a time t, n steps, or (neither) until Break."""
        self._break()
        now = self._now()
        self.running = True
        self._run_t0 = self._rec_t = now
        self._run_n0 = self._rec_n = self.steps_done
        self.run_until = now + float(t) if t is not None else None
        self.steps_left = int(n) if n is not None else None
        self.progress = 0
        self._fresh = True
        self._perf.clear()
        self._perf.append((time.perf_counter(), now, self.steps_done))

    def _set_method(self, method):
        self._ts().set_method(method)
        self.method, self.method_known = method, True

    def _break(self):
        self._record_run()
        self.running = False
        self.run_until = self.steps_left = None
        self._live_sample()
        self._update_torque()

    def _execute(self, c):
        name = c["cmd"]
        if name not in NO_RESET:
            self._fresh = True
            self.last_error = ""
        if name == "run":
            txt = str(c["text"]).strip()
            if txt.lower() in ("", "inf", "forever", "indefinite", "0"):
                self.say("Run()  # until Break", "in")
                self._start_run()
            else:
                self.say(f"Run({txt})", "in")
                self._start_run(t=self._eval(txt))
        elif name == "steps":
            self.say(f"Steps({c['text']})", "in")
            self._start_run(n=int(self._eval(c["text"])))
        elif name == "break":
            self._break()
        elif name in ("relax", "minimize"):
            self._break()
            self._blocking(name, getattr(self.world, name))
        elif name == "solver_type":
            self._set_method(c["value"])
            self._did(f"world.timesolver.set_method({c['value']!r})")
        elif name == "solver_set":
            self._solver_set(c["name"], c["text"])
        elif name == "param":
            self._set_param(c["target"], c["name"], c["text"])
        elif name == "switch":
            obj = self.targets[c["target"]]
            value = c["value"] if c.get("kind") == "bool" else self._eval(c["value"])
            setattr(obj, c["name"], value)
            self._did(f"{self._ref(c['target'])}.{c['name']} = {value!r}")
            self.describe()
        elif name == "world_set":
            vector = isinstance(getattr(self.world, c["name"]), (tuple, list))
            value = self._vec(c["text"], 3) if vector else self._eval(c["text"])
            setattr(self.world, c["name"], value)
            self._did(f"world.{c['name']} = {value!r}")
            self.describe()
        elif name == "set_pbc":
            value = tuple(int(self._eval(v)) for v in c["value"])
            if any(value):
                self.world.set_pbc(value)  # also fits mastergrid to the magnets
                self._did(f"world.set_pbc({value})")
            else:
                self.world.unset_pbc()
                self._did("world.unset_pbc()")
            self.refresh_targets()
            self.describe()
        elif name == "setm":
            self._set_m(c["target"], c["config"], c["args"])
        elif name == "region":
            self.region = int(self._eval(c["value"]))
            self.describe()
        elif name == "select":
            if c["target"] in self.targets:
                self.sel = c["target"]
                quantities = self.desc["targets"][self.sel]["quantities"]
                names = [q["name"] for q in quantities]
                for cfg in self.views:
                    if cfg["quant"] not in names:
                        cfg["quant"], cfg["comp"] = self._default_quantity(), ""
                self.describe()
                self._dirty = True
        elif name == "render":
            cfg = self._view(c.get("view", self.views[0]["id"] if self.views else None))
            for k in ("quant", "comp", "plane", "layer", "all"):
                if k in c:
                    cfg[k] = c[k]
            if "plane" in c and "layer" not in c:
                cfg["layer"] = None  # another plane: start in its middle
            _check_plane(cfg)
            self._dirty = True
        elif name == "view_add":
            return {"ok": True, "view": self.add_view()}
        elif name == "movie_start":
            self._movie_start(self._view(c["view"]), c)
        elif name == "movie_stop":
            return {"ok": True, "file": self._movie_stop(self._view(c["view"]))}
        elif name == "save_ovf":
            return {"ok": True, "file": self._save_ovf(self._view(c["view"]))}
        elif name == "view_autosave":
            self._view_autosave(self._view(c["view"]), self._eval(c["text"]))
        elif name == "view_remove":
            gone = [v for v in self.views if v["id"] == c["view"]]
            if gone and len(self.views) > 1:
                if gone[0].get("_movie"):
                    self._movie_stop(gone[0])  # keep what was recorded
                self.views.remove(gone[0])
                with self.lock:
                    self.frames.pop(c["view"], None)
        elif name == "checkpoint":
            self._checkpoint()
        elif name == "checkpoint_restore":
            self._restore(c["id"])
        elif name == "checkpoint_remove":
            self.checkpoints = [cp for cp in self.checkpoints if cp["id"] != c["id"]]
        elif name == "table_autosave":
            self._table_autosave(self._eval(c["text"]))
        elif name == "table_using":
            p = self._plot(c.get("plot", self.plots[0]["id"]))
            self._plot_set(p, c.get("x"), c.get("y"), c.get("logx"), c.get("logy"))
            self._prune_columns()
        elif name == "plot_quantity":  # any quantity of a magnet, as x or another y
            p = self._plot(c["plot"])
            col = self._series(c["target"], c["quant"], c.get("comp"))
            if c.get("axis") == "x":
                self._plot_set(p, x=col)
            else:
                ys = p["y"].split(",")
                self._plot_set(p, y=ys if col in ys else [*ys, col])
            self._prune_columns()
        elif name == "column_save":
            self.table.save(c["name"])
            self.say(f"saving {c['name']} in {self.table.path}")
        elif name == "table_custom":
            self._table_custom(c)
        elif name == "plot_add":
            return {"ok": True, "plot": self.add_plot()}
        elif name == "call":  # GUI methods the script calls while the loop runs
            return {"ok": True, "value": c["fn"]()}
        elif name == "plot_remove":
            if len(self.plots) > 1:
                self.plots = [p for p in self.plots if p["id"] != c["plot"]]
                self._prune_columns()
        elif name == "excite_preview":
            return {"ok": True, **self._excite_preview(c)}
        elif name == "excite":
            self._excite(c)
        elif name == "excite_remove":
            self._excite_remove(c.get("id"))
        elif name == "paint":
            self._paint(c)
        elif name == "paint_undo":
            self._undo_paint()
        elif name == "cli":
            return self._cli(c["code"])
        elif name == "script":
            return {"ok": True, "script": self.script_text()}
        elif name == "script_clear":
            self._record_run()  # forget the run so far too
            self.script, self._script_imports = [], []
        elif name == "refresh":
            self.refresh_targets()
            self.describe()
        else:
            raise ValueError(f"unknown command {name!r}")
        return {"ok": True}

    def _blocking(self, label, fn):
        self._did(f"world.{label}()")
        self.busy = f"{label}: busy, the page updates when finished"
        self._publish()
        t0 = time.perf_counter()
        try:
            fn()
        finally:
            self.busy = ""
        self.say(f"{label} done in {time.perf_counter() - t0:.2f} s")
        self._live_sample()
        self._update_torque()
        self._dirty = True

    def _solver_set(self, key, text):
        ts = self._ts()
        v = float(self._eval(text))
        if key == "fixdt":  # mumax3 FixDt: 0 = adaptive
            if v > 0:
                ts.adaptive_timestep = False
                ts.timestep = v
                self._did(
                    "world.timesolver.adaptive_timestep = False; "
                    f"world.timesolver.timestep = {v!r}"
                )
            else:
                ts.adaptive_timestep = True
                self._did("world.timesolver.adaptive_timestep = True")
        else:
            attr = {"maxerr": "max_error"}.get(key, key)
            setattr(ts, attr, v)
            self._did(f"world.timesolver.{attr} = {v!r}")
        self.describe()

    def _set_param(self, target, pname, text):
        obj = self.targets[target]
        q = getattr(obj, pname)
        value = self._vec(text, int(q.ncomp))
        ref = f"{self._ref(target)}.{pname}"
        if self.region < 0:
            attr = getattr(type(obj), pname)
            if attr.fset is not None:
                setattr(obj, pname, value)
            else:
                q.set(value)
            self._did(f"{ref} = {value!r}")
        else:
            q.set_in_region(self.region, value)
            self._did(f"{ref}.set_in_region({self.region}, {value!r})")
        self.describe()
        self._dirty = True

    def _set_m(self, target, config, args):
        obj = self.targets[target]
        imports = []
        if config == "Uniform":
            value = tuple(float(x) for x in self._eval(args))
            code = f"{self._ref(target)}.magnetization = {value}"
        else:
            fn = getattr(U, config)
            value = eval(f"fn{args}", {**MATH, **self.ns, "fn": fn})
            code = f"{self._ref(target)}.magnetization = {config}{args}"
            if self.ns.get(config) is not fn:  # the script did not import it
                imports.append(f"from mumaxplus.util import {config}")
        obj.magnetization = value
        self._did(code, imports)
        self._dirty = True
        self._live_sample()

    def _save_ovf(self, cfg, record=True):
        """Save the quantity of a display as an OVF file (mumax3: Save)."""
        if self.sel not in self.targets or not cfg["quant"]:
            raise ValueError("nothing to save")
        fq = getattr(self.targets[self.sel], cfg["quant"])
        if not hasattr(fq, "save_ovf"):
            raise ValueError(f"{cfg['quant']} cannot be saved as OVF")
        # the name save_ovf() picks
        count = fq._ovf_counts.get(fq.name, 0)
        file = os.path.abspath(fq.name.replace(":", "_") + f"_{count:06d}.ovf")
        fq.save_ovf()
        if record:
            self._did(f"{self._ref(self.sel)}.{cfg['quant']}.save_ovf()")
        self.say(f"saved {file}")
        return file

    def _view_autosave(self, cfg, period):
        """Save the display's quantity now and every ``period`` s (mumax3: AutoSave)."""
        cfg["autosave"] = max(float(period), 0.0)
        cfg["_next_save"] = None
        if cfg["autosave"] == 0:
            self.say(f"autosave of {cfg['quant']} off")
            return
        self.say(f"saving {cfg['quant']} every {cfg['autosave']:g} s")
        self._autosave_tick()

    def _autosave_tick(self):
        t = self._now()
        for cfg in self.views:
            period, due = cfg["autosave"], cfg["_next_save"]
            if period <= 0 or (due is not None and t < due * (1 - 1e-12)):
                continue
            try:
                self._save_ovf(cfg, record=False)
            except Exception as e:  # e.g. the quantity has gone: stop, do not repeat
                cfg["autosave"] = 0.0
                self.say(f"autosave of {cfg['quant']} stopped: {e}", "err")
                continue
            due = t if due is None else due
            while due <= t * (1 + 1e-12):
                due += period
            cfg["_next_save"] = due

    def _state_vars(self):
        """(target, attribute) of every field that evolves in time."""
        out = []
        for name, obj in self.targets.items():
            if isinstance(getattr(type(obj), "magnetization", None), property):
                out.append((name, "magnetization"))
            if ":sub" not in name and getattr(obj, "enable_elastodynamics", False):
                out += [(name, "elastic_displacement"), (name, "elastic_velocity")]
        return out

    def _checkpoint(self):
        """Keep a copy of the state in memory, to go back to it with _restore."""
        ts = self._ts()
        state = [
            (n, a, np.array(getattr(self.targets[n], a).eval()))
            for n, a in self._state_vars()
        ]
        cp = {
            "id": self._next_checkpoint,
            "time": float(ts.time),
            "steps": self.steps_done,
            "mb": sum(arr.nbytes for _, _, arr in state) / 1e6,
            "state": state,
        }
        self._next_checkpoint += 1
        self.checkpoints.append(cp)
        refs = ", ".join(f"{self._ref(n)}.{a}.eval()" for n, a, _ in state)
        self._did(f"checkpoint_{cp['id']} = (world.timesolver.time, {refs})")
        self.say(f"checkpoint {cp['id']} at t = {cp['time']:.6g} s ({cp['mb']:.3g} MB)")

    def _restore(self, cid):
        """Go back to a checkpoint: its fields, time and step count."""
        cp = next((cp for cp in self.checkpoints if cp["id"] == cid), None)
        if cp is None:
            raise ValueError(f"no checkpoint {cid!r}")
        missing = [n for n, _, _ in cp["state"] if n not in self.targets]
        if missing:
            raise ValueError(f"checkpoint {cid}: magnet(s) gone: {', '.join(missing)}")
        self._break()
        ts = self._ts()
        for n, a, arr in cp["state"]:
            setattr(self.targets[n], a, arr)
        ts.time = cp["time"]
        self.steps_done = cp["steps"]
        self._fresh = True
        refs = ", ".join(f"{self._ref(n)}.{a}" for n, a, _ in cp["state"])
        self._did(f"world.timesolver.time, {refs} = checkpoint_{cid}")
        self.table.rewind(cp["time"])
        self._live_sample()
        for cfg in self.views:  # AutoSave: save now and every period from here
            cfg["_next_save"] = None
        self._autosave_tick()
        self._update_torque()
        self._dirty = True

    # ---------------------------------------------------------------- excitations
    def _term(self, c):
        """Python source of an excitation's time term and mask, from the panel.

        ``c``: target, param, wave (sine, gauss, rect, sinc or expr), the wave's
        fields (A, f, phi, t0, sigma, tau, fc, or expr), and where (all, region,
        mask) with its value. An empty t0 is now. Returns a dict with the term's and
        the mask's source, the functions made from that same source, and the imports
        the recorded script needs.
        """
        obj = self.targets[c["target"]]
        param = getattr(obj, c["param"])
        ncomp = int(param.ncomp)
        ref = f"{self._ref(c['target'])}.{c['param']}"
        imports = ["import numpy as np"]
        env = {**self.ns, "np": np, **{n: getattr(np, n) for n in NP_NAMES}}
        if c["wave"] == "expr":
            body = str(c["expr"]).strip()
            used = {n.id for n in ast.walk(ast.parse(body)) if isinstance(n, ast.Name)}
            short = sorted(n for n in used & set(NP_NAMES) if n not in self.ns)
            if short:
                imports.append(f"from numpy import {', '.join(short)}")
        else:
            num = {}
            for k in re.findall(r"\{(\w+)\}", WAVEFORMS[c["wave"]]):
                text = str(c.get(k, "")).strip()
                num[k] = _num(
                    self._now() if k == "t0" and not text else self._eval(text)
                )
            wave = WAVEFORMS[c["wave"]].format(**num)
            amp = self._vec(c["A"], ncomp)
            amp = f"({', '.join(map(_num, amp))})" if ncomp > 1 else _num(amp)
            body = f"np.multiply({amp}, {wave})" if ncomp > 1 else f"{amp} * {wave}"
        term = f"lambda t: {body}"
        fn = eval(term, env)
        v = np.asarray(fn(self._now()), dtype=float)  # check it now
        if v.size != ncomp:
            raise ValueError(
                f"{c['param']} needs {ncomp} value(s) at a time, got {v.size}"
            )
        where, arg = c.get("where", "all"), str(c.get("whereValue", "")).strip()
        if where == "region":
            regions = f"{self._ref(c['target'])}.regions"
            mask = f"np.asarray({regions} == {int(self._eval(arg))}, float)"
        elif where == "mask":
            mask = f"lambda x, y, z: float({arg})"
        else:
            mask = None
        return {
            "ref": ref,
            "term": term,
            "fn": fn,
            "mask": mask,
            "mask_value": None if mask is None else eval(mask, env),
            "code": f"{ref}.add_time_term({term}"
            + (f", mask={mask})" if mask else ")"),
            "imports": imports,
        }

    def _excite_preview(self, c):
        """Sample the waveform of the panel over a time window, for its preview."""
        tm = self._term(c)
        t0 = self._now()
        num = lambda k, d: float(self._eval(c[k])) if str(c.get(k, "")).strip() else d
        start = num("t0", t0)
        window = {
            "sine": lambda: 5 / num("f", 1e9),
            "gauss": lambda: 8 * num("sigma", 1e-10),
            "rect": lambda: 1.4 * num("tau", 1e-10),
            "sinc": lambda: 20 / num("fc", 1e10),
        }.get(c["wave"], lambda: 1e-9)()
        if str(c.get("window", "")).strip():
            window = float(self._eval(c["window"]))
        if c["wave"] in ("gauss", "sinc"):  # centred pulses
            start -= window / 2
        elif c["wave"] == "rect":
            start -= window / 14
        ts = start + np.linspace(0, window, 401)
        vals = np.array([np.asarray(tm["fn"](t), dtype=float).reshape(-1) for t in ts])
        return {"t": ts.tolist(), "values": vals.T.tolist(), "term": tm["term"]}

    def _add_term(self, e):
        param = getattr(self.targets[e["target"]], e["param"])
        if e["mask_value"] is None:
            param.add_time_term(e["fn"])
        else:
            param.add_time_term(e["fn"], e["mask_value"])

    def _excite(self, c):
        """Add an excitation: a time term of a parameter (Parameter.add_time_term)."""
        tm = self._term(c)
        param = getattr(self.targets[c["target"]], c["param"])
        e = {
            "id": self._next_excitation,
            "target": c["target"],
            "param": c["param"],
            # the script's own time terms on this parameter (removing one of ours
            # removes those too)
            "foreign": bool(param.is_dynamic)
            and not any(
                x["target"] == c["target"] and x["param"] == c["param"]
                for x in self.excitations
            ),
            **tm,
        }
        self._next_excitation += 1
        self._add_term(e)
        e["label"] = f"{c['param']} += {tm['term'][len('lambda t: '):]}" + (
            f"  where {c['where']} {c.get('whereValue', '')}" if tm["mask"] else ""
        )
        self.excitations.append(e)
        self._did(tm["code"], tm["imports"])
        self._dirty = True

    def _excite_remove(self, eid):
        """Remove an excitation.

        mumax+ can only remove all time terms of a parameter, so the others added
        here are added again.
        """
        e = next((x for x in self.excitations if x["id"] == eid), None)
        if e is None:
            raise ValueError(f"no excitation {eid!r}")
        same = lambda x: x["target"] == e["target"] and x["param"] == e["param"]
        foreign = any(x["foreign"] for x in self.excitations if same(x))
        getattr(self.targets[e["target"]], e["param"]).remove_time_terms()
        self._did(f"{e['ref']}.remove_time_terms()")
        if foreign:
            self.say(
                f"this also removed the time terms the script added to {e['param']}"
            )
        self.excitations = [x for x in self.excitations if x is not e]
        for x in self.excitations:
            if same(x):
                x["foreign"] = False
                self._add_term(x)
                self._did(x["code"])
        self._dirty = True

    # ------------------------------------------------------------------- painting
    def _paint(self, c):
        """Paint on a field of the magnet under a display's image.

        ``c["points"]`` are positions in cells of the display's image (x right, y
        up, in its plane; cell i spans [i, i + 1)). The "brush" tool sets the quantity
        ``c["quant"]`` (m, or any parameter or variable) to ``c["value"]`` within
        ``c["radius"]`` cells of the points; "place" puts the m configuration
        ``c["config"]`` (vortex, skyrmion, ...) centred on the point, within the
        radius. Points with the same ``c["stroke"]`` make one stroke: one recorded
        piece of Python, and one undo.
        """
        with self.lock:
            meta = self.frames.get(c["view"], ({}, b""))[0]
        pts = [(float(x), float(y)) for x, y in c["points"]]
        st = self._stroke
        if st is None or st["id"] != c["stroke"]:
            st = self._new_stroke(c, meta, pts[0])
            if st is None:
                return
        pts = [(x - st["dx"], y - st["dy"]) for x, y in pts]  # cells of the magnet
        code, imports = self._paint_code(st, pts)
        env = {**self.ns, "np": np, **st["env"]}
        exec(compile(code, "<paint>", "exec"), env)  # the same code as recorded
        self._record_run()
        if st["entry"] is not None and st["entry"] == len(self.script) - 1:
            st["points"] += pts  # the stroke goes on: one piece of Python
            self.script[st["entry"]] = self._paint_code(st, st["points"])[0]
        else:
            if st["entry"] is not None:  # the run went on during the stroke
                self._paint_undo = None  # its m from before the run: no undo
            st["points"] = pts
            self.record(code, imports)
            st["entry"] = len(self.script) - 1
        if self._paint_undo is not None:
            self._paint_undo["entry"] = st["entry"]
        self._dirty = True
        self._live_sample()

    def _new_stroke(self, c, meta, first):
        """Find the magnet under the first point and remember how to undo."""
        dx = dy = dz = 0
        name = self.sel
        boxes = meta.get("boxes")
        if boxes:  # "show all magnets": the magnet the stroke starts on
            x, y, z = *first, int(c.get("layer", 0))
            on = lambda b, k, v: b[f"{k}0"] <= v < b[f"{k}0"] + b[f"n{k}"]  # noqa
            under = [b for b in boxes if on(b, "x", x) and on(b, "y", y)]
            # the one the shown slice goes through, if any (magnets overlap in a cut)
            under.sort(key=lambda b: not on(b, "z", z))
            if not under:
                return None
            b = under[0]
            name, dx, dy, dz = b["name"], b["x0"], b["y0"], b["z0"]
        obj = self.targets.get(name)
        tool = c.get("tool", "brush")
        quant = (
            "magnetization" if tool == "place" else c.get("quant") or "magnetization"
        )
        prop = getattr(type(obj), quant, None)
        if not isinstance(prop, property) or prop.fset is None:
            if quant == "magnetization":
                raise ValueError(
                    f"{name} has no magnetization to paint (a sublattice has)"
                )
            raise ValueError(f"{name} has no {quant} to paint")
        field = getattr(obj, quant)
        m = _mro(field)
        if not ("Variable" in m or ("Parameter" in m and "InterParameter" not in m)):
            raise ValueError(f"{quant} of {name} can not be painted")
        if "Parameter" in m and field.is_dynamic:  # set() would remove its time terms
            raise ValueError(
                f"{quant} of {name} has time terms, which painting would remove;"
                " remove them first, or paint before adding them"
            )
        axes = tuple(meta.get("axes", PLANES["xy"]))
        n = int(obj.grid.size[axes[2]])
        shown = int(c.get("layer", 0)) - dz  # the shown slice, in the magnet
        layer = None if c.get("all") else shown
        if layer is not None and not 0 <= layer < n:
            return None
        st = {
            "id": c["stroke"],
            "name": name,
            "ref": self._ref(name),
            "quant": quant,
            "ncomp": int(field.ncomp),
            "dx": dx,
            "dy": dy,
            "axes": axes,
            "layer": layer,
            "shown": shown,
            "radius": max(float(self._eval(c.get("radius", 1))), 0.5),
            "tool": tool,
            "points": [],
            "entry": None,
            "env": {},
        }
        if st["tool"] == "place":
            fn = getattr(U, str(c["config"]))
            st["config"], st["env"] = fn.__name__, {fn.__name__: fn}
            st["args"] = str(c.get("args", "")).strip().strip("()").strip()
            st["origin"], st["cell"] = tuple(obj.origin), self._cell_size()
        else:
            st["value"] = self._vec(c["value"], st["ncomp"])
        self._record_run()
        self._paint_undo = {
            "name": name,
            "quant": quant,
            "old": np.array(field.eval()),
            "entry": None,
        }
        self._stroke = st
        what = st.get("config") or f"{quant} = {st['value']!r}"
        self.say(f"paint {what} on {st['ref']}, radius {st['radius']:g} cells")
        return st

    def _cell_size(self):
        return tuple(float(v) for v in self.world.cellsize)

    def _paint_code(self, st, pts):
        """Python that paints ``pts`` of stroke ``st``, and the imports it needs."""
        ref, r2, (ah, av, an) = st["ref"], st["radius"] ** 2, st["axes"]
        z = ":" if st["layer"] is None else f"{st['layer']}:{st['layer'] + 1}"
        field = f"{ref}.{st['quant']}"
        # an xz or yz cut: paint a view of the array with the slices first, (c, n, v, h)
        cut = "" if an == 2 else f"np.moveaxis(_m, {3 - an}, 1)"
        lines = [f"_m = {field}.eval()"]
        imports = ["import numpy as np"]
        if st["tool"] == "place":  # the configuration centred on the point
            (x, y), o, c = pts[-1], st["origin"], st["cell"]
            pos = list(o)
            pos[ah], pos[av] = o[ah] + (x - 0.5) * c[ah], o[av] + (y - 0.5) * c[av]
            if an != 2:  # in the shown slice (xy: at the bottom layer's z, as before)
                pos[an] = o[an] + st["shown"] * c[an]
            args = ", ".join(f"{v:.6g}" for v in pos) + ")"
            args = f"({args}" + (f", {st['args']}" if st["args"] else "")
            lines.append(f"{ref}.magnetization = {st['config']}({args})")
            lines.append(f"_new = {ref}.magnetization.eval()")
            if cut:
                lines.append(f"_new = np.moveaxis(_new, {3 - an}, 1)")
            if self.ns.get(st["config"]) is not st["env"][st["config"]]:
                imports.append(f"from mumaxplus.util import {st['config']}")
            centres, value = [pts[-1]], f"_new[:, {z}, _on]"
        else:
            centres, value = pts, f"np.reshape({st['value']!r}, ({st['ncomp']}, 1, 1))"
        h, v = "xyz"[ah], "xyz"[av]
        lines += [
            f"_{v}, _{h} = np.mgrid[: _m.shape[{3 - av}], : _m.shape[{3 - ah}]] + 0.5"
            "  # cell centres",
            f"_on = np.zeros(_{h}.shape, bool)",
        ]
        if len(centres) == 1:
            ((x, y),) = centres
            lines.append(
                f"_on |= (_{h} - {x:.4g}) ** 2 + (_{v} - {y:.4g}) ** 2 <= {r2:.4g}"
            )
        else:
            cs = ", ".join(f"({x:.4g}, {y:.4g})" for x, y in centres)
            lines.append(f"for _c{h}, _c{v} in [{cs}]:")
            lines.append(
                f"    _on |= (_{h} - _c{h}) ** 2 + (_{v} - _c{v}) ** 2 <= {r2:.4g}"
            )
        lines += [f"{cut or '_m'}[:, {z}, _on] = {value}", f"{field} = _m"]
        return "\n".join(lines), imports

    def _undo_paint(self):
        """Undo the last paint stroke, if nothing changed the state since."""
        u = self._paint_undo
        self._record_run()
        if u is None or u["entry"] is None or u["entry"] != len(self.script) - 1:
            self._paint_undo = None
            raise ValueError("nothing to undo: the state changed since the last stroke")
        setattr(self.targets[u["name"]], u["quant"], u["old"])
        del self.script[u["entry"]]
        self._paint_undo = self._stroke = None
        self.say("undid the last paint stroke")
        self._dirty = True
        self._live_sample()

    # --------------------------------------------------------------------- movies
    def _movie_start(self, cfg, c):
        """Record a movie of a display: a frame every ``period`` s of simulation time.

        Frames are drawn here (plots.movie_frame) and kept as PNG files until
        _movie_stop makes the movie, so it records with or without the page.
        """
        if cfg.get("_movie"):
            raise ValueError("this display is already recording")
        period = float(self._eval(c["period"]))
        fps = float(self._eval(c.get("fps", 10)))
        fmt = c.get("format", "mp4")
        if period <= 0 or fps <= 0:
            raise ValueError("the period and the frame rate must be positive")
        if fmt not in ("mp4", "png"):
            raise ValueError(f"unknown movie format {fmt!r}")
        if fmt == "mp4" and not shutil.which("ffmpeg"):
            raise ValueError("MP4 needs ffmpeg, which is not installed: use PNG frames")
        base = cfg["quant"] + (f"_{cfg['comp']}" if cfg["comp"] != "" else "")
        stem = next(
            s
            for s in (f"{base}_movie_{i:03d}" for i in range(1000))
            if not Path(s).exists() and not Path(f"{s}.{fmt}").exists()
        )
        if fmt == "png":  # the frames are the result
            out = folder = Path(stem).resolve()
            out.mkdir()
        else:
            out = Path(f"{stem}.{fmt}").resolve()
            folder = Path(tempfile.mkdtemp(prefix="mumaxplus_movie_"))
        cfg["_movie"] = {
            "period": period,
            "fps": fps,
            "format": fmt,
            "arrows": int(c.get("arrows", 8)),
            "folder": folder,
            "out": out,
            "n": 0,
            "next": None,
        }
        self.say(f"recording {base}: a frame every {period:g} s, into {out}")
        self._movie_tick()  # the first frame: now

    def _movie_tick(self):
        """Add a frame to the movies that are due (at the step nearest their time)."""
        t, dt = self._now(), float(self._ts().timestep or 0.0)
        for cfg in self.views:
            mv = cfg.get("_movie")
            if not mv:
                continue
            tol = 0.5 * min(dt, mv["period"])  # t has rounding errors
            if mv["next"] is not None and t < mv["next"] - tol:
                continue
            try:
                self._movie_frame(cfg, mv)
            except Exception as e:  # stop recording, keep what was recorded
                self.say(f"movie of {cfg['quant']}: {type(e).__name__}: {e}", "err")
                self._movie_stop(cfg)
                continue
            due = t if mv["next"] is None else mv["next"]
            while due <= t + tol:
                due += mv["period"]
            mv["next"] = due

    def _movie_frame(self, cfg, mv):
        self._render_view(cfg)  # the display as it is now
        with self.lock:
            meta, buf = self.frames[cfg["id"]]
        n = meta["ncomp"] * meta["w"] * meta["h"]
        data = np.frombuffer(buf[: 4 * n], "<f4").reshape(
            meta["ncomp"], meta["h"], meta["w"]
        )
        rgba = np.frombuffer(buf[4 * n :], np.uint8) if meta["vector"] else None
        img = movie_frame(meta, data, rgba, mv["arrows"])
        img.save(mv["folder"] / f"{mv['n']:06d}.png")
        mv["n"] += 1

    def _movie_stop(self, cfg):
        """Stop recording and make the movie; return its file (or folder)."""
        mv = cfg.pop("_movie", None)
        if mv is None:
            raise ValueError("this display is not recording")
        if mv["format"] == "png":
            self.say(f"saved {mv['n']} frames in {mv['out']}")
            return str(mv["out"])
        try:
            if mv["n"] == 0:
                raise ValueError("no frames were recorded")
            self.busy = "making the movie"
            self._publish()
            write_movie(mv["folder"], mv["out"], mv["fps"])
        finally:
            self.busy = ""
            shutil.rmtree(mv["folder"], ignore_errors=True)
        self.outputs.add(str(mv["out"]))
        self.say(f"saved {mv['out']} ({mv['n']} frames at {mv['fps']:g} fps)")
        return str(mv["out"])

    def _table_autosave(self, period):
        period = float(period)
        self.table.period = max(period, 0.0)
        self.table.next_save = None
        if period > 0:
            self.table.clear()
        self.say(f"TableAutosave({period:g})", "in")

    # -------------------------------------------------------------------- console
    def _install_helpers(self):
        """mumax3-style functions available in the GUI console and the user's script."""

        def Run(t=None):
            self._start_run(t=None if t is None or t == float("inf") else float(t))

        def Steps(n):
            self._start_run(n=int(n))

        def Break():
            self._break()

        def Relax():
            self._blocking("relax", self.world.relax)

        def Minimize():
            self._blocking("minimize", self.world.minimize)

        def TableAdd(quantity, name=None, unit=None):
            self.table_add(quantity, name, unit)

        def TableAutosave(period):
            self._table_autosave(period)

        def TableSave():
            self.table.sample(self._now(), to_file=True)

        def SetSolver(method):
            self._set_method(method)
            self.record(f"world.timesolver.set_method({method!r})")

        helpers = [Run, Steps, Break, Relax, Minimize]
        helpers += [TableAdd, TableAutosave, TableSave, SetSolver]
        for fn in helpers:
            self.ns.setdefault(fn.__name__, fn)
        # the console lines that use them are not recorded as typed (Run, Relax, ...
        # record what they do themselves)
        self._helpers = {f.__name__ for f in helpers if self.ns[f.__name__] is f}

    def _cli(self, code):
        out = io.StringIO()
        self.say(code, "in")
        failed = False
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                try:
                    mode, tree = "eval", ast.parse(code, "<gui>", "eval")
                except SyntaxError:
                    mode, tree = "exec", ast.parse(code, "<gui>", "exec")
                self._record_run()
                at = len(self.script)  # before what Run, Relax, ... in it record
                result = eval(compile(tree, "<gui>", mode), self.ns)
                if result is not None:
                    print(repr(result))
                self._record_cli(code, tree, result, at)
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                lines = [f.lineno for f in tb if f.filename == "<gui>"]
                where = f" (line {lines[-1]})" if lines else ""
                print(f"{type(e).__name__}: {e}{where}")
                failed = True
        text = out.getvalue().rstrip()
        for line in text.splitlines():
            self.say(line, "err" if failed else "out")
        self.refresh_targets()
        self.describe()
        self._dirty = True
        return {"ok": not failed, "output": text}

    def _record_cli(self, code, tree, result, at):
        """Record a console line that worked, unless it only asked for a value."""
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        if names & self._helpers:  # Run, Relax, ... record what they do themselves
            call = tree.body if isinstance(tree, ast.Expression) else None
            lone = (  # just one call of a GUI command, like Run(1e-9)
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id in self._helpers
            )
            if not lone:
                self.script.insert(at, f"# console, uses GUI-only commands: {code}")
        elif isinstance(tree, ast.Module) or result is None:
            self.record(code)

    # --------------------------------------------------------------------- the loop
    def _steps(self, n):
        ts = self._ts()
        if self._fresh or not hasattr(ts, "_impl"):
            self._fresh = False
            ts.steps(n)  # public API: assures a sensible dt after a change
        else:
            ts._impl.steps(n)  # keep the adaptive dt between chunks (like solve())

    def _chunk(self):
        ts = self._ts()
        t0 = time.perf_counter()
        n = self._nchunk
        if self.steps_left is not None:
            n = min(n, self.steps_left)
        limits = [
            x
            for x in (
                self.run_until,
                self.table.next_save if self.table.period > 0 else None,
                *(v["_next_save"] for v in self.views if v["autosave"] > 0),
                *(v["_movie"]["next"] for v in self.views if v.get("_movie")),
            )
            if x is not None
        ]
        # land (almost) exactly on the stop / next table time,
        # overshooting by at most one step
        if limits:
            remaining = min(limits) - ts.time
            dt = ts.timestep or 0.0
            if remaining > 0 and dt > 0:
                n = max(1, min(n, int(0.5 * remaining / dt)))
        self._steps(n)
        self.steps_done += n
        self._measure()
        if n == self._nchunk:  # a whole chunk: aim for CHUNK_SECONDS of wall time
            wall = max(time.perf_counter() - t0, 1e-4)
            self._nchunk = max(1, min(int(n * CHUNK_SECONDS / wall), 2 * n))
        if self.steps_left is not None:
            self.steps_left -= n
        self._table_tick()
        self._autosave_tick()
        self._movie_tick()
        # progress and end of run, like mumax3's Run/Steps
        if self.run_until is not None:
            total = self.run_until - self._run_t0
            self.progress = (
                int(100 * min(1.0, (ts.time - self._run_t0) / total))
                if total > 0
                else 100
            )
            done = ts.time >= self.run_until * (1 - 1e-12)
        elif self.steps_left is not None:
            total = self.steps_left + (self.steps_done - self._run_n0)
            self.progress = int(100 * (self.steps_done - self._run_n0) / max(total, 1))
            done = self.steps_left <= 0
        else:
            done = False
        if done:
            self.progress = 100
            self._break()
            self._dirty = True

    def _measure(self):
        """Measure steps and simulated time per second of wall time, over PERF_WINDOW.

        The wall time includes everything the loop does (displays, table, commands),
        so this is the speed you get, not only that of the solver.
        """
        now, perf = time.perf_counter(), self._perf
        perf.append((now, self._now(), self.steps_done))
        while len(perf) > 2 and now - perf[1][0] > PERF_WINDOW:
            perf.popleft()
        (w0, t0, n0), (w1, t1, n1) = perf[0], perf[-1]
        if w1 - w0 > 0.2:  # too short a time says little
            self.perf = {
                "steps_s": (n1 - n0) / (w1 - w0),
                "sim_s": (t1 - t0) / (w1 - w0),
            }

    def _eta(self):
        """Wall time (s) left in the current Run / Steps, or None."""
        if not self.running:
            return None
        if self.run_until is not None and self.perf["sim_s"]:
            return max(self.run_until - self._now(), 0) / self.perf["sim_s"]
        if self.steps_left is not None and self.perf["steps_s"]:
            return self.steps_left / self.perf["steps_s"]
        return None

    def loop(self):
        """Drive the simulation on the calling thread until stop()."""
        self._render()
        self._publish()
        last_pub = 0.0
        while not self._stop:
            try:
                self._drain(timeout=0.0 if self.running else 0.05)
                if self.running:
                    self._chunk()
                    if time.perf_counter() - self._last_torque > TORQUE_INTERVAL:
                        self._update_torque()
                now = time.perf_counter()
                if self._dirty or (
                    self.running and now - self._last_render > RENDER_INTERVAL
                ):
                    self._render()
                if self.running or now - last_pub > 0.25:
                    self._publish()
                    last_pub = now
            except Exception as e:
                with contextlib.suppress(Exception):
                    self._record_run()
                self.running = False
                self.say(f"simulation stopped: {type(e).__name__}: {e}", "err")
                self.say(traceback.format_exc(limit=4).rstrip(), "err")
                self._publish()

    def stop(self):
        """Ask the simulation loop to stop (and end the pushed updates)."""
        with self.changed:
            self._stop = True
            self.changed.notify_all()
