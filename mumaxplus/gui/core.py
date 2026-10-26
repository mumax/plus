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

import contextlib
import inspect
import io
import math
from pathlib import Path
import queue
import re
import sys
import threading
import time
import traceback

import numpy as np

SOLVER_METHODS = ["Heun", "BogackiShampine", "CashKarp", "Fehlberg", "DormandPrince"]
DEFAULT_METHOD = "Fehlberg"  # documented default; mumax+ cannot report the current one
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
NO_RESET = {
    "break",
    "render",
    "refresh",
    "table_using",
    "region",
    "log_only",
    "view_add",
    "view_remove",
    "plot_add",
    "plot_remove",
    "call",
}


# ----------------------------------------------------------------- small helpers
def _mro(obj):
    return {c.__name__ for c in type(obj).__mro__}


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


def _fmt(v):
    """Short number format (float32-ish precision, like mumax3's float32(v))."""
    if isinstance(v, (tuple, list, np.ndarray)):
        return "(" + ", ".join(_fmt(x) for x in v) + ")"
    return f"{float(v):.7g}"


class Table:
    """The mumax3 table: columns (t, mx, my, mz, ...) sampled during the run."""

    def __init__(self, path="table.txt"):
        self.path = path
        self.cols = []  # (name, unit, fn) ; fn() -> float
        self.rows = []
        self.period = 0.0  # TableAutosave: 0 = live preview only (not saved)
        self.next_save = None
        self._header_written = False

    def names(self):
        """Return the column names, starting with t."""
        return ["t"] + [c[0] for c in self.cols]

    def clear(self):
        """Remove all rows; the header is written again on the next save."""
        self.rows.clear()
        self._header_written = False

    def add(self, name, unit, fn):
        """Add a column; return False if the name is already taken."""
        if any(c[0] == name for c in self.cols):
            return False
        self.cols.append((name, unit, fn))
        self.clear()
        return True

    def sample(self, t, to_file):
        """Append a row at time t, and write it to the table file if to_file."""
        row = [float(t)]
        for _, _, fn in self.cols:
            try:
                row.append(float(fn()))
            except Exception:
                row.append(float("nan"))
        self.rows.append(row)
        if len(self.rows) > TABLE_MAX_ROWS:
            self.rows[:] = self.rows[::2]
        if to_file:
            with open(self.path, "a") as f:
                if not self._header_written:
                    head = ["t (s)"] + [f"{n} ({u})" for n, u, _ in self.cols]
                    f.write("# " + "\t".join(head) + "\n")
                    self._header_written = True
                f.write("\t".join(f"{v:.8g}" for v in row) + "\n")

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
        self._fresh = (
            True  # TimeSolver.steps resets the adaptive dt; only wanted after a change
        )
        self.maxtorque = None
        self._last_torque = 0.0
        self.last_error = ""

        # what the page shows
        self.targets = {}
        self.sel = None  # selected magnet (display, parameters, mesh, initial m)
        self.region = -1
        self.views = (
            []
        )  # one dict per display panel: {"id", "quant", "comp", "layer", "all"}
        self._next_view = 0
        self.table = Table(table_file)
        self.plots = [
            {"id": 0, "x": "1", "y": "2"}
        ]  # one dict per plot panel: plot "table.txt" using x:y
        self._next_plot = 1
        self.desc = {}
        self.desc_version = 0
        self.frame_id = 0
        self.frames = {}  # view id -> (meta, bytes)
        self.snap = {}
        self.log = []
        self.log_id = 0
        self._dirty = True
        self._last_render = self._last_live = 0.0

        self._install_helpers()
        self.refresh_targets()
        self._default_table()
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
        for k, v in self.ns.items():
            if v is obj and not k.startswith("_"):
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
        keys = ("quant", "comp", "layer", "all")
        base = (
            {k: self.views[-1][k] for k in keys}
            if self.views
            else {
                "quant": self._default_quantity(),
                "comp": "",
                "layer": 0,
                "all": False,
            }
        )
        base.update({k: v for k, v in cfg.items() if k in keys})
        base["id"] = self._next_view
        self._next_view += 1
        self.views.append(base)
        self._dirty = True
        return base["id"]

    def add_plot(self, x=None, y=None):
        """Add a plot panel of table columns x:y (numbers or names).

        It starts as a copy of the last one.
        """
        p = dict(self.plots[-1], id=self._next_plot)
        self._next_plot += 1
        if x is not None:
            p["x"] = str(x)
        if y is not None:
            p["y"] = str(y)
        self.plots.append(p)
        return p["id"]

    def _view(self, vid):
        return next((v for v in self.views if v["id"] == vid), None)

    def _view_state(self, cfg):
        """Return what the page needs of a view: its settings and latest frame id."""
        frame = self.frames.get(cfg["id"])
        state = {k: v for k, v in cfg.items() if not k.startswith("_")}
        state["frame_id"] = frame[0]["id"] if frame else -1
        return state

    # --------------------------------------------------------------- introspection
    def _inspect(self, name, obj):
        info = {"quantities": [], "params": [], "switches": [], "doc": ""}
        cls = type(obj)
        for pname in sorted(dir(cls), key=str.lower):
            if pname.startswith("_"):
                continue
            prop = getattr(cls, pname, None)
            if not isinstance(prop, property):
                continue
            try:
                val = getattr(obj, pname)
            except Exception:
                continue
            m = _mro(val)
            doc = _doc(prop)
            try:
                if "Parameter" in m and "InterParameter" not in m:
                    info["params"].append(
                        {
                            "name": pname,
                            "ncomp": int(val.ncomp),
                            "unit": val.unit,
                            "doc": doc,
                        }
                    )
                    info["quantities"].append(
                        {
                            "name": pname,
                            "ncomp": int(val.ncomp),
                            "unit": val.unit,
                            "doc": doc,
                        }
                    )
                elif "FieldQuantity" in m or "Variable" in m:
                    info["quantities"].append(
                        {
                            "name": pname,
                            "ncomp": int(val.ncomp),
                            "unit": val.unit,
                            "doc": doc,
                        }
                    )
                elif isinstance(val, bool) and prop.fset is not None:
                    info["switches"].append(
                        {"name": pname, "kind": "bool", "value": val, "doc": doc}
                    )
                elif (
                    isinstance(val, (int, float))
                    and not isinstance(val, bool)
                    and prop.fset is not None
                ):
                    info["switches"].append(
                        {"name": pname, "kind": "num", "value": val, "doc": doc}
                    )
            except Exception:
                continue
        try:
            g = obj.grid.size
            c = obj.cellsize
            geom = np.asarray(obj.geometry)
            info["mesh"] = {
                "n": [int(v) for v in g],
                "c": [float(v) for v in c],
                "inside": float(geom.mean()),
            }
        except Exception:
            info["mesh"] = None
        return info

    def _param_values(self, name, obj, params):
        out = {}
        for p in params:
            try:
                q = getattr(obj, p["name"])
                if self.region < 0:
                    out[p["name"]] = _clean(q.uniform_value) if q.is_uniform else None
                else:
                    inside = np.asarray(obj.regions) == self.region
                    geom = np.asarray(obj.geometry, dtype=bool)
                    idx = np.argwhere(
                        inside & geom
                    )  # a cell that is really part of the magnet
                    if not len(idx):
                        idx = np.argwhere(inside)
                    if not len(idx):
                        out[p["name"]] = None
                        continue
                    iz, iy, ix = idx[0]
                    v = np.asarray(q.eval())[:, iz, iy, ix]
                    out[p["name"]] = (
                        float(v[0]) if p["ncomp"] == 1 else [float(x) for x in v]
                    )
            except Exception:
                out[p["name"]] = None
        return out

    def _config_info(self):
        out = [{"name": "Uniform", "doc": "Uniform magnetization", "args": "(1, 0, 0)"}]
        try:
            import mumaxplus.util as U
        except Exception:
            return out
        c = self._center()
        cell = self._cell()
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
            try:
                args = ", ".join(
                    defaults.get(p, "1") for p in inspect.signature(fn).parameters
                )
            except Exception:
                args = ""
            out.append({"name": n, "doc": _doc(fn), "args": f"({args})"})
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
        targets = {n: self._inspect(n, o) for n, o in self.targets.items()}
        for n, info in targets.items():
            info["values"] = self._param_values(n, self.targets[n], info["params"])
        ts = self._ts()
        solver = {}
        for key in (
            "timestep",
            "adaptive_timestep",
            "max_error",
            "headroom",
            "lower_bound",
            "upper_bound",
            "sensible_factor",
            "sensible_timestep_default",
        ):
            try:
                solver[key] = _clean(getattr(ts, key))
            except Exception:
                pass
        world_info = {"vectors": [], "numbers": [], "pbc": None}
        wcls = type(self.world)
        for pname in sorted(dir(wcls), key=str.lower):
            prop = getattr(wcls, pname, None)
            if (
                not isinstance(prop, property)
                or pname.startswith("_")
                or prop.fset is None
            ):
                continue
            try:
                v = getattr(self.world, pname)
            except Exception:
                continue
            if pname == "pbc_repetitions":
                world_info["pbc"] = [int(x) for x in v]
            elif (
                isinstance(v, (tuple, list))
                and len(v) == 3
                and all(isinstance(x, (int, float)) for x in v)
            ):
                world_info["vectors"].append(
                    {"name": pname, "value": [float(x) for x in v], "doc": _doc(prop)}
                )
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                world_info["numbers"].append(
                    {"name": pname, "value": float(v), "doc": _doc(prop)}
                )
        try:
            import mumaxplus as mx

            version = (
                f"mumax⁺ {getattr(mx, '__version__', '')}  ·  "
                f"{getattr(mx, 'FP_PRECISION', '')} precision"
            )
        except Exception:
            version = "mumax⁺"
        desc = {
            "title": self.title,
            "version": version,
            "targets": targets,
            "world": world_info,
            "solver": solver,
            "methods": SOLVER_METHODS,
            "configs": self._config_info(),
            "table_columns": self.table.names(),
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
        ncomp, nz, ny, nx = arr.shape
        layer = min(max(int(cfg["layer"]), 0), nz - 1)
        sl = arr[:, layer]  # (ncomp, ny, nx)
        comp = cfg["comp"]
        vector = ncomp == 3 and comp == ""
        rgba = b""
        if vector:
            from .plots import vector_rgba

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
            "nz": nz,
            "quant": cfg["quant"],
            "comp": comp,
            "vmin": float(finite.min()) if finite.size else 0.0,
            "vmax": float(finite.max()) if finite.size else 0.0,
            "unit": getattr(fq, "unit", ""),
            "fullnx": nx,
            "fullny": ny,
            "time": float(self._ts().time),
            "boxes": boxes,
            "origin": lo,  # boxes: one per magnet in "show all" mode, in cells
        }
        with self.lock:
            self.frame_id += 1
            meta["id"] = self.frame_id
            meta["view"] = cfg["id"]
            self.frames[cfg["id"]] = (_clean(meta), data.tobytes() + rgba)

    # ----------------------------------------------------------------------- table
    def _default_table(self):
        """mumax3's default table: t, mx, my, mz."""
        prim = self._primary()
        for name in prim:
            try:
                m = self.targets[name].magnetization
            except Exception:
                continue
            for i, comp in enumerate("xyz"):
                col = f"m{comp}" if len(prim) == 1 else f"{name}.m{comp}"
                self.table.add(
                    col, "", (lambda m=m, i=i: np.atleast_1d(m.average())[i])
                )

    def table_add(self, quantity, name=None, unit=None):
        """TableAdd(quantity).

        quantity is a name (of the selected magnet), a FieldQuantity (its average is
        added, per component), a ScalarQuantity, or any function without arguments
        that returns a number (e.g. a domain wall position).
        """
        obj = (
            getattr(self.targets[self.sel], quantity)
            if isinstance(quantity, str)
            else quantity
        )
        m = _mro(obj)
        if name is None:
            if isinstance(quantity, str):
                name = quantity
            elif "FieldQuantity" in m or "Variable" in m or "ScalarQuantity" in m:
                name = getattr(obj, "name", "quantity").replace(":", "_")
            else:
                name = getattr(obj, "__name__", "quantity")
        unit = getattr(obj, "unit", "") if unit is None else unit
        field = "FieldQuantity" in m or "Variable" in m
        n = int(obj.ncomp) if field else 1
        cols = [
            name if n == 1 else name + ("xyz"[i] if n == 3 else str(i))
            for i in range(n)
        ]
        taken = [col for col in cols if col in self.table.names()]
        if taken:
            raise ValueError(
                f"table already has a column {taken[0]!r}; "
                "choose another one with name=..."
            )
        if field:
            for i, col in enumerate(cols):
                self.table.add(
                    col, unit, (lambda o=obj, i=i: np.atleast_1d(o.average())[i])
                )
        elif "ScalarQuantity" in m:
            self.table.add(name, unit, lambda o=obj: o.eval())
        elif callable(obj):
            self.table.add(name, unit, obj)
        else:
            raise TypeError(
                f"cannot add {obj!r} to the table: "
                "give a quantity or a function returning a number"
            )
        cols = ", ".join(f"{i + 1}={n}" for i, n in enumerate(self.table.names()))
        self.say(f"table columns: {cols}")
        self.describe()

    def _live_sample(self):
        """Take a live-preview row (only when TableAutosave is 0).

        Never writes table.txt.
        """
        if self.table.period == 0:
            self._table_tick(force=True)

    def _table_tick(self, force=False):
        t = float(self._ts().time)
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

    def _publish(self):
        ts = self._ts()
        with self.lock:
            self.snap = _clean(
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
                    "error": self.last_error,
                    "targets": list(self.targets),
                }
            )

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
        env = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        env.update(self.ns)
        return eval(str(text), env)

    def _vec(self, text, ncomp):
        v = self._eval(text)
        if ncomp == 1:
            return float(v)
        v = tuple(float(x) for x in v)
        if len(v) != ncomp:
            raise ValueError(f"expected {ncomp} components, got {len(v)}")
        return v

    def _start_run(self, t=None, n=None):
        ts = self._ts()
        self.running = True
        self._run_t0, self._run_n0 = float(ts.time), self.steps_done
        self.run_until = float(ts.time) + float(t) if t is not None else None
        self.steps_left = int(n) if n is not None else None
        self.progress = 0
        self._fresh = True

    def _break(self):
        self.running = False
        self.run_until = self.steps_left = None
        self._live_sample()
        self._update_torque()

    def _execute(self, c):
        name = c["cmd"]
        ts = self._ts()
        if name not in NO_RESET:
            self._fresh = True
            self.last_error = ""
        if name == "run":
            self._break()
            txt = str(c["text"]).strip()
            if txt.lower() in (
                "",
                "inf",
                "forever",
                "indefinite",
                "0",
            ):  # run until Break
                self.say("Run()  # until Break", "in")
                self._start_run()
            else:
                self.say(f"Run({txt})", "in")
                self._start_run(t=self._eval(txt))
        elif name == "steps":
            self._break()
            self.say(f"Steps({c['text']})", "in")
            self._start_run(n=int(self._eval(c["text"])))
        elif name == "break":
            self._break()
        elif name in ("relax", "minimize"):
            self._break()
            self.say(f"world.{name}()", "in")
            self._blocking(name, lambda: getattr(self.world, name)())
        elif name == "solver_type":
            ts.set_method(c["value"])
            self.method, self.method_known = c["value"], True
            self.say(f"world.timesolver.set_method({c['value']!r})", "in")
        elif name == "solver_set":
            self._solver_set(c["name"], c["text"])
        elif name == "param":
            self._set_param(c["target"], c["name"], c["text"])
        elif name == "switch":
            obj = self.targets[c["target"]]
            value = c["value"] if c.get("kind") == "bool" else self._eval(c["value"])
            setattr(obj, c["name"], value)
            self.say(f"{self._ref(c['target'])}.{c['name']} = {value}", "in")
            self.describe()
        elif name == "world_set":
            cur = getattr(self.world, c["name"])
            value = (
                self._vec(c["text"], 3)
                if isinstance(cur, (tuple, list))
                else self._eval(c["text"])
            )
            setattr(self.world, c["name"], value)
            self.say(f"world.{c['name']} = {value}", "in")
            self.describe()
        elif name == "set_pbc":
            value = tuple(int(self._eval(v)) for v in c["value"])
            if any(value):
                self.world.set_pbc(value)  # also fits mastergrid to the magnets
                self.say(f"world.set_pbc({value})", "in")
            else:
                self.world.unset_pbc()
                self.say("world.unset_pbc()", "in")
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
                names = [
                    q["name"] for q in self.desc["targets"][self.sel]["quantities"]
                ]
                for cfg in self.views:
                    if cfg["quant"] not in names:
                        cfg["quant"], cfg["comp"] = self._default_quantity(), ""
                self.describe()
                self._dirty = True
        elif name == "render":
            cfg = self._view(c.get("view", self.views[0]["id"] if self.views else None))
            if cfg is None:
                raise ValueError(f"no display {c.get('view')!r}")
            for k in ("quant", "comp", "layer", "all"):
                if k in c:
                    cfg[k] = c[k]
            self._dirty = True
        elif name == "view_add":
            return {"ok": True, "view": self.add_view()}
        elif name == "view_remove":
            if len(self.views) > 1:
                self.views = [v for v in self.views if v["id"] != c["view"]]
                with self.lock:
                    self.frames.pop(c["view"], None)
        elif name == "table_autosave":
            self._table_autosave(self._eval(c["text"]))
        elif name == "table_using":
            pid = c.get("plot", self.plots[0]["id"])
            p = next((p for p in self.plots if p["id"] == pid), None)
            if p is None:
                raise ValueError(f"no plot {pid!r}")
            p["x"], p["y"] = str(c["x"]), str(c["y"])
        elif name == "plot_add":
            return {"ok": True, "plot": self.add_plot()}
        elif (
            name == "call"
        ):  # a function from the script (GUI methods called while the loop runs)
            return {"ok": True, "value": c["fn"]()}
        elif name == "plot_remove":
            if len(self.plots) > 1:
                self.plots = [p for p in self.plots if p["id"] != c["plot"]]
        elif name == "cli":
            return self._cli(c["code"])
        elif name == "refresh":
            self.refresh_targets()
            self.describe()
        else:
            raise ValueError(f"unknown command {name!r}")
        return {"ok": True}

    def _blocking(self, label, fn):
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
            else:
                ts.adaptive_timestep = True
            self.say(
                f"world.timesolver.adaptive_timestep = {v == 0}"
                + (f"; timestep = {v}" if v else ""),
                "in",
            )
        else:
            attr = {"maxerr": "max_error"}.get(key, key)
            setattr(ts, attr, v)
            self.say(f"world.timesolver.{attr} = {v}", "in")
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
            self.say(f"{ref} = {value}", "in")
        else:
            q.set_in_region(self.region, value)
            self.say(f"{ref}.set_in_region({self.region}, {value})", "in")
        self.describe()
        self._dirty = True

    def _set_m(self, target, config, args):
        obj = self.targets[target]
        if config == "Uniform":
            value = tuple(float(x) for x in self._eval(args))
            code = f"{self._ref(target)}.magnetization = {value}"
        else:
            import mumaxplus.util as U

            fn = getattr(U, config)
            value = eval(
                f"fn{args}",
                {
                    **{k: getattr(math, k) for k in dir(math) if not k.startswith("_")},
                    **self.ns,
                    "fn": fn,
                },
            )
            code = f"{self._ref(target)}.magnetization = {config}{args}"
        obj.magnetization = value
        self.say(code, "in")
        self._dirty = True
        self._live_sample()

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
            self._break()
            self._start_run(t=None if t is None or t == float("inf") else float(t))

        def Steps(n):
            self._break()
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
            self.table.sample(float(self._ts().time), to_file=True)

        def SetSolver(method):
            self._ts().set_method(method)
            self.method, self.method_known = method, True

        for fn in (
            Run,
            Steps,
            Break,
            Relax,
            Minimize,
            TableAdd,
            TableAutosave,
            TableSave,
            SetSolver,
        ):
            self.ns.setdefault(fn.__name__, fn)

    def _cli(self, code):
        out = io.StringIO()
        self.say(code, "in")
        failed = False
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                try:
                    result = eval(compile(code, "<gui>", "eval"), self.ns)
                    if result is not None:
                        print(repr(result))
                except SyntaxError:
                    exec(compile(code, "<gui>", "exec"), self.ns)
            except Exception as e:
                tb = [
                    f
                    for f in traceback.extract_tb(e.__traceback__)
                    if f.filename == "<gui>"
                ]
                print(
                    f"{type(e).__name__}: {e}"
                    + (f" (line {tb[-1].lineno})" if tb else "")
                )
                failed = True
        text = out.getvalue().rstrip()
        for line in text.splitlines():
            self.say(line, "err" if failed else "out")
        self.refresh_targets()
        self.describe()
        self._dirty = True
        return {"ok": not failed, "output": text}

    # --------------------------------------------------------------------- the loop
    def _call(self, name, arg):
        ts = self._ts()
        if self._fresh or not hasattr(ts, "_impl"):
            self._fresh = False
            return getattr(ts, name)(
                arg
            )  # public API: assures a sensible dt after a change
        return getattr(ts._impl, name)(
            arg
        )  # keep the adaptive dt between chunks (like TimeSolver.solve)

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
        self._call("steps", n)
        self.steps_done += n
        wall = max(time.perf_counter() - t0, 1e-4)
        self._nchunk = max(
            1,
            min(
                int(n * CHUNK_SECONDS / wall) if n == self._nchunk else self._nchunk,
                self._nchunk * 2,
            ),
        )
        if self.steps_left is not None:
            self.steps_left -= n
        self._table_tick()
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
                self.running = False
                self.say(f"simulation stopped: {type(e).__name__}: {e}", "err")
                self.say(traceback.format_exc(limit=4).rstrip(), "err")
                self._publish()

    def stop(self):
        """Ask the simulation loop to stop."""
        self._stop = True
