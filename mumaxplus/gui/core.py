"""Simulation controller of the mumax⁺ GUI.

Everything that touches the mumax⁺ world happens on ONE thread: the thread that calls
:meth:`Controller.loop`. It advances the solver in short chunks and executes the
commands that the web page queued in between.
HTTP threads only queue commands and read published snapshots, so a GUI edit never races
with a solver step, and nothing GPU-related is called from another thread.

The Controller is put together from the mixins of the other modules, one per part of
the page (table and plots, displays, movies, excitations, painting, checkpoints,
console); this module has the run control, the commands and the loop. A command
``{"cmd": name, ...}`` from the page runs the method ``_cmd_<name>`` of any of them.
"""

from __future__ import annotations

import collections
import contextlib
from pathlib import Path
import queue
import sys
import threading
import time
import traceback

import numpy as np

import mumaxplus.util as U

from .checkpoints import CheckpointMixin
from .common import MATH, _clean, _reached
from .console import ConsoleMixin
from .describe import DescribeMixin
from .display import RENDER_INTERVAL, DisplayMixin
from .excitation import ExcitationMixin
from .movie import MovieMixin
from .paint import PaintMixin
from .table import TableMixin

DEFAULT_METHOD = "Fehlberg"  # documented default; mumax⁺ cannot report the current one
CHUNK_SECONDS = 0.05  # wall time of one simulation chunk
TORQUE_INTERVAL = 0.5
PERF_WINDOW = 2.0  # wall time over which steps/s and simulated time/s are measured
# commands that do not change the simulation: they keep the adaptive time step and the
# error shown in the page (any other command resets both)
NO_RESET = {
    "break",
    "render",
    "table_using",
    "region",
    "select",
    "table_autosave",
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
    "complete",
    "script",
    "script_clear",
    "save_ovf",
    "view_autosave",
    "checkpoint",
    "checkpoint_remove",
}


class Controller(
    TableMixin,
    DisplayMixin,
    MovieMixin,
    DescribeMixin,
    ExcitationMixin,
    PaintMixin,
    CheckpointMixin,
    ConsoleMixin,
):
    """Drives a mumax⁺ world for the page (see the module docstring)."""

    def __init__(self, world, namespace, title=None):
        self.world = world
        self.lock = threading.RLock()  # Protect data shared between threads
        # command queue from HTTP threads to sim threads
        self.cmds: "queue.Queue" = queue.Queue()
        self._stop = False
        self._loop_thread = None  # the thread in loop(), if any
        self.title = title or (
            Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else "mumax⁺"
        )
        self.ns = namespace
        self.ns.setdefault("world", world)

        # run state (pause / Run / Steps / Break)
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
        self.region_note = ""  # why the region typed in the page was not used
        self.snap = {}
        self.snap_id = 0  # goes up when the snapshot changes
        self.changed = threading.Condition(self.lock)  # used to wake waiting pages
        self.log = []
        self.log_id = 0

        # performance: (wall time, simulation time, step) at the end of recent chunks
        self._perf = collections.deque()
        self.perf = {"steps_s": None, "sim_s": None}

        # activate every mixin class
        self._init_script()
        self._init_describe()
        self._init_display()
        self._init_table()
        self._init_movies()
        self._init_excitations()
        self._init_paint()
        self._init_checkpoints()
        self._install_helpers()
        self.refresh_targets()  # Lists magnets and sublattices
        self.sel = next(iter(self.targets), None)  # Select the first magnet
        self.add_view()  # and create the first display
        self.describe()  # Build the page description

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

    # -------------------------------------------------------------------- targets
    def refresh_targets(self):
        """Update the selectable magnets from the world."""
        targets = {}
        for name, magnet in self.world.magnets.items():
            targets[name] = magnet
            subs = getattr(type(magnet), "sublattices", None)
            if isinstance(subs, property):
                try:
                    for i, sub in enumerate(magnet.sublattices, start=1):
                        targets[f"{name}:sub{i}"] = sub
                except Exception:
                    pass
        if self.ns.get("magnets", self.targets) is self.targets:  # not the script's own
            self.ns["magnets"] = targets
        self.targets = targets

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

    # ------------------------------------------------------------------ snapshots
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
                "region_note": self.region_note,
                "views": [self._view_state(cfg) for cfg in self.views],
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

    def call(self, fn):
        """Run ``fn`` on the simulation thread: now if this is it or no loop runs."""
        thread = self._loop_thread
        if thread is None or thread is threading.current_thread():
            return fn()
        r = self.submit({"cmd": "call", "fn": fn}, wait=60.0)
        if r.get("ok") is False:
            raise RuntimeError(r.get("error"))
        return r.get("value")

    def _drain(self, timeout=0.0):
        try:
            item = self.cmds.get(timeout=timeout) if timeout else self.cmds.get_nowait()
        except queue.Empty:
            return False
        while True:
            cmd, box, done = item
            try:
                box["result"] = self._execute(cmd)
            except Exception as e:  # a bad command must never kill the loop
                self.say(f"{type(e).__name__}: {e}", "err")
                box["result"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            self._publish()
            done.set()
            try:
                item = self.cmds.get_nowait()
            except queue.Empty:
                return True

    # numbers typed in the page are expressions
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

    def _execute(self, c):
        name = c["cmd"]
        handler = getattr(self, f"_cmd_{name}", None)
        if handler is None:
            raise ValueError(f"unknown command {name!r}")
        if name not in NO_RESET:
            self._fresh = True
            self.last_error = ""
        return handler(c) or {"ok": True}

    def _cmd_run(self, c):
        txt = str(c["text"]).strip()
        if txt.lower() in ("", "inf", "forever", "indefinite", "0"):
            self.say("Run()  # until Break", "in")
            self._start_run()
        else:
            self.say(f"Run({txt})", "in")
            self._start_run(t=self._eval(txt))

    def _cmd_steps(self, c):
        self.say(f"Steps({c['text']})", "in")
        self._start_run(n=int(self._eval(c["text"])))

    def _cmd_break(self, c):
        self._break()

    def _cmd_relax(self, c):
        self._break()
        self._blocking("relax", self.world.relax)

    def _cmd_minimize(self, c):
        self._break()
        self._blocking("minimize", self.world.minimize)

    def _cmd_solver_type(self, c):
        self._set_method(c["value"])
        self._did(f"world.timesolver.set_method({c['value']!r})")

    def _cmd_solver_set(self, c):
        self._solver_set(c["name"], c["text"])

    def _cmd_param(self, c):
        self._set_param(c["target"], c["name"], c["text"])

    def _cmd_switch(self, c):
        obj = self.targets[c["target"]]
        value = c["value"] if c["kind"] == "bool" else self._eval(c["value"])
        setattr(obj, c["name"], value)
        self._did(f"{self._ref(c['target'])}.{c['name']} = {value!r}")
        self.describe()

    def _cmd_world_set(self, c):
        vector = isinstance(getattr(self.world, c["name"]), (tuple, list))
        value = self._vec(c["text"], 3) if vector else self._eval(c["text"])
        setattr(self.world, c["name"], value)
        self._did(f"world.{c['name']} = {value!r}")
        self.describe()

    def _cmd_set_pbc(self, c):
        value = tuple(int(self._eval(v)) for v in c["value"])
        if any(value):
            self.world.set_pbc(value)  # also fits mastergrid to the magnets
            self._did(f"world.set_pbc({value})")
        else:
            self.world.unset_pbc()
            self._did("world.unset_pbc()")
        self.refresh_targets()
        self.describe()

    def _cmd_setm(self, c):
        self._set_m(c["target"], c["config"], c["args"], c["field"])

    def _region_number(self, target, text, whole_magnet=False):
        """Return the region typed in the page, if the magnet has it; else raise.

        With ``whole_magnet``, -1 is allowed too. A region must have cells inside the
        geometry, as a value set or an excitation added elsewhere would do nothing.
        """
        value = self._eval(text)
        number = int(value)
        if number != value:
            raise ValueError(f"a region is a whole number, not {value}")
        if whole_magnet and number == -1:
            return number
        obj = self.targets[target]
        inside = np.asarray(obj.geometry, dtype=bool)
        regions = np.unique(np.asarray(obj.regions)[inside]).tolist()
        if number not in regions:
            listed = ", ".join(map(str, regions[:20])) + ", ..." * (len(regions) > 20)
            has = f"it has {listed}" if regions else "it has none"
            raise ValueError(f"{target} has no region {number}: {has}")
        return number

    def _cmd_region(self, c):
        try:
            self.region = self._region_number(self.sel, c["value"], whole_magnet=True)
        except Exception as e:
            self.region_note = str(e)
            raise
        if self.region_note:  # the refusal is over, also at the top of the page
            self.region_note = self.last_error = ""
        self.describe()

    def _cmd_select(self, c):
        if c["target"] in self.targets:
            self.sel = c["target"]
            quantities = self.desc["targets"][self.sel]["quantities"]
            names = [q["name"] for q in quantities]
            for cfg in self.views:
                if cfg["quant"] not in names:
                    cfg["quant"], cfg["comp"] = self._default_quantity(), ""
            self.region_note = ""
            try:
                self._region_number(self.sel, str(self.region), whole_magnet=True)
            except ValueError as e:  # the previous magnet had it
                self.region = -1
                self.region_note = f"{e}. Using the whole magnet."
                self.say(self.region_note, "err")
            self.describe()
            self._dirty = True

    def _cmd_call(self, c):
        """Run a method another thread calls while the loop runs (Controller.call)."""
        return {"ok": True, "value": c["fn"]()}

    # ----------------------------------------------------------------- run control
    def _start_run(self, t=None, n=None):
        """Run for a time t, n steps, or (neither) until Break."""
        self._break()
        if (t is not None and t <= 0) or (n is not None and n <= 0):
            return  # nothing to do; _chunk always takes at least one step
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

    def _blocking(self, label, fn):
        self._did(f"world.{label}()")
        self.busy = f"{label}: busy, the page updates when finished"
        self._publish()
        t0 = time.perf_counter()
        try:
            fn()
        finally:
            self.busy = ""
            self._fresh = True
        self.say(f"{label} done in {time.perf_counter() - t0:.2f} s")
        self._live_sample()
        self._update_torque()
        self._dirty = True

    def _solver_set(self, key, text):
        ts = self._ts()
        v = float(self._eval(text))
        if key == "fixdt":  # 0 = adaptive
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
            foreign = self._foreign_terms(target, pname)
            attr = getattr(type(obj), pname)
            if attr.fset is not None:
                setattr(obj, pname, value)
            else:
                q.set(value)
            self._did(f"{ref} = {value!r}")
            self._terms_removed(target, pname, foreign)  # Parameter.set removed them
        else:
            q.set_in_region(self.region, value)
            self._did(f"{ref}.set_in_region({self.region}, {value!r})")
        self.describe()
        self._dirty = True

    def _set_m(self, target, config, args, field):
        """Set the magnetization (or the elastic displacement) to a configuration."""
        if field not in ("magnetization", "elastic_displacement"):
            raise ValueError(f"can not set {field}")
        obj = self.targets[target]
        if field == "elastic_displacement" and not getattr(
            obj, "enable_elastodynamics", 0
        ):
            raise ValueError(f"enable_elastodynamics of {target} first")
        imports = []
        if config == "Uniform":
            value = tuple(float(x) for x in self._eval(args))
            code = f"{self._ref(target)}.{field} = {value}"
        else:
            fn = getattr(U, config)
            value = eval(f"fn{args}", {**MATH, **self.ns, "fn": fn})
            code = f"{self._ref(target)}.{field} = {config}{args}"
            if self.ns.get(config) is not fn:  # the script did not import it
                imports.append(f"from mumaxplus.util import {config}")
        setattr(obj, field, value)
        self._did(code, imports)
        self._dirty = True
        self._live_sample()

    # --------------------------------------------------------------------- the loop
    def _steps(self, n):
        ts = self._ts()
        if self._fresh:
            self._fresh = False
            ts.steps(n)  # public API: assures a sensible dt after a change
        else:
            ts._impl.steps(n)  # keep the adaptive dt between chunks (like solve())

    def _chunk(self):
        ts = self._ts()
        t0 = time.perf_counter()
        if self._fresh:  # after a change: a sensible dt, before looking at it below
            self._fresh = False
            ts._assure_sensible_timestep()
        if self.table.period > 0 and self.table.next_save is None:
            self._table_tick()  # sets the next save time: this chunk must land on it
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
        # land on the stop / next save time: approach it, then one step of exactly the
        # time left (as TimeSolver.run does); the next steps go on with the dt from
        # before
        dt_before = None
        if limits and not _reached(ts.time, min(limits)):
            remaining = min(limits) - ts.time
            dt = ts.timestep or 0.0
            if dt > 0 and remaining <= dt:
                n, dt_before = 1, dt
                ts.timestep = remaining
            elif dt > 0:  # (remaining can be inf)
                n = max(1, int(min(n, 0.5 * remaining / dt)))
        self._steps(n)
        if dt_before is not None:
            ts.timestep = dt_before
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
        # progress and end of run of Run/Steps
        if self.run_until is not None:
            total = self.run_until - self._run_t0
            self.progress = (
                int(100 * min(1.0, (ts.time - self._run_t0) / total))
                if total > 0
                else 100
            )
            done = _reached(ts.time, self.run_until)
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
        self._loop_thread = threading.current_thread()
        try:
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
        finally:
            self._finish_movies()
            self._loop_thread = None

    def stop(self):
        """Ask the simulation loop to stop (and end the pushed updates)."""
        with self.changed:
            self._stop = True
            self.changed.notify_all()
