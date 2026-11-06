"""The table (TableAdd, TableAutosave) and the plots of its columns."""

from __future__ import annotations

import math
import re
import threading
import time

import numpy as np

from .common import MATH, TIME_TOLERANCE, _col_name, _mro, _next_due, _reached

LIVE_SAMPLE_INTERVAL = 0.15  # live table sampling (wall time) when TableAutosave is 0

TABLE_MAX_ROWS = 20000


class Table:
    """The table: t and the columns the user added, sampled during the run.

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
        self._file_started = False  # the first row replaces the file of an earlier run
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
            with open(self.path, "a" if self._file_started else "w") as f:
                self._file_started = True
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
        with self.lock:  # columns and rows change together; rows change in place
            rows = [r[:] for r in self.rows]
            idx = list(range(len(self.cols) + 1))
            header = self._header(idx)
        return header + "".join(self._line(r, idx) for r in rows)

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


class TableMixin:
    """The Controller's table and plot panels."""

    def _init_table(self):
        self.table = Table()
        # one dict per plot panel: table columns x against y1,y2,... (names)
        self.plots = [{"id": 0, "x": "t", "y": "", "logx": False, "logy": False}]
        self._next_plot = 1
        self._last_live = 0.0

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
        name, expr, unit = (str(c[k]).strip() for k in ("name", "expr", "unit"))
        if not re.fullmatch(r"[A-Za-z_][\w.]*", name):
            raise ValueError("give the column a name: letters, digits, _ and .")
        code = compile(expr, "<table column>", "eval")

        def fn():
            return eval(code, {**MATH, "np": np, **self.ns})

        float(np.asarray(fn()))  # an error now, rather than NaN in every row
        self.table_add(fn, name, unit)
        p = self._plot(c["plot"])
        self._plot_set(p, y=[*p["y"].split(","), name])

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
            if tb.next_save is None:  # the first multiple of the period from now on
                k = math.ceil(t / tb.period * (1 - TIME_TOLERANCE))
                tb.next_save = k * tb.period
                if tb.rows and _reached(tb.rows[-1][0], tb.next_save):  # saved already
                    tb.next_save += tb.period
            if _reached(t, tb.next_save):
                # the time it landed on, without the rounding of the solver's time
                landed = abs(t - tb.next_save) <= TIME_TOLERANCE * tb.next_save
                tb.sample(tb.next_save if landed else t, to_file=True)
                tb.next_save = _next_due(
                    tb.next_save, t * (1 + TIME_TOLERANCE), tb.period
                )
        else:
            # live preview: sample every LIVE_SAMPLE_INTERVAL of wall time
            # (and at the end of a run)
            now = time.perf_counter()
            if force or now - self._last_live > LIVE_SAMPLE_INTERVAL:
                self._last_live = now
                if not tb.rows or t != tb.rows[-1][0]:
                    tb.sample(t, to_file=False)

    def _table_autosave(self, period):
        period = float(period)
        self.table.period = max(period, 0.0)
        self.table.next_save = None
        if period > 0:
            self.table.clear()
        self.say(f"TableAutosave({period:g})", "in")

    # ------------------------------------------------------------------- commands
    def _cmd_table_autosave(self, c):
        self._table_autosave(self._eval(c["text"]))

    def _cmd_table_using(self, c):
        p = self._plot(c["plot"])
        self._plot_set(p, c.get("x"), c.get("y"), c.get("logx"), c.get("logy"))
        self._prune_columns()

    def _cmd_plot_quantity(self, c):
        """Plot any quantity of a magnet, as x or another y."""
        p = self._plot(c["plot"])
        col = self._series(c["target"], c["quant"], c["comp"])
        if c["axis"] == "x":
            self._plot_set(p, x=col)
        else:
            ys = p["y"].split(",")
            self._plot_set(p, y=ys if col in ys else [*ys, col])
        self._prune_columns()

    def _cmd_column_save(self, c):
        self.table.save(c["name"])
        self.say(f"saving {c['name']} in {self.table.path}")

    def _cmd_table_custom(self, c):
        self._table_custom(c)

    def _cmd_plot_add(self, c):
        return {"ok": True, "plot": self.add_plot()}

    def _cmd_plot_remove(self, c):
        if len(self.plots) > 1:
            self.plots = [p for p in self.plots if p["id"] != c["plot"]]
            self._prune_columns()
