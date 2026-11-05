"""Painting on a display.

A brush for m or any parameter, or placing a vortex, skyrmion, ... of m (or a
Gaussian of u) at a point. Every stroke is recorded as Python, and can be undone.
"""

from __future__ import annotations

import numpy as np

import mumaxplus.util as U

from .common import _mro
from .display import PLANES


class PaintMixin:
    """The Controller's paint tools."""

    def _init_paint(self):
        # the stroke being painted, and how to undo the last one
        self._stroke = None
        self._paint_undo = None

    def _paint(self, c):
        """Paint on a field of the magnet under a display's image.

        ``c["points"]`` are positions in cells of the display's image (x right, y
        up, in its plane; cell i spans [i, i + 1)). The "brush" tool sets the quantity
        ``c["quant"]`` (m, or any parameter or variable) to ``c["value"]`` within
        ``c["radius"]`` cells of the points; "place" puts the configuration
        ``c["config"]`` (vortex, skyrmion, ... of m, or a Gaussian of u) of the field
        ``c["field"]`` centred on the point, within the radius. Points with the same
        ``c["stroke"]`` make one stroke: one recorded piece of Python, and one undo.
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
        if st["param"]:  # the value shown in the parameters panel
            self.describe()
        self._dirty = True
        self._live_sample()

    def _new_stroke(self, c, meta, first):
        """Find the magnet under the first point and remember how to undo."""
        dx = dy = dz = 0
        name = self.sel
        boxes = meta.get("boxes")
        if boxes:  # "show all magnets": the magnet the stroke starts on
            x, y, z = *first, int(c["layer"])
            on = lambda b, k, v: b[f"{k}0"] <= v < b[f"{k}0"] + b[f"n{k}"]  # noqa
            under = [b for b in boxes if on(b, "x", x) and on(b, "y", y)]
            # the one the shown slice goes through, if any (magnets overlap in a cut)
            under.sort(key=lambda b: not on(b, "z", z))
            if not under:
                return None
            b = under[0]
            name, dx, dy, dz = b["name"], b["x0"], b["y0"], b["z0"]
        obj = self.targets.get(name)
        tool = c["tool"]
        quant = c.get("field" if tool == "place" else "quant") or "magnetization"
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
        shown = int(c["layer"]) - dz  # the shown slice, in the magnet
        layer = None if c["all"] else shown
        if layer is not None and not 0 <= layer < n:
            return None
        st = {
            "id": c["stroke"],
            "name": name,
            "ref": self._ref(name),
            "quant": quant,
            "param": "Parameter" in m,
            "ncomp": int(field.ncomp),
            "dx": dx,
            "dy": dy,
            "axes": axes,
            "layer": layer,
            "shown": shown,
            "radius": max(float(self._eval(c["radius"])), 0.5),
            "tool": tool,
            "points": [],
            "entry": None,
            "env": {},
        }
        if st["tool"] == "place":
            fn = getattr(U, str(c["config"]))
            st["config"], st["env"] = fn.__name__, {fn.__name__: fn}
            st["args"] = str(c["args"]).strip().strip("()").strip()
            st["origin"], st["cell"] = tuple(obj.origin), self._cell_size()
        else:
            st["value"] = self._vec(c["value"], st["ncomp"])
        self._record_run()
        self._paint_undo = {
            "name": name,
            "quant": quant,
            "old": np.array(field.eval()),
            "param": st["param"],
            "entry": None,
        }
        self._stroke = st
        what = st.get("config") or f"{quant} = {st['value']!r}"
        self.say(f"paint {what} on {st['ref']}, radius {st['radius']:g} cells")
        return st

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
            if an != 2:  # in the shown slice (xy: at the bottom layer's z)
                pos[an] = o[an] + st["shown"] * c[an]
            args = ", ".join(f"{v:.6g}" for v in pos) + ")"
            args = f"({args}" + (f", {st['args']}" if st["args"] else "")
            lines.append(f"{field} = {st['config']}({args})")
            lines.append(f"_new = {field}.eval()")
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
        if u["param"]:
            self.describe()
        self._dirty = True
        self._live_sample()

    # ------------------------------------------------------------------- commands
    def _cmd_paint(self, c):
        self._paint(c)

    def _cmd_paint_undo(self, c):
        self._undo_paint()
