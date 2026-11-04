"""The display panels: what they show, their images, and saving them as OVF."""

from __future__ import annotations

import os
import time

import numpy as np

from .common import _clean, _next_due
from .plots import region_rgba, vector_rgba

RENDER_INTERVAL = 0.10  # minimum wall time between two display frames while running

# a display's plane: the world axes (0 = x, 1 = y, 2 = z) of the image's horizontal and
# vertical, and of the slice (the axis normal to it)
PLANES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}


def _check_plane(cfg):
    """Check a display's plane; a layer of None is its middle slice, xy's bottom one."""
    if cfg["plane"] not in PLANES:
        raise ValueError(f"plane must be xy, xz or yz, not {cfg['plane']!r}")
    if cfg.get("layer") is None and cfg["plane"] == "xy":
        cfg["layer"] = 0


class DisplayMixin:
    """The Controller's display panels."""

    def _init_display(self):
        # one dict per display panel: {"id", "quant", "comp", "plane", "layer", ...}
        self.views = []
        self._next_view = 0
        self.frame_id = 0
        self.frames = {}  # view id -> (meta, bytes)
        self._dirty = True  # render now, not only every RENDER_INTERVAL
        self._last_render = 0.0

    def _default_quantity(self):
        if self.sel is None:
            return None
        cls = type(self.targets[self.sel])
        # an antiferromagnet has no magnetization of its own, only of its sublattices
        for name in ("magnetization", "full_magnetization"):
            if isinstance(getattr(cls, name, None), property):
                return name
        return next(
            (n for n in dir(cls) if isinstance(getattr(cls, n, None), property)), None
        )

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
        base["autosave"] = 0.0  # AutoSave period of the shown quantity (OVF)
        base["_next_save"] = None
        base["id"] = self._next_view
        self._next_view += 1
        self.views.append(base)
        self._dirty = True
        return base["id"]

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

    def _eval_quant(self, name, quant):
        obj = self.targets[name]
        if quant == "regions":  # an array, not a quantity; NaN outside the geometry
            inside = np.asarray(obj.geometry, dtype=bool)
            arr = np.where(inside, np.asarray(obj.regions, dtype=float), np.nan)
        else:
            arr = np.asarray(getattr(obj, quant).eval())
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
        try:
            for cfg in self.views:
                self._render_view(cfg)
        finally:  # an error stops the loop's run once, not on every pass
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
        axes = PLANES[cfg["plane"]]
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
            if cfg["quant"] == "regions":  # a colour per region, not gray
                rgba = region_rgba(data[0]).tobytes()
        data = np.ascontiguousarray(data, dtype="<f4")
        finite = data[np.isfinite(data)]
        meta = {
            "vector": vector,
            "rgba": bool(rgba),  # the colours come with the data (else: gray)
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
            self.frames[cfg["id"]] = (_clean(meta), data.tobytes() + rgba)

    def _cell_size(self):
        return tuple(float(v) for v in self.world.cellsize)

    def _save_ovf(self, cfg, record=True):
        """Save the quantity of a display as an OVF file."""
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
        """Save the display's quantity now and every ``period`` s."""
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
            cfg["_next_save"] = _next_due(due, t * (1 + 1e-12), period)

    # ------------------------------------------------------------------- commands
    def _cmd_render(self, c):
        cfg = self._view(c["view"])
        new = {k: c[k] for k in ("quant", "comp", "plane", "layer", "all") if k in c}
        if "plane" in c and "layer" not in c:
            new["layer"] = None  # another plane: start in its middle
        new = {**cfg, **new}
        _check_plane(new)  # before the display changes
        cfg.update(new)
        self._dirty = True

    def _cmd_view_add(self, c):
        return {"ok": True, "view": self.add_view()}

    def _cmd_view_remove(self, c):
        gone = [v for v in self.views if v["id"] == c["view"]]
        if gone and len(self.views) > 1:
            if gone[0].get("_movie"):
                self._movie_end(gone[0])  # keep what was recorded
            self.views.remove(gone[0])
            with self.lock:
                self.frames.pop(c["view"], None)

    def _cmd_save_ovf(self, c):
        return {"ok": True, "file": self._save_ovf(self._view(c["view"]))}

    def _cmd_view_autosave(self, c):
        self._view_autosave(self._view(c["view"]), self._eval(c["text"]))
