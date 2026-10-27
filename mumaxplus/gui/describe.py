"""What the page needs to build itself (/api/page).

That is the magnets' quantities, parameters and switches, the world, the solver
settings and the initial magnetizations.
"""

from __future__ import annotations

import inspect
import shutil

import numpy as np

import mumaxplus
import mumaxplus.util as U

from .common import _clean, _doc, _mro, _param_json

SOLVER_METHODS = ["Heun", "BogackiShampine", "CashKarp", "Fehlberg", "DormandPrince"]

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


class DescribeMixin:
    """The Controller's description of the simulation for the page."""

    def _init_describe(self):
        self.desc = {}
        self.desc_version = 0  # goes up with every describe(): the page fetches it

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
                elif pname == "regions":  # an array: it can be shown, not plotted
                    item["doc"] = "The region index of every cell (a colour each)."
                    info["quantities"].append({**item, "ncomp": 1, "noplot": True})
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
