"""Excitations: time terms added to parameters in the page (Parameter.add_time_term)."""

from __future__ import annotations

import ast
import re

import numpy as np

from .common import _num

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


class ExcitationMixin:
    """The Controller's excitations."""

    def _init_excitations(self):
        self.excitations = []  # added in the page, see _excite
        self._next_excitation = 1

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
        where, arg = c["where"], str(c["whereValue"]).strip()
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
        param.add_time_term(e["fn"], e["mask_value"])

    def _excite(self, c):
        """Add an excitation: a time term of a parameter (Parameter.add_time_term)."""
        tm = self._term(c)
        e = {
            "id": self._next_excitation,
            "target": c["target"],
            "param": c["param"],
            # the script's own time terms on this parameter (removing one of ours
            # removes those too)
            "foreign": self._foreign_terms(c["target"], c["param"]),
            **tm,
        }
        self._next_excitation += 1
        self._add_term(e)
        e["label"] = f"{c['param']} += {tm['term'][len('lambda t: '):]}" + (
            f"  where {c['where']} {c['whereValue']}" if tm["mask"] else ""
        )
        self.excitations.append(e)
        self._did(tm["code"], tm["imports"])
        self._dirty = True

    def _excite_remove(self, eid):
        """Remove an excitation.

        mumax⁺ can only remove all time terms of a parameter, so the others added
        here are added again.
        """
        e = next((x for x in self.excitations if x["id"] == eid), None)
        if e is None:
            raise ValueError(f"no excitation {eid!r}")
        foreign = self._foreign_terms(e["target"], e["param"])
        getattr(self.targets[e["target"]], e["param"]).remove_time_terms()
        self._did(f"{e['ref']}.remove_time_terms()")
        self.excitations = [x for x in self.excitations if x is not e]
        self._terms_removed(e["target"], e["param"], foreign)
        self._dirty = True

    def _foreign_terms(self, target, param):
        """Return whether a parameter has time terms that were not added in the page."""
        own = [
            x for x in self.excitations if x["target"] == target and x["param"] == param
        ]
        if own:
            return any(x["foreign"] for x in own)
        return bool(getattr(getattr(self.targets[target], param), "is_dynamic", False))

    def _terms_removed(self, target, param, foreign):
        """All time terms of a parameter are gone: add those of the page back."""
        if foreign:
            self.say(f"this also removed the time terms the script added to {param}")
        for x in self.excitations:
            if x["target"] == target and x["param"] == param:
                x["foreign"] = False
                self._add_term(x)
                self._did(x["code"])

    # ------------------------------------------------------------------- commands
    def _cmd_excite_preview(self, c):
        return {"ok": True, **self._excite_preview(c)}

    def _cmd_excite(self, c):
        self._excite(c)

    def _cmd_excite_remove(self, c):
        self._excite_remove(c["id"])
