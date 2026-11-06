"""Small helpers shared by the modules of the GUI."""

from __future__ import annotations

import inspect
import math
import re

import numpy as np

from mumaxplus import FP_PRECISION

# what the numbers typed in the page may use besides the script's namespace
MATH = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}


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
    """First paragraph of a docstring on one line (shown as tooltips)."""
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


# the rounding of the solver's time, relative: a time that landed on a save time can end
# up just below it (the time is a single precision number in a single precision build)
TIME_TOLERANCE = (
    8 * np.finfo(np.float32 if FP_PRECISION == "SINGLE" else np.float64).eps
)


def _reached(t, due):
    """Return whether the time t has reached the time ``due``, within its rounding."""
    return t >= due - TIME_TOLERANCE * abs(due)


def _next_due(due, t, period):
    """Return the first time due + k * period (k = 0, 1, ...) after t.

    Computed, not stepped: a period below the precision of t, or far smaller than the
    time since ``due``, must not hang the simulation thread.
    """
    if due > t:
        return due
    due += (math.floor((t - due) / period) + 1) * period
    return due if due > t else math.nextafter(t, math.inf)


def _col_name(base, ncomp, i):
    """Table column of component i of a quantity: m, or mx, my, mz, or m0, m1, ..."""
    return base if ncomp == 1 else base + ("xyz"[i] if ncomp == 3 else str(i))
