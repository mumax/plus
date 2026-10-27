"""Small helpers shared by the modules of the GUI."""

from __future__ import annotations

import inspect
import math
import re

import numpy as np

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
