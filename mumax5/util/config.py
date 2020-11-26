"""Common magnetization configurations."""

import numpy as _np


def neelskyrmion(center, radius, charge, pol):
    """Create the Neel skyrmion magnetization configuration."""

    def func(x, y, z):
        x -= center[0]
        y -= center[1]
        r = _np.sqrt(x ** 2 + y ** 2)
        if r == 0.0:
            return (0, 0, pol)
        mz = 2 * pol * (_np.exp(-((r / radius) ** 2)) - 0.5)
        mx = (x * charge / r) * (1 - _np.abs(mz))
        my = (y * charge / r) * (1 - _np.abs(mz))
        return (mx, my, mz)

    return func
