"""Common magnetization configurations."""

import numpy as _np


def neelskyrmion(center, radius, charge, polarization):
    """Return a Neel skyrmion magnetization configuration.

    Parameters
    ----------
    center: tuple of three floats
        The position of the skyrmion.
    radius: float
        The radius of the skyrmion.
    charge: 1 or -1
        The charge of the skyrmion.
    polarization: 1 or -1
        The polarization of the skyrmion.
    """
    if charge != 1 and charge != -1:
        raise ValueError("The charge should be 1 or -1.")

    if polarization != 1 and polarization != -1:
        raise ValueError("The polarization should be 1 or -1.")

    x0, y0, _ = center

    def func(x, y, z):
        x -= x0
        y -= y0
        r = _np.sqrt(x ** 2 + y ** 2)
        if r == 0.0:
            return (0, 0, polarization)
        mz = 2 * polarization * (_np.exp(-((r / radius) ** 2)) - 0.5)
        mx = (x * charge / r) * (1 - _np.abs(mz))
        my = (y * charge / r) * (1 - _np.abs(mz))
        return (mx, my, mz)

    return func
