"""Common magnetization configurations."""

import numpy as _np

def twodomain(m1, m2, mw, wallposition, wallthickness=0.):
    """ Return a two-domain state magnetization configuration
    with a domain wall which is perpendicular to one of the
    three principal axes.

    Parameters
    ----------
    m1: tuple of three or six floats
        The magnetization of the first domain.
    m2: tuple of three or six floats
        The magnetization of the second domain.
    mw: tuple of three or six floats
        The magnetization inside the wall.
    wallposition: float
        The position of the domain wall.
    wallthickness: float
        The thickness of the wall.
        If given, wallposition corresponds to the center
        of the domain wall.
    """
    wallidx = int(_np.nonzero(wallposition)[0][0])
    '''
    def func(x, y, z):
         coo = (x, y, z)
         if coo[wallidx] < wallposition[wallidx] - wallthickness:
             return m1
         elif coo[wallidx] > wallposition[wallidx] + wallthickness:
             return m2
         return mw
    '''
    def func(x, y, z):
        if x < wallposition - wallthickness:
            return m1
        elif x > wallposition + wallthickness:
            return m2
        else:
            return mw
    return func


def vortex(position, diameter, circulation, polarization):
    """Return a vortex magnetization configuration.

    Parameters
    ----------
    position: tuple of three floats
        The position of the vortex center.
    diameter: float
        The diameter of the vortex center.
    circulation: 1 or -1
        Circulation of the vortex.
    polarization: 1 or -1
        The polarization of the vortex center.
    """
    if circulation != 1 and circulation != -1:
        raise ValueError("The circulation should be 1 or -1.")

    if polarization != 1 and polarization != -1:
        raise ValueError("The polarization should be 1 or -1.")

    x0, y0, _ = position

    def func(x, y, z):
        x -= x0
        y -= y0

        r2 = x ** 2 + y ** 2
        r = _np.sqrt(r2)

        if r == 0.0:
            return (0, 0, polarization)

        mx = -y * circulation / r
        my = x * circulation / r
        mz = 2 * polarization * _np.exp(-r2 / diameter ** 2)
        nrm = _np.sqrt(mx ** 2 + my ** 2 + mz ** 2)

        return (mx / nrm, my / nrm, mz / nrm)

    return func


def antivortex(position, diameter, circulation, polarization):
    """Return a antivortex magnetization configuration.

    Parameters
    ----------
    position: tuple of three floats
        The position of the center of the antivortex.
    diameter: float
        The diameter of the center of the antivortex.
    circulation: 1 or -1
        Circulation of the antivortex.
    polarization: 1 or -1
        The polarization of the center of the antivortex.
    """
    if circulation != 1 and circulation != -1:
        raise ValueError("The circulation should be 1 or -1.")

    if polarization != 1 and polarization != -1:
        raise ValueError("The polarization should be 1 or -1.")

    x0, y0, _ = position

    def func(x, y, z):
        x -= x0
        y -= y0

        r2 = x ** 2 + y ** 2
        r = _np.sqrt(r2)

        if r == 0.0:
            return (0, 0, polarization)

        mx = -x * circulation / r
        my = y * circulation / r
        mz = 2 * polarization * _np.exp(-r2 / diameter ** 2)
        nrm = _np.sqrt(mx ** 2 + my ** 2 + mz ** 2)

        return (mx / nrm, my / nrm, mz / nrm)

    return func


def neelskyrmion(position, radius, charge, polarization):
    """Return a Neel skyrmion magnetization configuration.

    Parameters
    ----------
    position: tuple of three floats
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

    x0, y0, _ = position

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


def blochskyrmion(position, radius, charge, polarization):
    """Return a Bloch skyrmion magnetization configuration.

    Parameters
    ----------
    position: tuple of three floats
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

    x0, y0, _ = position

    def func(x, y, z):
        x -= x0
        y -= y0
        r = _np.sqrt(x ** 2 + y ** 2)
        if r == 0.0:
            return (0, 0, polarization)
        mz = 2 * polarization * (_np.exp(-((r / radius) ** 2)) - 0.5)
        mx = -(y * charge / r) * (1 - _np.abs(mz))
        my = (x * charge / r) * (1 - _np.abs(mz))
        return (mx, my, mz)

    return func