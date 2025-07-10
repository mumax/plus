"""Common magnetization configurations."""

import numpy as _np
import math as _math

def twodomain(m1, mw, m2, wallposition, wallthickness=0.) -> callable:
    """Create a two-domain state magnetization configuration
    with a domain wall which is perpendicular to the x-axis.

    Parameters
    ----------
    m1: tuple of three floats
        The magnetization of the first domain.
    m2: tuple of three floats
        The magnetization of the second domain.
    mw: tuple of three floats
        The magnetization inside the wall.
    wallposition: float
        The position of the domain wall.
    wallthickness: float
        The thickness of the wall, which smooths out
        the wall with a Gaussian distribution.
        If given, wallposition corresponds to the center
        of the domain wall. If <= 0, there is no wall.

    Returns
    -------
    two-domains : callable
        a function which takes x-, y- and z-coordinates and returns a 3-tuple
        containing m_x, m_y and m_z.
    """

    def func(x, y, z):
        if x < wallposition:
            m = m1
        else:
            m = m2
        if wallthickness <= 0:
            return m
        gauss = _np.exp(-((x - wallposition)/wallthickness)**2)
        return ((1-gauss) * m[0] + gauss * mw[0],
                (1-gauss) * m[1] + gauss * mw[1],
                (1-gauss) * m[2] + gauss * mw[2])

    return func


def vortex(position, diameter, circulation, polarization) -> callable:
    """Create a vortex magnetization configuration.

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

    Returns
    -------
    vortex : callable
        a function which takes x-, y- and z-coordinates and returns a 3-tuple
        containing m_x, m_y and m_z.
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


def antivortex(position, diameter, circulation, polarization) -> callable:
    """Create an antivortex magnetization configuration.

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

    Returns
    -------
    antivortex : callable
        a function which takes x-, y- and z-coordinates and returns a 3-tuple
        containing m_x, m_y and m_z.
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


def neelskyrmion(position, radius, charge, polarization) -> callable:
    """Create a Neel skyrmion magnetization configuration.

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

    Returns
    -------
    Neel-skyrmion : callable
        a function which takes x-, y- and z-coordinates and returns a 3-tuple
        containing m_x, m_y and m_z.
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


def blochskyrmion(position, radius, charge, polarization) -> callable:
    """Create a Bloch skyrmion magnetization configuration.

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

    Returns
    -------
    Bloch-skyrmion : callable
        a function which takes x-, y- and z-coordinates and returns a 3-tuple
        containing m_x, m_y and m_z.
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

# --------------------------------------------------
# elastic displacement initial states

def gaussian_spherical_OoP(position, amplitude, sigma_x, sigma_y) -> callable:
    r"""Create an out-of-xy-plane gaussian distribution centered on the specified
    position with given standard deviations.
    
    .. math:: 
        A \exp\left(- \frac{(x - x_0)^2}{2\sigma_x^2} \frac{(y - y_0)^2}{2\sigma_y^2}\right) (0, 0, 1)

    Parameters
    ----------
    position : tuple of three floats
        The position of the Gaussian distribution.
    amplitude : float
        The amplitude of the Gaussian distribution.
    sigma_x : float
        The standard deviation in the x-direction of the Gaussian distribution.
    sigma_y : float
        The standard deviation in the y-direction of the Gaussian distribution.

    Returns
    -------
    Gaussian : callable
        a function which takes x-, y- and z-coordinates and returns a 3-tuple
        containing u_x, u_y and u_z.
    """
    
    x0, y0, _ = position
    denom = (4 * sigma_x*sigma_x * sigma_y*sigma_y)

    def func(x, y, z):
        uz = amplitude * _math.exp(- (x - x0)*(x - x0) * (y - y0)*(y - y0) / denom)
        return (0, 0, uz)
    
    return func

def gaussian_spherical_IP(position, amplitude, angle, sigma_x, sigma_y) -> callable:
    r"""Create an in-xy-plane vector field with uniform orientation specified by
    the given angle. The amplitude is modified by a gaussian distribution
    centered on the specified position with given standard deviations.
    
    .. math ::

        A \exp\left(- \frac{(x - x_0)^2}{2\sigma_x^2} \frac{(y - y_0)^2}{2\sigma_y^2}\right)
        (\cos(\theta), \sin(\theta), 0)

    Parameters
    ----------
    position : tuple of three floats
        The position of the Gaussian distribution.
    amplitude : float
        The amplitude of the uniform vector field.
    angle : float
        The angle in radians of the uniform vector field direction.
    sigma_x : float
        The standard deviation in the x-direction of the Gaussian distribution.
    sigma_y : float
        The standard deviation in the y-direction of the Gaussian distribution.
    
    Returns
    -------
    Gaussian : callable
        a function which takes x-, y- and z-coordinates and returns a 3-tuple
        containing u_x, u_y and u_z.
    """
    
    x0, y0, _ = position
    denom = 4 * sigma_x*sigma_x * sigma_y*sigma_y
    AC, AS = amplitude * _math.cos(angle), amplitude * _math.sin(angle)

    def func(x, y, z):
        E = _math.exp(- (x - x0)*(x - x0) * (y - y0)*(y - y0) / denom)
        return (AC*E, AS*E, 0)

    return func

def gaussian_uniform_IP(amplitude, theta, gausspos, sigma, phi) -> callable:
    r"""Create an in-xy-plane vector field with uniform orientation specified by
    the given angle theta. The amplitude is modified by a one-dimensional
    Gaussian distribution centered on gausspos, which varies along the
    transverse direction in the xy-plane specified by angle phi.

    .. math:: 
        A \exp\left(-\frac{(x'-x_0)^2}{2\sigma^2}\right)
        (\cos(\theta) ,\sin(\theta), 0)
    
    with x0 = gausspos and x' = x*cos(ϕ) + y*sin(ϕ)
    
    Parameters
    ----------
    amplitude : float
        The amplitude of the vector field.
    theta : float
        The angle in radians of the uniform vector field direction.
    gausspos : floats
        The center position of the Gaussian distribution.
    sigma : float
        The gaussian standard deviation.
    phi : float
        The angle in radians of the transverse direction.
    
    Returns
    -------
    Gaussian : callable
        a function which takes x-, y- and z-coordinates and returns a 3-tuple
        containing u_x, u_y and u_z.
    """

    denom = 2 * sigma*sigma
    Cphi, Sphi = _math.cos(phi), _math.sin(phi)
    ACtheta, AStheta = amplitude * _math.cos(theta), amplitude * _math.sin(theta)

    def func(x, y, z):
        x_ = Cphi*x + Sphi*y
        E = _math.exp(- (x_ - gausspos)*(x_ - gausspos) / denom)
        return (ACtheta * E, AStheta * E, 0)
    
    return func
