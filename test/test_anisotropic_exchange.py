import numpy as np
from mumaxplus import Altermagnet, Grid, World

SRTOL = 5e-7

def max_semirelative_error(result, wanted):
    return np.max(np.abs(result - wanted) / np.max(np.abs(wanted)))

def compute_second_order_derivative_numpy(magnet, exch):
    m = magnet.magnetization.get()
    cellsize = magnet.cellsize

    dx2 = cellsize[0] ** 2
    dy2 = cellsize[1] ** 2

    m_pad = np.pad(m, ((0,0), (0,0), (1,1), (1,1)), mode='constant', constant_values=0)

    # x-direction
    m_x_plus  = m_pad[:, :, 1:-1, 2:]
    m_x_minus = m_pad[:, :, 1:-1, :-2]
    # y-direction
    m_y_plus  = m_pad[:, :, 2:, 1:-1]
    m_y_minus = m_pad[:, :, :-2, 1:-1]

    deriv = ( exch[1] * (m_y_plus + m_y_minus - 2 * m) / dy2 +
              exch[0] * (m_x_plus + m_x_minus - 2 * m) / dx2)
    return deriv

def compute_mixed_derivative(magnet, exch):
    m = magnet.magnetization.get()
    cellsize = magnet.cellsize
    denom = 4 * cellsize[0] * cellsize[1]

    deriv = np.zeros_like(m)

    m_pad = np.pad(m, ((0,0), (0,0), (1,1), (1,1)), mode='constant', constant_values=0)

    m_pp = m_pad[:, :, 2:, 2:]
    m_pm = m_pad[:, :, 2:, :-2]
    m_mp = m_pad[:, :, :-2, 2:]
    m_mm = m_pad[:, :, :-2, :-2]

    deriv = exch * (m_pp - m_pm - m_mp + m_mm) / denom
    return deriv

def compute_anisotropic_exchange_numpy(magnet, sub, switch):
    angle = magnet.alterex_angle.uniform_value
    A1 = magnet.alterex_1.uniform_value
    A2 = magnet.alterex_2.uniform_value

    c = np.cos(angle)
    s = np.sin(angle)
    c2, s2 = c*c, s*s
    if switch:
        A1, A2 = A2, A1

    Axx = A1*c2 + A2*s2
    Ayy = A1*s2 + A2*c2
    Axy = 2*(A1-A2)*c*s

    deriv_diag = compute_second_order_derivative_numpy(sub, (Axx, Ayy))
    deriv_mixed = compute_mixed_derivative(sub, Axy)

    return (deriv_diag + deriv_mixed) / sub.msat.uniform_value

class TestAfmExchange:
    def test_anisotropic_exchange(self):

        world = World((1e-9, 2e-9, 3e-9))
        magnet = Altermagnet(world, Grid((16, 16, 1)))
        magnet.msat = 5.4e3
        magnet.alterex_1 = 32e-12
        magnet.alterex_2 = 5.4e-12
        magnet.alterex_angle = 1
        for i, sub in enumerate(magnet.sublattices):
            result = sub.anisotropic_exchange_field()
            wanted = compute_anisotropic_exchange_numpy(magnet, sub, i)
            assert max_semirelative_error(result, wanted) < SRTOL