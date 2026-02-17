import numpy as np
from mumaxplus import Altermagnet, Grid, World

SRTOL = 5e-7

def max_semirelative_error(result, wanted):
    return np.max(np.abs(result - wanted) / np.max(np.abs(wanted)))

def compute_second_order_derivative_numpy(magnet, exch):
    m = magnet.magnetization.get()
    cellsize = magnet.cellsize
    deriv = np.zeros(m.shape)

    m_ = np.roll(m, 1, axis=2)
    deriv[:, :, 1:, :] += exch[1] * (m_ - m)[:, :, 1:, :] / (cellsize[1] ** 2)

    m_ = np.roll(m, -1, axis=2)
    deriv[:, :, 0:-1, :] += exch[1] * (m_ - m)[:, :, 0:-1, :] / (cellsize[1] ** 2)

    m_ = np.roll(m, 1, axis=3)
    deriv[:, :, :, 1:] += exch[0] * (m_ - m)[:, :, :, 1:] / (cellsize[0] ** 2)

    m_ = np.roll(m, -1, axis=3)
    deriv[:, :, :, 0:-1] += exch[0] * (m_ - m)[:, :, :, 0:-1] / (cellsize[0] ** 2)

    return deriv

def compute_mixed_derivative(magnet, exch):
    m = magnet.magnetization.get()
    cellsize = magnet.cellsize
    denom = 4 * cellsize[0] * cellsize[1]

    deriv = np.zeros_like(m)

    m_pp = np.roll(np.roll(m, -1, axis=3), -1, axis=2)
    m_pm = np.roll(np.roll(m, -1, axis=3),  1, axis=2)
    m_mp = np.roll(np.roll(m,  1, axis=3), -1, axis=2)
    m_mm = np.roll(np.roll(m,  1, axis=3),  1, axis=2)
    deriv[:, :, 1:-1, 1:-1] = exch * (m_pp - m_pm - m_mp + m_mm)[:, :, 1:-1, 1:-1] / denom

    return deriv

def compute_anisotropic_exchange_numpy(magnet, sub, switch):
    angle = magnet.angle.uniform_value
    A1 = magnet.A1.uniform_value
    A2 = magnet.A2.uniform_value

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
        magnet.A1 = 32e-12
        magnet.A2 = 5.4e-12
        magnet.angle = 1
        for i, sub in enumerate(magnet.sublattices):
            result = sub.anisotropic_exchange_field()
            wanted = compute_anisotropic_exchange_numpy(magnet, sub, i)
            assert max_semirelative_error(result, wanted) < SRTOL