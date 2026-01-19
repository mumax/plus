import numpy as np

from mumaxplus import Ferromagnet, Grid, World

SRTOL = 3e-7

def max_semirelative_error(result, wanted):
    return np.max(np.abs(result - wanted) / np.max(np.abs(wanted)))

def compute_exchange_numpy(magnet):
    m = magnet.magnetization.get()
    cellsize = magnet.cellsize
    h_exch = np.zeros(m.shape)

    m_ = np.roll(m, 1, axis=1)
    h_exch[:, 1:, :, :] += (m_ - m)[:, 1:, :, :] / (cellsize[2] ** 2)

    m_ = np.roll(m, -1, axis=1)
    h_exch[:, :-1, :, :] += (m_ - m)[:, :-1, :, :] / (cellsize[2] ** 2)

    m_ = np.roll(m, 1, axis=2)
    h_exch[:, :, 1:, :] += (m_ - m)[:, :, 1:, :] / (cellsize[1] ** 2)

    m_ = np.roll(m, -1, axis=2)
    h_exch[:, :, 0:-1, :] += (m_ - m)[:, :, 0:-1, :] / (cellsize[1] ** 2)

    m_ = np.roll(m, 1, axis=3)
    h_exch[:, :, :, 1:] += (m_ - m)[:, :, :, 1:] / (cellsize[0] ** 2)

    m_ = np.roll(m, -1, axis=3)
    h_exch[:, :, :, 0:-1] += (m_ - m)[:, :, :, 0:-1] / (cellsize[0] ** 2)

    return 2 * magnet.aex.average()[0] * h_exch / magnet.msat.average()[0]


class TestExchange:
    def test_exchange(self):

        world = World((1e3, 2e3, 3e3))
        magnet = Ferromagnet(world, Grid((16, 16, 4)))
        magnet.aex = 3.2e7
        magnet.msat = 5.4

        result = magnet.exchange_field.eval()
        wanted = compute_exchange_numpy(magnet)

        assert max_semirelative_error(result, wanted) < SRTOL

    def test_exchange_spiral(self):
        """This test compares numerical and analytical exchange energy for spiral
        magnetizations as a function of the angle between neighboring spins.
        This is a recreation of figure 5 of the paper "The design and
        verification of MuMax3" up to 20°. https://doi.org/10.1063/1.4899186 """

        msat = 800e3
        aex = 5e-12
        mu0 = 4 * np.pi * 1e-7

        length, width, thickness = 100e-6, 1e-9, 1e-9
        nx, ny, nz = int(length/1e-9), 1, 1
        cx, cy, cz = length/nx, width/ny, thickness/nz
        world = World(cellsize=(cx, cy, cz))
        magnet = Ferromagnet(world, Grid((nx, ny, nz)))
        magnet.msat = msat
        magnet.aex = aex
        magnet.magnetization = (1, 0, 0)

        V = length * width * thickness
        Km = 0.5 * mu0 * msat**2

        # find exchange energy per angle
        angles, E_mumax, E_analytical = [], [], []
        kx = 0
        kx_step = 1e7
        X, _, _ = magnet.magnetization.meshgrid  # for fast magnetization setting
        while (max_angle := magnet.max_angle.eval()) < (20*np.pi/180):  # to 20°
            angles.append(max_angle * 180 / np.pi)
            E_mumax.append(magnet.exchange_energy.eval() / (Km * V))
            E_analytical.append(aex * kx**2 / Km)

            kx += kx_step
            magnet.magnetization = (np.cos(kx*X), np.sin(kx*X), np.zeros(shape=X.shape))

        # relative error
        error = [np.abs(a-e)/a for (e,a) in zip(E_mumax[1:], E_analytical[1:])]
        assert max(error) < 1e-2
