import numpy as np

from mumaxplus import NcAfm, Grid, World
from mumaxplus.util import *


def max_relative_error(result, wanted):
    return np.max(np.abs(result - wanted) / np.abs(wanted))

def compute_fm_exchange_numpy(magnet):
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

def compute_homo_exchange_numpy(magnet, sub2, sub3):
    l = magnet.latcon.average()[0]
    a = magnet.ncafmex_cell.average()[0]
    m2 = sub2.magnetization.get()
    m3 = sub3.magnetization.get()
    return np.full(m2.shape, 4 * a * (m2 + m3) / (l * l * sub2.msat.average()[0]))

def compute_inhomo_exchange_numpy(magnet, sub2, sub3):
    m2 = sub2.magnetization.get()
    m3 = sub3.magnetization.get()
    cellsize = sub2.cellsize
    h_exch = np.zeros(m2.shape)

    m2_ = np.roll(m2, 1, axis=1)
    m3_ = np.roll(m3, 1, axis=1)
    h_exch[:, 1:, :, :] += ((m2_ - m2)[:, 1:, :, :] + (m3_ - m3)[:, 1:, :, :]) / (cellsize[2] ** 2)

    m2_ = np.roll(m2, -1, axis=1)
    m3_ = np.roll(m3, -1, axis=1)
    h_exch[:, :-1, :, :] += ((m2_ - m2)[:, :-1, :, :] + (m3_ - m3)[:, :-1, :, :]) / (cellsize[2] ** 2)

    m2_ = np.roll(m2, 1, axis=2)
    m3_ = np.roll(m3, 1, axis=2)
    h_exch[:, :, 1:, :] += ((m2_ - m2)[:, :, 1:, :] + (m3_ - m3)[:, :, 1:, :]) / (cellsize[1] ** 2)

    m2_ = np.roll(m2, -1, axis=2)
    m3_ = np.roll(m3, -1, axis=2)
    h_exch[:, :, 0:-1, :] += ((m2_ - m2)[:, :, 0:-1, :] + (m3_ - m3)[:, :, 0:-1, :]) / (cellsize[1] ** 2)

    m2_ = np.roll(m2, 1, axis=3)
    m3_ = np.roll(m3, 1, axis=3)
    h_exch[:, :, :, 1:] += ((m2_ - m2)[:, :, :, 1:] + (m3_ - m3)[:, :, :, 1:]) / (cellsize[0] ** 2)

    m2_ = np.roll(m2, -1, axis=3)
    m3_ = np.roll(m3, -1, axis=3)
    h_exch[:, :, :, 0:-1] += ((m2_ - m2)[:, :, :, 0:-1] + (m3_ - m3)[:, :, :, 0:-1]) / (cellsize[0] ** 2)

    return magnet.ncafmex_nn.average()[0] * h_exch / sub2.msat.average()[0]

class TestNcAfmExchange:
    def test_fm_exchange(self):

        world = World((1e3, 2e3, 3e3))
        magnet = NcAfm(world, Grid((16, 16, 4)))
        magnet.aex = 3.2e7
        magnet.msat = 5.4
        for sub in magnet.sublattices:
            result = sub.exchange_field.eval()
            wanted = compute_fm_exchange_numpy(sub)
            assert max_relative_error(result, wanted) < 5e-3
    
    def test_homo_exchange(self):
        world = World((1e3, 2e3, 3e3))
        magnet = NcAfm(world, Grid((16, 16, 4)))
        magnet.msat = 5.4
        magnet.ncafmex_cell = -10e4

        for i in range(3):
            sub = magnet.sublattices[i]
            result = sub.homogeneous_exchange_field()
            sub2, sub3 = magnet.sublattices[(i+1)%3], magnet.sublattices[(i+2)%3]
            wanted = compute_homo_exchange_numpy(magnet, sub2, sub3)
            assert max_relative_error(result, wanted) < 5e-3

    def test_inhomo_exchange(self):
        world = World((1e3, 2e3, 3e3))
        magnet = NcAfm(world, Grid((16, 16, 4)))
        magnet.msat = 5.4e3
        magnet.ncafmex_nn = -10e4

        for i in range(3):
            sub = magnet.sublattices[i]
            sub2, sub3 = magnet.sublattices[(i+1)%3], magnet.sublattices[(i+2)%3]
            result = sub.inhomogeneous_exchange_field()
            wanted = compute_inhomo_exchange_numpy(magnet, sub2, sub3)
            assert max_relative_error(result, wanted) < 5e-3

    def test_exchange_spiral(self):
        """This test compares numerical and analytical exchange energy for spiral
        magnetizations as a function of the angle between neighboring spins.
        This is inspired by figure 5 of the paper "The design and verification of
        MuMax3" up to 20°. https://doi.org/10.1063/1.4899186.
        One of the three sublattice magnetizations is set to zero to compare with
        the AFM result."""

        msat = 800e3
        aex = 5e-12
        ncafmex_nn = -15e-12

        length, width, thickness = 100e-6, 1e-9, 1e-9
        nx, ny, nz = int(length/1e-9), 1, 1
        cx, cy, cz = length/nx, width/ny, thickness/nz
        world = World(cellsize=(cx, cy, cz))
        magnet = NcAfm(world, Grid((nx, ny, nz)))
        magnet.msat = msat
        magnet.sub3.msat = 0
        magnet.aex = aex
        magnet.ncafmex_nn = ncafmex_nn
        magnet.sub1.magnetization = (1, 0, 0)
        magnet.sub2.magnetization = (-1, 0, 0)

        V = length * width * thickness

        # find exchange energy per angle
        angles, E_fm_mumax, E_inhomo_mumax, E_fm_theory, E_inhomo_theory = [], [], [], [], []
        kx = 0
        kx_step = 1e7
        X, _, _ = magnet.sub1.magnetization.meshgrid  # for fast magnetization setting
        while (max_angle := magnet.sub1.max_angle()) < (20*np.pi/180):  # to 20°
            angles.append(max_angle * 180 / np.pi)
            E_fm_mumax.append(magnet.sub1.exchange_energy() / (Km(msat) * V))
            E_inhomo_mumax.append(magnet.sub1.inhomogeneous_exchange_energy() / (Km(msat) * V))
            E_fm_theory.append(aex * kx**2 / Km(msat))
            E_inhomo_theory.append(-0.5 * ncafmex_nn * kx * kx / Km(msat))

            kx += kx_step
            magnet.sub1.magnetization = (np.cos(kx*X), np.sin(kx*X), np.zeros(shape=X.shape))
            magnet.sub2.magnetization = (-np.cos(kx*X), -np.sin(kx*X), np.zeros(shape=X.shape))

        # relative error
        error = [np.abs(a-e)/a for (e,a) in zip(E_fm_mumax[1:], E_fm_theory[1:])]
        assert max(error) < 1e-2
        error = [np.abs(a-e)/a for (e,a) in zip(E_inhomo_mumax[1:], E_inhomo_theory[1:])]
        assert max(error) < 1e-2

    def test_exchange_spiral(self):
        """This test compares numerical and analytical exchange energy for
        magnetizations as a function of the angle between spins at a
        single simulation cell.
        This is inspired by figure 5 of the paper "The design and verification of
        MuMax3" up to 20°. https://doi.org/10.1063/1.4899186.
        One of the three sublattice magnetizations is set to zero to compare with
        the AFM result. """

        msat = 800e3
        ncafmex_nn = -15e-12
        ncafmex_cell = -100e-12
        l = 0.35e-9 # Default lattice constant

        length, width, thickness = 100e-6, 1e-9, 1e-9
        nx, ny, nz = int(length/1e-9), 1, 1
        cx, cy, cz = length/nx, width/ny, thickness/nz
        world = World(cellsize=(cx, cy, cz))
        magnet = NcAfm(world, Grid((nx, ny, nz)))
        magnet.msat = msat
        magnet.sub3.msat = 0
        magnet.ncafmex_nn = ncafmex_nn
        magnet.ncafmex_cell = ncafmex_cell
        magnet.sub1.magnetization = (1, 0, 0)
        magnet.sub2.magnetization = (-1, 0, 0)

        V = length * width * thickness

        # find exchange energy per phase angle
        angles, E_homo_mumax, E_inhomo_mumax, E_homo_theory, E_inhomo_theory = [], [], [], [], []
        kx = 1e7 #arbitrary k value
        phi = 0
        X, _, _ = magnet.sub1.magnetization.meshgrid  # for fast magnetization setting
        while (phi < np.pi):
            angles.append(phi * 180 / np.pi)
            E_homo_mumax.append(magnet.sub1.homogeneous_exchange_energy() / (Km(msat) * V))
            E_inhomo_mumax.append(magnet.sub1.inhomogeneous_exchange_energy() / (Km(msat) * V))
            E_homo_theory.append(2 * ncafmex_cell * np.cos(phi) /(l * l * Km(msat)))
            E_inhomo_theory.append(-0.5 * ncafmex_nn * kx * kx * np.cos(phi)/ Km(msat))
            
            phi += np.pi / 180
            magnet.sub1.magnetization = (np.cos(kx*X), np.sin(kx*X), np.zeros(shape=X.shape))
            magnet.sub2.magnetization = (-np.cos(kx*X + phi), -np.sin(kx*X + phi), np.zeros(shape=X.shape))

        # semi relative error
        error = [np.abs(a-e)/np.max(np.abs(E_homo_theory)) for (e,a) in zip(E_homo_mumax[1:], E_homo_theory[1:])]
        assert max(error) < 1e-2
        error = [np.abs(a-e)/np.max(np.abs(E_homo_theory)) for (e,a) in zip(E_inhomo_mumax[1:], E_inhomo_theory[1:])]
        assert max(error) < 1e-2
