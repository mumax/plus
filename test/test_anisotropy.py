import numpy as np

from mumax5 import Ferromagnet, Grid, World


def max_relative_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    relerr = err / np.linalg.norm(wanted, axis=0)
    return np.max(relerr)


def compute_anisotropy_field(magnet):
    """Computes the anisotropy field."""
    ku1 = magnet.ku1.eval()
    ku2 = magnet.ku2.eval()
    msat = magnet.msat.eval()
    u = magnet.anisU.eval()
    u /= np.sqrt(np.sum(u * u, axis=0))  # normalize u
    m = magnet.magnetization.get()
    mu = np.sum(m * u, axis=0)
    return (2 * ku1 * mu + 4 * ku2 * mu ** 3) * u / msat


def compute_anisotropy_energy_density(magnet):
    """Computes the anisotropy energy density."""
    ku1 = magnet.ku1.eval()
    ku2 = magnet.ku2.eval()
    u = magnet.anisU.eval()
    u /= np.sqrt(np.sum(u * u, axis=0))  # normalize u
    m = magnet.magnetization.get()
    mu = np.sum(m * u, axis=0)
    return -ku1 * mu ** 2 - ku2 * mu ** 4


def compute_anisotropy_energy(magnet):
    edens = compute_anisotropy_energy_density(magnet)
    (cx, cy, cz) = magnet.cellsize
    return np.sum(edens) * cx * cy * cz


class TestAnisotropy:
    """Test quantities related to anisotropy."""

    def setup_class(self):
        self.world = World(cellsize=(1e-9, 2e-9, 4e-9))
        self.magnet = Ferromagnet(self.world, Grid((16, 32, 4)))
        self.magnet.msat = 800e3
        self.magnet.ku1 = 4.1e6
        self.magnet.ku2 = 2.1e6
        self.magnet.anisU = (0.3, 0.4, 1.0)

    def test_anisotropy_field(self):
        result = (self.magnet.anisotropy_field(),)
        wanted = compute_anisotropy_field(self.magnet)
        assert max_relative_error(result, wanted) < 1e-3

    def test_anisotropy_energy_density(self):
        result = (self.magnet.anisotropy_energy_density(),)
        wanted = compute_anisotropy_energy_density(self.magnet)
        assert max_relative_error(result, wanted) < 1e-3

    def test_anisotropy_energy(self):
        result = self.magnet.anisotropy_energy()
        wanted = compute_anisotropy_energy(self.magnet)
        relative_error = (result - wanted) / wanted
        assert relative_error < 1e-3
