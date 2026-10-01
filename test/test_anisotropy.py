import numpy as np

from mumaxplus import Ferromagnet, Grid, World

RTOL = 2e-3  # fairly large, because energies can approach 0 -> divide by 0


def max_relative_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    relerr = err / np.linalg.norm(wanted, axis=0)
    return np.max(relerr)


# --- Uniaxial Anisotropy ---


def compute_uniaxial_anisotropy_field(magnet):
    """Computes the uniaxial anisotropy field."""
    ku1 = magnet.ku1.eval()
    ku2 = magnet.ku2.eval()
    msat = magnet.msat.eval()
    u = magnet.anisU.eval()
    u /= np.sqrt(np.sum(u * u, axis=0))  # normalize u
    m = magnet.magnetization.get()
    mu = np.sum(m * u, axis=0)
    return (2 * ku1 * mu + 4 * ku2 * mu**3) * u / msat


def compute_uniaxial_anisotropy_energy_density(magnet):
    """Computes the uniaxial anisotropy energy density."""
    ku1 = magnet.ku1.eval()
    ku2 = magnet.ku2.eval()
    u = magnet.anisU.eval()
    u /= np.sqrt(np.sum(u * u, axis=0))  # normalize u
    m = magnet.magnetization.get()
    mu = np.sum(m * u, axis=0)
    return -ku1 * mu**2 - ku2 * mu**4


def compute_uniaxial_anisotropy_energy(magnet):
    edens = compute_uniaxial_anisotropy_energy_density(magnet)
    (cx, cy, cz) = magnet.cellsize
    return np.sum(edens) * cx * cy * cz


class TestUniaxialAnisotropy:
    """Test quantities related to uniaxial anisotropy."""

    def setup_class(self):
        self.world = World(cellsize=(1e-9, 2e-9, 4e-9))
        self.magnet = Ferromagnet(self.world, Grid((16, 32, 4)))
        self.magnet.msat = 800e3
        self.magnet.ku1 = 4.1e6
        self.magnet.ku2 = 2.1e6
        self.magnet.anisU = (0.3, 0.4, 1.0)

    def test_anisotropy_field(self):
        result = (self.magnet.anisotropy_field(),)
        wanted = compute_uniaxial_anisotropy_field(self.magnet)
        assert max_relative_error(result, wanted) < RTOL

    def test_anisotropy_energy_density(self):
        result = (self.magnet.anisotropy_energy_density(),)
        wanted = compute_uniaxial_anisotropy_energy_density(self.magnet)
        assert max_relative_error(result, wanted) < RTOL

    def test_anisotropy_energy(self):
        result = self.magnet.anisotropy_energy()
        wanted = compute_uniaxial_anisotropy_energy(self.magnet)
        relative_error = (result - wanted) / wanted
        assert relative_error < RTOL


# --- Cubic Anisotropy ---


def compute_cubic_anisotropy_field(magnet):
    """Computes the cubic anisotropy field."""
    kc1, kc2, kc3 = magnet.kc1.eval(), magnet.kc2.eval(), magnet.kc3.eval()
    msat = magnet.msat.eval()
    c1, c2 = magnet.anisC1.eval(), magnet.anisC2.eval()
    c3 = np.cross(c1, c2, axis=0)
    c1 /= np.linalg.norm(c1, axis=0)
    c2 /= np.linalg.norm(c2, axis=0)
    c3 /= np.linalg.norm(c3, axis=0)
    m = magnet.magnetization.get()

    c1m, c2m, c3m = (
        np.sum(c1 * m, axis=0),
        np.sum(c2 * m, axis=0),
        np.sum(c3 * m, axis=0),
    )

    return -2/msat * (kc1 * ((c2m**2 + c3m**2)*c1m*c1 + \
                             (c1m**2 + c3m**2)*c2m*c2 + \
                             (c1m**2 + c2m**2)*c3m*c3) + \
                      kc2 * (c2m**2*c3m**2*c1m*c1 + \
                             c1m**2*c3m**2*c2m*c2 + \
                             c1m**2*c2m**2*c3m*c3) + \
                      2*kc3 * ((c2m**4 + c3m**4)*c1m**3*c1 + \
                               (c1m**4 + c3m**4)*c2m**3*c2 + \
                               (c1m**4 + c2m**4)*c3m**3*c3))  # fmt: skip


def compute_cubic_anisotropy_energy_density(magnet):
    """Computes the cubic anisotropy energy density."""
    kc1, kc2, kc3 = magnet.kc1.eval(), magnet.kc2.eval(), magnet.kc3.eval()
    c1, c2 = magnet.anisC1.eval(), magnet.anisC2.eval()
    c3 = np.cross(c1, c2, axis=0)
    c1 /= np.linalg.norm(c1, axis=0)
    c2 /= np.linalg.norm(c2, axis=0)
    c3 /= np.linalg.norm(c3, axis=0)
    m = magnet.magnetization.get()

    c1m, c2m, c3m = (
        np.sum(c1 * m, axis=0),
        np.sum(c2 * m, axis=0),
        np.sum(c3 * m, axis=0),
    )

    return (
        kc1 * (c1m**2 * c2m**2 + c1m**2 * c3m**2 + c2m**2 * c3m**2)
        + kc2 * (c1m**2 * c2m**2 * c3m**2)
        + kc3 * (c1m**4 * c2m**4 + c1m**4 * c3m**4 + c2m**4 * c3m**4)
    )


def compute_cubic_anisotropy_energy(magnet):
    edens = compute_cubic_anisotropy_energy_density(magnet)
    (cx, cy, cz) = magnet.cellsize
    return np.sum(edens) * cx * cy * cz


class TestCubicAnisotropy:
    """Test quantities related to cubic anisotropy."""

    def setup_class(self):
        self.world = World(cellsize=(1e-9, 2e-9, 4e-9))
        self.magnet = Ferromagnet(self.world, Grid((16, 32, 4)))
        self.magnet.msat = 800e3
        self.magnet.kc1 = 4.1e6
        self.magnet.kc2 = 2.1e6
        self.magnet.kc3 = 3.1e6
        c1 = np.random.random(size=3)  # some random direction
        self.magnet.anisC1 = c1
        self.magnet.anisC2 = np.cross(
            c1, np.random.random(size=3)
        )  # random perpendicular vector

    def test_anisotropy_field(self):
        result = (self.magnet.anisotropy_field(),)
        wanted = compute_cubic_anisotropy_field(self.magnet)
        assert max_relative_error(result, wanted) < RTOL

    def test_anisotropy_energy_density(self):
        result = (self.magnet.anisotropy_energy_density(),)
        wanted = compute_cubic_anisotropy_energy_density(self.magnet)
        assert max_relative_error(result, wanted) < RTOL

    def test_anisotropy_energy(self):
        result = self.magnet.anisotropy_energy()
        wanted = compute_cubic_anisotropy_energy(self.magnet)
        relative_error = (result - wanted) / wanted
        assert relative_error < RTOL
