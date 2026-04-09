"""
This script compares results of mumax⁺ to those from the paper of
Gomonay et al. (https://www.nature.com/articles/s44306-024-00042-3)
They describe an analytical model of an altermagnetic domain wall
and give theoretical results for the Néel vector and net magnetization.

The numerical material parameters used here are prototypical values.
"""

import pytest
import numpy as np

from mumaxplus import Altermagnet, Grid, World

SRTOL = 5e-3

cs = 0.5e-9 # Small to get good resolution in net magnetization profile
nx, ny, nz = 512, 1, 1
dw = 14  # Initial guess of number of cells in DW

def max_relative_error(result, wanted):
    err = np.abs(result - wanted)
    relerr = err / np.max(np.abs(wanted))
    return np.max(relerr)

def compute_domain_wall_width(magnet):
    """Computes the domain wall width"""
    A1 = magnet.alterex_1.uniform_value
    A2 = magnet.alterex_2.uniform_value
    A12 = magnet.atmex_nn.uniform_value
    K = magnet.sub1.ku1.uniform_value
    return np.sqrt((0.5*(A1+A2) - A12) / (2*K))

def compute_magnetization_prefactor(magnet):
    K = magnet.sub1.ku1.uniform_value
    Ms = magnet.sub1.msat.uniform_value
    a = magnet.latcon.uniform_value
    A1 = magnet.alterex_1.uniform_value
    A2 = magnet.alterex_2.uniform_value
    A0 = magnet.atmex_cell.uniform_value
    A12 = magnet.atmex_nn.uniform_value

    Han = 2 * K / Ms
    Hex = -8 * A0 / (Ms * a**2)
    return 0.5 * (Han/Hex) * (A1 - A2) / (0.5 * (A1+A2) - A12)

def neel_profile(x, position, width):
    """Walker ansatz to describe domain wall profile"""
    theta = 2 * np.arctan(np.exp((x - position) / width))
    return (np.zeros(x.shape),
            np.sin(-theta),
            np.cos(theta))

def net_magnetization_profile(x, position, width, prefactor):
    t = (x - position) / width
    denom = np.cosh(t)**3
    return (np.zeros(x.shape),
            -prefactor * np.sinh(t)**2 / denom,
             prefactor * np.sinh(t) / denom)

def initialize_and_minimize(magnet):
    """Create a two-domain state"""
    nx2 = nx // 2
    dw2 = dw // 2

    m = np.zeros(magnet.sub1.magnetization.shape)
    m[2, :, :,          0:nx2 - dw2] =  1 # Left domain
    m[1, :, :,  nx2 - dw2:nx2 + dw2] = -1 # Domain wall
    m[2, :, :,  nx2 + dw2:         ] = -1 # Right domain

    magnet.sub1.magnetization = m
    magnet.sub2.magnetization = -m
    magnet.minimize()

@pytest.mark.slow
class TestAltermagneticDomainWall:
    """Test Néel and net magnetization profiles of domain wall."""

    def setup_class(self):
        self.world = World((cs, cs, cs))
        self.magnet = Altermagnet(self.world, Grid((nx, ny, nz)))
        self.magnet.enable_demag = False
        self.magnet.msat = 2.9e5
        self.magnet.alpha = 0.01
        self.magnet.ku1 = 2e5
        self.magnet.anisU = (0, 0, 1)
        self.magnet.alterex_1 = 25e-12
        self.magnet.alterex_2 = 15e-12
        self.magnet.atmex_cell = -5e-12
        self.magnet.atmex_nn = -2.5e-12

        initialize_and_minimize(self.magnet)

    def test_neel_profile(self):
        width = compute_domain_wall_width(self.magnet)
        xx = np.linspace(0, nx*cs, nx)

        result = self.magnet.neel_vector()[:, 0, 0, :]
        wanted = neel_profile(xx, nx*cs/2, width)

        # Only look at DW itself when profile goes to zero inside the domains
        start = int(nx * 0.45)
        end = int(nx * 0.55)

        assert np.all(result[0] == 0) 
        assert max_relative_error(result[1][start:end], wanted[1][start:end]) < SRTOL
        assert max_relative_error(result[2], wanted[2]) < SRTOL

    def test_net_magnetization_profile(self):
        width = compute_domain_wall_width(self.magnet)
        prefactor = compute_magnetization_prefactor(self.magnet)
        xx = np.linspace(0, nx*cs, nx)

        wanted = net_magnetization_profile(xx, nx*cs/2, width, prefactor)
        result = 0.5 * (self.magnet.sub1.magnetization()
                        + self.magnet.sub2.magnetization())[:, 0, 0, :]

        # Only look at DW itself when profile goes to zero inside the domains
        start = int(nx * 0.45)
        end = int(nx * 0.55)

        assert np.all(result[0] == 0)
        assert max_relative_error(result[1][start:end], wanted[1][start:end]) < SRTOL
        assert max_relative_error(result[2][start:end], wanted[2][start:end]) < SRTOL