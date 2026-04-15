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

SRTOL = 7e-3

cs = 0.5e-9 # Small to get good resolution in net magnetization profile
nx, ny, nz = 512, 1, 1
dw = 10  # Initial guess of number of cells in DW

def max_normalized_error(result, wanted):
    err = np.abs(result - wanted)
    relerr = err / np.max(np.abs(wanted))
    return np.max(relerr)

def compute_domain_wall_width(magnet):
    """Computes the domain wall width"""
    A1 = magnet.alterex_1.uniform_value
    A2 = magnet.alterex_2.uniform_value
    A12 = magnet.afmex_nn.uniform_value
    K = magnet.sub1.ku1.uniform_value
    return np.sqrt((0.5*(A1+A2) - A12) / (2*K))

def compute_magnetization_prefactor(magnet):
    K   = magnet.sub1.ku1.uniform_value
    Ms  = magnet.sub1.msat.uniform_value
    a   = magnet.latcon.uniform_value
    A1  = magnet.alterex_1.uniform_value
    A2  = magnet.alterex_2.uniform_value
    A0  = magnet.afmex_cell.uniform_value
    A12 = magnet.afmex_nn.uniform_value

    Han = 2 * K / Ms
    Hex = -8 * A0 / (Ms * a**2)
    return 0.5 * (Han/Hex) * (A1 - A2) / (0.5 * (A1+A2) - A12)

def neel_profile(x, position, width, dw_comp, zero_comp):
    """Walker ansatz to describe domain wall profile"""
    theta = 2 * np.arctan(np.exp((x - position) / width))

    result = [None, None, None]
    result[zero_comp] = np.zeros(x.shape)
    result[dw_comp] = np.sin(-theta)
    result[2] = np.cos(theta)

    return result

def net_magnetization_profile(x, position, width, prefactor, dw_comp, zero_comp):
    t = (x - position) / width
    denom = np.cosh(t)**3

    result = [None, None, None]
    result[zero_comp] = np.zeros(x.shape)
    result[dw_comp] = -prefactor * np.sinh(t)**2 / denom
    result[2] =  prefactor * np.sinh(t) / denom

    return result

def initialize_and_minimize(magnet, dw_comp):
    """Create a two-domain state"""
    nx2 = nx // 2
    dw2 = dw // 2

    m = np.zeros(magnet.sub1.magnetization.shape)
    if dw_comp:
        m[2      , :, :,          0:nx2 - dw2] =  1 # Left domain
        m[dw_comp, :, :,  nx2 - dw2:nx2 + dw2] = -1 # Domain wall
        m[2      , :, :,  nx2 + dw2:         ] = -1 # Right domain
    else:
        m[2      , :,          0:nx2 - dw2, :] =  1 # Left domain
        m[dw_comp, :,  nx2 - dw2:nx2 + dw2, :] = -1 # Domain wall
        m[2      , :,  nx2 + dw2:         , :] = -1 # Right domain
    magnet.sub1.magnetization = m
    magnet.sub2.magnetization = -m
    magnet.minimize()

ORIENTATIONS = { "x": dict(grid = Grid((nx, ny, nz)),
                           angle = 0,
                           result_slice = lambda v: v[:, 0, 0, :],
                           dw_comp = 1,
                           zero_comp = 0),
                 "y": dict(grid = Grid((ny, nx, nz)),
                           angle = np.pi/2,
                           result_slice = lambda v: v[:, 0, :, 0],
                           dw_comp = 0,
                           zero_comp = 1)
}

def make_magnet(world, orientation):
    cfg = ORIENTATIONS[orientation]
    magnet = Altermagnet(world, cfg["grid"])
    magnet.enable_demag = False
    magnet.msat = 2.9e5
    magnet.alpha = 0.01
    magnet.ku1 = 2e5
    magnet.anisU = (0, 0, 1)
    magnet.alterex_1 = 25e-12
    magnet.alterex_2 = 15e-12
    magnet.alterex_angle = cfg["angle"]
    magnet.afmex_cell = -5e-12
    magnet.afmex_nn = -2.5e-12

    initialize_and_minimize(magnet, cfg["dw_comp"])
    return magnet
@pytest.mark.slow
class AltermagneticDomainWall:

    orientation: str

    def setup_class(self):
        self.world = World((cs, cs, cs))
        self.magnet = make_magnet(self.world, self.orientation)

    def test_neel_profile(self):
        cfg = ORIENTATIONS[self.orientation]

        width = compute_domain_wall_width(self.magnet)
        xx = np.linspace(0, nx * cs, nx)

        result = cfg["result_slice"](self.magnet.neel_vector())
        wanted = neel_profile(xx, nx*cs/2, width, cfg["dw_comp"], cfg["zero_comp"])

        # Only look at DW itself when profile goes to zero inside the domains
        start = int(nx * 0.45)
        end = int(nx * 0.55)

        assert np.all(result[cfg["zero_comp"]] == 0)
        assert max_normalized_error(result[cfg["dw_comp"]][start:end], wanted[cfg["dw_comp"]][start:end]) < SRTOL
        assert max_normalized_error(result[2], wanted[2]) < SRTOL

    def test_net_magnetization_profile(self):
        cfg = ORIENTATIONS[self.orientation]

        width = compute_domain_wall_width(self.magnet)
        prefactor = compute_magnetization_prefactor(self.magnet)
        xx = np.linspace(0, nx * cs, nx)

        result = cfg["result_slice"]( 0.5 * (self.magnet.sub1.magnetization()
                                           + self.magnet.sub2.magnetization()))
        wanted = net_magnetization_profile(xx, nx*cs/2, width, prefactor, cfg["dw_comp"], cfg["zero_comp"])

        # Only look at DW itself when profile goes to zero inside the domains
        start = int(nx * 0.45)
        end = int(nx * 0.55)

        assert np.all(result[cfg["zero_comp"]] == 0)
        assert max_normalized_error(result[cfg["dw_comp"]][start:end], wanted[cfg["dw_comp"]][start:end]) < SRTOL
        assert max_normalized_error(result[2][start:end], wanted[2][start:end]) < SRTOL

@pytest.mark.slow
class TestAltermagneticDomainWallAlongX(AltermagneticDomainWall):
    """Domain wall along x-direction."""
    orientation = "x"

@pytest.mark.slow
class TestAltermagneticDomainWallAlongY(AltermagneticDomainWall):
    """Domain wall along y-direction."""
    orientation = "y"