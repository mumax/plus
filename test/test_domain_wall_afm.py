"""
This script compares results of mumaxplus to those from the paper of
Sánchez-Tejerina et al. (https://doi.org/10.1103/PhysRevB.101.014433)
They describe an analytical model of antiferromagnetic domain wall
motion driven by a current and give theoretical results for it's
width and speed.

The numerical material parameters used here can be found
in the mentioned article.
"""


import pytest

import numpy as np
from scipy.optimize import curve_fit  # SCIPY NEEDED

from mumaxplus import Antiferromagnet, Grid, World
from mumaxplus.util import *


RTOL = 5e-3

def max_relative_error(result, wanted):
    err = np.abs(result - wanted)
    relerr = err / np.abs(wanted)
    return np.max(relerr)

def compute_domain_wall_width(magnet):
    """Computes the domain wall width"""
    ku = magnet.sub1.ku1.uniform_value
    aex = magnet.sub1.aex.uniform_value
    afmex_nn = magnet.afmex_nn.uniform_value
    return np.sqrt((2*aex - afmex_nn) / (2*ku))

def compute_domain_wall_speed(magnet):
    """Computes the domain wall speed"""
    pol = magnet.sub1.pol.uniform_value
    J = magnet.sub1.jcur.uniform_value[-1]
    FL = magnet.sub1.free_layer_thickness.uniform_value
    Ms = magnet.sub1.msat.uniform_value
    alpha = magnet.sub1.alpha.uniform_value

    Hsh = HBAR * pol * J / (2 * QE * FL)
    L = Ms / GAMMALL
    return np.pi * compute_domain_wall_width(magnet) * Hsh / (alpha * 2 * L)

def DW_profile(x, position, width):
    """Walker ansatz to describe domain wall profile"""
    return np.cos(2 * np.arctan(np.exp(-(x - position) / width)))

def fit_domain_wall(magnet):
    """Fit Walker ansatz to domain wall profile"""
    mz = magnet.sub1.magnetization()[0,]
    profile = mz[:, 50, 0]  # middle row of the grid
    z = np.linspace(-200e-9, 200e-9, 200)
    popt, pcov = curve_fit(DW_profile, z, profile, p0=(0, 5e-9))
    return popt

def get_domain_wall_speed(self):
    """Find stationary value of the velocity"""
    t = 1e-11
    self.world.timesolver.run(t/2)
    q1 = fit_domain_wall(self.magnet)[0]
    self.world.timesolver.run(t/2)
    q2 = fit_domain_wall(self.magnet)[0]
    return 2 * np.abs(q1-q2) / t

def initialize(self):
    """Create a two-domain state"""
    m = np.zeros(self.magnet.sub1.magnetization.shape)
    m[0,0:99,:,:] = -1
    m[2,99:101,:,:] = 1  # Domain wall has a width of 4 nm.
    m[0,101:,:,:] = 1
    self.magnet.sub1.magnetization = m
    self.magnet.sub2.magnetization = -m
    self.world.timesolver.run(1e-11)  # instead of minimize for better performance


@pytest.mark.slow
class TestStaticDomainWall:
    """Test width of stable domain wall."""

    def setup_class(self):
        self.world = World(cellsize=(2e-9, 2e-9, 2e-9))
        self.magnet = Antiferromagnet(self.world, Grid((1, 100, 200)))
        self.magnet.msat = 0.4e6
        self.magnet.alpha = 0.1
        self.magnet.ku1 = 64e3
        self.magnet.anisU = (1, 0, 0)
        self.magnet.aex = 10e-12
        self.magnet.afmex_cell = -25e-12
        self.magnet.afmex_nn = -5e-12
        self.magnet.sub1.dmi_tensor.set_interfacial_dmi(0.11e-3)
        self.magnet.sub2.dmi_tensor.set_interfacial_dmi(0.11e-3)

        initialize(self)

    def test_domain_wall_width(self):
        result = fit_domain_wall(self.magnet)[-1]
        wanted = compute_domain_wall_width(self.magnet)
        assert max_relative_error(result, wanted) < RTOL


@pytest.mark.slow
class TestDynamicDomainWall:
    """Test speed of domain wall motion."""

    def setup_class(self):
        self.world = World(cellsize=(2e-9, 2e-9, 2e-9))
        self.magnet = Antiferromagnet(self.world, Grid((1, 100, 200)))
        self.magnet.msat = 0.4e6
        self.magnet.alpha = 0.1
        self.magnet.ku1 = 64e3
        self.magnet.anisU = (1, 0, 0)
        self.magnet.aex = 10e-12
        self.magnet.afmex_cell = -15e-12
        self.magnet.afmex_nn = -5e-12
        self.magnet.sub1.dmi_tensor.set_interfacial_dmi(0.11e-3)
        self.magnet.sub2.dmi_tensor.set_interfacial_dmi(0.11e-3)

        initialize(self)

        # Slonczewski parameters
        self.magnet.pol = 0.044
        self.magnet.fixed_layer = (0, 1, 0)
        self.magnet.Lambda = 1
        self.magnet.free_layer_thickness = 2e-9
        self.magnet.jcur = (0, 0, 1e12)

    def test_domain_wall_speed(self):
        result = get_domain_wall_speed(self)
        wanted = compute_domain_wall_speed(self.magnet)
        assert max_relative_error(result, wanted) < RTOL
