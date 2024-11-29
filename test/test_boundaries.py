"""Test if the canting at the end of a nanowire corresponds to the 1D analytical
result if open boundary conditions are used. This test is similar to the
standard test proposed in arXiv:1803.11174

If the nanowire consists only out of one row of cells, the analytical canting
matches with the simulated canting if open (or periodic) boundary conditions are
used. The Neumann BC yields a different canting.

This does not mean that Neumann BC are wrong. To be more precise, the analytical
result, as well as the numerical results obtained with open and Neumann BC are
slightly wrong because the width of the nanowire is not taken into account
properly.

Due to this difference, the Neumann result is compared with the mumax³ Neumann
result.

The mumax³ results are as follows:
Neumann:   0.44716486657643906
Open:      0.5958252040658198
Periodic:  0.5958256832017506
"""


import pytest
import numpy as np

from mumaxplus import Ferromagnet, Grid, World


RTOL = 2e-4  # 0.02%

ncell = 1024
cs = 0.05

DMI = 0.9 * 4/np.pi  # 90% of critical DMI strength

minimizerstop = 1e-7


def relative_error(simulated, wanted):
    return np.abs(simulated - wanted)/wanted


def canting_x(magnet):  
    return np.arctan2(magnet.magnetization.eval()[0,0,0,0],  
                      magnet.magnetization.eval()[2,0,0,0]) 

def canting_y(magnet):  
    return np.arctan2(magnet.magnetization.eval()[1,0,0,0],  
                      magnet.magnetization.eval()[2,0,0,0]) 


def analytic():
    theta0 = np.arcsin(DMI/2)
    cant_analytic = 2*np.arctan(np.exp(-cs/2)*np.tan(theta0/2)) # shift towards center of the cell
    return cant_analytic


def simulation(x_dir):
    """Sets up the mumax⁺ simulation"""

    world = World(cellsize=(cs, cs, cs))

    if x_dir:
        magnet = Ferromagnet(world, Grid((ncell, 1, 1)))
    else:
        magnet = Ferromagnet(world, Grid((1, ncell, 1)))

    magnet.dmi_tensor.set_interfacial_dmi(DMI)
    magnet.enable_demag = False
    magnet.anisU = (0,0,1)
    magnet.aex = 1.
    magnet.ku1 = 1.
    magnet.msat = 1.
    magnet.magnetization = (0,0,1)

    return world, magnet

class TestBoundaries:
    def setup_class(self):
        """Creates a world and magnet in the x and y direction
        together with the desired result
        """
        self.world_x, self.magnet_x = simulation(True)  # Wire in the x-direction.
        self.world_y, self.magnet_y = simulation(False)  # Wire in the y-direction
        self.wanted = analytic()
    
    def test_neumann_x(self):
        """Compare the Neumann boundary conditions with mumax³
        for a wire in the x-direction."""
        self.magnet_x.enable_openbc = False
        self.magnet_x.minimize(minimizerstop)
        cant_neumann = canting_x(self.magnet_x)
        err = relative_error(cant_neumann, 0.44716486657643906)
        assert err < RTOL
    
    def test_open_x(self):
        """Compare the open boundary conditions with the analytical result
        for a wire in the x-direction."""
        self.magnet_x.enable_openbc = True
        self.magnet_x.minimize(minimizerstop)
        cant_open = canting_x(self.magnet_x)
        err = relative_error(cant_open, self.wanted)
        assert err < RTOL
    
    def test_periodic_x(self):
        """Compare the periodic boundary conditions with the analytical result
        for a wire in the x-direction."""
        self.magnet_x.enable_openbc = False
        self.world_x.set_pbc((0,1,0))
        self.magnet_x.minimize(minimizerstop)
        cant_period = canting_x(self.magnet_x)
        err = relative_error(cant_period, self.wanted)
        assert err < RTOL
    
    def test_neumann_y(self):
        """Compare the Neumann boundary conditions with mumax³
        for a wire in the y-direction."""
        self.magnet_y.enable_openbc = False
        self.magnet_y.minimize(minimizerstop)
        cant_neumann = canting_y(self.magnet_y)
        err = relative_error(cant_neumann, 0.44716486657643906)
        assert err < RTOL
    
    def test_open_y(self):
        """Compare the open boundary conditions with the analytical result
        for a wire in the y-direction."""
        self.magnet_y.enable_openbc = True
        self.magnet_y.minimize(minimizerstop)
        cant_open = canting_y(self.magnet_y)
        err = relative_error(cant_open, self.wanted)
        assert err < RTOL
    
    def test_periodic_y(self):
        """Compare the periodic boundary conditions with the analytical result
        for a wire in the y-direction."""
        self.magnet_y.enable_openbc = False
        self.world_y.set_pbc((1,0,0))
        self.magnet_y.minimize(minimizerstop)
        cant_period = canting_y(self.magnet_y)
        err = relative_error(cant_period, self.wanted)
        assert err < RTOL
