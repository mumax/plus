"""This test is based on the 2D case in
   https://iopscience.iop.org/article/10.1088/1367-2630/aaea1c
   It compares the final magnetization of both sublattices in an antiferromagnet
   with that of a ferromagnet. All AFM exchanges are set to 0 for this test."""

import numpy as np
from mumaxplus import Antiferromagnet, Ferromagnet, Grid, World
from mumaxplus.util.shape import Cylinder
from mumaxplus.util.config import neelskyrmion


ATOL = 1e-4
def max_absolute_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    return np.max(err)

def simulations(openBC):
    """This simulates a 2D circle with interfacial DMI and a Neél skyrmion
       in a ferromagnet and an antiferromagnet."""
    
    # constants
    A = 13e-12
    D = 3e-3
    Ku = 0.4e6
    anisU = (0,0,1)
    Ms = 0.86e6

    # charge and polarization of the skyrmion
    charge, pol = -1, 1

    # diameter and thickness of the skyrmion
    diam = 100e-9
    thickness = 2e-9
    skyrmion_radius = 25e-9

    nx, ny, nz = 50, 50, 1
    dx, dy, dz = 2e-9, 2e-9, 2e-9

    gridsize = (nx, ny, nz)
    cellsize = (dx, dy, dz)

    # ferromagnet simulation
    world = World(cellsize=cellsize)
    geo = Cylinder(diam, thickness).translate((nx*dx-dx)/2, (ny*dy-dy)/2, 0)
    magnet = Ferromagnet(world, Grid(gridsize), geometry=geo)

    magnet.enable_demag = False
    magnet.enable_openbc = openBC

    magnet.msat = Ms
    magnet.aex = A
    magnet.ku1 = Ku
    magnet.anisU = anisU

    magnet.dmi_tensor.set_interfacial_dmi(D)
    magnet.magnetization = neelskyrmion(magnet.center, skyrmion_radius, charge, pol)
    magnet.minimize()

    # antiferromagnet simulation
    world_AFM = World(cellsize=cellsize)
    geo = Cylinder(diam, thickness).translate((nx*dx-dx)/2, (ny*dy-dy)/2, 0)
    magnet_AFM = Antiferromagnet(world_AFM, Grid(gridsize), geometry=geo)

    magnet_AFM.enable_demag = False
    magnet_AFM.enable_openbc = openBC

    magnet_AFM.msat = Ms
    magnet_AFM.aex = A
    magnet_AFM.afmex_cell = 0
    magnet_AFM.afmex_nn = 0
    magnet_AFM.ku1 = Ku
    magnet_AFM.anisU = anisU

    for sub in magnet_AFM.sublattices:
        sub.dmi_tensor.set_interfacial_dmi(D)
    magnet_AFM.magnetization = neelskyrmion(magnet.center, skyrmion_radius, charge, pol)
    magnet_AFM.minimize()

    return  magnet, magnet_AFM


class TestDMI2D:
    """Compare the results of the simulations by comparing the magnetizations.
    """

    def test_closed(self):
        magnet, magnet_AFM = simulations(False)
        err = max_absolute_error(magnet.magnetization.eval(), magnet_AFM.sub1.magnetization.eval())
        assert err < ATOL
        err = max_absolute_error(magnet.magnetization.eval(), magnet_AFM.sub2.magnetization.eval())
        assert err < ATOL

    def test_open(self):
        magnet, magnet_AFM = simulations(True)
        err = max_absolute_error(magnet.magnetization.eval(), magnet_AFM.sub1.magnetization.eval())
        assert err < ATOL
        err = max_absolute_error(magnet.magnetization.eval(), magnet_AFM.sub2.magnetization.eval())
        assert err < ATOL