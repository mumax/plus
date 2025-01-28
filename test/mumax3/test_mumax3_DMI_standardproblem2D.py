"""This test is based on the 2D case in
   https://iopscience.iop.org/article/10.1088/1367-2630/aaea1c"""

import pytest
import numpy as np
from mumaxplus import Ferromagnet, Grid, World
from mumax3 import Mumax3Simulation
from mumaxplus.util.shape import Cylinder
from mumaxplus.util.config import neelskyrmion


ATOL = 1e-3
def max_absolute_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    return np.max(err)

def simulations(openBC):
    """This simulates a 2D circle with interfacial DMI and a Neél skyrmion
       in both mumax³ and mumax⁺."""
    
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

    # mumax⁺ simulation
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

    # mumax³ simulation
    mumax3sim = Mumax3Simulation(
        f"""
            // Set mesh and disk geometry
            SetGridSize{tuple(gridsize)}
            SetCellSize{tuple(cellsize)}
            SetGeom(Circle({diam}))

            Msat        = {Ms}
            Aex         = {A}
            Ku1         = {Ku}
            anisU       = vector{tuple(anisU)}
            Dind        = {D}
            // No Demag:
            EnableDemag = false

            openBC = {openBC}

            // Initial state
            m = Neelskyrmion({charge}, {pol})

            minimize()
            SaveAs(m, "m.ovf")
        """
    )

    return  magnet, mumax3sim


@pytest.mark.mumax3
class TestDMI2D:
    """Compare the results of the simulations by comparing the magnetizations.
    """

    def test_closed(self):
        magnet, mumax3sim = simulations(False)
        err = max_absolute_error(magnet.magnetization.eval(), mumax3sim.get_field("m"))
        assert err < ATOL

    def test_open(self):
        magnet, mumax3sim = simulations(True)
        err = max_absolute_error(magnet.magnetization.eval(), mumax3sim.get_field("m"))
        assert err < ATOL