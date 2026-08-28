"""This test is based on the 3D case in
   https://iopscience.iop.org/article/10.1088/1367-2630/aaea1c"""

import pytest
import numpy as np
from mumaxplus import Ferromagnet, Grid, World
from mumax3 import Mumax3Simulation
from mumaxplus.util.shape import Cylinder
from mumaxplus.util.config import blochskyrmion


ATOL = 2e-3
def max_absolute_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    return np.max(err)

def simulations(openBC):
    """This simulates a 3D cylinder with bulk DMI and a bloch skyrmion
       in both mumax³ and mumax⁺."""
    
    # constants
    A = 8.78e-12
    D = 1.58e-3
    Ms = 0.384e6
    Bz = 0.4

    # diameter and thickness of the skyrmion
    diam, thickness = 183e-9, 21e-9
    skyrmion_radius = 36e-9

    # charge and polarization of the skyrmion
    charge, pol = 1, -1

    nx, ny, nz = 183, 183, 21
    dx, dy, dz = 1e-9, 1e-9, 1e-9

    gridsize = (nx, ny, nz)
    cellsize = (dx, dy, dz)

    # mumax⁺ simulation
    world = World(cellsize=cellsize)
    geo = Cylinder(diam, thickness).translate((nx*dx-dx)/2, (ny*dy-dy)/2, (nz*dz-dz)/2)
    magnet = Ferromagnet(world, Grid(gridsize), geometry=geo)

    magnet.enable_demag = False
    magnet.enable_openbc = openBC
    magnet.msat = Ms
    magnet.aex = A
    magnet.dmi_tensor.set_bulk_dmi(D)

    magnet.bias_magnetic_field = (0,0,Bz)

    magnet.magnetization = blochskyrmion(magnet.center, skyrmion_radius, charge, pol)

    magnet.minimize()

    # mumax³ simulation
    mumax3sim = Mumax3Simulation(
        f"""
            lx := 183e-9
            ly := 183e-9
            lz := 21e-9

            SetGridSize{tuple(gridsize)}
            SetCellSize{tuple(cellsize)}

            // Define the cylinder
            SetGeom(Circle({diam}))

            Msat         = {Ms}
            Aex          = {A}
            Dbulk        = {D}

            // External field in T
            B_ext = vector(0, 0, {Bz})

            // No Demag
            EnableDemag = false

            openBC = {openBC}

            m = BlochSkyrmion({charge}, {pol})

            // Relax with conjugate gradient:
            minimize();
            SaveAs(m, "m")
        """
    )

    return  magnet, mumax3sim


@pytest.mark.mumax3
@pytest.mark.slow
class TestDMI3D:
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