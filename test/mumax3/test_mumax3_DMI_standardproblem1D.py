"""This test is based on the 1D case in
   https://iopscience.iop.org/article/10.1088/1367-2630/aaea1c
   The magnet lies along the z-direction, because mumax³ ignores derivatives in
   the z-direction when there are no cells in that direction. See issue #352 on
   the mumax³ github repository https://github.com/mumax/3/issues/352.
   """

import pytest
import numpy as np
from mumaxplus import Ferromagnet, Grid, World
from mumax3 import Mumax3Simulation

RTOL = 1e-5
def max_relative_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    relerr = err / np.linalg.norm(wanted, axis=0)
    return np.max(relerr)

def simulations(openbc, interfacial):
    """This simulates a 1D wire with bulk or interfacial DMI in both
       mumax³ and mumax⁺."""

    # constants
    A = 13e-12
    D = 3e-3
    Ku = 0.4e6
    anisU = (1,0,0)
    Ms = 0.86e6

    cellsize = (1e-9, 1e-9, 1e-9)
    gridsize = (1, 1, 100)
    magnetization = (1,0,0)

    # mumax⁺ simulation
    world = World(cellsize=cellsize)
    magnet = Ferromagnet(world, Grid(gridsize))
    magnet.enable_demag = False
    magnet.enable_openbc = openbc

    magnet.msat = Ms
    magnet.aex = A
    magnet.ku1 = Ku
    magnet.anisU = anisU

    if interfacial:
        DMI = "Dind"
        magnet.dmi_tensor.set_interfacial_dmi(D)
    else:
        DMI = "Dbulk"
        magnet.dmi_tensor.set_bulk_dmi(D)
    
    magnet.magnetization = magnetization
    magnet.minimize()
    
    # mumax³ simulation
    mumax3sim = Mumax3Simulation(
        f"""
            setcellsize{cellsize}
            setgridsize{gridsize}
            msat = {Ms}
            aex = {A}
            Ku1 = {Ku}
            anisU = vector{anisU}
            {DMI} = {D}
            enabledemag = False
            openBC = {openbc}

            m = Uniform{tuple(magnetization)}
            minimize()
            saveas(m, "m.ovf")
        """
    )

    return  magnet, mumax3sim


@pytest.mark.mumax3
class TestDMI1D:
    """Compare the results of the simulations by comparing the magnetizations.
    """

    def test_closed_inter(self):
        magnet, mumax3sim = simulations(False, True)
        err = max_relative_error(magnet.magnetization.eval(), mumax3sim.get_field("m"))
        assert err < RTOL

    def test_closed_bulk(self):
        magnet, mumax3sim = simulations(False, False)
        err = max_relative_error(magnet.magnetization.eval(), mumax3sim.get_field("m"))
        assert err < RTOL

    def test_open_inter(self):
        magnet, mumax3sim = simulations(True, True)
        err = max_relative_error(magnet.magnetization.eval(), mumax3sim.get_field("m"))
        assert err < RTOL

    def test_open_bulk(self):
        magnet, mumax3sim = simulations(True, False)
        err = max_relative_error(magnet.magnetization.eval(), mumax3sim.get_field("m"))
        assert err < RTOL