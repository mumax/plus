import pytest
import numpy as np

from mumax3 import Mumax3Simulation
from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.util import *

RTOL = 1e-5  # 0.001%

def max_relative_error(result, wanted):
    err = np.linalg.norm(result - wanted, axis=0)
    relerr = err / np.linalg.norm(wanted, axis=0)
    return np.max(relerr)

xi1, xi2, xi3, xi4 = 0.0, 0.05, 0.1, 0.5
@pytest.fixture(scope="class", params=[xi1, xi2, xi3, xi4])
def simulations(request):
    """Sets up a simulation to check the STT for both mumaxplus and mumax3, given
    a specific non-adiabacity xi. This is meant to be a quicker test than STP5.
    """

    # === specifications ===
    length, width, thickness = 100e-9, 100e-9, 10e-9
    nx, ny, nz = 50, 50, 5  # following mumax3 paper
    cellsize = (length/nx, width/ny, thickness/nz)
    gridsize = (nx, ny, nz)

    msat = 800e3
    aex = 13e-12
    alpha = 0.1

    xi = request.param
    pol = 1.0  # purely polarized current
    jcur = (1e12, 0, 0)

    # === mumax3 ===
    mumax3sim = Mumax3Simulation(
        f"""
            setcellsize{tuple(cellsize)}
            setgridsize{tuple(gridsize)}
            DisableSlonczewskiTorque = True

            msat = {msat}
            aex = {aex}
            alpha = {alpha}

            m = vortex(1, 1)
            minimize()
            SaveAs(m, "m_initial.ovf")

            xi = {xi}
            Pol = {pol}
            J = vector{tuple(jcur)}

            saveas(STTorque, "STT.ovf")
        """
    )

    # === mumaxplus ===

    world = World(cellsize=cellsize)
    magnet = Ferromagnet(world, Grid(gridsize))
    magnet.enable_slonczewski_torque = False  # Only check Zhang-Li torque

    magnet.msat = msat
    magnet.aex = aex
    magnet.alpha = alpha

    magnet.magnetization = mumax3sim.get_field("m_initial")

    magnet.xi = xi
    magnet.pol = pol
    magnet.jcur = jcur

    return  world, magnet, mumax3sim

@pytest.mark.mumax3
class TestZhangLi:
    """Compare the results of the simulations by comparing the STT and total torque."""

    def test_STT(self, simulations):
        world, magnet, mumax3sim = simulations
        err = max_relative_error(magnet.spin_transfer_torque.eval(), mumax3sim.get_field("STT") * GAMMALL)
        assert err < RTOL
    
    def test_total(self, simulations):
        world, magnet, mumax3sim = simulations
        err = max_relative_error(magnet.torque.eval() - magnet.llg_torque.eval(), magnet.spin_transfer_torque.eval())
        assert err < RTOL
