"""This test compares the spin transfer torque of mumax⁺ against mumax³ when both
the Zhang-Li and the Slonczewski torques are present: the torque on a magnetization
that varies along z, with an in-plane current and a current along z, and the
dynamics of a vortex driven by both torques.
"""

from mumax3 import Mumax3Simulation
import numpy as np
import pytest

from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.util import *

RTOL = 1e-5  # torque, relative to its maximum
ATOL = 1e-3  # average magnetization


def max_norm(field):
    return np.max(np.linalg.norm(field, axis=0))


def max_semirelative_error(result, wanted):
    """Maximum error for vector quantities, divided by the maximum of wanted."""
    return max_norm(result - wanted) / max_norm(wanted)


def max_absolute_error(result, wanted):
    return np.max(abs(result - wanted))


@pytest.fixture(scope="class", params=[True, False])
def torque_simulations(request):
    """Sets up a random magnetization, which varies along z, with both torques
    present, given the position of the fixed layer. mumax³ saves the spin transfer
    torque with both torques, with the Zhang-Li torque alone and with the
    Slonczewski torque alone.
    """

    # === specifications ===
    nx, ny, nz = 32, 24, 4
    cellsize = (2e-9, 2e-9, 2e-9)
    gridsize = (nx, ny, nz)

    msat = 800e3
    aex = 13e-12
    alpha = 0.1

    pol = 0.6
    xi = 0.05
    Lambda = 2
    epsilon_prime = 0.1
    jcur = (5e11, -2.5e11, 1e12)  # all components for Zhang-Li, jz for Slonczewski
    fixed_layer = (1, 0.5, -0.3)
    fixed_layer_on_top = request.param
    position = "FIXEDLAYER_TOP" if fixed_layer_on_top else "FIXEDLAYER_BOTTOM"

    # === mumax³ ===
    mumax3sim = Mumax3Simulation(
        f"""
            setcellsize{tuple(cellsize)}
            setgridsize{tuple(gridsize)}

            msat = {msat}
            aex = {aex}
            alpha = {alpha}

            m = RandomMagSeed(1234)
            SaveAs(m, "m_initial.ovf")

            xi = {xi}
            Pol = {pol}
            Lambda = {Lambda}
            EpsilonPrime = {epsilon_prime}
            FixedLayer = vector{tuple(fixed_layer)}
            FixedLayerPosition = {position}
            J = vector{tuple(jcur)}
            SaveAs(STTorque, "STT.ovf")

            DisableSlonczewskiTorque = true
            SaveAs(STTorque, "STT_zhang_li.ovf")
            DisableSlonczewskiTorque = false
            DisableZhangLiTorque = true
            SaveAs(STTorque, "STT_slonczewski.ovf")
        """
    )

    # === mumax⁺ ===
    world = World(cellsize=cellsize)
    magnet = Ferromagnet(world, Grid(gridsize))

    magnet.msat = msat
    magnet.aex = aex
    magnet.alpha = alpha

    magnet.magnetization = mumax3sim.get_field("m_initial")

    magnet.xi = xi
    magnet.pol = pol
    magnet.Lambda = Lambda
    magnet.epsilon_prime = epsilon_prime
    magnet.fixed_layer = fixed_layer
    magnet.fixed_layer_on_top = fixed_layer_on_top
    magnet.jcur = jcur

    return magnet, mumax3sim


@pytest.mark.mumax3
class TestZhangLiSlonczewskiTorque:
    """Compare the spin transfer torque of mumax⁺ and mumax³, with both torques and
    with each torque alone."""

    def test_both(self, torque_simulations):
        magnet, mumax3sim = torque_simulations
        err = max_semirelative_error(
            magnet.spin_transfer_torque.eval(),
            mumax3sim.get_field("STT") * GAMMALL_DEFAULT,
        )
        assert err < RTOL

    def test_zhang_li(self, torque_simulations):
        magnet, mumax3sim = torque_simulations
        magnet.enable_slonczewski_torque = False
        result = magnet.spin_transfer_torque.eval()
        magnet.enable_slonczewski_torque = True
        wanted = mumax3sim.get_field("STT_zhang_li") * GAMMALL_DEFAULT
        assert max_semirelative_error(result, wanted) < RTOL

    def test_slonczewski(self, torque_simulations):
        magnet, mumax3sim = torque_simulations
        magnet.enable_zhang_li_torque = False
        result = magnet.spin_transfer_torque.eval()
        magnet.enable_zhang_li_torque = True
        wanted = mumax3sim.get_field("STT_slonczewski") * GAMMALL_DEFAULT
        assert max_semirelative_error(result, wanted) < RTOL


@pytest.fixture(scope="class")
def dynamics_simulations():
    """Sets up and runs a vortex for mumax⁺ and mumax³, driven by an in-plane
    current through the Zhang-Li torque and by a current along z through the
    Slonczewski torque of an in-plane fixed layer.
    """

    # === specifications ===
    length, width, thickness = 100e-9, 100e-9, 10e-9
    nx, ny, nz = 50, 50, 5  # as in standard problem 5
    cellsize = (length / nx, width / ny, thickness / nz)
    gridsize = (nx, ny, nz)

    msat = 800e3
    aex = 13e-12
    alpha = 0.1

    pol = 0.6
    xi = 0.05
    Lambda = 1
    epsilon_prime = 0.05
    jcur = (1e12, 0, 2e10)
    fixed_layer = (1, 0, 0)

    max_time = 1e-9
    step_time = 5e-12

    # === mumax³ ===
    mumax3sim = Mumax3Simulation(
        f"""
            setcellsize{tuple(cellsize)}
            setgridsize{tuple(gridsize)}

            msat = {msat}
            aex = {aex}
            alpha = {alpha}

            m = vortex(1, 1)
            minimize()
            SaveAs(m, "m_initial.ovf")

            xi = {xi}
            Pol = {pol}
            Lambda = {Lambda}
            EpsilonPrime = {epsilon_prime}
            FixedLayer = vector{tuple(fixed_layer)}
            J = vector{tuple(jcur)}

            tableautosave({step_time})
            run({max_time})
        """
    )

    # === mumax⁺ ===
    world = World(cellsize=cellsize)
    magnet = Ferromagnet(world, Grid(gridsize))

    magnet.msat = msat
    magnet.aex = aex
    magnet.alpha = alpha

    magnet.magnetization = mumax3sim.get_field("m_initial")
    magnet.minimize()

    magnet.xi = xi
    magnet.pol = pol
    magnet.Lambda = Lambda
    magnet.epsilon_prime = epsilon_prime
    magnet.fixed_layer = fixed_layer
    magnet.jcur = jcur

    timepoints = np.arange(0, max_time + 0.5 * step_time, step_time)
    outputquantities = {
        "mx": lambda: magnet.magnetization.average()[0],
        "my": lambda: magnet.magnetization.average()[1],
        "mz": lambda: magnet.magnetization.average()[2],
    }
    mumaxplusoutput = world.timesolver.solve(timepoints, outputquantities)

    return mumaxplusoutput, mumax3sim


@pytest.mark.mumax3
class TestZhangLiSlonczewskiDynamics:
    """Compare the average magnetization of a vortex driven by both torques."""

    def test_magnetization_x(self, dynamics_simulations):
        mumaxplusoutput, mumax3sim = dynamics_simulations
        err = max_absolute_error(
            result=mumaxplusoutput["mx"], wanted=mumax3sim.get_column("mx")
        )
        assert err < ATOL

    def test_magnetization_y(self, dynamics_simulations):
        mumaxplusoutput, mumax3sim = dynamics_simulations
        err = max_absolute_error(
            result=mumaxplusoutput["my"], wanted=mumax3sim.get_column("my")
        )
        assert err < ATOL

    def test_magnetization_z(self, dynamics_simulations):
        mumaxplusoutput, mumax3sim = dynamics_simulations
        err = max_absolute_error(
            result=mumaxplusoutput["mz"], wanted=mumax3sim.get_column("mz")
        )
        assert err < ATOL
