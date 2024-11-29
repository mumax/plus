import pytest
import numpy as np

from mumax3 import Mumax3Simulation
from mumaxplus import Ferromagnet, Grid, World

# fairly large tolerance because timepoints don't exactly match up
# => large error for large change in magnetization
ATOL = 2e-2  # 2%

def max_absolute_error(result, wanted):
    return np.max(abs(result - wanted))


@pytest.fixture(scope="class", params=[True, False])
def simulations(request):
    """Sets up and runs a Slonczewski STT test for mumax⁺ and mumax³.
    This is based on a test in the paper "The design and verification of MuMax3".
    https://doi.org/10.1063/1.4899186 """

    length, width, thickness = 160e-9, 80e-9, 5e-9
    nx, ny, nz = 64, 32, 1
    cx, cy, cz = length/nx, width/ny, thickness/nz

    # permalloy-like
    msat = 800e3
    aex = 13e-12
    alpha = 0.01
    magnetization = (1, 0, 0)

    # Slonczewski parameters chosen specifically so no term is zero
    pol = 0.5669
    Lambda = 2
    epsilon_prime = 1
    jz = -6e-3/(length*width)
    jcur = (0, 0, jz)
    mp = (np.cos(20*np.pi/180), np.sin(20*np.pi/180), 0)  # fixed layer mag

    max_time = 0.5e-9
    step_time = 0.5e-12


    # === mumax⁺ ===
    world = World(cellsize=(cx, cy, cz))
    magnet = Ferromagnet(world, Grid((nx, ny, nz)))
    magnet.msat = msat
    magnet.aex = aex
    magnet.alpha = alpha
    magnet.magnetization = magnetization
    magnet.minimize()

    magnet.pol = pol
    magnet.Lambda = Lambda
    magnet.epsilon_prime = epsilon_prime
    magnet.jcur = jcur
    magnet.fixed_layer = mp
    magnet.fixed_layer_on_top = request.param

    timepoints = np.arange(0, max_time + 0.5*step_time, step_time)
    outputquantities = {"mx": lambda: magnet.magnetization.average()[0],
                        "my": lambda: magnet.magnetization.average()[1],
                        "mz": lambda: magnet.magnetization.average()[2]}
    mumaxplusoutput = world.timesolver.solve(timepoints, outputquantities)


    # === mumax³ ===
    if request.param:
        FixedLayerPosition = "FIXEDLAYER_TOP"
    else:
        FixedLayerPosition = "FIXEDLAYER_BOTTOM"

    mumax3sim = Mumax3Simulation(
        f"""
            SetGridSize{(nx, ny, nz)}
            SetCellSize{(cx, cy, cz)}

            Msat = {msat}
            Aex = {aex}
            alpha = {alpha}
            m = Uniform{tuple(magnetization)}
            minimize()

            Pol = {pol}
            Lambda = {Lambda}
            EpsilonPrime = {epsilon_prime}
            J = vector{tuple(jcur)}
            FixedLayer = vector{tuple(mp)}
            FixedLayerPosition = {FixedLayerPosition}

            TableAutoSave({step_time})
            Run({max_time})
            """
        )
    return mumaxplusoutput, mumax3sim


@pytest.mark.mumax3
class TestSlonczewskiSTT:
    """Compare the results of a Slonczewski STT test of mumax⁺ against mumax³.
    This is based on a test in the paper "The design and verification of MuMax3".
    https://doi.org/10.1063/1.4899186 """

    def test_magnetization_x(self, simulations):
        # absolute error: mx goes through 0, but is unitless
        mumaxplusoutput, mumax3sim = simulations
        err = max_absolute_error(result=mumaxplusoutput["mx"],
                                 wanted=mumax3sim.get_column("mx"))
        assert err < ATOL

    def test_magnetization_y(self, simulations):
        # absolute error: my goes through 0, but is unitless
        mumaxplusoutput, mumax3sim = simulations
        err = max_absolute_error(result=mumaxplusoutput["my"],
                                 wanted=mumax3sim.get_column("my"))
        assert err < ATOL

    def test_magnetization_z(self, simulations):
        # absolute error: mz goes through 0, but is unitless
        mumaxplusoutput, mumax3sim = simulations
        err = max_absolute_error(result=mumaxplusoutput["mz"],
                                 wanted=mumax3sim.get_column("mz"))
        assert err < ATOL
