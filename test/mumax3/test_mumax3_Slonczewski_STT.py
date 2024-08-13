import pytest
import numpy as np
import math

from mumax3 import Mumax3Simulation
from mumaxplus import Ferromagnet, Grid, World

# fairly large tolerance because timepoints don't exactly match up
# => large error for large change in magnetization
ATOL = 2e-2  # 2%

def max_absolute_error(result, wanted):
    return np.max(abs(result - wanted))


@pytest.mark.mumax3
class TestSlonczewskiSTT:
    """Compare the results of a Slonczewski STT test of mumaxplus against mumax3.
    This is based on a test in the paper "The design and verification of MuMax3".
    https://doi.org/10.1063/1.4899186 """

    def setup_class(self):        
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
        eps_prime = 1
        jz = -6e-3/(length*width)
        jcur = (0, 0, jz)
        mp = (math.cos(20*np.pi/180), math.sin(20*np.pi/180), 0)  # fixed layer mag

        max_time = 0.5e-9
        step_time = 0.5e-12


        # === mumaxplus ===
        world = World(cellsize=(cx, cy, cz))
        magnet = Ferromagnet(world, Grid((nx, ny, nz)))
        magnet.msat = msat
        magnet.aex = aex
        magnet.alpha = alpha
        magnet.magnetization = magnetization
        magnet.minimize()

        magnet.pol = pol
        magnet.Lambda = Lambda
        magnet.eps_prime = eps_prime
        magnet.jcur = jcur
        magnet.FixedLayer = mp

        timepoints = np.arange(0, max_time + 0.5*step_time, step_time)
        outputquantities = {"mx": lambda: magnet.magnetization.average()[0],
                            "my": lambda: magnet.magnetization.average()[1],
                            "mz": lambda: magnet.magnetization.average()[2]}
        self.mumaxplusoutput = world.timesolver.solve(timepoints, outputquantities)


        # === mumax3 ===
        self.mumax3sim = Mumax3Simulation(
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
                EpsilonPrime = {eps_prime}
                J = vector{tuple(jcur)}
                FixedLayer = vector{tuple(mp)}

                TableAutoSave({step_time})
                Run({max_time})
                """
            )


    def test_magnetization_x(self):
        # absolute error: mx goes through 0, but is unitless
        err = max_absolute_error(result=self.mumaxplusoutput["mx"],
                                 wanted=self.mumax3sim.get_column("mx"))
        assert err < ATOL

    def test_magnetization_y(self):
        # absolute error: my goes through 0, but is unitless
        err = max_absolute_error(result=self.mumaxplusoutput["my"],
                                 wanted=self.mumax3sim.get_column("my"))
        assert err < ATOL

    def test_magnetization_z(self):
        # absolute error: mz goes through 0, but is unitless
        err = max_absolute_error(result=self.mumaxplusoutput["mz"],
                                 wanted=self.mumax3sim.get_column("mz"))
        assert err < ATOL
