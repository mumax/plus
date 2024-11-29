import pytest
import numpy as np

from mumax3 import Mumax3Simulation
from mumaxplus import Ferromagnet, Grid, World

RTOL = 3e-2  # 3% is quite large :(

def max_absolute_error(result, wanted):
    return np.max(abs(result - wanted))

def max_relative_error(result, wanted):
    return np.max(abs((result - wanted)/wanted))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))


B1 = (-24.6e-3, 4.3e-3, 0)  # field 1
B2 = (-35.5e-3, -6.3e-3, 0)  # field 2
@pytest.fixture(scope="class", params=[B1, B2])
def simulations(request):
    """Sets up and runs standard problem 4 for both mumax⁺ and mumax³, given
    a specific magnetic field. The magnetization and energies throughout time
    can later be compared quickly.
    """
    B_mag = request.param

    # === specifications ===
    length, width, thickness = 500e-9, 125e-9, 3e-9
    nx, ny, nz = 128, 32, 1

    msat = 800e3
    aex = 13e-12
    alpha = 0.02

    magnetization = (1, 0.1, 0)

    max_time = 1e-9
    step_time = 1e-12
    

    # === mumax⁺ ===
    world = World(cellsize=(length / nx, width / ny, thickness / nz))
    magnet = Ferromagnet(world, Grid((nx, ny, nz)))
    
    magnet.msat = msat
    magnet.aex = aex
    magnet.alpha = alpha

    magnet.magnetization = magnetization
    magnet.minimize()

    world.bias_magnetic_field = B_mag

    timepoints = np.arange(0, max_time + 0.5*step_time, step_time)
    outputquantities = {
        "t": lambda: world.timesolver.time,
        "mx": lambda: magnet.magnetization.average()[0],
        "my": lambda: magnet.magnetization.average()[1],
        "mz": lambda: magnet.magnetization.average()[2],
        "E_total": magnet.total_energy,
        "E_exch": magnet.exchange_energy,
        "E_Zeeman": magnet.zeeman_energy,
        "E_demag": magnet.demag_energy
    }
    mumaxplusoutput = world.timesolver.solve(timepoints, outputquantities)


    # === mumax³ ===
    mumax3sim = Mumax3Simulation(
        f"""
            setcellsize{tuple(world.cellsize)}
            setgridsize{tuple(magnet.grid.size)}

            msat = {msat}
            aex = {aex}
            alpha = {alpha}

            m = uniform{tuple(magnetization)}
            minimize()

            tableadd(E_total)
            tableadd(E_exch)
            tableadd(E_zeeman)
            tableadd(E_demag)

            tableautosave({step_time})
            B_ext = vector{tuple(B_mag)}

            run({max_time})
        """
    )

    return mumaxplusoutput, mumax3sim

@pytest.mark.mumax3
class TestStandardProblem4:
    """Compare the results of standard problem #4 of mumax⁺ against mumax³.
    Standard Problems: http://www.ctcms.nist.gov/~rdm/mumag.org.html
    Number 4: https://www.ctcms.nist.gov/~rdm/std4/spec4.html
    """

    # TODO: Add test for requested output 2:
    # An image of the  magnetization at the time when the x-component
    # of the spatially averaged magnetization first crosses zero

    def test_magnetization_x(self, simulations):
        mumaxplusoutput, mumax3sim = simulations
        # absolute error: mx goes through 0, but is unitless
        err = max_absolute_error(
            result=mumaxplusoutput["mx"], wanted=mumax3sim.get_column("mx")
        )
        assert err < RTOL

    def test_magnetization_y(self, simulations):
        mumaxplusoutput, mumax3sim = simulations
        # absolute error: my goes through 0, but is unitless
        err = max_absolute_error(
            result=mumaxplusoutput["my"], wanted=mumax3sim.get_column("my")
        )
        assert err < RTOL

    def test_magnetization_z(self, simulations):
        mumaxplusoutput, mumax3sim = simulations
        # absolute error: mz goes through 0, but is unitless
        err = max_absolute_error(
            result=mumaxplusoutput["mz"], wanted=mumax3sim.get_column("mz")
        )
        assert err < RTOL

    def test_total_energy(self, simulations):
        mumaxplusoutput, mumax3sim = simulations
        # semirelative: E_total goes through 0, but has a unit
        err = max_semirelative_error(
            result=mumaxplusoutput["E_total"], wanted=mumax3sim.get_column("E_total")
        )
        assert err < RTOL

    def test_exchange_energy(self, simulations):
        mumaxplusoutput, mumax3sim = simulations
        # semirelative: E_exch is always positive, but sometimes too close to 0
        err = max_semirelative_error(
            result=mumaxplusoutput["E_exch"], wanted=mumax3sim.get_column("E_exch")
        )
        assert err < RTOL

    def test_zeeman_energy(self, simulations):
        mumaxplusoutput, mumax3sim = simulations
        # semirelative: E_Zeeman goes through 0, but has a unit
        err = max_semirelative_error(
            result=mumaxplusoutput["E_Zeeman"], wanted=mumax3sim.get_column("E_Zeeman")
        )
        assert err < RTOL

    def test_demag_energy(self, simulations):
        mumaxplusoutput, mumax3sim = simulations
        # semirelative: E_demag is probably always positive, but sometimes too close to 0
        err = max_semirelative_error(
            result=mumaxplusoutput["E_demag"], wanted=mumax3sim.get_column("E_demag")
        )
        assert err < RTOL
