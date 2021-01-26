from math import isclose

import numpy as np
from numpy import arccos, arctan, cos, exp, pi, sin, sqrt, tan
import pytest

from mumax5 import Ferromagnet, Grid, World

VALID_METHOD_NAMES = [
    "Heun",
    "BogackiShampine",
    "CashKarp",
    "Fehlberg",
    "DormandPrince",
]


def magnetic_moment_precession(time, initial_magnetization, hfield_z, damping=0.0):
    """Return the analytical solution of the LLG equation for a single magnetic
    moment and an applied field along the z direction.
    """
    mx, my, mz = initial_magnetization
    theta0 = arccos(mz / sqrt(mx ** 2 + my ** 2 + mz ** 2))
    phi0 = arctan(my / mx)
    gammaLL = 1.76086e11
    freq = gammaLL * hfield_z / (1 + damping ** 2)
    phi = phi0 + freq * time
    theta = pi - 2 * arctan(exp(damping * freq * time) * tan(pi / 2 - theta0 / 2))
    return {"mx": sin(theta) * cos(phi), "my": sin(theta) * sin(phi), "mz": cos(theta)}


@pytest.fixture
def test_world():
    """Return a new empty world for testing purposes."""
    return World(cellsize=(1, 1, 1))


def test_initial_time_is_zero(test_world):
    assert isclose(test_world.timesolver.time, 0.0, rel_tol=1e-5)


@pytest.mark.parametrize("time", [0.0, 1e-9, 10, -1])
def test_set_time(test_world, time):
    test_world.timesolver.time = time
    assert isclose(test_world.timesolver.time, time, rel_tol=1e-5)


@pytest.mark.parametrize("method", VALID_METHOD_NAMES)
def test_set_method(test_world, method):
    test_world.timesolver.set_method(method)


@pytest.mark.parametrize("invalid_method", ["", 2, "asdf", None])
def test_set_invalid_method(test_world, invalid_method):
    with pytest.raises((TypeError, ValueError)):
        test_world.timesolver.set_method(invalid_method)


@pytest.mark.parametrize("timestep", [1e-9, 1.2, 7e-14])
def test_set_timestep(test_world, timestep):
    test_world.timesolver.timestep = timestep
    assert isclose(test_world.timesolver.timestep, timestep, rel_tol=1e-5)


def test_adaptive_timestep_default_is_true(test_world):
    assert test_world.timesolver.adaptive_timestep == True


@pytest.mark.parametrize("adaptive", [True, False])
def test_set_adaptive_timestep(test_world, adaptive):
    test_world.timesolver.adaptive_timestep = adaptive
    assert test_world.timesolver.adaptive_timestep == adaptive


def test_fixed_timestep(test_world):
    dt, nsteps = 2.3e-9, 103
    test_world.timesolver.adaptive_timestep = False
    test_world.timesolver.timestep = dt
    test_world.timesolver.steps(nsteps)
    assert isclose(test_world.timesolver.time, nsteps * dt, rel_tol=1e-5)


@pytest.mark.parametrize("method", VALID_METHOD_NAMES)
def test_solve_single_system(test_world, method):
    """Test the solver for a single system.

    The system which is used for the test here is a single magnetic moment which
    precesses around a magnetic field along the z direction. The output is checked
    against the analytical solution.
    """
    # arbitrary parameters used for this test
    hfield_z = 0.11
    damping = 0.2
    m0 = (1, 0, 0)
    timepoints = np.linspace(0, 5e-10, 20)

    magnet = Ferromagnet(test_world, grid=Grid((1, 1, 1)))
    magnet.enable_demag = False
    magnet.magnetization = m0
    magnet.alpha = damping

    test_world.bias_magnetic_field = (0, 0, hfield_z)
    test_world.timesolver.set_method(method)

    output = test_world.timesolver.solve(
        timepoints=timepoints,
        quantity_dict={
            "mx": lambda: magnet.magnetization.average()[0],
            "my": lambda: magnet.magnetization.average()[1],
            "mz": lambda: magnet.magnetization.average()[2],
        },
    )

    exact = magnetic_moment_precession(timepoints, m0, hfield_z, damping)

    for mc in ["mx", "my", "mz"]:
        max_error = np.max(np.abs(output[mc] - exact[mc]))
        assert max_error < 1e-2

    # # uncomment for visual testing
    # import matplotlib.pyplot as plt
    # plt.plot(exact['mx'], exact['my'])
    # plt.plot(output['mx'], output['my'])
    # plt.show()


@pytest.mark.parametrize("method", VALID_METHOD_NAMES)
def test_solve_multiple_systems(test_world, method):
    """Test the solver for multiple systems.

    The systems which are used for the test here are two uncoupled magnetic moments
    which precess around a magnetic field along the z direction. The output is
    checked against the analytical solution.
    """
    # arbitrary parameters used for this test
    hfield_z = 0.11
    damping = 0.2
    m0 = (1, 0, 0)
    timepoints = np.linspace(0, 5e-11, 10)

    magnets = []
    for i in [0, 1]:
        magnet = Ferromagnet(test_world, grid=Grid((1, 1, 1), origin=(i, 0, 0)))
        magnet.enable_demag = False
        magnet.magnetization = m0
        magnet.alpha = damping
        magnets.append(magnet)

    test_world.bias_magnetic_field = (0, 0, hfield_z)
    test_world.timesolver.set_method(method)

    output = test_world.timesolver.solve(
        timepoints=timepoints,
        quantity_dict={
            "mx_0": lambda: magnets[0].magnetization.average()[0],
            "my_0": lambda: magnets[0].magnetization.average()[1],
            "mz_0": lambda: magnets[0].magnetization.average()[2],
            "mx_1": lambda: magnets[1].magnetization.average()[0],
            "my_1": lambda: magnets[1].magnetization.average()[1],
            "mz_1": lambda: magnets[1].magnetization.average()[2],
        },
    )

    exact = magnetic_moment_precession(timepoints, m0, hfield_z, damping)

    for magnet_idx in [0, 1]:
        for mc in ["mx", "my", "mz"]:
            max_error = np.max(np.abs(output[mc + f"_{magnet_idx}"] - exact[mc]))
            assert max_error < 1e-2
