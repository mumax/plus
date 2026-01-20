"""
The test used here is based on examples/errorscaling.py, since the
error scaling is highly dependent on the floating-point precision.
"""

import numpy as np
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Literal

import mumaxplus


def magnetic_moment_precession(time, initial_magnetization, hfield_z, damping):
    """Return the analytical solution of the LLG equation for a single magnetic
    moment and an applied field along the z direction.
    """
    mx, my, mz = initial_magnetization
    theta0 = np.acos(mz)
    phi0 = np.atan(my / mx)
    freq = mumaxplus.util.constants.GAMMALL_DEFAULT * hfield_z / (1 + damping ** 2)
    phi = phi0 + freq * time
    theta = np.pi - 2 * np.atan(np.exp(damping * freq * time) * np.tan(np.pi / 2 - theta0 / 2))
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def single_system(method, dt):
    """This function simulates a single spin in a magnetic field of 0.1 T without damping.

    Returns the absolute error between the simulation and the exact solution.

    Parameters:
    method -- The used simulation method
    dt     -- The time step
    """
    # --- Setup ---
    world = mumaxplus.World(cellsize=(1e-9, 1e-9, 1e-9))
    
    magnetization = (1/np.sqrt(2), 0, 1/np.sqrt(2))
    damping = 0.001
    hfield_z = 0.1  # External field strength
    duration = 2*np.pi/(mumaxplus.util.constants.GAMMALL_DEFAULT * hfield_z) * (1 + damping**2) * 10  # Time of 10 precessions

    magnet = mumaxplus.Ferromagnet(world, grid=mumaxplus.Grid((1, 1, 1)))
    magnet.enable_demag = False
    magnet.magnetization = magnetization
    magnet.alpha = damping
    magnet.aex = 10e-12
    magnet.msat = 1/mumaxplus.util.constants.MU0
    world.bias_magnetic_field = (0, 0, hfield_z)

    # --- Run the simulation ---
    world.timesolver.set_method(method)
    world.timesolver.adaptive_timestep = False
    world.timesolver.timestep = dt
    
    world.timesolver.run(duration)
    output = magnet.magnetization.average()

    # --- Compare with exact solution ---
    exact = magnetic_moment_precession(duration, magnetization, hfield_z, damping)
    error = np.linalg.norm(exact - output)

    return error


def run_with_precision(precision: Literal["SINGLE", "DOUBLE"]):
    FP_PRECISION_CPP = {"SINGLE": 1,
                        "DOUBLE": 2}.get(precision)
    PREC_TEST = {"SINGLE": '> 1e-4',
                 "DOUBLE": '< 2e-6'}.get(precision)
    
    code = textwrap.dedent(f"""
        try: import mumaxplus
        except ModuleNotFoundError: exit()
        from {Path(__file__).stem} import single_system
        
        assert mumaxplus._cpp.FP_PRECISION == {FP_PRECISION_CPP}
        assert mumaxplus.FP_PRECISION == "{precision}"
        assert single_system("DormandPrince", 1.1e-11) {PREC_TEST}
    """)
    subprocess.check_output([sys.executable, "-c", code, "--mumaxplus-fp-precision", precision], cwd=Path(__file__).parent)
    subprocess.check_output([sys.executable, "-c", code], cwd=Path(__file__).parent, env=os.environ | {"MUMAXPLUS_FP_PRECISION": precision})


def test_single_precision():
    run_with_precision("SINGLE")


def test_double_precision():
    run_with_precision("DOUBLE")
