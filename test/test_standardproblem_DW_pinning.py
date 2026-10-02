"""This script solves the proposed micromagnetic standard problem about domain
wall pinning at a boundary between a soft and a hard magnetic phase, with an
external magnetic field pointing to the right. The question is at what magnetic
field strength the domain wall unpins and moves to the right.

The problem specification can be found in the following paper.
https://doi.org/10.1016/j.jmmm.2021.168875
"""

import numpy as np
import pytest

from mumaxplus import World, Grid, Ferromagnet
from mumaxplus.util.constants import MU0


def analytical_pinning_field(A1, A2, K1, K2, Ms1, Ms2):
    """Returns the minimum field strength in Tesla at which the domain wall
    unpins from the boundary between phases 1 and 2.
    """
    return 2 * K2 / Ms2 * (1 - K1/K2 * A1/A2) / (1 + np.sqrt(Ms1/Ms2 * A1/A2)) ** 2


@pytest.mark.slow
def test_stdp_DW_pinning():
    # material parameters
    Asoft = 0.25e-11
    Ahard = 1.0e-11
    Ksoft = 1e5
    Khard = 1e6
    Ms_soft = 0.25 / MU0
    Ms_hard = 1 / MU0

    # choose combination of soft and hard parameters for phases 1 and 2
    A1, K1, Ms1 = Asoft, Ksoft, Ms_soft
    A2, K2, Ms2 = Ahard, Khard, Ms_hard

    cx, cy, cz = 1e-9, 1e-9, 1e-9
    nx, ny, nz = 80, 1, 1  # each phase 40 nm long

    world = World((cx, cy, cz))
    magnet = Ferromagnet(world, Grid((nx, ny, nz)),
                regions=lambda x,y,z: 1 if x < (nx - 1) / 2 * cx else 2)  # left 1, right 2

    magnet.enable_demag = False  # disabled for simplicity
    magnet.anisU = (1, 0, 0)
    magnet.alpha = 1.0

    # left
    magnet.aex.set_in_region(1, A1)
    magnet.ku1.set_in_region(1, K1)
    magnet.msat.set_in_region(1, Ms1)

    # right
    magnet.aex.set_in_region(2, A2)
    magnet.ku1.set_in_region(2, K2)
    magnet.msat.set_in_region(2, Ms2)

    # magnetic field steps
    B_pin_anal = analytical_pinning_field(A1, A2, K1, K2, Ms1, Ms2)
    Bx_min = 0.1 * B_pin_anal  # prevent DW from being pushed out on the left side
    Bx_max = 2
    Bx_steps = 200 + 1
    Bx_array = np.linspace(Bx_min, Bx_max, Bx_steps)

    # initial configuration
    magnet.bias_magnetic_field = (Bx_min, 0, 0)
    magnet.magnetization = lambda x, y, z: (1.0, 0.3, 0.0) if x < magnet.center[0] else (-1.0, 0.3, 0.0)
    magnet.relax()  # relax to initial configuration

    mx_array = np.zeros_like(Bx_array)
    for i, Bx in enumerate(Bx_array):
        magnet.bias_magnetic_field = (Bx, 0, 0)
        magnet.minimize()
        mx_array[i] = magnet.magnetization.average()[0]


    # find sudden big change in mx
    mx_diff_tol = 0.6
    B_pin_numeric = Bx_max  # default value
    B_pin_tol = 0.5 * (Bx_max - Bx_min) / (Bx_steps - 1)  # half distance between steps

    for i in range(1, Bx_steps):
        if mx_array[i] - mx_array[i - 1] > mx_diff_tol:
            # (de)pinning field is in the interval (Bx_array[i-1], Bx_array[i]]
            B_pin_numeric = 0.5 * (Bx_array[i] + Bx_array[i - 1])
            break

    assert abs(B_pin_numeric - B_pin_anal) <= B_pin_tol
