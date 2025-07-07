"""This file tests every term of the elastic force individually by comparing
the calculated elastic force to the analytical solutions.
Periodic boundry conditions are used in the direction(s) corresponding to the
derivative direction(s) so everything stays smooth, while zero derivative is
assumed in all other directions.
All stiffness constants are uniform in these tests for simplicity!
"""

import numpy as np
import math

from mumaxplus import Grid, World, Ferromagnet


SRTOL = 1e-4
SRTOL_MIX = 1e-4

cx, cy, cz = 1.5e-9, 2e-9, 2.5e-9
cellsize = (cx, cy, cz)
N1, N2 = 128, 256
P1, P2 = 2, 3  # number of sine periods
A = 1e-15  # displacement amplitude
C11 = 283e9
C44 = 58e9
C12 = 166e9


def max_absolute_error(result, wanted):
    """Maximum error for vector quantities."""
    return np.max(np.linalg.norm(result - wanted, axis=0))

def max_semirelative_error(result, wanted):
    """Like relative error, but divides by the maximum of wanted.
    Useful when removing units but the results go through zero.
    """
    return max_absolute_error(result, wanted) / np.max(abs(wanted))


def make_long_magnet(d_comp):
    """Makes a world with a magnet of the appropriate size according to the
    direction of the double derivative.
    """
    gridsize, pbc_repetitions = [0, 0, 0], [0, 0, 0]
    gridsize[d_comp], pbc_repetitions[d_comp] = N1, 1  # set for PBC grid
    world = World(cellsize, mastergrid=Grid(gridsize), pbc_repetitions=pbc_repetitions)

    gridsize = [1, 1, 1]  # set for magnet
    gridsize[d_comp] = N1
    magnet =  Ferromagnet(world, Grid(gridsize))
    magnet.enable_elastodynamics = True

    return magnet


def set_and_check_sine_force(magnet, d_comp, u_comp, C):
    """Sets given component of the displacement of the given magnet to a sine
    wave, the periodicity of which depends on the direction of the derivative.
    Then checks if the simulated force corresponds to the expected analytical force.
    You need to set C yourself.
    """

    L = N1*cellsize[d_comp]
    k = P1 * 2*math.pi/L
    def displacement_func(x, y, z):
        u = [0., 0., 0.]
        relevant_coord = (x, y, z)[d_comp]
        u[u_comp] = A * math.sin(k * relevant_coord)
        return tuple(u)

    magnet.elastic_displacement = displacement_func
    
    force_num = magnet.internal_body_force.eval()
    force_anal = - k**2 * C * magnet.elastic_displacement.eval()

    assert max_semirelative_error(force_num, force_anal) < SRTOL


# ==================================================
# Tests for double derivative along the corresponding direction
# C11 is constant TODO: vary C11 as well

def test_dx_dx_ux():
    magnet = make_long_magnet(d_comp=0)
    magnet.C11 = C11
    set_and_check_sine_force(magnet, d_comp=0, u_comp=0, C=C11)

def test_dy_dy_uy():
    magnet = make_long_magnet(d_comp=1)
    magnet.C11 = C11
    set_and_check_sine_force(magnet, d_comp=1, u_comp=1, C=C11)

def test_dz_dz_uz():
    magnet = make_long_magnet(d_comp=2)
    magnet.C11 = C11
    set_and_check_sine_force(magnet, d_comp=2, u_comp=2, C=C11)

# ==================================================
# Tests for double derivative along the different direction
# C44 is constant TODO: vary C44 as well (careful of mixed derivatives!)

def test_dy_dy_ux():
    magnet = make_long_magnet(d_comp=1)
    magnet.C44 = C44
    magnet.C12 = -C44  # to remove mixed derivative
    set_and_check_sine_force(magnet, d_comp=1, u_comp=0, C=C44)

def test_dz_dz_ux():
    magnet = make_long_magnet(d_comp=2)
    magnet.C44 = C44
    magnet.C12 = -C44  # to remove mixed derivative
    set_and_check_sine_force(magnet, d_comp=2, u_comp=0, C=C44)


def test_dx_dx_uy():
    magnet = make_long_magnet(d_comp=0)
    magnet.C44 = C44
    magnet.C12 = -C44  # to remove mixed derivative
    set_and_check_sine_force(magnet, d_comp=0, u_comp=1, C=C44)

def test_dz_dz_uy():
    magnet = make_long_magnet(d_comp=2)
    magnet.C44 = C44
    magnet.C12 = -C44  # to remove mixed derivative
    set_and_check_sine_force(magnet, d_comp=2, u_comp=1, C=C44)


def test_dx_dx_uz():
    magnet = make_long_magnet(d_comp=0)
    magnet.C44 = C44
    magnet.C12 = -C44  # to remove mixed derivative
    set_and_check_sine_force(magnet, d_comp=0, u_comp=2, C=C44)

def test_dy_dy_uz():
    magnet = make_long_magnet(d_comp=1)
    magnet.C44 = C44
    magnet.C12 = -C44  # to remove mixed derivative
    set_and_check_sine_force(magnet, d_comp=1, u_comp=2, C=C44)


# ==================================================
# Tests for mixed derivatives
# f_i += C12 ∂j(∂i(u_j))
# C12 is constant TODO: vary C12 as well (no C44 so no double derivative!)

def analytical_mixed_force(k_outer, k_inner, d_comp_outer, d_comp_inner, mgrid):
    force = np.zeros_like(mgrid)
    force[d_comp_inner, ...] = A * C12 * k_outer * k_inner * \
                                np.cos(k_inner * mgrid[d_comp_inner]) *\
                                np.cos(k_outer * mgrid[d_comp_outer])
    return force

def check_mixed_derivative(d_comp_outer, d_comp_inner):
    """Makes a world with a rectangular magnet of the appropriate size according
    to the directions of the mixed derivatives.

    Then sets the d_comp_outer component of the displacement of the given magnet
    to a product of sine waves, the periodicity of which depends on the direction
    of the derivatives.

    Then checks if the simulated force corresponds to the expected analytical
    force at d_comp_inner.
    """
    
    # make world
    gridsize, pbc_repetitions = [0, 0, 0], [0, 0, 0]
    gridsize[d_comp_outer], gridsize[d_comp_inner] = N1, N2
    pbc_repetitions[d_comp_outer], pbc_repetitions[d_comp_inner] = 1, 1
    world = World(cellsize, mastergrid=Grid(gridsize), pbc_repetitions=pbc_repetitions)

    # make magnet
    gridsize = [1, 1, 1]
    gridsize[d_comp_outer], gridsize[d_comp_inner] = N1, N2
    magnet =  Ferromagnet(world, Grid(gridsize))
    magnet.enable_elastodynamics = True

    magnet.C12 = C12  # enabling only mixed derivative

    # set displacement to A * sin(ki * i) * sin(kj * j)
    L_outer = N1*cellsize[d_comp_outer]
    L_inner = N2*cellsize[d_comp_inner]
    k_outer = P1 * 2*math.pi/L_outer
    k_inner = P2 * 2*math.pi/L_inner

    def displacement_func(x, y, z):
        u = [0., 0., 0.]
        outer_coord = (x, y, z)[d_comp_outer]
        inner_coord = (x, y, z)[d_comp_inner]
        u[d_comp_outer] = A * (math.sin(k_outer * outer_coord) *
                               math.sin(k_inner * inner_coord))
        return tuple(u)

    magnet.elastic_displacement = displacement_func

    # compare forces
    force_num = magnet.internal_body_force.eval()
    force_anal = analytical_mixed_force(k_outer, k_inner,
                                        d_comp_outer, d_comp_inner,
                                        mgrid=magnet.internal_body_force.meshgrid)

    assert max_semirelative_error(force_num, force_anal) < SRTOL_MIX


def test_dy_dx_uy():
    check_mixed_derivative(d_comp_outer=1, d_comp_inner=0)

def test_dz_dx_uz():
    check_mixed_derivative(d_comp_outer=2, d_comp_inner=0)

def test_dx_dy_ux():
    check_mixed_derivative(d_comp_outer=0, d_comp_inner=1)

def test_dz_dy_uz():
    check_mixed_derivative(d_comp_outer=2, d_comp_inner=1)

def test_dx_dz_ux():
    check_mixed_derivative(d_comp_outer=0, d_comp_inner=2)

def test_dy_dz_uy():
    check_mixed_derivative(d_comp_outer=1, d_comp_inner=2)


if __name__ == "__main__":
    # for manual testing
    pass
