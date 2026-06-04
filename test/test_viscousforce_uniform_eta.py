"""This file tests every term of the viscous damping force individually by comparing
the calculated elastic force to the analytical solutions.
Periodic boundry conditions are used in the direction(s) corresponding to the
derivative direction(s) so everything stays smooth, while zero derivative is
assumed in all other directions.
All viscosity constants are uniform in these tests for simplicity!
"""

import numpy as np
import math
import pytest

from mumaxplus import Grid, World, Ferromagnet


SRTOL = 1e-4
SRTOL_MIX = 1e-4

cx, cy, cz = 1.5e-9, 2e-9, 2.5e-9
cellsize = (cx, cy, cz)
N1, N2 = 128, 256
P1, P2 = 2, 3  # number of sine periods
A = 1e-8  # velocity amplitude

# using viscosity tensor
eta11 = 283e-5
eta44 = 58e-5
eta12 = 166e-5

# or using Rayleigh damping
beta = 5e-14 / math.pi
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


def set_ignored_values(magnet):
    """Sets all stiffness constants and the rayleigh damping stiffness
    coefficient of a magnet to non-zero values. These should be ignored when
    setting any viscosity tensor component."""
    magnet.C11 = C11
    magnet.C12 = C12
    magnet.C44 = C44
    magnet.stiffness_damping = beta


def set_and_check_sine_force(magnet, d_comp, v_comp, eta):
    """Sets given component of the velocity of the given magnet to a sine
    wave, the periodicity of which depends on the direction of the derivative.
    Then checks if the simulated force corresponds to the expected analytical force.
    You need to set eta yourself.
    """

    L = N1*cellsize[d_comp]
    k = P1 * 2*math.pi/L
    def velocity_func(x, y, z):
        v = [0., 0., 0.]
        relevant_coord = (x, y, z)[d_comp]
        v[v_comp] = A * math.sin(k * relevant_coord)
        return tuple(v)

    magnet.elastic_velocity = velocity_func
    
    force_num = magnet.internal_body_force.eval()
    force_anal = - k**2 * eta * magnet.elastic_velocity.eval()

    assert max_semirelative_error(force_num, force_anal) < SRTOL


# ==================================================
# Tests for double derivative along the corresponding direction
# eta11 is constant TODO: vary eta11 as well

# using viscosity tensor

def test_dx_dx_vx_tensor():
    magnet = make_long_magnet(d_comp=0)
    magnet.eta11 = eta11
    set_ignored_values(magnet)
    set_and_check_sine_force(magnet, d_comp=0, v_comp=0, eta=eta11)

def test_dy_dy_vy_tensor():
    magnet = make_long_magnet(d_comp=1)
    magnet.eta11 = eta11
    set_ignored_values(magnet)
    set_and_check_sine_force(magnet, d_comp=1, v_comp=1, eta=eta11)

def test_dz_dz_vz_tensor():
    magnet = make_long_magnet(d_comp=2)
    magnet.eta11 = eta11
    set_ignored_values(magnet)
    set_and_check_sine_force(magnet, d_comp=2, v_comp=2, eta=eta11)

# -------------------------
# using Rayleigh damping stiffness coefficient

def test_dx_dx_vx_Rayleigh():
    magnet = make_long_magnet(d_comp=0)
    magnet.C11 = C11
    magnet.stiffness_damping = beta
    set_and_check_sine_force(magnet, d_comp=0, v_comp=0, eta=beta*C11)

def test_dy_dy_vy_Rayleigh():
    magnet = make_long_magnet(d_comp=1)
    magnet.C11 = C11
    magnet.stiffness_damping = beta
    set_and_check_sine_force(magnet, d_comp=1, v_comp=1, eta=beta*C11)

def test_dz_dz_vz_Rayleigh():
    magnet = make_long_magnet(d_comp=2)
    magnet.C11 = C11
    magnet.stiffness_damping = beta
    set_and_check_sine_force(magnet, d_comp=2, v_comp=2, eta=beta*C11)

# ==================================================
# Tests for double derivative along the different direction
# eta44 is constant TODO: vary eta44 as well (careful of mixed derivatives!)

# using viscosity tensor

def test_dy_dy_vx_tensor():
    magnet = make_long_magnet(d_comp=1)
    magnet.eta44 = eta44
    magnet.eta12 = -eta44  # to remove mixed derivative
    set_ignored_values(magnet)
    set_and_check_sine_force(magnet, d_comp=1, v_comp=0, eta=eta44)

def test_dz_dz_vx_tensor():
    magnet = make_long_magnet(d_comp=2)
    magnet.eta44 = eta44
    magnet.eta12 = -eta44  # to remove mixed derivative
    set_ignored_values(magnet)
    set_and_check_sine_force(magnet, d_comp=2, v_comp=0, eta=eta44)


def test_dx_dx_vy_tensor():
    magnet = make_long_magnet(d_comp=0)
    magnet.eta44 = eta44
    magnet.eta12 = -eta44  # to remove mixed derivative
    set_ignored_values(magnet)
    set_and_check_sine_force(magnet, d_comp=0, v_comp=1, eta=eta44)

def test_dz_dz_vy_tensor():
    magnet = make_long_magnet(d_comp=2)
    magnet.eta44 = eta44
    magnet.eta12 = -eta44  # to remove mixed derivative
    set_ignored_values(magnet)
    set_and_check_sine_force(magnet, d_comp=2, v_comp=1, eta=eta44)


def test_dx_dx_vz_tensor():
    magnet = make_long_magnet(d_comp=0)
    magnet.eta44 = eta44
    magnet.eta12 = -eta44  # to remove mixed derivative
    set_ignored_values(magnet)
    set_and_check_sine_force(magnet, d_comp=0, v_comp=2, eta=eta44)

def test_dy_dy_vz_tensor():
    magnet = make_long_magnet(d_comp=1)
    magnet.eta44 = eta44
    magnet.eta12 = -eta44  # to remove mixed derivative
    set_ignored_values(magnet)
    set_and_check_sine_force(magnet, d_comp=1, v_comp=2, eta=eta44)

# -------------------------
# using Rayleigh damping stiffness coefficient

def test_dy_dy_vx_Rayleigh():
    magnet = make_long_magnet(d_comp=1)
    magnet.C44 = C44
    magnet.C12 = -C44  # to remove mixed derivative
    magnet.stiffness_damping = beta
    set_and_check_sine_force(magnet, d_comp=1, v_comp=0, eta=beta*C44)

def test_dz_dz_vx_Rayleigh():
    magnet = make_long_magnet(d_comp=2)
    magnet.C44 = C44
    magnet.C12 = -C44  # to remove mixed derivative
    magnet.stiffness_damping = beta
    set_and_check_sine_force(magnet, d_comp=2, v_comp=0, eta=beta*C44)


def test_dx_dx_vy_Rayleigh():
    magnet = make_long_magnet(d_comp=0)
    magnet.C44 = C44
    magnet.C12 = -C44  # to remove mixed derivative
    magnet.stiffness_damping = beta
    set_and_check_sine_force(magnet, d_comp=0, v_comp=1, eta=beta*C44)

def test_dz_dz_vy_Rayleigh():
    magnet = make_long_magnet(d_comp=2)
    magnet.C44 = C44
    magnet.C12 = -C44  # to remove mixed derivative
    magnet.stiffness_damping = beta
    set_and_check_sine_force(magnet, d_comp=2, v_comp=1, eta=beta*C44)


def test_dx_dx_vz_Rayleigh():
    magnet = make_long_magnet(d_comp=0)
    magnet.C44 = C44
    magnet.C12 = -C44  # to remove mixed derivative
    magnet.stiffness_damping = beta
    set_and_check_sine_force(magnet, d_comp=0, v_comp=2, eta=beta*C44)

def test_dy_dy_vz_Rayleigh():
    magnet = make_long_magnet(d_comp=1)
    magnet.C44 = C44
    magnet.C12 = -C44  # to remove mixed derivative
    magnet.stiffness_damping = beta
    set_and_check_sine_force(magnet, d_comp=1, v_comp=2, eta=beta*C44)

# ==================================================
# Tests for mixed derivatives
# f_i += eta12 ∂j(∂i(v_j))
# eta12 is constant TODO: vary eta12 as well (no eta44 so no double derivative!)

# Parametrized fixture
@pytest.fixture(params=[True, False])
def use_tensor(request):
    return request.param

def analytical_mixed_force(k_outer, k_inner, d_comp_outer, d_comp_inner, mgrid, eta=eta12):
    force = np.zeros_like(mgrid)
    force[d_comp_inner, ...] = A * eta * k_outer * k_inner * \
                                np.cos(k_inner * mgrid[d_comp_inner]) *\
                                np.cos(k_outer * mgrid[d_comp_outer])
    return force

def check_mixed_derivative(d_comp_outer, d_comp_inner, use_tensor):
    """Makes a world with a rectangular magnet of the appropriate size according
    to the directions of the mixed derivatives.

    Then sets the d_comp_outer component of the velocity of the given magnet
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

    if use_tensor:
        magnet.eta12 = eta12  # enabling only mixed derivative
        set_ignored_values(magnet)
    else:
        magnet.C12 = C12
        magnet.stiffness_damping = beta

    # set displacement to A * sin(ki * i) * sin(kj * j)
    L_outer = N1*cellsize[d_comp_outer]
    L_inner = N2*cellsize[d_comp_inner]
    k_outer = P1 * 2*math.pi/L_outer
    k_inner = P2 * 2*math.pi/L_inner

    def velocity_func(x, y, z):
        v = [0., 0., 0.]
        outer_coord = (x, y, z)[d_comp_outer]
        inner_coord = (x, y, z)[d_comp_inner]
        v[d_comp_outer] = A * (math.sin(k_outer * outer_coord) *
                               math.sin(k_inner * inner_coord))
        return tuple(v)

    magnet.elastic_velocity = velocity_func

    # compare forces
    force_num = magnet.internal_body_force.eval()
    force_anal = analytical_mixed_force(k_outer, k_inner,
                                        d_comp_outer, d_comp_inner,
                                        mgrid=magnet.internal_body_force.meshgrid,
                                        eta=eta12 if use_tensor else beta*C12)

    assert max_semirelative_error(force_num, force_anal) < SRTOL_MIX


def test_dy_dx_vy(use_tensor):
    check_mixed_derivative(d_comp_outer=1, d_comp_inner=0, use_tensor=use_tensor)

def test_dz_dx_vz(use_tensor):
    check_mixed_derivative(d_comp_outer=2, d_comp_inner=0, use_tensor=use_tensor)

def test_dx_dy_vx(use_tensor):
    check_mixed_derivative(d_comp_outer=0, d_comp_inner=1, use_tensor=use_tensor)

def test_dz_dy_vz(use_tensor):
    check_mixed_derivative(d_comp_outer=2, d_comp_inner=1, use_tensor=use_tensor)

def test_dx_dz_vx(use_tensor):
    check_mixed_derivative(d_comp_outer=0, d_comp_inner=2, use_tensor=use_tensor)

def test_dy_dz_vy(use_tensor):
    check_mixed_derivative(d_comp_outer=1, d_comp_inner=2, use_tensor=use_tensor)
