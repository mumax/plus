"""This test checks the spin transfer torque when both the Zhang-Li and the
Slonczewski torques are present, and each torque on its own against its
closed form.
"""

import numpy as np
import pytest

from mumaxplus import Antiferromagnet, Ferromagnet, Grid, World
from mumaxplus.util.constants import HBAR, MUB, QE

RTOL = 1e-5  # closed forms, single precision finite differences
SUM_RTOL = 1e-6  # the total against its two parts, expected to be exact

cellsize = (2e-9, 3e-9, 1.5e-9)
nx, ny, nz = 16, 12, 4

msat = 800e3
alpha = 0.1
pol = 0.6
xi = 0.05
jcur = (5e11, -2.5e11, 1e12)  # all components for Zhang-Li, jz for Slonczewski
Lambda = 2
epsilon_prime = 0.1

# Arbitrary non-uniform magnetization, so that the Zhang-Li torque is non-zero
rng = np.random.default_rng(seed=1234)
magnetization = rng.normal(size=(3, nz, ny, nx))
magnetization /= np.linalg.norm(magnetization, axis=0)

# Fixed layer which is zero in a quarter of the cells, so that the Slonczewski
# torque vanishes there while the Zhang-Li torque does not
fixed_layer = np.zeros((3, nz, ny, nx))
fixed_layer[0], fixed_layer[1], fixed_layer[2] = 1.0, 0.5, -0.3
fixed_layer[:, :, : ny // 2, : nx // 2] = 0

# Variations of the setup: frozen cells, cells outside of the geometry, pol = 0 or
# jcur = 0 in a block. Each block overlaps the quarter without a fixed layer and
# the rest of the magnet.
CASES = ["plain", "frozen", "geometry", "pol_block", "jcur_block"]


def max_norm(field):
    return np.max(np.linalg.norm(field, axis=0))


def max_semirelative_error(result, wanted):
    """Maximum error for vector quantities, divided by the maximum of wanted."""
    return max_norm(result - wanted) / max_norm(wanted)


def set_parameters(magnet, case="plain", fixed_layer_on_top=True):
    magnet.msat = msat
    magnet.alpha = alpha
    magnet.pol = pol
    magnet.jcur = jcur
    magnet.xi = xi  # Zhang-Li
    magnet.Lambda = Lambda  # Slonczewski
    magnet.epsilon_prime = epsilon_prime
    magnet.fixed_layer = fixed_layer
    magnet.fixed_layer_on_top = fixed_layer_on_top

    if case == "frozen":
        frozen_spins = np.zeros((1, nz, ny, nx))
        frozen_spins[:, 1:3, 3:9, 4:12] = 1
        magnet.frozen_spins = frozen_spins
    elif case == "pol_block":
        # no Zhang-Li torque in the block, the epsilon_prime term remains
        pol_ = np.full((1, nz, ny, nx), pol)
        pol_[:, :, 2:8, 5:12] = 0
        magnet.pol = pol_
    elif case == "jcur_block":
        jcur_ = np.empty((3, nz, ny, nx))
        jcur_[:] = np.reshape(jcur, (3, 1, 1, 1))
        jcur_[:, 1:4, 4:10, 3:10] = 0
        magnet.jcur = jcur_


def create_magnet(case="plain", fixed_layer_on_top=True):
    geometry = None
    if case == "geometry":
        geometry = np.ones((nz, ny, nx), dtype=bool)
        geometry[:, 3:9, 4:12] = False  # a hole through the whole thickness
        geometry[0, -2:, -3:] = False  # and a missing corner in the bottom layer

    world = World(cellsize=cellsize)
    magnet = Ferromagnet(world, Grid((nx, ny, nz)), geometry=geometry)
    magnet.magnetization = magnetization
    set_parameters(magnet, case, fixed_layer_on_top)
    return magnet


def cell_values(parameter):
    """Values of a parameter in every cell, shape (ncomp, nz, ny, nx)."""
    return parameter.eval().astype(np.float64)


def active_cells(magnet):
    """Cells which get a torque: inside the geometry and not frozen."""
    geometry = np.asarray(magnet.geometry, dtype=bool)
    return geometry & (cell_values(magnet.frozen_spins)[0] == 0)


def neighbor(m, axis, sign):
    """Magnetization of the neighboring cell, zero outside of the grid."""
    result = np.zeros_like(m)
    source, target = [slice(None)] * 4, [slice(None)] * 4
    if sign > 0:
        source[axis], target[axis] = slice(1, None), slice(None, -1)
    else:
        source[axis], target[axis] = slice(None, -1), slice(1, None)
    result[tuple(target)] = m[tuple(source)]
    return result


def zhang_li_torque(magnet):
    """Zhang-Li torque, discretized with central differences. A neighbor outside
    of the grid or without msat does not contribute."""
    m = cell_values(magnet.magnetization)
    msat_ = cell_values(magnet.msat)[0]
    pol_ = cell_values(magnet.pol)[0]
    xi_ = cell_values(magnet.xi)[0]
    alpha_ = cell_values(magnet.alpha)[0]
    j = cell_values(magnet.jcur)
    b = np.zeros_like(msat_)  # msat is zero outside of the geometry
    np.divide(MUB * pol_, QE * msat_ * (1 + xi_**2), out=b, where=msat_ != 0)
    u = b * j

    m_neighbors = np.where(msat_ != 0, m, 0)
    hspin = np.zeros_like(m)
    for u_i, c_i, axis in zip(u, cellsize, (3, 2, 1)):  # x, y and z axes of m
        m_plus = neighbor(m_neighbors, axis, 1)
        m_minus = neighbor(m_neighbors, axis, -1)
        hspin += u_i * (m_plus - m_minus) / (2 * c_i)

    mxh = np.cross(m, hspin, axis=0)
    mxmxh = np.cross(m, mxh, axis=0)
    torque = (-1 / (1 + alpha_**2)) * (
        (1 + xi_ * alpha_) * mxmxh + (xi_ - alpha_) * mxh
    )
    return torque * active_cells(magnet)


def slonczewski_torque(magnet):
    """Slonczewski torque."""
    m = cell_values(magnet.magnetization)
    msat_ = cell_values(magnet.msat)[0]
    pol_ = cell_values(magnet.pol)[0]
    alpha_ = cell_values(magnet.alpha)[0]
    lambda2 = cell_values(magnet.Lambda)[0] ** 2
    eps_p = cell_values(magnet.epsilon_prime)[0]
    gamma = cell_values(magnet.gamma)[0]
    jz = cell_values(magnet.jcur)[2]
    d = cell_values(magnet.free_layer_thickness)[0]
    if not magnet.fixed_layer_on_top:
        d *= -1

    p = cell_values(magnet.fixed_layer)
    p_norm = np.linalg.norm(p, axis=0)
    p = np.divide(p, p_norm, out=np.zeros_like(p), where=p_norm != 0)

    B = np.zeros_like(msat_)  # msat is zero outside of the geometry
    np.divide((HBAR / QE) * jz, msat_ * d, out=B, where=msat_ * d != 0)
    eps = pol_ * lambda2 / ((lambda2 + 1) + (lambda2 - 1) * np.sum(m * p, axis=0))
    pxm = np.cross(p, m, axis=0)
    mxpxm = np.cross(m, pxm, axis=0)
    torque = (
        ((eps + eps_p * alpha_) * mxpxm + (eps_p - eps * alpha_) * pxm)
        * (B / (1 + alpha_**2))
        * gamma
    )
    return torque * active_cells(magnet)


def torque_and_parts(magnet):
    """Spin transfer torque with both torques, then each torque alone."""
    both = magnet.spin_transfer_torque.eval()

    magnet.enable_slonczewski_torque = False
    zhang_li = magnet.spin_transfer_torque.eval()
    magnet.enable_slonczewski_torque = True

    magnet.enable_zhang_li_torque = False
    slonczewski = magnet.spin_transfer_torque.eval()
    magnet.enable_zhang_li_torque = True

    return both, zhang_li, slonczewski


@pytest.mark.parametrize("fixed_layer_on_top", [True, False])
@pytest.mark.parametrize("case", CASES)
def test_both_torques_are_added(case, fixed_layer_on_top):
    """With both torques present, the spin transfer torque is the Zhang-Li torque
    plus the Slonczewski torque, each evaluated with the other one disabled."""
    magnet = create_magnet(case, fixed_layer_on_top)
    both, zhang_li, slonczewski = torque_and_parts(magnet)

    # both torques contribute substantially
    assert max_norm(zhang_li) > 0.1 * max_norm(both)
    assert max_norm(slonczewski) > 0.1 * max_norm(both)

    err = max_semirelative_error(both, zhang_li + slonczewski)
    assert err < SUM_RTOL


@pytest.mark.parametrize("fixed_layer_on_top", [True, False])
@pytest.mark.parametrize("case", CASES)
def test_both_torques_closed_form(case, fixed_layer_on_top):
    """With both torques present, compare to the sum of the closed forms."""
    magnet = create_magnet(case, fixed_layer_on_top)
    result = magnet.spin_transfer_torque.eval()
    wanted = zhang_li_torque(magnet) + slonczewski_torque(magnet)
    err = max_semirelative_error(result, wanted)
    assert err < RTOL


@pytest.mark.parametrize("switch_off", ["flag", "parameter"])
@pytest.mark.parametrize("case", CASES)
def test_zhang_li_alone(case, switch_off):
    """Only the Zhang-Li torque, the Slonczewski torque being disabled or zero."""
    magnet = create_magnet(case)
    if switch_off == "flag":
        magnet.enable_slonczewski_torque = False
    else:
        magnet.fixed_layer = (0, 0, 0)

    result = magnet.spin_transfer_torque.eval()
    err = max_semirelative_error(result, zhang_li_torque(magnet))
    assert err < RTOL


@pytest.mark.parametrize("switch_off", ["flag", "parameter"])
@pytest.mark.parametrize("case", CASES)
def test_slonczewski_alone(case, switch_off):
    """Only the Slonczewski torque, the Zhang-Li torque being disabled or zero."""
    magnet = create_magnet(case)
    if switch_off == "flag":
        magnet.enable_zhang_li_torque = False
    else:
        magnet.pol = 0  # leaves the epsilon_prime term of the Slonczewski torque

    result = magnet.spin_transfer_torque.eval()
    err = max_semirelative_error(result, slonczewski_torque(magnet))
    assert err < RTOL


def test_in_plane_current_with_fixed_layer():
    """An in-plane current gives a Zhang-Li torque, even if a fixed layer is set;
    the Slonczewski torque is zero without jz."""
    magnet = create_magnet()
    magnet.jcur = (jcur[0], jcur[1], 0)
    result = magnet.spin_transfer_torque.eval()
    wanted = zhang_li_torque(magnet)
    assert max_norm(wanted) > 0
    err = max_semirelative_error(result, wanted)
    assert err < RTOL


@pytest.mark.parametrize("fixed_layer_on_top", [True, False])
def test_antiferromagnet_sublattices(fixed_layer_on_top):
    """Both torques are added on each sublattice of an antiferromagnet."""
    world = World(cellsize=cellsize)
    magnet = Antiferromagnet(world, Grid((nx, ny, nz)))
    set_parameters(magnet, fixed_layer_on_top=fixed_layer_on_top)
    noise = np.random.default_rng(seed=5678).normal(size=(3, nz, ny, nx))
    magnet.sub1.magnetization = magnetization
    magnet.sub2.magnetization = -magnetization + 0.5 * noise

    for sub in magnet.sublattices:
        both, zhang_li, slonczewski = torque_and_parts(sub)
        assert max_norm(zhang_li) > 0.1 * max_norm(both)
        assert max_norm(slonczewski) > 0.1 * max_norm(both)
        assert max_semirelative_error(both, zhang_li + slonczewski) < SUM_RTOL

        wanted = zhang_li_torque(sub) + slonczewski_torque(sub)
        assert max_semirelative_error(both, wanted) < RTOL
