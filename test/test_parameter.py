from typing import Tuple

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from mumaxplus import Ferromagnet, Grid, World


@pytest.fixture
def test_parameters() -> Tuple[World, Ferromagnet]:
    nx, ny, nz = 4, 7, 3
    w = World(cellsize=(1e-9, 1e-9, 1e-9))
    w.timesolver.time = 0.5
    magnet = Ferromagnet(w, Grid((nx, ny, nz)))
    return w, magnet


def test_assign_scalar_value(test_parameters: Tuple[World, Ferromagnet]):
    _, magnet = test_parameters
    ku1_value = 1e6

    magnet.ku1 = ku1_value
    assert magnet.ku1.is_dynamic is False
    assert magnet.ku1.is_uniform is True
    assert_almost_equal(magnet.ku1.eval(), ku1_value)


def test_assign_vector_value(test_parameters: Tuple[World, Ferromagnet]):
    _, magnet = test_parameters
    ncomp = 3
    b_value = (0, 0, 1.5)
    expected_value = np.zeros(shape=(ncomp, *magnet.grid.shape))
    expected_value[2, :, :, :] = b_value[2]

    magnet.bias_magnetic_field = b_value
    assert magnet.bias_magnetic_field.is_dynamic is False
    assert magnet.bias_magnetic_field.is_uniform is True
    assert_almost_equal(magnet.bias_magnetic_field.eval(), expected_value)


def test_assign_array_value(test_parameters: Tuple[World, Ferromagnet]):
    _, magnet = test_parameters
    alpha_value = np.random.rand(1, *magnet.grid.shape)
    b_value = np.random.rand(3, *magnet.grid.shape)

    magnet.alpha = alpha_value
    assert magnet.alpha.is_uniform is False
    assert_almost_equal(magnet.alpha.eval(), alpha_value)

    magnet.bias_magnetic_field = b_value
    assert magnet.bias_magnetic_field.is_uniform is False
    assert_almost_equal(magnet.bias_magnetic_field.eval(), b_value)


def test_assign_scalar_time_dependent_term(test_parameters: Tuple[World, Ferromagnet]):
    world, magnet = test_parameters

    term = lambda t: 24 * np.sinc(t)

    magnet.ku1 = term
    assert magnet.ku1.is_dynamic is True
    assert magnet.ku1.is_uniform is True
    assert_almost_equal(magnet.ku1.eval(), term(world.timesolver.time))

    magnet.ku1.remove_time_terms()
    assert magnet.ku1.is_dynamic is False
    assert_almost_equal(magnet.ku1.eval(), 0)


def test_assign_scalar_time_dependent_term_mask(
    test_parameters: Tuple[World, Ferromagnet]
):
    world, magnet = test_parameters
    ncomp = 1
    mask_value1 = 0.2
    mask_value2 = 0.8
    term = lambda t: 24 * np.sinc(t)

    # correct mask shape
    test_mask1 = mask_value1 * np.ones(shape=(ncomp, *magnet.grid.shape))
    magnet.ku1 = (term, test_mask1)
    assert magnet.ku1.is_dynamic is True
    assert magnet.ku1.is_uniform is False
    assert_almost_equal(magnet.ku1.eval(), mask_value1 * term(world.timesolver.time))

    # mask without components
    test_mask2 = mask_value2 * np.ones(shape=magnet.grid.shape)
    magnet.ku1 = (term, test_mask2)
    assert magnet.ku1.is_dynamic is True
    assert magnet.ku1.is_uniform is False
    assert_almost_equal(magnet.ku1.eval(), mask_value2 * term(world.timesolver.time))

    # incorrect mask shape
    test_mask3 = np.ones(shape=(ncomp, 4, 7, 3))
    with pytest.raises(ValueError):
        magnet.ku2 = (term, test_mask3)


def test_assign_scalar_time_dependent_term_mask_func(
    test_parameters: Tuple[World, Ferromagnet]
):
    world, magnet = test_parameters
    # ncomp = 1
    mask_value1 = 0.2
    mask_value2 = 0.8
    term = lambda t: 24 * np.sinc(t)

    # correct mask shape
    test_mask1 = lambda x, y, z: (mask_value1)
    magnet.ku1 = (term, test_mask1)
    assert magnet.ku1.is_dynamic is True
    assert magnet.ku1.is_uniform is False
    assert_almost_equal(magnet.ku1.eval(), mask_value1 * term(world.timesolver.time))

    # mask without components
    test_mask2 = lambda x, y, z: mask_value2
    magnet.ku1 = (term, test_mask2)
    assert magnet.ku1.is_dynamic is True
    assert magnet.ku1.is_uniform is False
    assert_almost_equal(magnet.ku1.eval(), mask_value2 * term(world.timesolver.time))

    # incorrect mask shape
    test_mask3 = lambda x, y, z: (mask_value1, mask_value2)
    with pytest.raises(ValueError):
        magnet.ku2 = (term, test_mask3)


def test_add_multiple_scalar_time_dependent_terms(
    test_parameters: Tuple[World, Ferromagnet]
):
    world, magnet = test_parameters
    ncomp = 1
    alpha_value = 0.5
    mask_value2 = 0.001
    term1 = lambda t: 24 * np.sinc(t)
    term2 = lambda t: -24 * np.sinc(t)

    magnet.alpha = alpha_value
    magnet.alpha.add_time_term(term1)
    magnet.alpha.add_time_term(
        term2, mask_value2 * np.ones(shape=(ncomp, *magnet.grid.shape))
    )

    assert magnet.alpha.is_dynamic is True
    assert magnet.alpha.is_uniform is False
    assert_almost_equal(
        magnet.alpha.eval(),
        alpha_value
        + term1(world.timesolver.time)
        + mask_value2 * term2(world.timesolver.time),
    )
    magnet.alpha.remove_time_terms()
    assert magnet.alpha.is_dynamic is False
    assert_almost_equal(magnet.alpha.eval(), alpha_value)


def test_assign_vector_time_dependent_term(test_parameters: Tuple[World, Ferromagnet]):
    world, magnet = test_parameters
    ncomp = 3
    term = lambda t: (0, 1, 0.25 * np.sin(t))
    expected_value = np.zeros(shape=(ncomp, *magnet.grid.shape))
    term_value = term(world.timesolver.time)
    expected_value[0, :, :, :] = term_value[0]
    expected_value[1, :, :, :] = term_value[1]
    expected_value[2, :, :, :] = term_value[2]

    magnet.bias_magnetic_field = term

    magnet.bias_magnetic_field.eval()

    assert magnet.bias_magnetic_field.is_dynamic is True
    assert magnet.bias_magnetic_field.is_uniform is True
    assert_almost_equal(magnet.bias_magnetic_field.eval(), expected_value)

    magnet.bias_magnetic_field.remove_time_terms()
    assert magnet.bias_magnetic_field.is_dynamic is False
    assert_almost_equal(magnet.bias_magnetic_field.eval(), 0)


def test_assign_vector_time_dependent_term_mask(
    test_parameters: Tuple[World, Ferromagnet]
):
    world, magnet = test_parameters
    ncomp = 3
    mask_value1 = 0.2
    mask_value2 = 0.8
    mask_values3 = (0.1, 0.2, 0.3)
    term = lambda t: (0, 1, 0.25 * np.sin(t))
    expected_value = np.zeros(shape=(ncomp, *magnet.grid.shape))
    term_value = term(world.timesolver.time)
    expected_value[0, :, :, :] = term_value[0]
    expected_value[1, :, :, :] = term_value[1]
    expected_value[2, :, :, :] = term_value[2]

    # correct mask shape
    test_mask1 = mask_value1 * np.ones(shape=(ncomp, *magnet.grid.shape))
    expected_value1 = mask_value1 * expected_value
    magnet.bias_magnetic_field = (term, test_mask1)
    assert magnet.bias_magnetic_field.is_dynamic is True
    assert magnet.bias_magnetic_field.is_uniform is False
    assert_almost_equal(magnet.bias_magnetic_field.eval(), expected_value1)

    # correct mask without components
    test_mask2 = mask_value2 * np.ones(shape=magnet.grid.shape)
    magnet.bias_magnetic_field = (term, test_mask2)
    expected_value2 = mask_value2 * expected_value
    assert magnet.bias_magnetic_field.is_dynamic is True
    assert magnet.bias_magnetic_field.is_uniform is False
    assert_almost_equal(magnet.bias_magnetic_field.eval(), expected_value2)

    # correct mask shape with different components
    test_mask3 = np.ones(shape=(ncomp, *magnet.grid.shape))
    test_mask3[0, ...] = mask_values3[0]
    test_mask3[1, ...] = mask_values3[1]
    test_mask3[2, ...] = mask_values3[2]
    expected_value3 = expected_value
    expected_value3[0, ...] *= mask_values3[0]
    expected_value3[1, ...] *= mask_values3[1]
    expected_value3[2, ...] *= mask_values3[2]
    magnet.bias_magnetic_field = (term, test_mask3)
    assert magnet.bias_magnetic_field.is_dynamic is True
    assert magnet.bias_magnetic_field.is_uniform is False
    assert_almost_equal(magnet.bias_magnetic_field.eval(), expected_value3)

    # incorrect mask shape
    test_mask3 = np.ones(shape=(1, 4, 7, 3))
    with pytest.raises(ValueError):
        magnet.bias_magnetic_field = (term, test_mask3)


def test_assign_vector_time_dependent_term_mask_func(
    test_parameters: Tuple[World, Ferromagnet]
):
    world, magnet = test_parameters
    ncomp = 3
    mask_value1 = 0.2
    mask_value2 = 0.8
    mask_values3 = (0.1, 0.2, 0.3)
    term = lambda t: (0, 1, 0.25 * np.sin(t))
    expected_value = np.zeros(shape=(ncomp, *magnet.grid.shape))
    term_value = term(world.timesolver.time)
    expected_value[0, :, :, :] = term_value[0]
    expected_value[1, :, :, :] = term_value[1]
    expected_value[2, :, :, :] = term_value[2]

    # correct mask shape
    test_mask1 = lambda x, y, z: (mask_value1, mask_value1, mask_value1)
    expected_value1 = mask_value1 * expected_value
    magnet.bias_magnetic_field = (term, test_mask1)
    assert magnet.bias_magnetic_field.is_dynamic is True
    assert magnet.bias_magnetic_field.is_uniform is False
    assert_almost_equal(magnet.bias_magnetic_field.eval(), expected_value1)

    # correct mask without components
    test_mask2 = lambda x, y, z: mask_value2
    magnet.bias_magnetic_field = (term, test_mask2)
    expected_value2 = mask_value2 * expected_value
    assert magnet.bias_magnetic_field.is_dynamic is True
    assert magnet.bias_magnetic_field.is_uniform is False
    assert_almost_equal(magnet.bias_magnetic_field.eval(), expected_value2)

    # correct mask shape with different components
    test_mask3 = lambda x, y, z: mask_values3
    expected_value3 = expected_value
    expected_value3[0, ...] *= mask_values3[0]
    expected_value3[1, ...] *= mask_values3[1]
    expected_value3[2, ...] *= mask_values3[2]
    magnet.bias_magnetic_field = (term, test_mask3)
    assert magnet.bias_magnetic_field.is_dynamic is True
    assert magnet.bias_magnetic_field.is_uniform is False
    assert_almost_equal(magnet.bias_magnetic_field.eval(), expected_value3)

    # incorrect mask shape
    test_mask3 = lambda x, y, z: (mask_value1, mask_value2)
    with pytest.raises(ValueError):
        magnet.bias_magnetic_field = (term, test_mask3)


def test_add_multiple_vector_time_dependent_terms(
    test_parameters: Tuple[World, Ferromagnet]
):
    world, magnet = test_parameters
    ncomp = 3
    b_value = (1, 1, 0)
    mask_value2 = 0.001
    term1 = lambda t: np.array((0, 1, 0.25 * np.sin(t)))
    term2 = lambda t: np.array((0, 0, -24 * np.sinc(t)))

    expected_value = np.zeros(shape=(ncomp, *magnet.grid.shape))
    term_value = (
        b_value
        + term1(world.timesolver.time)
        + mask_value2 * term2(world.timesolver.time)
    )
    expected_value[0, :, :, :] = term_value[0]
    expected_value[1, :, :, :] = term_value[1]
    expected_value[2, :, :, :] = term_value[2]

    magnet.bias_magnetic_field = b_value
    magnet.bias_magnetic_field.add_time_term(term1)
    magnet.bias_magnetic_field.add_time_term(
        term2, mask_value2 * np.ones(shape=(ncomp, *magnet.grid.shape))
    )

    assert magnet.bias_magnetic_field.is_dynamic is True
    assert magnet.bias_magnetic_field.is_uniform is False

    assert_almost_equal(magnet.bias_magnetic_field.eval(), expected_value)

    magnet.bias_magnetic_field.remove_time_terms()

    expected_value2 = np.zeros(shape=(ncomp, *magnet.grid.shape))
    expected_value2[0, :, :, :] = b_value[0]
    expected_value2[1, :, :, :] = b_value[1]
    expected_value2[2, :, :, :] = b_value[2]

    assert magnet.bias_magnetic_field.is_dynamic is False
    assert_almost_equal(magnet.bias_magnetic_field.eval(), expected_value2)
