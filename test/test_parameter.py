from typing import Tuple

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from mumax5 import Ferromagnet, Grid, World


@pytest.fixture
def test_parameters() -> Tuple[World, Ferromagnet]:
    nx, ny, nz = 4, 7, 3
    w = World(cellsize=(1e-9, 1e-9, 1e-9))
    w.timesolver.time = 0.5
    magnet = Ferromagnet(w, Grid((nx, ny, nz)))
    return w, magnet


def test_assign_scalar_value(test_parameters: Tuple[World, Ferromagnet]):
    _, magnet = test_parameters
    ku1_value = 1E6

    magnet.ku1 = ku1_value
    assert magnet.ku1.is_dynamic == False
    assert magnet.ku1.is_uniform == True
    assert np.all(np.equal(magnet.ku1.eval(), ku1_value))


def test_assign_vector_value(test_parameters: Tuple[World, Ferromagnet]):
    _, magnet = test_parameters
    ncomp = 3
    b_value = (0, 0, 1.5)
    expected_value = np.zeros(shape=(ncomp, *magnet.grid.shape))
    expected_value[2, :, :, :] = b_value[2]

    magnet.bias_magnetic_field = b_value
    assert magnet.bias_magnetic_field.is_dynamic == False
    assert magnet.bias_magnetic_field.is_uniform == True
    assert_almost_equal(magnet.bias_magnetic_field.eval(), expected_value)


def test_assign_array_value(test_parameters: Tuple[World, Ferromagnet]):
    _, magnet = test_parameters
    alpha_value = np.random.rand(1, *magnet.grid.shape)
    b_value = np.random.rand(3, *magnet.grid.shape)

    magnet.alpha = alpha_value
    assert magnet.alpha.is_uniform == False
    assert_almost_equal(magnet.alpha.eval(), alpha_value)

    magnet.bias_magnetic_field = b_value
    assert magnet.bias_magnetic_field.is_uniform == False
    assert_almost_equal(magnet.bias_magnetic_field.eval(), b_value)


def test_assign_scalar_time_dependent_term(test_parameters: Tuple[World, Ferromagnet]):
    world, magnet = test_parameters

    term = lambda t: 24 * np.sinc(t)

    magnet.ku1 = term
    assert magnet.ku1.is_dynamic == True
    assert magnet.ku1.is_uniform == True
    assert np.all(np.equal(magnet.ku1.eval(),
                           term(world.timesolver.time)))

    magnet.ku1.remove_time_terms()
    assert magnet.ku1.is_dynamic == False


def test_assign_scalar_time_dependent_term_mask(test_parameters: Tuple[World, Ferromagnet]):
    world, magnet = test_parameters
    ncomp = 1
    mask_value = 0.2
    term = lambda t: 24 * np.sinc(t)

    # correct mask shape
    test_mask1 = mask_value * np.ones(shape=(ncomp,*magnet.grid.shape))
    magnet.ku1 = (term, test_mask1)
    assert magnet.ku1.is_dynamic == True
    assert magnet.ku1.is_uniform == False
    assert np.all(np.equal(magnet.ku1.eval(),
                           mask_value * term(world.timesolver.time)))

    # incorrect mask shape
    test_mask2 = np.ones(shape=(ncomp,4, 7, 3))
    with pytest.raises(RuntimeError):
        magnet.ku2 = (term, test_mask2)


def test_add_multiple_scalar_time_dependent_terms(test_parameters: Tuple[World, Ferromagnet]):
    world, magnet = test_parameters
    ncomp = 1
    alpha_value = 0.5
    mask_value2 = 0.001
    term1 = lambda t: 24 * np.sinc(t)
    term2 = lambda t: -24 * np.sinc(t)

    magnet.alpha = alpha_value
    magnet.alpha.add_time_terms(term1)
    magnet.alpha.add_time_terms(term2, mask_value2 *
        np.ones(shape=(ncomp,*magnet.grid.shape)))

    assert magnet.alpha.is_dynamic == True
    # assert magnet.bias_magnetic_field.is_uniform == False
    assert np.all(np.equal(magnet.alpha.eval(),
                           alpha_value + term1(world.timesolver.time)
                           + mask_value2 * term2(world.timesolver.time)))

    magnet.alpha.remove_time_terms()
    assert magnet.alpha.is_dynamic == False


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

    assert magnet.bias_magnetic_field.is_dynamic == True
    assert magnet.bias_magnetic_field.is_uniform == True
    assert_almost_equal(magnet.bias_magnetic_field.eval(),
                        expected_value)

    magnet.bias_magnetic_field.remove_time_terms()
    assert magnet.bias_magnetic_field.is_dynamic == False


def test_assign_vector_time_dependent_term_mask(test_parameters: Tuple[World, Ferromagnet]):
    world, magnet = test_parameters
    ncomp = 3
    mask_value = 0.2
    term = lambda t: (0, 1, 0.25 * np.sin(t))
    expected_value = np.zeros(shape=(ncomp, *magnet.grid.shape))
    term_value = term(world.timesolver.time)
    expected_value[0, :, :, :] = term_value[0]
    expected_value[1, :, :, :] = term_value[1]
    expected_value[2, :, :, :] = term_value[2]
    expected_value *= mask_value

    # correct mask shape
    test_mask1 = mask_value * np.ones(shape=(ncomp, *magnet.grid.shape))
    magnet.bias_magnetic_field = (term, test_mask1)
    assert magnet.bias_magnetic_field.is_dynamic == True
    assert magnet.bias_magnetic_field.is_uniform == False
    assert_almost_equal(magnet.bias_magnetic_field.eval(),
                        expected_value)

    # incorrect mask shape
    test_mask2 = np.ones(shape=(1, 4, 7, 3))
    with pytest.raises(RuntimeError):
        magnet.bias_magnetic_field = (term, test_mask2)


def test_add_multiple_vector_time_dependent_terms(test_parameters: Tuple[World, Ferromagnet]):
    world, magnet = test_parameters
    ncomp = 3
    b_value = (1, 1, 0)
    mask_value2 = 0.001
    term1 = lambda t: np.array((0, 1, 0.25 * np.sin(t)))
    term2 = lambda t: np.array((0, 0, -24 * np.sinc(t)))

    expected_value = np.zeros(shape=(ncomp, *magnet.grid.shape))
    term_value = b_value + term1(world.timesolver.time) +\
        mask_value2 * term2(world.timesolver.time)
    expected_value[0, :, :, :] = term_value[0]
    expected_value[1, :, :, :] = term_value[1]
    expected_value[2, :, :, :] = term_value[2]

    magnet.bias_magnetic_field = b_value
    magnet.bias_magnetic_field.add_time_terms(term1)
    magnet.bias_magnetic_field.add_time_terms(term2, mask_value2 *
        np.ones(shape=(ncomp, *magnet.grid.shape)))

    assert magnet.bias_magnetic_field.is_dynamic == True
    assert magnet.bias_magnetic_field.is_uniform == False

    assert_almost_equal(magnet.bias_magnetic_field.eval(),
                        expected_value)

    magnet.bias_magnetic_field.remove_time_terms()
    assert magnet.bias_magnetic_field.is_dynamic == False
