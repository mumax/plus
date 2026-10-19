from typing import Tuple

import numpy as np
import pytest

from mumaxplus import Ferromagnet, Grid, World
from mumaxplus.util.config import vortex


@pytest.fixture
def test_parameters() -> Tuple[Ferromagnet, np.ndarray]:
    nx, ny, nz = 4, 7, 3
    w = World(cellsize=(1e-9, 1e-9, 1e-9))
    g = Grid((nx, ny, nz))
    r = np.random.randint(0, 12345, size=g.shape)
    magnet = Ferromagnet(w, g, regions=r)
    return magnet, r


def test_uniform_parameter_in_region(test_parameters: Tuple[Ferromagnet, np.ndarray]):
    magnet, regions = test_parameters
    ku1_value = 1e6

    some_index = np.random.choice(np.unique(regions))
    magnet.ku1.set_in_region(some_index, ku1_value)

    mask = regions == some_index
    assert np.all(magnet.ku1()[0, mask] == ku1_value)
    assert np.all(magnet.ku1()[0, ~mask] == 0.0)


def test_uniform_vectorparameter_in_region(
    test_parameters: Tuple[Ferromagnet, np.ndarray]
):
    magnet, regions = test_parameters
    bias_value = np.array((1, 0.1, 3))

    some_index = np.random.choice(np.unique(regions))
    magnet.bias_magnetic_field.set_in_region(some_index, bias_value)

    mask = regions == some_index
    assert np.allclose(magnet.bias_magnetic_field()[:, mask], bias_value[:, None])
    assert np.all(magnet.bias_magnetic_field()[:, ~mask] == 0.0)


def test_uniform_variable_in_region(test_parameters: Tuple[Ferromagnet, np.ndarray]):
    magnet, regions = test_parameters

    initial_vector = np.array((1, 0, 0))
    region_value = np.array((0, 1, 0))

    magnet.magnetization = initial_vector

    some_index = np.random.choice(np.unique(regions))
    magnet.magnetization.set_in_region(some_index, region_value)

    mask = regions == some_index
    assert np.allclose(magnet.magnetization()[:, mask], region_value[:, None])
    assert np.allclose(magnet.magnetization()[:, ~mask], initial_vector[:, None])


def test_vectorized_function_parameter_in_region(
    test_parameters: Tuple[Ferromagnet, np.ndarray]
):
    magnet, regions = test_parameters

    def func(x, y, z):
        return x + y + z

    some_index = np.random.choice(np.unique(regions))
    magnet.ku1.set_in_region(some_index, func)

    x, y, z = magnet.ku1.meshgrid
    mask = regions == some_index

    assert np.allclose(magnet.ku1()[0, mask], func(x[mask], y[mask], z[mask]))
    assert np.all(magnet.ku1()[0, ~mask] == 0.0)


def test_non_vectorized_function_parameter_in_region(
    test_parameters: Tuple[Ferromagnet, np.ndarray]
):
    magnet, regions = test_parameters

    initial_value = 1e-12
    magnet.aex = initial_value

    def values_function(x, y, z):
        return x + y + z

    def set_func(x, y, z):
        return values_function(x, y, z)

    x, y, z = magnet.aex.meshgrid

    some_index = np.random.choice(np.unique(regions))
    magnet.aex.set_in_region(some_index, set_func)
    mask = regions == some_index

    assert np.allclose(
        magnet.aex()[0, mask], values_function(x[mask], y[mask], z[mask])
    )
    assert np.all(magnet.aex()[0, ~mask] == initial_value)


def test_vectorized_function_vectorparameter_in_region(
    test_parameters: Tuple[Ferromagnet, np.ndarray]
):
    magnet, regions = test_parameters

    def func(x, y, z):
        return (x, y**2, 3 * x)

    some_index = np.random.choice(np.unique(regions))
    magnet.bias_magnetic_field.set_in_region(some_index, func)

    x, y, z = magnet.bias_magnetic_field.meshgrid
    mask = regions == some_index

    assert np.allclose(
        magnet.bias_magnetic_field()[:, mask],
        np.array(func(x[mask], y[mask], z[mask]))[:, None],
    )
    assert np.all(magnet.bias_magnetic_field()[:, ~mask] == 0.0)


def test_non_vectorized_function_vectorparameter_in_region(
    test_parameters: Tuple[Ferromagnet, np.ndarray]
):
    magnet, regions = test_parameters

    initial_vector = np.array((1.0, 0.0, 0.0))
    magnet.bias_magnetic_field = initial_vector

    def values_func(x, y, z):
        return (x, y**2, 3 * x)

    def set_func(x, y, z):
        return values_func(x, y, z)

    x, y, z = magnet.magnetization.meshgrid
    some_index = np.random.choice(np.unique(regions))
    magnet.bias_magnetic_field.set_in_region(some_index, set_func)
    mask = regions == some_index

    assert np.allclose(
        magnet.bias_magnetic_field()[:, mask], values_func(x[mask], y[mask], z[mask])
    )
    assert np.all(magnet.bias_magnetic_field()[:, ~mask] == initial_vector[:, None])


def test_vectorized_function_variable_in_region(
    test_parameters: Tuple[Ferromagnet, np.ndarray]
):
    magnet, regions = test_parameters

    def func(x, y, z):
        vec = np.array((x, y**2, z + 3e-9))
        return vec / np.linalg.norm(vec, axis=0)

    initial_vector = np.array((1.0, 0.0, 0.0))
    magnet.magnetization = initial_vector

    some_index = np.random.choice(np.unique(regions))
    magnet.magnetization.set_in_region(some_index, func)

    x, y, z = magnet.magnetization.meshgrid
    mask = regions == some_index

    assert np.allclose(magnet.magnetization()[:, mask], func(x[mask], y[mask], z[mask]))
    assert np.all(magnet.magnetization()[:, ~mask] == initial_vector[:, None])


def test_config_variable_in_region(test_parameters: Tuple[Ferromagnet, np.ndarray]):
    magnet, regions = test_parameters

    initial_vector = np.array((1.0, 0.0, 0.0))
    magnet.magnetization = initial_vector

    func = vortex(magnet.center, 5e-9, 1, 1)
    some_index = np.random.choice(np.unique(regions))
    magnet.magnetization.set_in_region(some_index, func)

    x, y, z = magnet.magnetization.meshgrid
    mask = regions == some_index

    assert np.allclose(magnet.magnetization()[:, mask], func(x[mask], y[mask], z[mask]))
    assert np.all(magnet.magnetization()[:, ~mask] == initial_vector[:, None])


def test_incompatible_uniform_value_in_region_scalar_parameter(test_parameters):
    magnet, regions = test_parameters
    some_index = np.random.choice(np.unique(regions))

    with pytest.raises(TypeError):
        magnet.ku1.set_in_region(some_index, (1.0, 2.0))


def test_incompatible_uniform_value_in_region_vector_parameter(test_parameters):
    magnet, regions = test_parameters
    some_index = np.random.choice(np.unique(regions))

    with pytest.raises(TypeError):
        magnet.bias_magnetic_field.set_in_region(some_index, (1.0, 2.0))


def test_incompatible_uniform_value_in_region_variable(test_parameters):
    magnet, regions = test_parameters
    some_index = np.random.choice(np.unique(regions))

    with pytest.raises(TypeError):
        magnet.magnetization.set_in_region(some_index, (2.0, 2.0))


def test_incompatible_function_in_region_scalar_parameter(test_parameters):
    magnet, regions = test_parameters

    def bad_func(x, y, z):
        return (x, y)

    some_index = np.random.choice(np.unique(regions))

    with pytest.raises(ValueError):
        magnet.ku1.set_in_region(some_index, bad_func)


def test_incompatible_function_in_region_vector_parameter(test_parameters):
    magnet, regions = test_parameters

    def bad_func(x, y, z):
        return (x, y)

    some_index = np.random.choice(np.unique(regions))

    with pytest.raises(ValueError):
        magnet.bias_magnetic_field.set_in_region(some_index, bad_func)


def test_incompatible_function_in_region_variable(test_parameters):
    magnet, regions = test_parameters

    def bad_func(x, y, z):
        return (x, y)

    some_index = np.random.choice(np.unique(regions))

    with pytest.raises(ValueError):
        magnet.magnetization.set_in_region(some_index, bad_func)
