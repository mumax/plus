import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from mumaxplus import Ferromagnet, Grid, World


@pytest.fixture
def test_parameters():
    nx, ny, nz = 4, 7, 3
    w = World(cellsize=(1e-9, 1e-9, 1e-9))
    magnet = Ferromagnet(w, Grid((nx, ny, nz)))
    return w, magnet


def test_magnetization_normalization(test_parameters):
    _, magnet = test_parameters

    grid_shape = magnet.grid.shape
    print(grid_shape)
    print(*grid_shape)
    m_not_normalized = 10 * np.random.rand(3, *grid_shape) - 5
    magnet.magnetization.set(m_not_normalized)

    m = magnet.magnetization.get()
    norms = np.linalg.norm(m, axis=0)

    assert np.max(np.abs(norms - 1)) < 1e-5


def test_magnet_bias_field(test_parameters):
    _, magnet = test_parameters

    grid_shape = magnet.grid.shape
    bias_field_default = [0, 0, 0]
    bias_field_new = [1, 0, 0]
    expected_field = np.zeros([3, *grid_shape])
    expected_field[0, :, :, :] = 1

    assert magnet.bias_magnetic_field.average() == bias_field_default

    magnet.bias_magnetic_field = bias_field_new
    assert_almost_equal(magnet.external_field.eval(), expected_field)


@pytest.mark.parametrize("size", [(2, 4, 0), (-1, 2, 3), (1, 0, 1)])
def test_invalid_grid_size(size):
    world = World((1, 1, 1))
    with pytest.raises(ValueError):
        magnet = Ferromagnet(world, Grid(size))


def test_max_torque(test_parameters):
    _, magnet = test_parameters

    wanted = np.max(np.linalg.norm(magnet.torque.eval(), axis=0))
    result = magnet.max_torque.eval()
    relerr = abs(wanted - result) / wanted
    assert relerr < 1e-5
