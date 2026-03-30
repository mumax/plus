"""Tests for the InterParameter class."""

import numpy as np
import pytest
from mumaxplus import World, Grid, Ferromagnet


def lut_index(i, j):
    """Mirror the C++ getLutIndex: lower-triangular LUT index for (i, j)."""
    if i > j:
        i, j = j, i
    return j * (j - 1) // 2 + i

n_regions = 5
nx, ny, nz = 32, 16, 1

@pytest.fixture(scope="module")  # reuse across tests
def magnet():
    """Return a Ferromagnet whose cells are randomly assigned to n_regions."""
    world = World((1e-9, 1e-9, 1e-9))
    regions = np.random.randint(0, n_regions, size=(nz, ny, nx))
    magnet = Ferromagnet(world, Grid((nx, ny, nz)), regions=regions)
    return magnet

@pytest.fixture
def interex(magnet):
    """The inter_exchange InterParameter of the re-used module-scoped magnet
    fixture. inter_exchange is reset to a known uniform zero state before each
    test.
    """
    magnet.inter_exchange.set(0.0)  # reset state before each test
    return magnet.inter_exchange

# -------------------- Tests --------------------

class TestRegions:
    def test_reported_number_of_regions(self, interex):
        assert interex.number_of_regions == n_regions

    def test_actual_number_of_regions(self, interex):
        indices = interex.region_indices
        assert len(indices) == n_regions

    def test_region_indices(self, interex):
        for idx in interex.region_indices:
            assert 0 <= idx < interex.number_of_regions


class TestUniformState:
    def test_set_value(self, interex):
        val = 2
        interex.set(val)
        assert interex.is_uniform is True
        assert interex.uniform_value == val

    def test_set_uniform_value(self, interex):
        val = 3
        interex.uniform_value = val  # different way of setting
        assert interex.is_uniform is True
        assert interex.uniform_value == val

    def test_uniform_triangular_matrix(self, interex):
        val = 5
        interex.set(val)
        matrix = interex.eval()

        i_s, j_s = np.tril_indices(n_regions, k=-1)  # lower triangle, no diagonal
        assert np.all(matrix[i_s, j_s] == val)

    def test_diagonal_is_zero(self, interex):
        """Diagonal has no physical meaning for inter-region parameters."""
        interex.set(5.0)
        matrix = interex.eval()
        assert np.all(np.diag(matrix) == 0.0)


class TestSetGetBetween:
    def test_set_between_makes_non_uniform(self, interex):
        interex.set_between(0, 1, 99.0)
        assert interex.is_uniform is False

    def test_get_between_after_set_between(self, interex):
        interex.set_between(1, 3, 42.0)
        assert interex.get_between(1, 3) == pytest.approx(42.0)

    def test_get_between_is_symmetric(self, interex):
        interex.set_between(2, 4, 7.0)
        assert interex.get_between(2, 4) == interex.get_between(4, 2)

    def test_all_set_between(self, interex):
        # also tests for symmetry of eval at the same time

        for i in range(n_regions):
            for j in range(i):  # small range
                val = float(lut_index(i, j))
                interex.set_between(i, j, val)

        matrix = interex.eval()
        for i in range(n_regions):
            for j in range(n_regions):  # larger range than setter
                if i == j:
                    assert matrix[i, j] == 0
                else:
                    assert matrix[i, j] == lut_index(i, j)

    def test_set_between_same_index_exception(self, interex):
        with pytest.raises(Exception):
            interex.set_between(2, 2, 1.0)

    def test_get_between_same_index_exception(self, interex):
        with pytest.raises(Exception):
            interex.get_between(3, 3)

    def test_set_between_with_uniform_value(self, interex):
        """Setting a pair to the current uniform value should keep is_uniform True."""
        interex.set(5.0)
        interex.set_between(1, 2, 5.0)
        assert interex.is_uniform is True

    def test_uniform_value_exception_when_not_uniform(self, interex):
        interex.set_between(0, 1, 99.0)
        assert interex.is_uniform is False
        with pytest.raises(Exception):
            _ = interex.uniform_value


class TestEval:
    def test_eval_shape(self, interex):
        assert interex.eval().shape == (n_regions, n_regions)

    def test_callable_matches_eval(self, interex):
        interex.set(3.0)
        assert np.all(interex() == interex.eval())
