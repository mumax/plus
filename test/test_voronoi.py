from mumaxplus import Grid, World
from mumaxplus.util import VoronoiTessellator

import pytest
import numpy as np


@pytest.fixture
def test_parameters(request):
    world = World(cellsize=(1e-9, 1e-9, 1e-9))
    grid = Grid((request.param))
    return world, grid

@pytest.mark.parametrize("test_parameters", [(13, 17, 1), (5, 51, 87)], indirect=True)
class TestVoronoiTessellation:
    """Unit tests of the VoronoiTessellator."""

    def test_pbc(self, test_parameters):
        world_nopbc, grid = test_parameters
        world_pbc = World(world_nopbc.cellsize, pbc_repetitions=(1, 1, 1), mastergrid=Grid(grid.size))
        tess = VoronoiTessellator(grainsize=5e-9)
        result_pbc = tess.generate(world_pbc, grid)
        result_nopbc = tess.generate(world_nopbc, grid)
        assert not np.array_equal(result_pbc, result_nopbc)

    def test_region_of_center(self, test_parameters):
        world, grid = test_parameters
        tess = VoronoiTessellator(grainsize=5e-9, region_of_center=lambda coo: int(coo[0]) % 5)
        result = np.ravel(tess.generate(world, grid))
        assert np.all((0 <= result) & (result < 5))

    def test_max_index(self, test_parameters):
        world, grid = test_parameters
        tess = VoronoiTessellator(grainsize=5e-9, max_idx=100)
        result = np.ravel(tess.generate(world, grid))
        assert np.all((0 <= result) & (result < 101))

    def test_seed(self, test_parameters):
        world, grid = test_parameters
        tess1 = VoronoiTessellator(grainsize=5e-9, seed=12345)
        tess2 = VoronoiTessellator(grainsize=5e-9, seed=12345)

        result1 = tess1.generate(world, grid)
        result2 = tess2.generate(world, grid)

        assert np.array_equal(result1, result2)

    def test_grid_compatibility(self, test_parameters):
        world, grid = test_parameters
        tess = VoronoiTessellator(grainsize=5e-9)
        result = tess.generate(world, grid)
        assert result.shape == grid.size[::-1]
