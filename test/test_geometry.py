import numpy as np
import pytest

from mumax5 import Ferromagnet, Grid, World


class TestGeometry:
    def test_none_geometry(self):
        """Test if geometry contains all cells if no geometry is specified."""
        world = World(cellsize=(3e-9, 4e-9, 5e-9))
        magnet = Ferromagnet(world, Grid((128, 64, 4)))
        assert np.all(magnet.geometry)

    def test_ndarray_geometry(self):
        """Checks if a geometry is correctly set when using an ndarray."""
        world = World(cellsize=(3e-9, 4e-9, 5e-9))
        grid = Grid((128, 64, 4))
        geometry = np.random.randint(low=0, high=2, size=grid.shape)
        magnet = Ferromagnet(world=world, grid=grid, geometry=geometry)
        assert np.all(magnet.geometry == geometry.astype(bool))

    def test_invalid_ndarray_geometry(self):
        """Test if a geometry array with a wrong shape will raise an error."""
        world = World(cellsize=(3e-9, 4e-9, 5e-9))
        grid = Grid((128, 64, 4))

        # create a geometry array with a wrong shape
        nx, ny, nz = grid.size
        geometry = np.ones((nz, ny, nx + 1), dtype=bool)  # wooops, wrong nx

        with pytest.raises(ValueError):
            magnet = Ferromagnet(world=world, grid=grid, geometry=geometry)

    def test_function_geometry(self):
        """Test if a geometry is correctly set when using a function."""
        world = World(cellsize=(3e-9, 4e-9, 5e-9))
        grid = Grid((128, 64, 4))
        geomfunc = lambda x, y, z: x ** 2 + y ** 2 < (40e-9) ** 2
        magnet = Ferromagnet(world=world, grid=grid, geometry=geomfunc)

        # let's compute the geometry array ourselves here
        x, y, z = magnet.magnetization.meshgrid
        geometry = np.vectorize(geomfunc, otypes=[bool])(x, y, z)

        # which should match the geometry stored in the magnet
        assert np.all(geometry == magnet.geometry)

    def test_invalid_function_geometry(self):
        """Test if an array is raised when an invalid geometry function is used."""
        world = World(cellsize=(3e-9, 4e-9, 5e-9))
        grid = Grid((128, 64, 4))
        geomfunc = lambda x, y: True
        with pytest.raises(TypeError):
            magnet = Ferromagnet(world=world, grid=grid, geometry=geomfunc)
