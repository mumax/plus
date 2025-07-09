import numpy as _np
import warnings as _w

import _mumaxpluscpp as _cpp
from mumaxplus.world import World
from mumaxplus.grid import Grid
class VoronoiTessellator:


    def __init__(self, grainsize, seed=None, max_idx=255, region_of_center=None):
        """Create a Voronoi tessellator instance.
        
        This class is used to generate a Voronoi tessellation, which can
        be done using either the `generate` or the `coo_to_idx` method.

        **Important:** other methods in this class cannot be used unless
        `generate` has been called. E.g. retrieving a list of region
        indices requires a specified world and grid.

        Parameters
        ----------
        grainsize : float
            The average grain diameter.
        seed : int (default=None)
            The seed of the used random number generators. This seed affects
            the values of the generated region indices and the number and
            positions of the Voronoi centers. When set to `None` (default), a random
            integer from a uniform distribution [0,1234567) is chosen as the seed.
        max_idx : int (default=255)
            The (inclusive) maximum region index within the tessellation. This value
            has no upper bound.
        region_of_center : callable, optional
            A function with signature tuple(float)->int which assigns region indices to
            generated Voronoi centers. If not specified, a random region index will be generated.
        """
        self.seed = seed if seed is not None else _np.random.randint(1234567)
        self._impl = _cpp.VoronoiTessellator(grainsize, self.seed, max_idx, region_of_center)

    def generate(self, world, grid, make_2d=False):
        """Generates a Voronoi tessellation.

        If make_2d is set to True (default: False) the VoronoiTesselator ignores the z-direction,
        resulting in a tessellation where a region index is uniform in the z-direction.

        **Important:** 2D grids are assumed to lie in the xy-plane.

        Returns an ndarray of shape (nz, ny, nx) which is filled
        with region indices.
        """

        has_pbc = world.pbc_repetitions != (0,0,0)
        self.tessellation = self._impl.generate(grid._impl, world.cellsize, has_pbc, make_2d)
        return self.tessellation
    
    def coo_to_idx(self, x, y, z):
        """Returns the region index (int) of the given coordinate within the
        Voronoi tessellation.

        **Important:** This method has no information about the used world and
        grid. E.g. this means that periodic boundary conditions will not apply.
        This can be overriden by calling `generate` before assinging this function
        to the `Magnet`'s regions parameter.
        """
        return self._impl.coo_to_idx((x,y,z))

    @property
    def indexDictionary(self):
        """Create a dictionary where each region (key) is linked
        to a list of grid coordinates (value)."""

        if not hasattr(self, 'tessellation'):
            _w.warn(
            "The full tessellation has not been generated yet."
             " Call `generate(world, grid)` before accessing `indexDictionary`.", UserWarning
            )
            return None

        from collections import defaultdict
        tessellation = self.tessellation
        nz, ny, nx = tessellation.shape
        
        idxs = tessellation.flatten()
        coords = _np.array(_np.meshgrid(range(nx), range(ny), range(nz), indexing='ij')
                          ).reshape(3, -1).T
        
        idxDict = defaultdict(list)
        
        for idx, coord in zip(idxs, coords):
            idxDict[idx].append(tuple(coord))
            
        idxDict = dict(idxDict)
        return idxDict

    @property
    def indices(self):
        """Returns list of unique region indices."""
        if not hasattr(self, 'tessellation'):
            _w.warn(
            "The full tessellation has not been generated yet."
             " Call `generate(world, grid)` before accessing `indices`.", UserWarning
            )
            return None
        return _np.unique(_np.ravel(self.tessellation)).astype(int)

    @property
    def number_of_regions(self):
        """Returns number of unique region indices."""
        if not hasattr(self, 'tessellation'):
            _w.warn(
            "The full tessellation has not been generated yet."
             " Call `generate(world, grid)` before accessing `number_of_regions`.", UserWarning
            )
            return None
        return _np.unique(_np.ravel(self.tessellation)).size

    # TODO: implement (C++) function which returns neighbouring regions of a certain idx
