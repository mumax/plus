import numpy as _np

import _mumaxpluscpp as _cpp
from mumaxplus.world import World
from mumaxplus.grid import Grid
class VoronoiTessellator:

    def __init__(self, grainsize, max_idx=256, seed=1234567):
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
        max_idx : int (default=256)
            The maximum region index within the tessellation. This value
            has no upper bound.
        seed : int (default=1234567)
            The seed of the used random number generators. This seed affects
            the values of the generated region indices and the number and
            positions of the Voronoi centers
        """
        self._impl = _cpp.VoronoiTessellator(grainsize, max_idx, seed)

    def generate(self, world, grid):
        """Generates a Voronoi tessellation.

        Returns an ndarray of shape (nz, ny, nx) which is filled
        with region indices."""

        # is this the cleanest way to check PBC?
        # do we need this check in the C++ module?
        has_pbc = world.pbc_repetitions != (0,0,0)

        self.tessellation = self._impl.generate(grid._impl, world.cellsize, has_pbc)

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
        from collections import defaultdict
        tessellation = self.tessellation
        _, nz, ny, nx = tessellation.shape
        
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
        return _np.unique(_np.ravel(self.tessellation)).astype(int)

    @property
    def number_of_regions(self):
        """Returns number of unique region indices."""
        return _np.unique(_np.ravel(self.tessellation)).size

    # TODO: implement (C++) function which returns neighbouring regions of a certain idx
