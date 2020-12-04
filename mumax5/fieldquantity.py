"""FieldQuantity implementation."""

import numpy as _np

from .grid import Grid


class FieldQuantity:
    """A functor representing a physical field quantity.

    The class wraps the C++ FieldQuantity class.
    """

    def __init__(self, impl):
        self._impl = impl

    def __repr__(self):
        """Return FieldQuantity string representation."""
        return (
            f"FieldQuantity(grid={self.grid}, name='{self.name}', "
            f"ncomp={self.ncomp}, unit={self.unit})"
        )

    @property
    def name(self):
        """Return isntance's name."""
        return self._impl.name

    @property
    def unit(self):
        """Return the unit of the quantity.

        Returns
        -------
        unit : str
            The unit of the quantity, the default value is an empty string.
        """
        return self._impl.unit

    @property
    def ncomp(self):
        """Return the number of components of the quantity.

        In most cases this would be either 1 (scalar field) or 3 (vector fields).

        Returns
        -------
        ncomp : int
            The number of components of the quantity.
        """
        return self._impl.ncomp

    @property
    def grid(self):
        """Return grid on which the quantity will be evaluated.

        Returns
        -------
        grid : Grid
            The grid on which the quantity will be evaluated.
        """
        return Grid._from_impl(self._impl.grid)

    @property
    def shape(self):
        """Return the shape of the output numpy array of this quantity.

        Returns
        -------
        shape : tuple of ints
            The shape of the output numpy array of this quantity.
        """
        return (self.ncomp, *self.grid.shape)

    def eval(self):
        """Evaluate the quantity."""
        return self._impl.eval()

    def __call__(self):
        """Evaluate the quantity."""
        return self.eval()

    def average(self):
        """Evaluate the quantity and return the average of each component."""
        # TODO add return type.
        return self._impl.average()

    @property
    def meshgrid(self):
        """Return a numpy meshgrid with the x, y, and z coordinate of each cell."""
        # TODO: it might make more sense to put this somewhere else
        nx, ny, nz = self.grid.size
        cellsize = self._impl.system.cellsize
        origin = self._impl.system.cell_position((0, 0, 0))

        mgrid_idx = _np.flip(_np.mgrid[0:nz, 0:ny, 0:nx], axis=0)

        mgrid = _np.zeros(mgrid_idx.shape, dtype=_np.float32)
        for c in [0, 1, 2]:
            mgrid[c] = origin[c] + mgrid_idx[c] * cellsize[c]

        return mgrid

    def _bench(self, ntimes=100):
        import time

        start = time.time()
        for i in range(ntimes):
            self._impl.exec()
        stop = time.time()
        return (stop - start) / ntimes
