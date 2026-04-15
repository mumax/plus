"""FieldQuantity implementation."""

import numpy as _np
import pyovf
from .grid import Grid


class FieldQuantity:
    """A functor representing a physical field quantity."""
    _ovf_counts = {} # It keeps resetting to 0 if it is in the __init__
    def __init__(self, impl):
        self._impl = impl

    def __repr__(self):
        """Return FieldQuantity string representation."""
        return (
            f"FieldQuantity(grid={self.grid}, name='{self.name}', "
            f"ncomp={self.ncomp}, unit={self.unit})"
        )

    @property
    def name(self) -> str:
        """Return instance's name."""
        return self._impl.name

    @property
    def unit(self) -> str:
        """Return the unit of the quantity.

        Returns
        -------
        unit : str
            The unit of the quantity, the default value is an empty string.
        """
        return self._impl.unit

    @property
    def ncomp(self) -> int:
        """Return the number of components of the quantity.

        In most cases this would be either 1 (scalar field) or 3 (vector fields).

        Returns
        -------
        ncomp : int
            The number of components of the quantity.
        """
        return self._impl.ncomp

    @property
    def grid(self) -> Grid:
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

    def average(self) -> float:
        """Evaluate the quantity and return the average of each component."""
        # TODO add return type.
        return self._impl.average()

    def get_rgb(self):
        """Evaluate the vector field quantity and return its rgb representation
        as a numpy ndarray of the same shape (3, nz, ny, nx).
        
        Note
        ----
        The final color scheme is different from mumax³. In this case, the
        saturation does not depend on the z-component anymore, meaning the z=0 plane
        remains unchanged, but other colors will appear slightly less saturated.
        This ensures that the color sphere is continuous everywhere, particularly
        when crossing the xz- or yz-plane with a normalized length less than 1,
        where the colors will fade through gray.
        """
        assert self.ncomp == 3, \
            "The rgb representation can only be calculated for vector fields."
        return self._impl.get_rgb().get()  # Field to ndarray

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

    def save_ovf(self, name=""):
        """Save the FieldQuantity as an OVF file.

        Parameters
        ----------
        name : str (default="")
            The name of the OVF file. If the name is empty (the default), the name of the FieldQuantity will be used.
            
        Warning
        -------
        The shape of the array in the OVF file is (nz, ny, nx, ncomp) and not (ncomp, nz, ny, nx) in order to have the correct metadata.
        
        Warning
        -------
        self.name returns a string with colons (:). To avoid issues on Windows, these colons are changed to underscores (_)."""
        cx, cy, cz = self._impl.system.cellsize
        ovf = pyovf.create(_np.moveaxis(self.eval(), 0, -1), xstepsize=cx, ystepsize=cy, zstepsize=cz, title=self.name)
        ovf.TotalSimTime = self._impl.system.time
        if name == "":
            count = self._ovf_counts.get(self.name, 0)
            name = self.name.replace(":", "_") + f"{count:06d}.ovf"
            self._ovf_counts[self.name] = count + 1
        pyovf.write(name, ovf)

    def load_ovf(self, name):
        """Load an OVF file as a FieldQuantity.

        Parameters
        ----------
        name : str (default="")
            The name of the OVF file."""
        ovf = pyovf.read(name)
        if self.ncomp == 1:
            data = _np.array([ovf.data])
        else:
            # _np.ascontiguousarray is used so data is the transformed array. Otherwise the C++ layer still uses ovf.data
            data = _np.ascontiguousarray(_np.moveaxis(ovf.data, -1, 0))
        self.set(data)