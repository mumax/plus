"""Grid implementation."""

import _mumaxpluscpp as _cpp


class Grid:
    """Create a Grid instance."""
    def __init__(self, size, origin=(0, 0, 0)):
        """
        Parameters
        ----------
        size : tuple[int] of size 3
            The grid size.
        origin : tuple[int] of size 3, optional
            The origin of the grid. The default value is (0, 0, 0).
        """
        if len(size) != 3:
            raise ValueError("'size' should have three dimensions.")

        if len(origin) != 3:
            raise ValueError("'origin' should have three dimensions.")

        self._impl = _cpp.Grid(size, origin)

    def __repr__(self):
        """Return Grid string representation."""
        return f"Grid(size={self.size}, origin={self.origin})"

    def __eq__(self, other: "Grid") -> bool:
        return self._impl == other._impl

    def __ne__(self, other: "Grid") -> bool:
        return self._impl != other._impl

    @classmethod
    def _from_impl(cls, impl):
        grid = cls.__new__(cls)
        grid._impl = impl
        return grid

    @property
    def size(self) -> tuple[int]:
        """Return the number of cells in the x, y, and z direction."""
        return self._impl.size

    @property
    def origin(self) -> tuple[float]:
        """Return the origin the grid."""
        return self._impl.origin

    @property
    def shape(self) -> tuple[int]:
        """Return the number of cells in the x, y, and z direction.

        The shape property is similar to the size property, but with the order of
        directions reversed (z, y, x).
        """
        Nx, Ny, Nz = self.size
        return (Nz, Ny, Nx)

    @property
    def ncells(self) -> int:
        """Return the total number of cells in this grid."""
        return self._impl.ncells
