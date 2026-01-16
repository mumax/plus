"""Variable implementation."""

import numpy as _np

from .fieldquantity import FieldQuantity


class Variable(FieldQuantity):
    """Represent a physical variable field, e.g. magnetization."""
    def __init__(self, impl):
        super().__init__(impl)

    def set(self, value):
        """Set the variable value.

        Parameters
        ----------
        value : tuple of floats, ndarray, or callable
            The new value for the variable field can be set uniformly with a tuple
            of floats. The number of floats should match the number of components.
            Or the new value can be set cell by cell with an ndarray with the same
            shape as this variable, or with a function which returns the cell value
            as a function of the position.
        """
        if hasattr(value, "__call__"):
            self._set_func(value)
        else:
            self._impl.set(value)

    def set_in_region(self, region_idx, value):
        """
        Set a static value in a specified region.

        Parameters
        ----------
        region_idx : int
            The index of the region the variable must be set in.
        value : float, tuple of floats, or callable
            Value to assign within the specified region. The value may be either a
            uniform scalar or vector matching the number of variable components, or
            a callable that takes grid coordinates and returns a compatible value.

        See Also
        --------
        :func:`set`
        """

        # uniform value
        if isinstance(value, (float, int)) or (
           isinstance(value, tuple) and len(value) == 3):
            self._impl.set_in_region(region_idx, value)

        # evaluate value based on function
        elif callable(value):
            regions = self._impl.system.regions
            mask = (regions == region_idx)
            x, y, z = self.meshgrid

            field = self.eval().copy()
            data = value(x[mask], y[mask], z[mask])

            if self.ncomp == 1:
                if isinstance(data, (tuple, list)):
                    raise ValueError("Function must return a scalar value.")
                field[0][mask] = data
            else:
                if len(data) != self.ncomp:
                    raise ValueError(f"Function must return values with {self.ncomp} components, "+
                                     f"got {len(data)} instead.")
                for c in range(self.ncomp):
                    field[c][mask] = data[c]
            self._impl.set(field)

        else:
            raise TypeError("Value must be uniform or returned by a function.")

    def _set_func(self, func):
        X, Y, Z = self.meshgrid
        self._impl.set(_np.vectorize(func, otypes=[float]*self.shape[0])(X, Y, Z))

    def get(self):
        """Get the variable value."""
        return self._impl.get()
