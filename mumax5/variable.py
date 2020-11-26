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

    def _set_func(self, func):
        value = _np.zeros(self.shape, dtype=_np.float32)

        for iz in range(value.shape[1]):
            for iy in range(value.shape[2]):
                for ix in range(value.shape[3]):

                    pos = self._impl.system.cell_position((ix, iy, iz))
                    cell_value = func(*pos)

                    for ic in range(value.shape[0]):
                        value[ic, iz, iy, ix] = cell_value[ic]

        self._impl.set(value)

    def get(self):
        """Get the variable value."""
        return self._impl.get()
