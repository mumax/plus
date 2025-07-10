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
        X, Y, Z = self.meshgrid
        self._impl.set(_np.vectorize(func, otypes=[float]*self.shape[0])(X, Y, Z))

    def get(self):
        """Get the variable value."""
        return self._impl.get()
