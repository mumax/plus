"""Patameter implementation."""

import numpy as _np

from .fieldquantity import FieldQuantity


class Parameter(FieldQuantity):
    """Represent a physical material parameter, e.g. the exchange stiffness."""

    def __init__(self, impl):
        """Initialize a python Parameter from a c++ Parameter instance.

        Parameters should only have to be initialized within the mumax5 module and not
        by the end user.
        """
        self._impl = impl

    @property
    def is_uniform(self):
        """Return true if parameter is uniform, otherwise false."""
        self._impl.is_uniform

    def set(self, value):
        """Set the parameter value.

        Parameter
        ---------
        value: float, tuple of floats, numpy array, or callable
            The new value for the parameter. Use a single float to set a uniform scalar
            parameter or a tuple of three floats for a uniform vector parameter. To set
            the values of an inhomogeneous parameter, use a numpy array or a function
            which returns the parameter value as a function of the position.
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
                    cell_value = _np.array(func(*pos), ndmin=1)

                    for ic in range(value.shape[0]):
                        value[ic, iz, iy, ix] = cell_value[ic]

        self._impl.set(value)
