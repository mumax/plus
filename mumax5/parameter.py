"""Patameter implementation."""

from typing import Type, ValuesView
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

    def __repr__(self):
        """Return Parameter string representation."""
        return super().__repr__().replace("FieldQuantity", "Parameter")

    @property
    def is_uniform(self):
        """Return True if a Parameter instance is uniform, otherwise False."""
        return self._impl.is_uniform()

    @property
    def is_dynamic(self):
        """Return True if a Parameter instance has time dependent terms, otherwise False."""
        return self._impl.is_dynamic()

    def add_time_terms(self, term, mask=None):
        """Add a time-dependent term.
        
        If mask is None, then the value of the time-dependent term will be the same for
        every grid cell and the final parameter value will be:
            a) uniform_value + term(t)
            b) cell_value + term(t)
        where t is a time value in seconds.
        If mask is not None, then the value of the time-dependent term will be
        multiplied by the mask values and the parameter instance will be estimated as:
            a) uniform_value + term(t) * mask
            b) cell_value + term(t) * cell_mask_value

        Parameter can have multiple time-dependent terms. All their values will be
        weighted by their mask values and summed, prior to being added to the static
        parameter value.

        Parameters
        ----------
        term : callable
            Time-dependent function that will be added to the static parameter values.
            Possible signatures are (float)->float and (float)->tuple(float).
        mask : numpy.ndarray
            An numpy array defining how the magnitude of the time-dependent function
            should be weighted depending on the cell coordinates. In example, it can
            be an array of 0s and 1s. The number of components of the Parameter
            instance and mask should be the same. Default value is None.
        """
        if mask is None:
            self._impl.add_time_terms(term)
        else:
            # if mask.shape != self.ncomp: should be a different check function
            #     raise ValueError(
            #         f"mask has unexpected shape. Expected {mask.shape}, provided "
            #         f"{self.ncomp}"
            #     )

            self._impl.add_time_terms(term, mask)


    def remove_time_terms(self):
        """Remove all time dependent terms."""
        self._impl.remove_time_terms()


    def set(self, value):
        # change docs to show that we can set time-functions as well
        """Set the parameter value.

        Parameters
        ----------
        value: float, tuple of floats, numpy array, or callable
            The new value for the parameter. Use a single float to set a uniform scalar
            parameter or a tuple of three floats for a uniform vector parameter. To set
            the values of an inhomogeneous parameter, use a numpy array or a function
            which returns the parameter value as a function of the position.
        """
        if callable(value):
            # test whether given function takes 1 or 3 arguments
            time_func = True
            try:
                value(0)
            except TypeError:
                time_func = False

            if time_func:
                self.set(0)
                self.add_time_terms(value)
            else:
                self._set_func(value)
        elif isinstance(value, tuple) and callable(value[0]):
            # first term is time-function, second term is a mask
            self.set(0)
            self.add_time_terms(*value)
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
