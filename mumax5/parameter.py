"""Patameter implementation."""

import numpy as np
from numpy.core.fromnumeric import ndim, shape

import _mumax5cpp as _cpp
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

    def eval(self):
        """Estimate parameter value on the given grid.
        
        Returns
        -------
        numpy.ndarray
            Parameter value.
        """
        return self._impl.eval()

    @property
    def is_uniform(self):
        """Return True if a Parameter instance is uniform, otherwise False."""
        return self._impl.is_uniform()

    @property
    def is_dynamic(self):
        """Return True if a Parameter instance has time dependent terms, otherwise False."""
        return self._impl.is_dynamic()

    def add_time_term(self, term, mask=None):
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
            instance and the shape of mask should conform. Default value is None.
        """
        if isinstance(self._impl, _cpp.VectorParameter):
            # The VectorParameter value should be a sequence of size 3
            # here we convert that sequence to a numpy array
            original_term = term
            term = lambda t: np.array(original_term(t), dtype=float)
            # change mask dimensions to include components dimension
            mask = self._check_mask_shape(mask, ncomp=3)
        elif isinstance(self._impl, _cpp.Parameter):
            # change mask dimensions to include components dimension
            mask = self._check_mask_shape(mask, ncomp=1)

        if mask is None:
            self._impl.add_time_term(term)
        else:
            self._impl.add_time_term(term, mask)

    def _check_mask_shape(self, mask, ncomp):
        """Change mask shape to have 4 dimensions and correct components."""
        ndim = 4
        if (mask is not None):
            if len(mask.shape) != ndim:
                expected_mask_shape = (ncomp, *mask.shape)
            elif mask.shape[0] != ncomp:
                expected_mask_shape = (ncomp, *mask.shape[1:])
            else:
                expected_mask_shape = mask.shape
            
            if expected_mask_shape != mask.shape:
                new_mask = np.zeros(shape=expected_mask_shape)

                for i in range(ncomp):
                    new_mask[i] = mask
            else:
                new_mask = mask

            return new_mask

    def remove_time_terms(self):
        """Remove all time dependent terms."""
        self._impl.remove_time_terms()

    def set(self, value):
        """Set the parameter value.

        Use a single float to set a uniform scalar parameter or a tuple of three
        floats for a uniform vector parameter.

        To set the values of an inhomogeneous parameter, use a numpy array or a function
        which returns the parameter value as a function of the position, i.e.
        (x: float, y: float, z: float) -> float or
        (x: float, y: float, z: float) -> sequence[float] of size 3.
        
        To assign time-dependant terms using this method use either a single-argument
        function, i.e. (float t) -> float or (t: float) -> sequence[float] of size 3;
        or a tuple of size two consisting of a time-dependent term as its first entry
        and the mask of the function as its second entry, i.e.
        ((float t) -> float, numpy.ndarray) or ((float t) -> [float], numpy.ndarray).

        Parameters
        ----------
        value: float, tuple of floats, numpy array, or callable
            The new value for the parameter.
        """
        if callable(value):
            # test whether given function takes 1 or 3 arguments
            is_time_function = True
            try:
                value(0)
            except TypeError:
                is_time_function = False

            if is_time_function:
                self.remove_time_terms()
                self.add_time_term(value)
            else:
                self._set_func(value)
        elif isinstance(value, tuple) and callable(value[0]):
            # first term is time-function, second term is a mask
            self.remove_time_terms()
            self.add_time_term(*value)
        else:
            self._impl.set(value)

    def _set_func(self, func):
        value = np.zeros(self.shape, dtype=np.float32)

        for iz in range(value.shape[1]):
            for iy in range(value.shape[2]):
                for ix in range(value.shape[3]):

                    pos = self._impl.system.cell_position((ix, iy, iz))
                    cell_value = np.array(func(*pos), ndmin=1)

                    for ic in range(value.shape[0]):
                        value[ic, iz, iy, ix] = cell_value[ic]

        self._impl.set(value)
