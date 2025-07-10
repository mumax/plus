"""InterParameter implementation."""

import _mumaxpluscpp as _cpp
import numpy as _np

class InterParameter():
    """Represent a physical material parameter which acts between
    different regions, i.e. `inter_exchange`.
    """

    def __init__(self, impl):
        """Initialize a python InterParameter from a C++ InterParameter instance.

        InterParameters should only have to be initialized within the mumax⁺
        module and not by the end user.
        """
        self._impl = impl

    def __repr__(self):
        """Return InterParameter string representation."""
        return (
            f"InterParameter(number_of_regions={self.number_of_regions}, "
            f"name='{self.name}', ncomp={self.ncomp}, unit={self.unit})"
        )

    def eval(self):
        """Evaluate the quantity.
        Return a numpy array containing a symmetric matrix where each
        region index corresponds to a row/column index. The elements of this
        matrix correspond to the values of self between the two regions.
        """
        values = self._impl.eval()
        if not values:
            raise ValueError(f"The InterParameter '{self.name}' is not defined.")

        N = int(_np.max(self.region_indices)) + 1  # TODO: use property instead of max+1

        value_matrix = _np.zeros((N, N))
        i_s, j_s = _np.tril_indices(N, k=-1)
        value_matrix[i_s, j_s] = values  # lower triangle
        value_matrix[j_s, i_s] = values  # upper triangle
        return value_matrix

    def __call__(self):
        """Evaluate the quantity.
        Return a numpy array containing a symmetric matrix where each
        region index corresponds to a row/column index. The elements of this
        matrix correspond to the values of self between the two regions.
        """
        return self.eval()

    @property
    def name(self) -> str:
        """Return instance's name."""
        return self._impl.name

    @property
    def unit(self) -> str:
        """Return unit of the quantity."""
        return self._impl.unit

    @property
    def ncomp(self) -> int:
        """Return the number of components of the quantity."""
        return self._impl.ncomp

    @property
    def number_of_regions(self) -> int:
        """Return the number of regions between which the quantity
        is active."""
        return self._impl.number_of_regions

    @property
    def region_indices(self) -> list[int]:
        """Return list of unique region indices."""
        return self._impl.unique_regions

    @property
    def is_uniform(self) -> bool:
        """Return True if an InterParameter instance is uniform, otherwise False.
        
        See Also
        --------
        uniform_value
        """
        return self._impl.is_uniform

    @property
    def uniform_value(self) -> float:
        """Return the uniform value of the InterParameter instance if it exists.
        
        See Also
        --------
        is_uniform
        """
        return self._impl.uniform_value

    @uniform_value.setter
    def uniform_value(self, value):
        """Set the InterParameter value between every different region
        to the same value. This functions the same as simply setting the
        InterParameter with `= value` or `set(value)`.

        See Also
        --------
        set
        """
        assert isinstance(value, (float, int)
                          ), "The value should be uniform and static."
        self._impl.uniform_value = value

    def set(self, value):
        """Set the InterParameter value between every different region
        to the same value.

        See Also
        --------
        set_between
        """
        assert isinstance(value, (float, int)
                          ), "The value should be uniform and static."
        self._impl.set(value)

    def set_between(self, i, j, value):
        """Set InterParameter value between regions i and j.
        
        See Also
        --------
        get_between, set
        """
        self._impl.set_between(i, j, value)

    def get_between(self, i, j) -> float:
        """Get the InterParameter value between regions i and j.
        
        See Also
        --------
        set_between
        """
        return self._impl.get_between(i, j)
