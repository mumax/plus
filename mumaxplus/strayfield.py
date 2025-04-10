"""StrayField implementation."""

import _mumaxpluscpp as _cpp

from .fieldquantity import FieldQuantity


class StrayField(FieldQuantity):
    """Represent a stray field of a magnet in a specific grid.

    Parameters
    ----------
    magnet : mumaxplus.Ferromagnet
        Magnet instance which is the field source.
    grid : mumaxplus.Grid
        Grid instance on which the stray field will be computed.
    """

    def __init__(self, magnet, grid):
        super().__init__(_cpp.StrayField(magnet._impl, grid._impl))

    @classmethod
    def _from_impl(cls, impl):
        sf = cls.__new__(cls)
        sf._impl = impl
        return sf

    def set_method(self, method):
        """Set the computation method for the stray field.

        Parameters
        ----------
        method : {"brute", "fft"}, optional
            The default value is "fft".
        """
        self._impl.set_method(method)

    def set_order(self, order):
        """Set the order of 1/R in the asymptotic expansion of the demag kernel.
        The default value is 11.
        """
        self._impl.set_order(order)
