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
        """Set the order of 1/R, where R is the distance between cells, in the
        asymptotic expansion of the demag kernel.
        The default value is 11.
        """
        assert isinstance(order, int), "The order should be an integer."
        self._impl.set_order(order)

    def set_switch_radius(self, R=-1):
        """Set the radius from which the asymptotic expantion should be used.
        Default is -1, then the OOMMF method is used:
        Assume the following errors on the analytical and asymptotic result
        E_analytic = eps R³/V
        E_asymptotic = V R²/(5(R²-dmax²)) dmax^(n-3)/R^(n)
        Here V is dx*dy*dz, dmax = max(dx,dy,dz), n is the order of asymptote
        and eps = 5e-10 is a constant determined by trial and error.
        Use the analytical model when
        E_analytic / E_asymptotic < 1
        """
        self._impl.set_switching_radius(R)
