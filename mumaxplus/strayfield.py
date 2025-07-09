"""StrayField implementation."""

import _mumaxpluscpp as _cpp

from .fieldquantity import FieldQuantity


class StrayField(FieldQuantity):
    """Represent a stray field of a magnet in a specific grid.

    Parameters
    ----------
    magnet : mumaxplus.Magnet
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

    @property
    def order(self):
        """Set the maximum order of 1/R, where R is the distance between cells, in the
        asymptotic expansion of the demag kernel.

        This value should be an integer and 3 <= order <= 12. Choosing an even
        order gives the same result as that order minus 1.

        The default value is 11.

        See Also
        --------
        epsilon, switch_radius
        """
        return self._impl.order

    @order.setter
    def order(self, value):
        assert isinstance(value, int), "The order should be an integer."
        self._impl.order = value

    @property
    def epsilon(self):
        """Set epsilon to calculate the analytical error. The error is defined
        as epsilon * R³ / V.

        The default value is 5e-10.

        See Also
        --------
        order, switch_radius
        """
        return self._impl.epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._impl.epsilon = value

    @property
    def switch_radius(self):
        """Set the radius R, in meters, from which point the asymptotic expantion
        should be used.
        Default is -1, then the OOMMF error estimations are used:

        Assume the following errors on the analytical and asymptotic result

        E_analytic = eps R³/V

        E_asymptotic = V R²/(5(R²-dmax²)) dmax^(n-3)/R^(n)

        Here V is dx*dy*dz, dmax = max(dx,dy,dz), n is the order of asymptote
        and eps is a constant.
        Use the analytical model when

        E_analytic / E_asymptotic < 1

        See Also
        --------
        set_order, set_epsilon
        """
        return self._impl.switching_radius

    @switch_radius.setter
    def switch_radius(self, value=-1):
        self._impl.switching_radius = value
