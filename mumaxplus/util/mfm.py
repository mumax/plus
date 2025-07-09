import _mumaxpluscpp as _cpp
from mumaxplus import FieldQuantity

class MFM(FieldQuantity):

    def __init__(self, input, grid):
        """Create an MFM instance.
        
        This class is used to create a magnetic force microscopy image. The
        needle is simulated as a pair of monopoles with a charge of 1/µ0 at a
        distance `tipsize` from one another. Here µ0 is the vacuum permeability.
        The height of the needle is determined by the origin of the input `grid`,
        which gets multiplied by the cellsize of the world, and the `lift` of
        the needle.

        When the `eval` function is called for this instance, it returns a numpy
        array of the same size as the input `grid` which contains the potential
        energy due to the interaction of the magnet(s) and the needle.
        
        Parameters
        ----------
        input : can be either a World or Magnet instance. If it is a
                World, all magnets in that world will be used to calculate
                the potential energy at the tip.
        grid : this is a Grid instance used as a scanning surface.
               Physically, this is the plane on which the MFM needle moves."""
        self._impl = _cpp.MFM(input._impl, grid._impl)

    @property
    def name(self):
        """The name of the MFM instance"""
        return self._impl.name
    
    @property
    def unit(self):
        """The unit of the MFM instance. This is always J."""
        return self._impl.unit
    
    @property
    def lift(self):
        """The height of tip of the needle above the z component of the origin
        of the input `grid`.

        default = 10e-9 m."""
        return self._impl.lift
    
    @lift.setter
    def lift(self, value):
        self._impl.lift = value

    @property
    def tipsize(self):
        """The distance between the two monopoles in the needle.
        
        default = 1e-3 m."""
        return self._impl.tipsize
    
    @tipsize.setter
    def tipsize(self, value):
        self._impl.tipsize = value