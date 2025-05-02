import _mumaxpluscpp as _cpp
from mumaxplus import FieldQuantity

class MFM(FieldQuantity):

    def __init__(self, input, grid):
        self._impl = _cpp.MFM(input._impl, grid._impl)
    
    @property
    def lift(self):
        return self._impl.lift
    
    @lift.setter
    def lift(self, value):
        self._impl.lift.set(value)

    @property
    def tipsize(self):
        return self._impl.tipsize
    
    @tipsize.setter
    def tipsize(self, value):
        self._impl.tipsize = value