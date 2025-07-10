"""Scalar quantity implementation."""


class ScalarQuantity:
    """Functor representing a physical scalar quantity."""
    def __init__(self, impl):
        self._impl = impl

    def __repr__(self):
        """Return ScalarQuantity string representation."""
        return f"ScalarQuantity(name='{self.name}', unit={self.unit})"

    @property
    def name(self) -> str:
        """Return the instance's name."""
        return self._impl.name

    @property
    def unit(self) -> str:
        """Return the instance's unit."""
        return self._impl.unit

    def eval(self):
        """Evaluate and return scalar value."""
        return self._impl.eval()

    def __call__(self):
        """Evaluate and return scalar value."""
        return self.eval()
