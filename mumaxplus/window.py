class Window:
    """Simulation window of the world.

    Each world already has its own Window. This Window can be accessed through
    the world.window property.

    Windows should not be initialized by the end user.
    """

    def __init__(self, impl, origin=None):
        self._impl = impl
        if origin is not None:
            self._impl._set_origin(origin)

    def _check_boundary(self, boundary):
        if not isinstance(boundary, int) or boundary not in (0, 1):
            raise ValueError(f"Invalid boundary: {boundary}. Must be one of (0, 1).")


    def insert_magnetization(self, boundary, value):
        """
        Set magnetization value at a given boundary.
        If none are given (or set to zero), the current edge value is used.

        Parameters
        ----------
        boundary : int
            Boundary index: 0 = Left/Bottom, 1 = Right/Top
        value : tuple of 3 floats
            Magnetization vector (x, y, z)
        """
        self._check_boundary(boundary)
        self._impl.insert_magnetization(boundary, value)

    @property
    def position(self):
        """Returns the current position of the simulation window."""
        return self._impl.position

    @property
    def velocity(self):
        """Returns the current velocity of the simulation window."""
        return self._impl.velocity

    @property
    def total_shift(self):
        """Returns the total amount shifted by the simulation window."""
        return self._impl.total_shift