class Window:
    """Simulation window of the world.

    Each world already has its own Window. This Window can be accessed through
    the world.window property.

    Windows should not be initialized by the end user.
    """

    def __init__(self, impl):
        self._impl = impl

    def _check_boundary(self, boundary):
        if not isinstance(boundary, int) or boundary not in (0, 1, 2, 3):
            raise ValueError(f"Invalid boundary: {boundary}. Must be one of (0, 1, 2, 3).")


    def insert_magnetization(self, boundary, value):
        """
        Set magnetization value at a given boundary.

        Parameters
        ----------
        boundary : int
            Boundary index: 0 = Left, 1 = Right, 2 = Top, 3 = Bottom
        value : tuple of 3 floats
            Magnetization vector (x, y, z)
        """
        self._check_boundary(boundary)
        self._impl.insert_magnetization(boundary, value)

    def insert_geometry(self, boundary, value):
        """
        Set geometry value at a given boundary.

        Parameters
        ----------
        boundary : int
            Boundary index: 0 = Left, 1 = Right, 2 = Top, 3 = Bottom
        value : bool
        """
        self._check_boundary(boundary)
        self._impl.insert_geometry(boundary, value)

    def insert_region_index(self, boundary, value):
        """
        Set region value at a given boundary.

        Parameters
        ----------
        boundary : int
            Boundary index: 0 = Left, 1 = Right, 2 = Top, 3 = Bottom
        value : int
        """
        self._check_boundary(boundary)
        self._impl.insert_region_index(boundary, value)

    @property
    def total_shift(self):
        """Returns the total amount shifted by the simulation window."""
        return self._impl.total_shift

    @property
    def velocity(self):
        """Returns the velocity of the simulation window."""
        return self._impl.velocity