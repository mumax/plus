"""BoundaryTraction implementation."""

from .parameter import Parameter

class BoundaryTraction:
    """Contains the traction parameters of a Magnet's boundary at each of the 6
    sides of rectangular cells. The traction is applied as a boundary condition
    of the stress tensor σ during calculation of the internal force.
    
    t = σ·n = n·σ

    Here t is the applied traction vector at the boundary with normal vector n,
    with n equal to ± e_x, ± e_y or ± e_z, represented by :attr:`pos_x_side`,
    :attr:`neg_x_side`, :attr:`pos_y_side`, :attr:`neg_y_side`,
    :attr:`pos_z_side` and :attr:`neg_z_side`.

    The traction set in any cell without a relevant boundary (e.g. in bulk
    cells) is thus ignored.

    BoundaryTraction should not be initialized by the end user.
    """

    def __init__(self, impl):
        """Create BoundaryTraction from a cpp BoundaryTraction instance.

        warning
        -------
        The end user should not create BoundaryTraction instances. Each Magnet
        already has BoundaryTraction as an attribute which can be used to set
        the traction parameters. See :func:`Magnet.boundary_traction`.

        Parameters
        ----------
        impl: _mumaxpluscpp.BoundaryTraction
        """
        self._impl = impl

    def make_zero(self):
        """Set all traction parameters to zero."""
        self.pos_x_side = (0., 0., 0.)
        self.neg_x_side = (0., 0., 0.)
        self.pos_y_side = (0., 0., 0.)
        self.neg_y_side = (0., 0., 0.)
        self.pos_z_side = (0., 0., 0.)
        self.neg_z_side = (0., 0., 0.)

    @property
    def pos_x_side(self):
        """External traction vector (Pa)
        applied at the boundary with normal vector n = (+1, 0, 0).
        """
        return Parameter(self._impl.pos_x_side)

    @pos_x_side.setter
    def pos_x_side(self, value):
        self.pos_x_side.set(value)

    @property
    def neg_x_side(self):
        """External traction vector (Pa)
        applied at the boundary with normal vector n = (-1, 0, 0).
        """
        return Parameter(self._impl.neg_x_side)

    @neg_x_side.setter
    def neg_x_side(self, value):
        self.neg_x_side.set(value)


    @property
    def pos_y_side(self):
        """External traction vector (Pa)
        applied at the boundary with normal vector n = (0, +1, 0).
        """
        return Parameter(self._impl.pos_y_side)

    @pos_y_side.setter
    def pos_y_side(self, value):
        self.pos_y_side.set(value)

    @property
    def neg_y_side(self):
        """External traction vector (Pa)
        applied at the boundary with normal vector n = (0, -1, 0).
        """
        return Parameter(self._impl.neg_y_side)

    @neg_y_side.setter
    def neg_y_side(self, value):
        self.neg_y_side.set(value)
    

    @property
    def pos_z_side(self):
        """External traction vector (Pa)
        applied at the boundary with normal vector n = (0, 0, +1).
        """
        return Parameter(self._impl.pos_z_side)

    @pos_z_side.setter
    def pos_z_side(self, value):
        self.pos_z_side.set(value)

    @property
    def neg_z_side(self):
        """External traction vector (Pa)
        applied at the boundary with normal vector n = (0, 0, -1).
        """
        return Parameter(self._impl.neg_z_side)

    @neg_z_side.setter
    def neg_z_side(self, value):
        self.neg_z_side.set(value)
