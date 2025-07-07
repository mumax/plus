"""World imlementation."""

import _mumaxpluscpp as _cpp

from .timesolver import TimeSolver
from .grid import Grid
from .ferromagnet import Ferromagnet
from .antiferromagnet import Antiferromagnet
from .ncafm import NcAfm

import warnings

class World:
    """Construct a world with a given cell size.

    Parameters
    ----------
    cellsize : tuple[float] of size 3
        A tuple of three floating pointing numbers which represent the dimensions
        of the cells in the x, y, and z direction.

    pbc_repetitions : tuple[int] of size 3, default=(0,0,0)
        The number of repetitions for everything inside mastergrid in the
        x, y and z directions to create periodic boundary conditions.
        The number of repetitions determines the cutoff range for the
        demagnetization.
    
    mastergrid : Grid, default=Grid((0,0,0))
        Mastergrid defines a periodic simulation box. If it has zero size in
        a direction, then it is considered to be infinitely large
        (no periodicity) in that direction.
        A 0 in `mastergrid` should correspond to a 0 in `pbc_repetitions`.
        All subsequently added magnets need to fit inside this mastergrid.

    ``pbc_repetitions`` and ``mastergrid`` can be changed later using ``set_pbc``.

    See Also
    --------
    cellsize, pbc_repetitions, mastergrid
    """

    def __init__(self, cellsize,
                 pbc_repetitions=(0,0,0), mastergrid=Grid((0,0,0))):
        if len(cellsize) != 3:
            raise ValueError("'cellsize' should have three dimensions.")
        if len(pbc_repetitions) != 3:
            raise ValueError("'pbc_repetitions' should have three dimensions.")
        if not all(isinstance(item, int) for item in pbc_repetitions):
            raise ValueError("All elements of `pbc_repetitions` must be integers.")

        self._impl = _cpp.World(cellsize, mastergrid._impl, pbc_repetitions)

    def __repr__(self):
        """Return World string representation."""
        return f"World(cellsize={self.cellsize})"

    @classmethod
    def _from_impl(cls, impl):
        world = cls.__new__(cls)
        world._impl = impl
        return world

    @property
    def timesolver(self):
        """Time solver for this world."""
        return TimeSolver(self._impl.timesolver)

    def get_ferromagnet(self, name):
        """Get a ferromagnet by its name.
        Raises KeyError if there is no magnet with the given name."""
        magnet_impl = self._impl.get_ferromagnet(name)
        if magnet_impl is None:
            raise KeyError(f"No magnet named {name}")
        return Ferromagnet._from_impl(magnet_impl)
    
    def get_antiferromagnet(self, name):
        """Get an antiferromagnet by its name.
        Raises KeyError if there is no magnet with the given name."""
        magnet_impl = self._impl.get_antiferromagnet(name)
        if magnet_impl is None:
            raise KeyError(f"No magnet named {name}")
        return Antiferromagnet._from_impl(magnet_impl)

    def get_ncafm(self, name):
        """Get a non-collinear antiferromagnet by its name.
        Raises KeyError if there is no magnet with the given name."""
        magnet_impl = self._impl.get_ncafm(name)
        if magnet_impl is None:
            raise KeyError(f"No magnet named {name}")
        return NcAfm._from_impl(magnet_impl)

    @property
    def ferromagnets(self):
        """Get a dictionairy of ferromagnets by name."""
        return {key: Ferromagnet._from_impl(impl) for key, impl in
                self._impl.ferromagnets.items()}
    
    @property
    def antiferromagnets(self):
        """Get a dictionairy of antiferromagnets by name."""
        return {key: Antiferromagnet._from_impl(impl) for key, impl in
                self._impl.antiferromagnets.items()}

    @property
    def ncafms(self):
        """Get a dictionairy of non-collinear antiferromagnets by name."""
        return {key: NcAfm._from_impl(impl) for key, impl in
                self._impl.ncafms.items()}
    
    def minimize(self, tol=1e-6, nsamples=10):
        """Minimize the total energy.

        Fast energy minimization of the world as a whole, but less
        robust than `relax` when starting from a high energy state.

        Parameters
        ----------
        tol : int / float (default=1e-6)
            The maximum allowed difference between consecutive magnetization
            evaluations when advancing toward an energy minimum.

        nsamples : int (default=10)
            The number of consecutive magnetization evaluations that must not
            differ by more than the tolerance "tol".

        See Also
        --------
        relax
        """
        self._impl.minimize(tol, nsamples)
    
    def relax(self, tol=1e-9):
        """Relax the state to an energy minimum.

        The system evolves in time without precession (pure damping) until
        the total energy (i.e. the sum of all magnets in this world) hits
        the noise floor.
        Hereafter, relaxation keeps on going until the maximum torque is
        minimized.

        Parameters
        ----------
        tol : float, default=1e-9
            The lowest maximum error of the timesolver.

        See Also
        --------
        RelaxTorqueThreshold
        minimize
        """

        if tol >= 1e-5:
            warnings.warn("The set tolerance is greater than or equal to the default value"
                          + " used for the timesolver (1e-5). Using this value results"
                          + " in no torque minimization, only energy minimization.", UserWarning)
        self._impl.relax(tol)

    @property
    def RelaxTorqueThreshold(self):
        """Threshold torque used for relaxing the system (default = -1).

        If set to a negative value (default behaviour),
        the system relaxes until the total torque (i.e. the sum of all
        magnets in this world) is steady or increasing.

        If set to a positive value,
        the system relaxes until the total torque is smaller than or
        equal to this threshold.

        See Also
        --------
        relax
        """
        return self._impl.RelaxTorqueThreshold
        
    @RelaxTorqueThreshold.setter
    def RelaxTorqueThreshold(self, value):
        assert value != 0, "The relax threshold should not be zero."
        self._impl.RelaxTorqueThreshold = value

    @property
    def cellsize(self):
        """Return the cell size of the world.

        Returns
        -------
        tuple[float] of size 3
            A tuple of three floating pointing numbers which represent the dimensions
            of the cells in the x, y, and z direction.
        """
        return self._impl.cellsize

    @property
    def bias_magnetic_field(self):
        """Return a uniform magnetic field which extends over the whole world."""
        return self._impl.bias_magnetic_field

    @bias_magnetic_field.setter
    def bias_magnetic_field(self, value):
        """Set a uniform magnetic field which extends over the whole world."""
        self._impl.bias_magnetic_field = value

    @property
    def mastergrid(self):
        """The master grid of the world.

        Mastergrid defines a periodic simulation box. If it has zero size in a
        direction, then it is considered to be infinitely large (no periodicity) in
        that direction.

        It is advised to set ``mastergrid`` using ``set_pbc``.

        See Also
        --------
        pbc_repetitions, set_pbc
        """
        return Grid._from_impl(self._impl.mastergrid)

    @mastergrid.setter
    def mastergrid(self, mastergrid: 'Grid'):
        """Set the PBC mastergrid.

        It is advised to set ``mastergrid`` using ``set_pbc``.
        
        This will recalculate all strayfield kernels of all magnets in the world.

        Parameters
        ----------
        mastergrid : Grid
            The PBC mastergrid.
        """
        self._impl.mastergrid = mastergrid._impl

    @property
    def pbc_repetitions(self):
        """The number of repetitions for everything inside mastergrid in the
        x, y and z directions to create periodic boundary conditions. The number of
        repetitions determines the cutoff range for the demagnetization.

        For example (2,0,1) means that, for the strayFieldKernel computation,
        all magnets are essentially copied twice to the right, twice to the left,
        but not in the y direction. That row is then copied once up and once down,
        creating a 5x1x3 grid.

        It is advised to set ``pbc_repetitions`` using ``set_pbc``.

        See Also
        --------
        mastergrid, set_pbc
        """
        return self._impl.pbc_repetitions

    @pbc_repetitions.setter
    def pbc_repetitions(self, value):
        """Set the PBC repetitions. 

        This will recalculate all strayfield kernels of all magnets in the world.

        It is advised to set ``pbc_repetitions`` using ``set_pbc``.

        Parameters
        ----------
        value : tuple[int] of size 3
            The number of repetitions to set in each direction.
        """
        if len(value) != 3:
            raise ValueError("'pbc_repetitions' should have three dimensions.")
        if not all(isinstance(item, int) for item in value):
            raise ValueError("All elements of `pbc_repetitions` must be integers.")
        self._impl.pbc_repetitions = value

    @property
    def bounding_grid(self):
        """Returns Grid which is the minimum bounding box of all magnets
        currently in the world.
        """
        return Grid._from_impl(self._impl.bounding_grid)

    def set_pbc(self, pbc_repetitions, mastergrid=None):
        """Set the periodic boundary conditions.

        This will recalculate all strayfield kernels of all magnets in the world.
        
        Parameters
        ----------
        pbc_repetitions : tuple[int] of size 3
            The number of repetitions for everything inside mastergrid in the
            x, y and z directions to create periodic boundary conditions.
            The number of repetitions determines the cutoff range for the
            demagnetization.
        
        mastergrid : Grid, default=None
            Mastergrid defines a periodic simulation box. If it has zero size in
            a direction, then it is considered to be infinitely large
            (no periodicity) in that direction.
            A 0 in `mastergrid` should correspond to a 0 in `pbc_repetitions`.

            If set to `None` (default), the `mastergrid` will be set to the
            minimum bounding box of the magnets currently inside the world, but
            infinitely large (size 0, no periodicity) for any direction set to 0
            in `pbc_repetitions`.
            This reflects the behavior of the mumax³ SetPBC function.
            This will throw an error if there are no magnets in the world.

        See Also
        --------
        pbc_repetitions, mastergrid
        unset_pbc
        """
        if len(pbc_repetitions) != 3:
            raise ValueError("'pbc_repetitions' should have three dimensions.")
        if not all(isinstance(item, int) for item in pbc_repetitions):
            raise ValueError("All elements of `pbc_repetitions` must be integers.")

        if mastergrid is None:
            self._impl.set_pbc(pbc_repetitions)
        else:
            self._impl.set_pbc(mastergrid._impl, pbc_repetitions)

    def unset_pbc(self):
        """Unset the periodic boundary conditions.

        This will recalculate all strayfield kernels of all magnets in the world.
        
        See Also
        --------
        set_pbc
        """
        self._impl.unset_pbc()
