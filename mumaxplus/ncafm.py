"""Non-collinear antiferromagnet implementation."""

import numpy as _np
import warnings

import _mumaxpluscpp as _cpp

from .dmitensor import DmiTensor, DmiTensorGroup
from .magnet import Magnet
from .fieldquantity import FieldQuantity
from .ferromagnet import Ferromagnet
from .interparameter import InterParameter
from .parameter import Parameter
from .scalarquantity import ScalarQuantity


class NcAfm(Magnet):
    """Create a non-collinear antiferromagnet instance."""

    def __init__(self, world, grid, name="", geometry=None, regions=None):
        """
        Parameters
        ----------
        world : World
            World in which the non-collinear antiferromagnet lives.
        grid : Grid
            The number of cells in x, y, z the non-collinear antiferromagnet should be
            divided into.
        geometry : None, ndarray, or callable (default=None)
            The geometry of the non-collinear antiferromagnet can be set in three ways.

            1. If the geometry contains all cells in the grid, then use None (the default)
            2. Use an ndarray which specifies for each cell wheter or not it is in the
               geometry.
            3. Use a function which takes x, y, and z coordinates as arguments and returns
               true if this position is inside the geometry and false otherwise.
        
        regions : None, ndarray, or callable (default=None)
            The regional structure of a non-collinear antiferromagnet can be set in the
            same three ways as the geometry. This parameter indexes each grid cell to a
            certain region.
        name : str (default="")
            The non-collinear antiferromagnet's identifier. If the name is empty (the default),
            a name for the non-collinear antiferromagnet will be created.
        """
        super().__init__(world._impl.add_ncafm,
                         world, grid, name, geometry, regions)

    def __repr__(self):
        """Return non-collinear antiferromagnet string representation."""
        return f"NcAfm(grid={self.grid}, name='{self.name}')"

    def __setattr__(self, name, value):
        """Set non-collinear antiferromagnet or sublattice properties.
        
            If the non-collinear antiferromagnet doesn't have the named attribute, then the corresponding
            attributes of all sublattices are set.
            e.g. to set the saturation magnetization of all sublattices to the
            same value, one could use:
                NcAfm.msat = 800e3
            which is equal to
                NcAfm.sub1.msat = 800e3
                NcAfm.sub2.msat = 800e3
                NcAfm.sub3.msat = 800e3
        """
        if hasattr(NcAfm, name) or name == "_impl":
            # set attribute of yourself, without causing recursion
            object.__setattr__(self, name, value)
        elif hasattr(Ferromagnet, name):
            setattr(self.sub1, name, value)
            setattr(self.sub2, name, value)
            setattr(self.sub3, name, value)
        else:
            raise AttributeError(
                r'Both non-collinear antiferromagnet and Ferromagnet have no attribute "{}".'.format(name))

    @property
    def sub1(self) -> Ferromagnet:
        """First sublattice instance."""
        return Ferromagnet._from_impl(self._impl.sub1())
    
    @property
    def sub2(self) -> Ferromagnet:
        """Second sublattice instance."""
        return Ferromagnet._from_impl(self._impl.sub2())

    @property
    def sub3(self) -> Ferromagnet:
        """Third sublattice instance."""
        return Ferromagnet._from_impl(self._impl.sub3())
    
    @property
    def sublattices(self) -> tuple[Ferromagnet]:
        return (self.sub1, self.sub2, self.sub3)

    def other_sublattices(self, sub: "Ferromagnet") -> tuple[Ferromagnet]:
        """Returns sister sublattices of given sublattice."""
        return self._impl.other_sublattices(sub._impl)

    @property
    def bias_magnetic_field(self) -> Parameter:
        """Uniform bias magnetic field which will affect a non-collinear antiferromagnet.

        The value should be specifed in Teslas.
        """
        return self.sub1.bias_magnetic_field

    @bias_magnetic_field.setter
    def bias_magnetic_field(self, value):
        self.sub1.bias_magnetic_field.set(value)
        self.sub2.bias_magnetic_field.set(value)
        self.sub3.bias_magnetic_field.set(value)

    @property
    def enable_demag(self) -> bool:
        """Enable/disable demagnetization switch of all sublattices.

        Default = True.
        """
        return self.sub1.enable_demag

    @enable_demag.setter
    def enable_demag(self, value):
        self.sub1.enable_demag = value
        self.sub2.enable_demag = value
        self.sub3.enable_demag = value

    def minimize(self, tol=1e-6, nsamples=30):
        """Minimize the total energy.

        Fast energy minimization, but less robust than `relax`
        when starting from a high energy state.

        Parameters
        ----------
        tol : int / float (default=1e-6)
            The maximum allowed difference between consecutive magnetization
            evaluations when advancing toward an energy minimum.

        nsamples : int (default=30)
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
        the total energy (i.e. the sum of sublattices) hits the noise floor.
        Hereafter, relaxation keeps on going until the maximum torque is
        minimized.

        Compared to `minimize`, this function takes a longer time to execute,
        but is more robust when starting from a high energy state (i.e. random).

        Parameters
        ----------
        tol : float, default=1e-9
            The lowest maximum error of the timesolver.

        See Also
        --------
        minimize
        """
        if tol >= 1e-5:
            warnings.warn("The set tolerance is greater than or equal to the default value"
                          + " used for the timesolver (1e-5). Using this value results"
                          + " in no torque minimization, only energy minimization.", UserWarning)
        self._impl.relax(tol)


    # ----- MATERIAL PARAMETERS -----------

    @property
    def ncafmex_cell(self) -> Parameter:
        """Intracell non-collinear antiferromagnetic exchange constant (J/m).
        This parameter plays the role of exchange constant of the
        antiferromagnetic homogeneous exchange interaction in a single
        simulation cell.
        
        See Also
        --------
        ncafmex_nn
        latcon
        """
        return Parameter(self._impl.ncafmex_cell)

    @ncafmex_cell.setter
    def ncafmex_cell(self, value):
        self.ncafmex_cell.set(value)

        warn = False
        if self.ncafmex_cell.is_uniform:
            warn = self.ncafmex_cell.uniform_value > 0
        else:
            warn = _np.any(self.ncafmex_cell.eval() > 0)
        
        if warn:
            warnings.warn("The non-collinear antiferromagnetic exchange constant ncafmex_cell"
                          + " is set to a positive value, instead of negative (or zero)."
                          + " Make sure this is intentional!", UserWarning)

    @property
    def ncafmex_nn(self) -> Parameter:
        """Intercell non-collinear antiferromagnetic exchange constant (J/m).
        This parameter plays the role of exchange constant of the
        antiferromagnetic inhomogeneous exchange interaction between
        neighbouring simulation cells.
        
        See Also
        --------
        ncafmex_cell
        """
        return Parameter(self._impl.ncafmex_nn)

    @ncafmex_nn.setter
    def ncafmex_nn(self, value):
        self.ncafmex_nn.set(value)

        warn = False
        if self.ncafmex_nn.is_uniform:
            warn = self.ncafmex_nn.uniform_value > 0
        elif _np.any(self.ncafmex_nn.eval() > 0):
            warn = True
        
        if warn:
            warnings.warn("The non-collinear antiferromagnet exchange constant ncafmex_nn"
                          + " is set to a positive value, instead of negative (or zero)."
                          + " Make sure this is intentional!", UserWarning)

    @property
    def inter_ncafmex_nn(self) -> Parameter:
        """Interregional non-collinear antiferromagnetic exchange constant (J/m).
        If set to zero (default), then the harmonic mean of the exchange constants
        of the two regions are used.

        When no exchange interaction between different regions
        is wanted, set `scale_ncafmex_nn` to zero.

        This parameter should be set with

        >>> magnet.inter_ncafmex_nn.set_between(region1, region2, value)

        See Also
        --------
        ncafmex_nn, Ferromagnet.inter_exchange, scale_ncafmex_nn
        Ferromagnet.scale_exchange
        """
        return InterParameter(self._impl.inter_ncafmex_nn)

    @inter_ncafmex_nn.setter
    def inter_ncafmex_nn(self, value):
        if value > 0:
            warnings.warn("The non-collinear antiferromagnetic exchange constant"
                          + " inter_ncafmex_nn is set to a positive value, instead"
                          + "of negative (or zero). Make sure this is intentional!", UserWarning)
        self.inter_ncafmex_nn.set(value)

    @property
    def scale_ncafmex_nn(self) -> Parameter:
        """Scaling of the non-collinear antiferromagneticic exchange constant
        between different regions. This factor is multiplied by the harmonic
        mean of the exchange constants of the two regions.

        If :attr:`inter_ncafmex_nn` is set to a non-zero value, then this
        overrides :attr:`scale_ncafmex_nn`, i.e. :attr:`scale_ncafmex_nn` is
        automatically set to zero when `inter_ncafmex_nn` is not.

        This parameter should be set with

        >>> magnet.scale_ncafmex_nn.set_between(region1, region2, value)

        See Also
        --------
        ncafmex_nn, inter_ncafmex_nn, Ferromagnet.inter_exchange
        Ferromagnet.scale_exchange
        """
        return InterParameter(self._impl.scale_ncafmex_nn)

    @scale_ncafmex_nn.setter
    def scale_ncafmex_nn(self, value):
        self.scale_ncafmex_nn.set(value)

    @property
    def latcon(self) -> Parameter:
        """Lattice constant (m).

        Physical lattice constant of the non-collinear antiferromagnet. This
        doesn't break the micromagnetic character of the simulation package,
        but is only used to calculate the homogeneous exchange field, i.e.
        the non-collinear antiferromagnetic exchange interaction between spins
        at the same site.

        Default = 0.35 nm.

        See Also
        --------
        ncafmex_cell
        """
        return Parameter(self._impl.latcon)
    
    @latcon.setter
    def latcon(self, value):
        self.latcon.set(value)

    @property
    def dmi_tensor(self) -> DmiTensor:
        """
        Get the DMI tensor of this non-collinear antiferromagnet.
        This tensor describes intersublattice DMI exchange.

        Note that individual sublattices can have their own tensor
        to describe intrasublattice DMI exchange.

        Returns
        -------
        DmiTensor
            The DMI tensor of this non-collinear antiferromagnet.
        
        See Also
        --------
        DmiTensor, dmi_tensors
        """
        return DmiTensor(self._impl.dmi_tensor)

    @property
    def dmi_tensors(self) -> DmiTensorGroup:
        """ Returns the DMI tensor of self, self.sub1, self.sub2 and self.sub3.

        This group can be used to set the intersublattice and all intrasublattice
        DMI tensors at the same time.

        For example, to set interfacial DMI in the whole system to the same value,
        one could use

        >>> magnet = NcAfm(world, grid)
        >>> magnet.dmi_tensors.set_interfacial_dmi(1e-3)

        Or to set an individual tensor element, one could use
        
        >>> magnet.dmi_tensors.xxy = 1e-3

        See Also
        --------
        DmiTensor, dmi_tensor
        """
        return DmiTensorGroup([
            self.dmi_tensor, self.sub1.dmi_tensor, self.sub2.dmi_tensor, self.sub3.dmi_tensor
            ])

    @property
    def dmi_vector(self) -> Parameter:
        """ DMI vector D (J/m³) associated with the homogeneous DMI (in a single simulation cell),
         defined by the energy density ε = D . (m1 x m2 + m2 x m3 + m3 x m1) with m1, m2
         and m3 being the sublattice magnetizations.

        See Also
        --------
        DmiTensor, dmi_tensor, dmi_tensors
         """
        return Parameter(self._impl.dmi_vector)

    @dmi_vector.setter
    def dmi_vector(self, value):
        self.dmi_vector.set(value)

    # ----- QUANTITIES ----------------------
    @property
    def octupole_vector(self) -> FieldQuantity:
        """Weighted dimensionless octupole vector of a non-collinear
        antiferromagnet as defined in https://doi.org/10.1038/s41563-023-01620-2.
        """
        return FieldQuantity(_cpp.octupole_vector(self._impl))

    @property
    def full_magnetization(self) -> FieldQuantity:
        """Full non-collinear antiferromagnetic magnetization M1 + M2 + M3 (A/m).
        
        See Also
        --------
        Ferromagnet.full_magnetization
        """
        return FieldQuantity(_cpp.full_magnetization(self._impl))

    @property
    def angle_field(self) -> FieldQuantity:
        """Returns the deviation from the optimal angle (120°) between
        magnetization vectors in the same cell which are coupled by the
        intracell exchange interaction (rad).
        The first component respresents the angle deviation for sub1 and sub2.
        The second component respresents the angle deviation for sub1 and sub3.
        The third component respresents the angle deviation for sub2 and sub3.


        See Also
        --------
        max_intracell_angle_between
        ncafmex_cell
        """
        return FieldQuantity(_cpp.angle_field(self._impl))

    def max_intracell_angle_between(self, sub_i : "Ferromagnet", sub_j : "Ferromagnet"):
        """The maximal deviation from 120° between sublattice spins in the same simulation
        cell (rad). Input should be two integers from {1, 2, 3} denoting a sublattice index.

        See Also
        --------
        angle_field
        ncafmex_cell
        Ferromagnet.max_angle
        """
        return _cpp.max_intracell_angle_between(sub_i._impl, sub_j._impl)

    @property
    def total_energy_density(self) -> FieldQuantity:
        """Total energy density of all sublattices combined (J/m³). Kinetic and
        elastic energy densities of the non-collinear antiferromagnet are also
        included if elastodynamics is enabled.

        See Also
        --------
        total_energy
        Magnet.enable_elastodynamics, Magnet.elastic_energy_density
        Magnet.kinetic_energy_density
        """
        return FieldQuantity(_cpp.total_energy_density(self._impl))

    @property
    def total_energy(self) -> ScalarQuantity:
        """Total energy of all sublattices combined (J). Kinetic and elastic
        energies of the non-collinear antiferromagnet are also included if
        elastodynamics is enabled.

        See Also
        --------
        total_energy_density
        Magnet.enable_elastodynamics, Magnet.elastic_energy
        Magnet.kinetic_energy
        """
        return ScalarQuantity(_cpp.total_energy(self._impl))