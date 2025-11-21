"""Altermagnet implementation."""

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


class Altermagnet(Magnet):
    """Create an altermagnet instance."""
    def __init__(self, world, grid, name="", geometry=None, regions=None):
        """        
        Parameters
        ----------
        world : World
            World in which the altermagnet lives.
        grid : Grid
            The number of cells in x, y, z the altermagnet should be divided into.
        geometry : None, ndarray, or callable (default=None)
            The geometry of the altermagnet can be set in three ways.

            1. If the geometry contains all cells in the grid, then use None (the default)
            2. Use an ndarray which specifies for each cell whether or not it is in the
               geometry.
            3. Use a function which takes x, y, and z coordinates as arguments and returns
               true if this position is inside the geometry and false otherwise.

        regions : None, ndarray, or callable (default=None)
            The regional structure of an altermagnet can be set in the same three ways
            as the geometry. This parameter indexes each grid cell to a certain region.
        name : str (default="")
            The altermagnet's identifier. If the name is empty (the default), a name for the
            altermagnet will be created.
        """
        super().__init__(world._impl.add_altermagnet,
                         world, grid, name, geometry, regions)

    def __repr__(self):
        """Return Altermagnet string representation."""
        return f"Altermagnet(grid={self.grid}, name='{self.name}')"

    def __setattr__(self, name, value):
        """Set ATM or sublattice properties.
        
            If the ATM doesn't have the named attribute, then the corresponding
            attributes of both sublattices are set.
            e.g. to set the saturation magnetization of both sublattices to the
            same value, one could use:
                altermagnet.msat = 800e3
            which is equal to
                altermagnet.sub1.msat = 800e3
                altermagnet.sub2.msat = 800e3
        """
        if hasattr(Altermagnet, name) or name == "_impl":
            # set attribute of yourself, without causing recursion
            object.__setattr__(self, name, value)
        elif hasattr(Ferromagnet, name):
            setattr(self.sub1, name, value)
            setattr(self.sub2, name, value)
        else:
            raise AttributeError(
                r'Both Altermagnet and Ferromagnet have no attribute "{}".'.format(name))

    @property
    def sub1(self) -> Ferromagnet:
        """First sublattice instance."""
        return Ferromagnet._from_impl(self._impl.sub1())
    
    @property
    def sub2(self) -> Ferromagnet:
        """Second sublattice instance."""
        return Ferromagnet._from_impl(self._impl.sub2())
    
    @property
    def sublattices(self) -> tuple[Ferromagnet]:
        """Both sublattice instances"""
        return (self.sub1, self.sub2)

    def other_sublattice(self, sub: "Ferromagnet") -> Ferromagnet:
        """Returns sister sublattice of given sublattice."""
        return Ferromagnet._from_impl(self._impl.other_sublattice(sub._impl))

    @property
    def bias_magnetic_field(self) -> Parameter:
        """Uniform bias magnetic field which will affect an altermagnet.

        The value should be specifed in Teslas.
        """
        return self.sub1.bias_magnetic_field

    @bias_magnetic_field.setter
    def bias_magnetic_field(self, value):
        self.sub1.bias_magnetic_field.set(value)
        self.sub2.bias_magnetic_field.set(value)

    @property
    def enable_demag(self) -> bool:
        """Enable/disable demagnetization switch of both sublattices.

        Default = True.
        """
        return self.sub1.enable_demag

    @enable_demag.setter
    def enable_demag(self, value):
        self.sub1.enable_demag = value
        self.sub2.enable_demag = value

    def minimize(self, tol=1e-6, nsamples=20):
        """Minimize the total energy.

        Fast energy minimization, but less robust than :func:`relax`
        when starting from a high energy state.

        Parameters
        ----------
        tol : int / float (default=1e-6)
            The maximum allowed difference between consecutive magnetization
            evaluations when advancing toward an energy minimum.

        nsamples : int (default=20)
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

        Compared to :func:`minimize`, this function takes a longer time to execute,
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
    def atmex_cell(self) -> Parameter:
        """Intracell altermagnetic exchange constant (J/m).
        This parameter plays the role of exchange constant of
        the altermagnetic homogeneous exchange interaction
        in a single simulation cell.
        
        See Also
        --------
        atmex_ani, atmex_nn, latcon
        """
        return Parameter(self._impl.atmex_cell)

    @atmex_cell.setter
    def atmex_cell(self, value):
        self.atmex_cell.set(value)

        warn = False
        if self.atmex_cell.is_uniform:
            warn = self.atmex_cell.uniform_value > 0
        elif _np.any(self.atmex_cell.eval() > 0):
            warn = True
        
        if warn:
            warnings.warn("The altermagnetic exchange constant atmex_cell"
                          + " is set to a positive value, instead of negative (or zero)."
                          + " Make sure this is intentional!", UserWarning)

    @property
    def atmex_nn(self) -> Parameter:
        """Intercell altermagnetic exchange constant (J/m).
        This parameter plays the role of exchange constant of
        the altermagnetic inhomogeneous exchange interaction
        between neighbouring simulation cells.
        
        See Also
        --------
        atmex_ani, atmex_cell
        """
        return Parameter(self._impl.atmex_nn)

    @atmex_nn.setter
    def atmex_nn(self, value):
        self.atmex_nn.set(value)

        warn = False
        if self.atmex_nn.is_uniform:
            warn = self.atmex_nn.uniform_value > 0
        elif _np.any(self.atmex_nn.eval() > 0):
            warn = True
        
        if warn:
            warnings.warn("The altermagnetic exchange constant atmex_nn"
                          + " is set to a positive value, instead of negative (or zero)."
                          + " Make sure this is intentional!", UserWarning)

    @property
    def atmex_ani(self) -> Parameter:
        """Intercell anisotropic exchange constant (J/m).
        This parameter plays the role of exchange constant of
        the altermagnetic anisotropic exchange interaction
        between neighbouring simulation cells.
        
        See Also
        --------
        atmex_cell, atmex_nn
        """
        return Parameter(self._impl.atmex_ani)

    @atmex_ani.setter
    def atmex_ani(self, value):
        self.atmex_ani.set(value)

    @property
    def inter_atmex_nn(self) -> InterParameter:
        """Interregional altermagnetic exchange constant (J/m).
        If set to zero (default), then the harmonic mean of
        the exchange constants of the two regions are used.

        When no exchange interaction between different regions
        is wanted, set `scale_atmex_nn` to zero.

        This parameter should be set with
        
        >>> magnet.inter_atmex_nn.set_between(region1, region2, value)

        See Also
        --------
        atmex_nn, Ferromagnet.inter_exchange, scale_atmex_nn, Ferromagnet.scale_exchange
        """
        return InterParameter(self._impl.inter_atmex_nn)

    @inter_atmex_nn.setter
    def inter_atmex_nn(self, value):
        if value > 0:
            warnings.warn("The altermagnetic exchange constant inter_atmex_nn"
                          + " is set to a positive value, instead of negative (or zero)."
                          + " Make sure this is intentional!", UserWarning)
        self.inter_atmex_nn.set(value)

    @property
    def scale_atmex_nn(self) -> InterParameter:
        """Scaling of the altermagnetic exchange constant between
        different regions. This factor is multiplied by the harmonic
        mean of the exchange constants of the two regions.

        If `inter_atmex_nn` is set to a non-zero value, then this
        overrides `scale_atmex_nn`, i.e. `scale_atmex_nn` is
        automatically set to zero when `inter_atmex_nn` is not.

        This parameter should be set with
        
        >>> magnet.scale_atmex_nn.set_between(region1, region2, value)

        See Also
        --------
        atmex_nn, inter_atmex_nn, Ferromagnet.inter_exchange, Ferromagnet.scale_exchange
        """
        return InterParameter(self._impl.scale_atmex_nn)

    @scale_atmex_nn.setter
    def scale_atmex_nn(self, value):
        self.scale_atmex_nn.set(value)

    @property
    def latcon(self) -> Parameter:
        """Lattice constant (m).

        Physical lattice constant of the Altermagnet. This doesn't break the
        micromagnetic character of the simulation package, but is only used to
        calculate the homogeneous exchange field, i.e. the altermagnetic
        exchange interaction between spins at the same site.

        Default = 0.35 nm.

        See Also
        --------
        atmex_cell
        """
        return Parameter(self._impl.latcon)
    
    @latcon.setter
    def latcon(self, value):
        self.latcon.set(value)

    @property
    def dmi_tensor(self) -> DmiTensor:
        """
        Get the DMI tensor of this Altermagnet. This tensor
        describes intersublattice DMI exchange.

        Note that individual sublattices can have their own tensor
        to describe intrasublattice DMI exchange.

        Returns
        -------
        DmiTensor
            The DMI tensor of this Altermagnet.
        
        See Also
        --------
        DmiTensor, dmi_tensors, dmi_vector
        """
        return DmiTensor(self._impl.dmi_tensor)

    @property
    def dmi_tensors(self) -> DmiTensorGroup:
        """ Returns the DMI tensor of self, self.sub1 and self.sub2.

        This group can be used to set the intersublattice and both intrasublattice
        DMI tensors at the same time.

        For example, to set interfacial DMI in the whole system to the same value,
        one could use
        
        >>> magnet = Altermagnet(world, grid)
        >>> magnet.dmi_tensors.set_interfacial_dmi(1e-3)

        Or to set an individual tensor element, one could use
        
        >>> magnet.dmi_tensors.xxy = 1e-3

        See Also
        --------
        DmiTensor, dmi_tensor, dmi_vector
        """
        return DmiTensorGroup([self.dmi_tensor, self.sub1.dmi_tensor, self.sub2.dmi_tensor])

    @property
    def dmi_vector(self):
        """ DMI vector D (J/m³) associated with the homogeneous DMI (in a single simulation cell),
         defined by the energy density ε = D . (m1 x m2) with m1 and m2 being the sublattice
         magnetizations.

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
    def neel_vector(self) -> FieldQuantity:
        """Weighted dimensionless Neel vector of an altermagnet.
        (msat1*m1 - msat2*m2) / (msat1 + msat2)
        """
        return FieldQuantity(_cpp.neel_vector(self._impl))
    
    @property
    def full_magnetization(self) -> FieldQuantity:
        """Full altermagnetic magnetization M1 + M2 (A/m).
        
        See Also
        --------
        Ferromagnet.full_magnetization
        """
        return FieldQuantity(_cpp.full_magnetization(self._impl))
    ''' 
    @property
    def angle_field(self) -> FieldQuantity:
        """Returns the deviation from the optimal angle (180°) between
        magnetization vectors in the same cell which are coupled by the
        intracell exchange interaction (rad).

        See Also
        --------
        max_intracell_angle
        atmex_cell
        """
        return FieldQuantity(_cpp.angle_field(self._impl))
    
    @property
    def max_intracell_angle(self) -> ScalarQuantity:
        """The maximal deviation from 180° between ATM-exchange coupled magnetization
        vectors in the same simulation cell (rad).

        See Also
        --------
        angle_field
        atmex_cell
        Ferromagnet.max_angle
        """
        return ScalarQuantity(_cpp.max_intracell_angle(self._impl))
    '''
    @property
    def total_energy_density(self) -> FieldQuantity:
        """Total energy density of both sublattices combined (J/m³). Kinetic and
        elastic energy densities of the altermagnet are also included if
        elastodynamics is enabled.
        
        See Also
        --------
        total_energy
        Magnet.enable_elastodynamics, Magnet.elastic_energy_density, Magnet.kinetic_energy_density
        """
        return FieldQuantity(_cpp.total_energy_density(self._impl))

    @property
    def total_energy(self) -> ScalarQuantity:
        """Total energy of both sublattices combined (J). Kinetic and elastic
        energies of the altermagnet are also included if elastodynamics is
        enabled.
        
        See Also
        --------
        total_energy_density
        Magnet.enable_elastodynamics, Magnet.elastic_energy, Magnet.kinetic_energy
        """
        return ScalarQuantity(_cpp.total_energy(self._impl))