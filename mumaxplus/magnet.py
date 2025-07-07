"""Magnet implementation."""

import numpy as _np
from abc import ABC, abstractmethod

import _mumaxpluscpp as _cpp

from .fieldquantity import FieldQuantity
from .grid import Grid
from .parameter import Parameter
from .scalarquantity import ScalarQuantity
from .strayfield import StrayField
from .variable import Variable

class Magnet(ABC):
    """A Magnet should never be initialized by the user. It contains no physics.
    Use ``Ferromagnet`` or ``Antiferromagnet`` instead.

    Parameters
    ----------
    _impl_function : callable
        The appropriate `world._impl` method of the child magnet, for example
        `world._impl.add_ferromagnet` or `world._impl.add_antiferromagnet`.
    world : mumaxplus.World
        World in which the magnet lives.
    grid : mumaxplus.Grid
        The number of cells in x, y, z the magnet should be divided into.
    geometry : None, ndarray, or callable (default=None)
        The geometry of the magnet can be set in three ways.

        1. If the geometry contains all cells in the grid, then use None (the default)
        2. Use an ndarray which specifies for each cell wheter or not it is in the
           geometry.
        3. Use a function which takes x, y, and z coordinates as arguments and returns
           true if this position is inside the geometry and false otherwise.
       
    regions : None, ndarray, or callable (default=None)
        The regional structure of a magnet can be set in the same three ways
        as the geometry. This parameter indexes each grid cell to a certain region.
    name : str (default="")
        The magnet's identifier. If the name is empty (the default), a name for the
        magnet will be created.
    """
    
    @abstractmethod  # TODO: does this work?
    def __init__(self, _impl_function, world, grid, name="", geometry=None, regions=None):
   
        geometry_array = self._get_mask_array(geometry, grid, world, "geometry")
        regions_array = self._get_mask_array(regions, grid, world, "regions")
        self._impl = _impl_function(grid._impl, geometry_array, regions_array, name)

    @staticmethod
    def _get_mask_array(input, grid, world, input_name):
        if input is None:
            return None
        
        T = bool if input_name == "geometry" else int
        if callable(input):
            # construct meshgrid of x, y, and z coordinates for the grid
            nx, ny, nz = grid.size
            cs = world.cellsize
            idxs = _np.flip(_np.mgrid[0:nz, 0:ny, 0:nx], axis=0)  # meshgrid of indices
            x, y, z = [(grid.origin[i] + idxs[i]) * cs[i] for i in [0, 1, 2]]

            # evaluate the input function for each position in this meshgrid
            return _np.vectorize(input, otypes=[T])(x, y, z)

        # When here, the input is not None, not callable, so it should be an
        # ndarray or at least should be convertable to ndarray
        input_array = _np.array(input, dtype=T)
        if input_array.shape != grid.shape:
            raise ValueError(
                "The dimensions of the {} do not match the dimensions "
                + "of the grid.".format(input_name)
                    )
        return input_array

    @abstractmethod
    def __repr__(self):
        """Return Magnet string representation."""
        return f"Magnet(grid={self.grid}, name='{self.name}')"

    @classmethod
    def _from_impl(cls, impl):
        magnet = cls.__new__(cls)
        magnet._impl = impl
        return magnet

    @property
    def name(self):
        """Name of the magnet."""
        return self._impl.name

    @property
    def grid(self):
        """Return the underlying grid of the magnet."""
        return Grid._from_impl(self._impl.system.grid)

    @property
    def cellsize(self):
        """Dimensions of the cell."""
        return self._impl.system.cellsize

    @property
    def geometry(self):
        """Geometry of the magnet."""
        return self._impl.system.geometry

    @property
    def regions(self):
        """Regions of the magnet."""
        return self._impl.system.regions

    @property
    def origin(self):
        """Origin of the magnet.

        Returns
        -------
        origin: tuple[float] of size 3
            xyz coordinate of the origin of the magnet.
        """
        return self._impl.system.origin

    @property
    def center(self):
        """Center of the magnet.

        Returns
        -------
        center: tuple[float] of size 3
            xyz coordinate of the center of the magnet.
        """
        return self._impl.system.center

    @property
    def world(self):
        """Return the World of which the magnet is a part."""
        from .world import World  # imported here to avoid circular imports
        return World._from_impl(self._impl.world)

    @property
    def meshgrid(self):
        """Return a numpy meshgrid with the x, y, and z coordinate of each cell."""
        nx, ny, nz = self.grid.size
        mgrid_idx = _np.flip(_np.mgrid[0:nz, 0:ny, 0:nx], axis=0)

        mgrid = _np.zeros(mgrid_idx.shape, dtype=_np.float32)
        for c in [0, 1, 2]:
            mgrid[c] = self.origin[c] + mgrid_idx[c] * self.cellsize[c]

        return mgrid

    @property
    def enable_as_stray_field_source(self):
        """Enable/disable this magnet (self) as the source of stray fields felt
        by other magnets. This does not influence demagnetization.
        
        Default = True.

        See Also
        --------
        enable_as_stray_field_destination
        """
        return self._impl.enable_as_stray_field_source

    @enable_as_stray_field_source.setter
    def enable_as_stray_field_source(self, value):
        self._impl.enable_as_stray_field_source = value

    @property
    def enable_as_stray_field_destination(self):
        """Enable/disable whether this magnet (self) is influenced by the stray
        fields of other magnets. This does not influence demagnetization.
        
        Default = True.

        See Also
        --------
        enable_as_stray_field_source
        """
        return self._impl.enable_as_stray_field_destination

    @enable_as_stray_field_destination.setter
    def enable_as_stray_field_destination(self, value):
        self._impl.enable_as_stray_field_destination = value

    # ----- ELASTIC VARIABLES  -------

    @property
    def elastic_displacement(self):
        """Elastic displacement vector (m).

        The elastic displacement is uninitialized (does not exist) if the
        elastodynamics are disabled.
        
        See Also
        --------
        elastic_velocity
        enable_elastodynamics
        """
        return Variable(self._impl.elastic_displacement)

    @elastic_displacement.setter
    def elastic_displacement(self, value):
        self.elastic_displacement.set(value)

    @property
    def elastic_velocity(self):
        """Elastic velocity vector (m/s).
        
        The elastic velocity is uninitialized (does not exist) if the
        elastodynamics are disabled.

        elastic_displacement
        enable_elastodynamics
        """
        return Variable(self._impl.elastic_velocity)

    @elastic_velocity.setter
    def elastic_velocity(self, value):
        self.elastic_velocity.set(value)

    @property
    def enable_elastodynamics(self):
        """Enable/disable elastodynamic time evolution.

        If elastodynamics are disabled (default), the elastic displacement and
        velocity are uninitialized to save memory.

        Elastodynamics can not be used together with rigid normal and shear
        strain, where strain is set by the user instead.

        See Also
        --------
        rigid_norm_strain, rigid_shear_strain
        """
        return self._impl.enable_elastodynamics

    @enable_elastodynamics.setter
    def enable_elastodynamics(self, value):
        self._impl.enable_elastodynamics = value

    # ----- ELASTIC PARAMETERS -------

    @property
    def external_body_force(self):
        """External body force density f_ext that is added to the effective body
        force density (N/m³).

        See Also
        --------
        effective_body_force
        """
        return Parameter(self._impl.external_body_force)

    @external_body_force.setter
    def external_body_force(self, value):
        self.external_body_force.set(value)

    @property
    def C11(self):
        """Stiffness constant C11 = C22 = C33 of the stiffness tensor (N/m²).
        
        See Also
        --------
        C12, C44, stress_tensor
        """
        return Parameter(self._impl.C11)

    @C11.setter
    def C11(self, value):
        self.C11.set(value)

    @property
    def C12(self):
        """Stiffness constant C12 = C13 = C23 of the stiffness tensor (N/m²).
        
        See Also
        --------
        C11, C44, stress_tensor
        """
        return Parameter(self._impl.C12)

    @C12.setter
    def C12(self, value):
        self.C12.set(value)

    @property
    def C44(self):
        """Stiffness constant C44 = C55 = C66 of the stiffness tensor (N/m²).

        For isotropic materials, this is equal to the shear modulus.

        See Also
        --------
        C11, C12, stress_tensor
        """
        return Parameter(self._impl.C44)

    @C44.setter
    def C44(self, value):
        self.C44.set(value)

    @property
    def eta(self):
        """Phenomenological elastic damping constant (kg/m³s)."""
        return Parameter(self._impl.eta)

    @eta.setter
    def eta(self, value):
        self.eta.set(value)

    @property
    def stiffness_damping(self):
        """Rayleigh damping stiffness coefficient β (s).

        Default = 0.05 / (1e12 * pi) s.

        This corresponds to a damping ratio of 5% at a frequency of 1 THz.
        This is meant to stabilize the high frequency modes, with a typical
        wavelength of only a few cellsizes in length.

        The viscosity tensor η is assumed to be proportional to the stiffness
        tensor C, with this coefficient β as the proportionality constant.

        η = β * C

        This parameter is completely **ignored** when any component of the viscosity
        tensor (:attr:`eta11`, :attr:`et12` or :attr:`eta44`) has been set.

        See Also
        --------
        eta11, eta12, eta44
        C11, C12, C44
        strain_rate, viscous_stress
        """
        return Parameter(self._impl.stiffness_damping)

    @stiffness_damping.setter
    def stiffness_damping(self, value):
        self.stiffness_damping.set(value)

    @property
    def eta11(self):
        """Viscosity constant eta11 = eta22 = eta33 of the viscosity tensor (Pa s)
        in Voigt notation, which connects strain rate to viscous stress via
        σ = η : dε/dt.

        When set, this parameter overrules the :attr:`stiffness_damping`.
        
        See Also
        --------
        eta12, eta44
        stiffness_damping
        strain_rate, viscous_stress
        """
        return Parameter(self._impl.eta11)

    @eta11.setter
    def eta11(self, value):
        self.eta11.set(value)

    @property
    def eta12(self):
        """Viscosity constant eta12 = eta13 = eta23 of the viscosity tensor (Pa s)
        in Voigt notation, which connects strain rate to viscous stress via
        σ = η : dε/dt.

        When set, this parameter overrules the :attr:`stiffness_damping`.
        
        See Also
        --------
        eta11, eta44
        stiffness_damping
        strain_rate, viscous_stress
        """
        return Parameter(self._impl.eta12)

    @eta12.setter
    def eta12(self, value):
        self.eta12.set(value)

    @property
    def eta44(self):
        """Viscosity constant eta44 = eta55 = eta66 of the viscosity tensor (Pa s)
        in Voigt notation, which connects strain rate to viscous stress via
        σ = η : dε/dt.

        For isotropic materials, this is equal to the shear viscosity.

        When set, this parameter overrules the :attr:`stiffness_damping`.
        
        See Also
        --------
        eta11, eta12
        stiffness_damping
        strain_rate, visocus_stress
        """
        return Parameter(self._impl.eta44)

    @eta44.setter
    def eta44(self, value):
        self.eta44.set(value)

    @property
    def rho(self):
        """Mass density (kg/m³).
        
        Default = 1.0 kg/m³
        """
        return Parameter(self._impl.rho)

    @rho.setter
    def rho(self, value):
        self.rho.set(value)

    @property
    def rigid_norm_strain(self):
        """The applied normal strain (m/m).

        This quantity has three components (εxx, εyy, εzz),
        which forms the diagonal of the symmetric strain tensor::

               εxx  0   0
                0  εyy  0
                0   0  εzz

        The rigid strain can not be used together with elastodynamics.
        Here the strain is set by the user as a parameter for the magnetoelastic
        field, instead of calculated dynamically (strain_tensor).

        See Also
        --------
        rigid_shear_strain
        enable_elastodynamics, strain_tensor
        """
        return Parameter(self._impl.rigid_norm_strain)
    
    @rigid_norm_strain.setter
    def rigid_norm_strain(self, value):
        if self.enable_elastodynamics:
            raise Exception("Can not use normal strain with elastodynamics enabled.")
        self.rigid_norm_strain.set(value)
    
    @property
    def rigid_shear_strain(self):
        """The applied shear strain (m/m).

        This quantity has three components (εxy, εxz, εyz),
        which forms the off-diagonal of the symmetric strain tensor::

                 0  εxy εxz
                εxy  0  εyz
                εxz εyz  0

        The rigid strain can not be used together with elastodynamics.
        Here the strain is set by the user as a parameter for the magnetoelastic
        field, instead of calculated dynamically (strain_tensor).

        See Also
        --------
        rigid_norm_strain
        enable_elastodynamics, strain_tensor
        """
        return Parameter(self._impl.rigid_shear_strain)

    @rigid_shear_strain.setter
    def rigid_shear_strain(self, value):
        if self.enable_elastodynamics:
            raise Exception("Can not use shear strain with elastodynamics enabled.")
        self.rigid_shear_strain.set(value)

    # ----- ELASTIC QUANTITIES -------

    @property
    def strain_tensor(self):
        """Strain tensor (m/m), calculated according to ε = 1/2 (∇u + (∇u)^T),
        with u the elastic displacement.

        This quantity has six components (εxx, εyy, εzz, εxy, εxz, εyz),
        which forms the symmetric strain tensor::

               εxx εxy εxz
               εxy εyy εyz
               εxz εyz εzz

        Note that the strain corresponds to the real strain and not the
        engineering strain, which would be (εxx, εyy, εzz, 2*εxy, 2*εxz, 2*εyz).

        If you want to set the strain as a parameter yourself, use rigid normal
        and shear strain.

        See Also
        --------
        elastic_energy, elastic_energy_density, elastic_displacement, stress_tensor
        rigid_norm_strain, rigid_shear_strain
        """
        return FieldQuantity(_cpp.strain_tensor(self._impl))

    @property
    def strain_rate(self):
        """Time derivative of the strain tensor (1/s), calculated according to
        dε/dt = 1/2 (∇v + (∇v)^T), with v the elastic velocity.

        See Also
        --------
        strain_tensor
        elastic_velocity
        """
        return FieldQuantity(_cpp.strain_rate(self._impl))
    
    @property
    def elastic_stress(self):
        """Elastic stress tensor (N/m²), calculated according to Hooke's law
        σ = c:ε.

        See Also
        --------
        C11, C12, C44
        strain_tensor
        stress_tensor
        """
        return FieldQuantity(_cpp.elastic_stress(self._impl))

    @property
    def viscous_stress(self):
        """Viscous stress tensor (N/m²) due to isotropic viscous damping,
        calculated according to

        σ = η : dε/dt

        with η the viscosity tensor and dε/dt the strain rate tensor.

        See Also
        --------
        eta11, eta12, eta44
        stiffness_damping
        strain_rate
        stress_tensor
        """
        return FieldQuantity(_cpp.viscous_stress(self._impl))

    @property
    def stress_tensor(self):
        """Total stress tensor (N/m²), including elastic stress and viscous stress.
        
        This quantity has six components (σxx, σyy, σzz, σxy, σxz, σyz),
        which forms the symmetric stress tensor::

               σxx σxy σxz
               σxy σyy σyz
               σxz σyz σzz

        See Also
        --------
        elastic_stress, viscous_stress
        internal_body_force
        """
        return FieldQuantity(_cpp.stress_tensor(self._impl))

    @property
    def internal_body_force(self):
        """Internal body force density due to stress divergence (N/m³).

        f = ∇·σ
        
        See Also
        --------
        stress_tensor
        effective_body_force
        """
        return FieldQuantity(_cpp.internal_body_force(self._impl))

    @property
    def effective_body_force(self):
        """Elastic effective body force density is the sum of elastic,
        magnetoelastic and external body force densities (N/m³).
        Elastic damping is not included.

        f_eff = f_int + f_mel + f_ext

        In the case of this Magnet being a host (antiferromagnet),
        f_mel is the sum of all magnetoelastic body forces of all sublattices.

        See Also
        --------
        external_body_force, internal_body_force, magnetoelastic_force
        """
        return FieldQuantity(_cpp.effective_body_force(self._impl))

    @property
    def elastic_damping(self):
        """Elastic damping body force density proportional to η and velocity: -ηv (N/m³).

        See Also
        --------
        eta, elastic_velocity
        """
        return FieldQuantity(_cpp.elastic_damping(self._impl))

    @property
    def elastic_acceleration(self):
        """Elastic acceleration includes all effects that influence the elastic
        velocity including elastic, magnetoelastic and external body force
        densities, and elastic damping (m/s²).

        See Also
        --------
        rho
        effective_body_force, elastic_damping
        """
        return FieldQuantity(_cpp.elastic_acceleration(self._impl))

    @property
    def kinetic_energy_density(self):
        """Kinetic energy density related to the elastic velocity (J/m³).
        
        See Also
        --------
        elastic_velocity, kinetic_energy, rho
        """
        return FieldQuantity(_cpp.kinetic_energy_density(self._impl))

    @property
    def kinetic_energy(self):
        """Kinetic energy related to the elastic velocity (J/m³).
        
        See Also
        --------
        elastic_velocity, kinetic_energy_density, rho
        """
        return ScalarQuantity(_cpp.kinetic_energy(self._impl))

    @property
    def elastic_energy_density(self):
        """Potential energy density related to elastics (J/m³).
        This is given by 1/2 σ:ε
        
        See Also
        --------
        elastic_energy, strain_tensor, stress_tensor
        """
        return FieldQuantity(_cpp.elastic_energy_density(self._impl))

    @property
    def elastic_energy(self):
        """Potential energy related to elastics (J).
        
        See Also
        --------
        elastic_energy_density, strain_tensor, stress_tensor
        """
        return ScalarQuantity(_cpp.elastic_energy(self._impl))

    @property
    def poynting_vector(self):
        """Poynting vector (W/m2).
        This is given by P = - σv
        
        See Also
        --------
        elastic_velocity, stress_tensor
        """
        return FieldQuantity(_cpp.poynting_vector(self._impl))

    # --- stray field ---

    def stray_field_from_magnet(self, source_magnet: "Magnet"):
        """Return the magnetic field created by the given input `source_magnet`,
        felt by this magnet (`self`). This raises an error if there exists no
        `StrayField` instance between these two magnets.

        Parameters
        ----------
        source_magnet : Magnet
            The magnet acting as the source of the requested stray field.
        
        Returns
        -------
        stray_field : StrayField
            StrayField with the given `source_magnet` as source and the Grid of
            this magnet (`self`) as destination.

        See Also
        --------
        StrayField
        """
        return StrayField._from_impl(
                        self._impl.stray_field_from_magnet(source_magnet._impl))

    @property
    def demag_field(self):
        """Demagnetization field (T).

        Ferromagnetic sublattices of other host magnets don't have a demag field.
        
        See Also
        --------
        stray_field_from_magnet
        StrayField
        """
        return self.stray_field_from_magnet(self)
