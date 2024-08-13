"""Ferromagnet implementation."""

import numpy as _np

import _mumax5cpp as _cpp

from .magnet import Magnet
from .dmitensor import DmiTensor
from .fieldquantity import FieldQuantity
from .parameter import Parameter
from .poissonsystem import PoissonSystem
from .scalarquantity import ScalarQuantity
from .variable import Variable

import warnings
# from .world import World  # imported below to avoid circular imports


class Ferromagnet(Magnet):
    """Create a ferromagnet instance.

    Parameters
    ----------
    world : mumax5.World
        World in which the ferromagnet lives.
    grid : mumax5.Grid
        The number of cells in x, y, z the ferromagnet should be divided into.
    geometry : None, ndarray, or callable (default=None)
        The geometry of the ferromagnet can be set in three ways.

        1. If the geometry contains all cells in the grid, then use None (the default)
        2. Use an ndarray which specifies for each cell wheter or not it is in the
           geometry.
        3. Use a function which takes x, y, and z coordinates as arguments and returns
           true if this position is inside the geometry and false otherwise.

    name : str (default="")
        The ferromagnet's identifier. If the name is empty (the default), a name for the
        ferromagnet will be created.
    """

    def __init__(self, world, grid, name="", geometry=None):
        super().__init__(world._impl.add_ferromagnet, world, grid, name, geometry)

    def __repr__(self):
        """Return Ferromagnet string representation."""
        return f"Ferromagnet(grid={self.grid}, name='{self.name}')"

    @property
    def magnetization(self):
        """Direction of the magnetization (normalized)."""
        return Variable(self._impl.magnetization)

    @magnetization.setter
    def magnetization(self, value):
        self.magnetization.set(value)

    @property
    def enable_demag(self):
        """Enable/disable demagnetization switch.
        
        Default = True.
        """
        return self._impl.enable_demag

    @enable_demag.setter
    def enable_demag(self, value):
        self._impl.enable_demag = value

    @property
    def enable_openbc(self):
        """Enable/disable open boundary conditions.
        
        When set to False (default), Neumann boundary conditions are applied.
        These affect the calculation of DMI and exchange field terms.
        """
        return self._impl.enable_openbc
    
    @enable_openbc.setter
    def enable_openbc(self, value):
        self._impl.enable_openbc = value

    @property
    def enable_zhang_li_torque(self):
        """Enable/disable Zhang-Li spin transfer torque.
        
        Default = True.

        See Also
        --------
        enable_slonczewski_torque
        """
        return self._impl.enable_zhang_li_torque

    @enable_zhang_li_torque.setter
    def enable_zhang_li_torque(self, value):
        self._impl.enable_zhang_li_torque = value

    @property
    def enable_slonczewski_torque(self):
        """Enable/disable Slonczewski spin transfer torue.
        
        Default = True.

        See Also
        --------
        enable_zhang_li_torque
        """
        return self._impl.enable_slonczewski_torque

    @enable_slonczewski_torque.setter
    def enable_slonczewski_torque(self, value):
        self._impl.enable_slonczewski_torque = value

    @property
    def bias_magnetic_field(self):
        """Uniform bias magnetic field which will affect a ferromagnet.
        
        The value should be specifed in Teslas.
        """
        return Parameter(self._impl.bias_magnetic_field)

    @bias_magnetic_field.setter
    def bias_magnetic_field(self, value):
        self.bias_magnetic_field.set(value)

    def minimize(self, tol=1e-6, nsamples=10):
        """Minimize the total energy.

        Fast energy minimization, but less robust than `relax`
        when starting from a high energy state.

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
        the total energy hits the noise floor.
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
        the system relaxes until the torque is steady or increasing.
        
        If set to a positive value,
        the system relaxes until the torque is smaller than or equal
        to this threshold.

        See Also
        --------
        relax
        """
        return self._impl.RelaxTorqueThreshold
        
    @RelaxTorqueThreshold.setter
    def RelaxTorqueThreshold(self, value):
        assert value != 0, "The relax threshold should not be zero."
        self._impl.RelaxTorqueThreshold = value

    # ----- MATERIAL PARAMETERS -----------

    @property
    def msat(self):
        """Saturation magnetization (A/m)."""
        return Parameter(self._impl.msat)

    @msat.setter
    def msat(self, value):
        self.msat.set(value)

    @property
    def alpha(self):
        """LLG damping parameter."""
        return Parameter(self._impl.alpha)

    @alpha.setter
    def alpha(self, value):
        self.alpha.set(value)

    @property
    def aex(self):
        """Exchange constant (J/m)."""
        return Parameter(self._impl.aex)

    @aex.setter
    def aex(self, value):
        self.aex.set(value)

    @property
    def ku1(self):
        """Uniaxial anisotropy parameter Ku1 (J/m³).
        
        See Also
        --------
        ku2, anisU
        """
        return Parameter(self._impl.ku1)

    @ku1.setter
    def ku1(self, value):
        self.ku1.set(value)

    @property
    def ku2(self):
        """Uniaxial anisotropy parameter Ku2 (J/m³).
        
        See Also
        --------
        ku1, anisU
        """
        return Parameter(self._impl.ku2)

    @ku2.setter
    def ku2(self, value):
        self.ku2.set(value)

    @property
    def anisU(self):
        """Uniaxial anisotropy direction (the easy axis).
        
        See Also
        --------
        ku1, ku2
        """
        return Parameter(self._impl.anisU)

    @anisU.setter
    def anisU(self, value):
        self.anisU.set(value)

    @property
    def kc1(self):
        """Cubic anisotropy parameter Kc1 (J/m³).
        
        See Also
        --------
        kc2, kc3, anisC1, anisC2
        """
        return Parameter(self._impl.kc1)
    
    @kc1.setter
    def kc1(self, value):
        self.kc1.set(value)

    @property
    def kc2(self):
        """Cubic anisotropy parameter Kc2 (J/m³).

        See Also
        --------
        kc1, kc3, anisC1, anisC2
        """
        return Parameter(self._impl.kc2)
    
    @kc2.setter
    def kc2(self, value):
        self.kc2.set(value)
        
    @property
    def kc3(self):
        """Cubic anisotropy parameter Kc3 (J/m³).
        
        See Also
        --------
        kc1, kc2, anisC1, anisC2
        """
        return Parameter(self._impl.kc3)
    
    @kc3.setter
    def kc3(self, value):
        self.kc3.set(value)

    @property
    def anisC1(self):
        """First cubic anisotropy direction.
        
        See Also
        --------
        kc1, kc2, kc3, anisC2
        """
        return Parameter(self._impl.anisC1)
    
    @anisC1.setter
    def anisC1(self, value):
        self.anisC1.set(value)

    @ property
    def anisC2(self):
        """Second cubic anisotropy direction.
        
        See Also
        --------
        kc1, kc2, kc3, anisC1
        """
        return Parameter(self._impl.anisC2)
    
    @anisC2.setter
    def anisC2(self, value):
        self.anisC2.set(value)

    @property
    def Lambda(self):
        """Slonczewski Λ parameter.
        
        See Also
        --------
        eps_prime, jcur, pol, FixedLayer, FreeLayerThickness
        """
        return Parameter(self._impl.Lambda)
    
    @Lambda.setter
    def Lambda(self, value):
        self.Lambda.set(value)
    
    @property
    def FreeLayerThickness(self):
        """Slonczewski free layer thickness (m). If set to zero (default),
        then the thickness will be deduced from the mesh size.
        
        See Also
        --------
        eps_prime, jcur, Lambda, pol, FixedLayer
        """
        return Parameter(self._impl.FreeLayerThickness)
    
    @FreeLayerThickness.setter
    def FreeLayerThickness(self, value):
        self.FreeLayerThickness.set(value)

    @property
    def eps_prime(self):
        """Slonczewski secondary STT term ε'.
        
        See Also
        --------
        jcur, Lambda, pol, FixedLayer, FreeLayerThickness
        """
        return Parameter(self._impl.eps_prime)
    
    @eps_prime.setter
    def eps_prime(self, value):
        self.eps_prime.set(value)

    @property
    def FixedLayer(self):
        """Slonczewski fixed layer polarization.
        
        See Also
        --------
        eps_prime, jcur, Lambda, pol, FreeLayerThickness
        """
        return Parameter(self._impl.FixedLayer)
    
    @FixedLayer.setter
    def FixedLayer(self, value):
        self.FixedLayer.set(value)


    @property
    def xi(self):
        """Non-adiabaticity of the Zhang-Li spin-transfer torque.
        
        See Also
        --------
        jcur, pol
        """
        return Parameter(self._impl.xi)

    @xi.setter
    def xi(self, value):
        self.xi.set(value)

    @property
    def pol(self):
        """Electrical current polarization.
        
        See Also
        --------
        eps_prime, jcur, Lambda, FixedLayer, FreeLayerThickness, xi
        """
        return Parameter(self._impl.pol)

    @pol.setter
    def pol(self, value):
        self.pol.set(value)

    @property
    def jcur(self):
        """Electrical current density (A/m²).

        See Also
        --------
        eps_prime, Lambda, pol, FixedLayer, FreeLayerThickness, xi
        """
        return Parameter(self._impl.jcur)

    @jcur.setter
    def jcur(self, value):
        self.jcur.set(value)

    @property
    def temperature(self):
        """Temperature (K).
        
        See Also
        --------
        thermal_noise
        """
        return Parameter(self._impl.temperature)

    @temperature.setter
    def temperature(self, value):
        self.temperature.set(value)

    @property
    def dmi_tensor(self):
        """Get the DMI tensor of this Ferromagnet.

        See Also
        --------
        DmiTensor

        Returns
        -------
        DmiTensor
            The DMI tensor of this ferromagnet.
        """
        return DmiTensor(self._impl.dmi_tensor)

    @property
    def applied_potential(self):
        """The applied electrical potential (V).

        Cells with Nan values do not have an applied potential.

        See Also
        --------
        conductivity, conductivity_tensor
        electrical_potential
        """
        return Parameter(self._impl.applied_potential)

    @applied_potential.setter
    def applied_potential(self, value):
        self.applied_potential.set(value)

    @property
    def conductivity(self):
        """Conductivity without considering anisotropic magneto resistance (S/m).
        
        See Also
        --------
        conductivity_tensor
        applied_potential, electrical_potential
        """
        return Parameter(self._impl.conductivity)

    @conductivity.setter
    def conductivity(self, value):
        self.conductivity.set(value)

    @property
    def amr_ratio(self):
        """Anisotropic magneto resistance ratio."""
        return Parameter(self._impl.amr_ratio)

    @amr_ratio.setter
    def amr_ratio(self, value):
        self.amr_ratio.set(value)

    # ----- POISSON SYSTEM ----------------------

    @property
    def poisson_system(self):
        """Get the poisson solver which computes the electric potential.
        
        See Also
        --------
        electric_potential
        applied_potential, conductivity, conductivity_tensor
        """
        return PoissonSystem(self._impl.poisson_system)

    # ----- QUANTITIES ----------------------

    @property
    def torque(self):
        """Total torque on the magnetization (rad/s)."""
        return FieldQuantity(_cpp.torque(self._impl))

    @property
    def llg_torque(self):
        """Torque on the magnetization exerted by the total effective field (rad/s)."""
        return FieldQuantity(_cpp.llg_torque(self._impl))

    @property
    def damping_torque(self):
        """Torque used by the relax function (rad/s). This is the term in the
        Landau-Liftshitz-Gilbert torque with the damping factor α.
        
        See Also
        --------
        relax, llg_torque, alpha
        """
        return FieldQuantity(_cpp.damping_torque(self._impl))

    @property
    def spin_transfer_torque(self):
        """Spin transfer torque exerted on the magnetization (rad/s)."""
        return FieldQuantity(_cpp.spin_transfer_torque(self._impl))
    
    @property
    def max_torque(self):
        """The maximum value of the torque over all cells (rad/s)."""
        return ScalarQuantity(_cpp.max_torque(self._impl))
    
    @property
    def demag_field(self):
        """Demagnetization field (T).
        
        See Also
        --------
        demag_energy_density, demage_energy
        """
        return FieldQuantity(_cpp.demag_field(self._impl))

    @property
    def demag_energy_density(self):
        """Energy density related to the demag field (J/m³).
        
        See Also
        --------
        demag_energy, demag_field
        """
        return FieldQuantity(_cpp.demag_energy_density(self._impl))

    @property
    def demag_energy(self):
        """Energy related to the demag field (J).
        
        See Also
        --------
        demag_energy_density, demag_field
        """
        return ScalarQuantity(_cpp.demag_energy(self._impl))

    @property
    def anisotropy_field(self):
        """Anisotropic effective field term (T).
        
        See Also
        --------
        anisotropy_energy_density, anisotropy_energy
        """
        return FieldQuantity(_cpp.anisotropy_field(self._impl))

    @property
    def anisotropy_energy_density(self):
        """Energy density related to the magnetic anisotropy (J/m³).
        
        See Also
        --------
        anisotropy_energy, anisotropy_field
        """
        return FieldQuantity(_cpp.anisotropy_energy_density(self._impl))

    @property
    def anisotropy_energy(self):
        """Energy related to the magnetic anisotropy (J).
        
        See Also
        --------
        anisotropy_energy_density, anisotropy_field
        """
        return ScalarQuantity(_cpp.anisotropy_energy(self._impl))

    @property
    def exchange_field(self):
        """Effective field of the exchange interaction (T).
        
        See Also
        --------
        exchange_energy_density, exchange_energy
        """
        return FieldQuantity(_cpp.exchange_field(self._impl))

    @property
    def exchange_energy_density(self):
        """Energy density related to the exchange interaction (J/m³).
        
        See Also
        --------
        exchange_energy, exchange_field
        """
        return FieldQuantity(_cpp.exchange_energy_density(self._impl))

    @property
    def exchange_energy(self):
        """Energy related to the exchange interaction (J).

        See Also
        --------
        exchange_energy_density, exchange_field
        """
        return ScalarQuantity(_cpp.exchange_energy(self._impl))

    @property
    def max_angle(self):
        """Maximal angle difference of the magnetization between exchange\
         coupled cells (rad)."""
        return ScalarQuantity(_cpp.max_angle(self._impl))

    @property
    def dmi_field(self):
        """Effective field of the Dzyaloshinskii-Moriya interaction (T).

        Returns a FieldQuantity which evaluates the effective field corresponding to the
        DMI energy density.

        Returns
        -------
        dmi_field : mumax5.FieldQuantity

        See Also
        --------
        dmi_energy_density, dmi_energy
        dmi_tensor
        """
        return FieldQuantity(_cpp.dmi_field(self._impl))

    @property
    def dmi_energy_density(self):
        r"""Energy density related to the Dzyaloshinskii-Moriya interaction (J/m³).

        Returns a FieldQuantity which evaluates the Dzyaloshinskii-Moriya energy
        density:

        .. math:: \varepsilon_{\text{DMI}} = \frac{1}{2} D_{ijk}
              \left[ m_j \frac{d}{dx_i} m_k - m_k \frac{d}{dx_i} m_j \right]

        Returns
        -------
        dmi_energy_density : mumax5.FieldQuantity

        See Also
        --------
        dmi_energy, dmi_field
        dmi_tensor
        """
        return FieldQuantity(_cpp.dmi_energy_density(self._impl))

    @property
    def dmi_energy(self):
        """Energy related to the Dzyaloshinskii-Moriya interaction (J).

        Returns
        -------
        dmi_energy_density : float

        See Also
        --------
        dmi_energy_density, dmi_field
        dmi_tensor
        """
        return ScalarQuantity(_cpp.dmi_energy(self._impl))
    
    @property
    def external_field(self):
        """Sum of external fields (T).
        
        See Also
        --------
        bias_magnetic_field
        """
        return FieldQuantity(_cpp.external_field(self._impl))

    @property
    def zeeman_energy_density(self):
        """Energy density related to external fields (J/m³).
        
        See Also
        --------
        zeeman_energy, external_field
        """
        return FieldQuantity(_cpp.zeeman_energy_density(self._impl))

    @property
    def zeeman_energy(self):
        """Energy related to external fields (J).
        
        See Also
        --------
        zeeman_energy_density, external_field
        """
        return ScalarQuantity(_cpp.zeeman_energy(self._impl))

    @property
    def effective_field(self):
        """Sum of all effective field terms (T)."""
        return FieldQuantity(_cpp.effective_field(self._impl))

    @property
    def total_energy_density(self):
        """Energy density related to the total effective field (J/m³).
        
        See Also
        --------
        total_energy
        """
        return FieldQuantity(_cpp.total_energy_density(self._impl))

    @property
    def total_energy(self):
        """Energy related to the total effective field (J).

        See Also
        --------
        total_energy_density
        """
        return ScalarQuantity(_cpp.total_energy(self._impl))

    @property
    def electrical_potential(self):
        """Electrical potential (V).

        Calculates the electrical potential with a Poisson solver, using the
        `applied_potential` and `conductivity(_tensor)`.
        
        See Also
        --------
        applied_potential, conductivity, conductivity_tensor
        poisson_system
        """
        return FieldQuantity(_cpp.electrical_potential(self._impl))

    @property
    def conductivity_tensor(self):
        """Conductivity tensor taking into account AMR (S/m).

        This quantity has six components (Cxx, Cyy, Czz, Cxy, Cxz, Cyz)
        which forms the symmetric conductivity tensor::

               Cxx Cxy Cxz
               Cxy Cyy Cyz
               Cxz Cyz Czz

        See Also
        --------
        conductivity
        applied_potential, electrical_potential
        """
        return FieldQuantity(_cpp.conductivity_tensor(self._impl))

    @property
    def thermal_noise(self):
        """Thermal noise on the magnetization.
        
        See Also
        --------
        temperature
        """
        return FieldQuantity(_cpp.thermal_noise(self._impl))

    @property
    def full_magnetization(self):
        """Unnormalized magnetization (A/m).
        
        See Also
        --------
        magnetization, msat
        """
        return FieldQuantity(_cpp.full_magnetization(self._impl))
