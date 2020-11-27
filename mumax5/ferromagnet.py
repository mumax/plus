"""Ferromagnet implementation."""

import _mumax5cpp as _cpp

from .fieldquantity import FieldQuantity
from .grid import Grid
from .scalarquantity import ScalarQuantity
from .variable import Variable


class Ferromagnet:
    """Create a ferromagnet instance.

    Parameters
    ----------
    world : mumax5.World
        World in which the ferromagnet lives.
    grid : mumax5.Grid
        The number of cells in x, y, z the ferromagnet should be divided into.
    name : str, default=""
        The ferromagnet's identifier.
    """

    def __init__(self, world, grid, name=""):
        self._impl = world._impl.add_ferromagnet(grid._impl, name)

    @property
    def name(self):
        """Name of the ferromagnet."""
        return self._impl.name

    @property
    def grid(self):
        """Return the underlying grid of the ferromagnet."""
        return Grid._from_impl(self._impl.grid)

    @property
    def cellsize(self):
        """Dimensions of the cell."""
        return self._impl.cellsize

    @property
    def magnetization(self):
        """Direction of the magnetization (normalized)."""
        return Variable(self._impl.magnetization)

    @magnetization.setter
    def magnetization(self, value):
        self.magnetization.set(value)

    @property
    def enable_demag(self):
        """Enable/disable demagnetization switch."""
        return self._impl.enable_demag

    @enable_demag.setter
    def enable_demag(self, value):
        self._impl.enable_demag = value

    def minimize(self):
        """Minimize the total energy."""
        self._impl.minimize()

    # ----- MATERIAL PARAMETERS -----------

    @property
    def msat(self):
        """Saturation magnetization."""
        return self._impl.msat

    @msat.setter
    def msat(self, value):
        self._impl.msat.set(value)

    @property
    def alpha(self):
        """LLG damping parameter."""
        return self._impl.alpha

    @alpha.setter
    def alpha(self, value):
        self._impl.alpha.set(value)

    @property
    def aex(self):
        """Exchange constant."""
        return self._impl.aex

    @aex.setter
    def aex(self, value):
        self._impl.aex.set(value)

    @property
    def ku1(self):
        """Uniaxial anisotropy parameter Ku1."""
        return self._impl.ku1

    @ku1.setter
    def ku1(self, value):
        self._impl.ku1.set(value)

    @property
    def ku2(self):
        """Uniaxial anisotropy parameter Ku2."""
        return self._impl.ku2

    @ku2.setter
    def ku2(self, value):
        self._impl.ku2.set(value)

    @property
    def anisU(self):
        """Uniaxial anisotropy direction (the easy axis)."""
        return self._impl.anisU

    @anisU.setter
    def anisU(self, value):
        self._impl.anisU.set(value)

    @property
    def idmi(self):
        """Interfacial DMI strength."""
        return self._impl.idmi

    @idmi.setter
    def idmi(self, value):
        self._impl.idmi.set(value)

    @property
    def xi(self):
        """Non-adiabaticity of the spin-transfer torque."""
        return self._impl.xi

    @xi.setter
    def xi(self, value):
        self._impl.xi.set(value)

    @property
    def pol(self):
        """Electrical current polarization."""
        return self._impl.pol

    @pol.setter
    def pol(self, value):
        self._impl.pol.set(value)

    @property
    def jcur(self):
        """Electrical current density."""
        return self._impl.jcur

    @jcur.setter
    def jcur(self, value):
        self._impl.jcur.set(value)

    @property
    def temperature(self):
        """Temperature."""
        return self._impl.temperature

    @temperature.setter
    def temperature(self, value):
        self._impl.temperature.set(value)

    # ----- QUANTITIES ----------------------

    @property
    def torque(self):
        """Total torque on the magnetization."""
        return FieldQuantity(_cpp.torque(self._impl))

    @property
    def llg_torque(self):
        """Torque on the magnetization exerted by the total effective field."""
        return FieldQuantity(_cpp.llg_torque(self._impl))

    @property
    def spin_transfer_torque(self):
        """Spin transfer torque exerted on the magnetization."""
        return FieldQuantity(_cpp.spin_transfer_torque(self._impl))

    @property
    def demag_field(self):
        """Demagnetization field."""
        return FieldQuantity(_cpp.demag_field(self._impl))

    @property
    def demag_energy_density(self):
        """Energy density related to the demag field."""
        return FieldQuantity(_cpp.demag_energy_density(self._impl))

    @property
    def demag_energy(self):
        """Energy related to the demag field."""
        return ScalarQuantity(_cpp.demag_energy(self._impl))

    @property
    def anisotropy_field(self):
        """Anisotropic effective field term."""
        return FieldQuantity(_cpp.anisotropy_field(self._impl))

    @property
    def anisotropy_energy_density(self):
        """Energy density related to the magnetic anisotropy."""
        return FieldQuantity(_cpp.anisotropy_energy_density(self._impl))

    @property
    def anisotropy_energy(self):
        """Energy related to the magnetic anisotropy."""
        return ScalarQuantity(_cpp.anisotropy_energy(self._impl))

    @property
    def exchange_field(self):
        """Effective field of the exchange interaction."""
        return FieldQuantity(_cpp.exchange_field(self._impl))

    @property
    def exchange_energy_density(self):
        """Energy density related to the exchange interaction."""
        return FieldQuantity(_cpp.exchange_energy_density(self._impl))

    @property
    def exchange_energy(self):
        """Energy related to the exchange interaction."""
        return ScalarQuantity(_cpp.exchange_energy(self._impl))

    @property
    def max_angle(self):
        """Maximal angle difference of the magnetization between exchange\
         coupled cells."""
        return ScalarQuantity(_cpp.max_angle(self._impl))

    @property
    def interfacialdmi_field(self):
        """Effective field of the Dzyaloshinskii-Moriya interaction."""
        return FieldQuantity(_cpp.interfacialdmi_field(self._impl))

    @property
    def interfacialdmi_energy_density(self):
        """Energy density related to the interfacial Dzyaloshinskii-Moriya\
         interaction."""
        return FieldQuantity(_cpp.interfacialdmi_energy_density(self._impl))

    @property
    def interfacialdmi_energy(self):
        """Energy related to the interfacial Dzyaloshinskii-Moriya interaction."""
        return ScalarQuantity(_cpp.interfacialdmi_energy(self._impl))

    @property
    def external_field(self):
        """Sum of external field."""
        return FieldQuantity(_cpp.external_field(self._impl))

    @property
    def zeeman_energy_density(self):
        """Energy density related to external fields."""
        return FieldQuantity(_cpp.zeeman_energy_density(self._impl))

    @property
    def zeeman_energy(self):
        """Energy related to external fields."""
        return ScalarQuantity(_cpp.zeeman_energy(self._impl))

    @property
    def effective_field(self):
        """Sum of all effective field terms."""
        return FieldQuantity(_cpp.effective_field(self._impl))

    @property
    def total_energy_density(self):
        """Energy density related to the total effective field."""
        return FieldQuantity(_cpp.total_energy_density(self._impl))

    @property
    def total_energy(self):
        """Energy related to the total effective field."""
        return ScalarQuantity(_cpp.total_energy(self._impl))

    @property
    def thermal_noise(self):
        """Thermal noise on the magnetization."""
        return FieldQuantity(_cpp.thermal_noise(self._impl))
