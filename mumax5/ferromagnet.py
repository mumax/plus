"""Ferromagnet implementation."""

import numpy as _np

import _mumax5cpp as _cpp

from .dmitensor import DmiTensor
from .fieldquantity import FieldQuantity
from .grid import Grid
from .parameter import Parameter
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
        if geometry is None:
            self._impl = world._impl.add_ferromagnet(grid._impl, name)
            return

        if callable(geometry):
            # construct meshgrid of x, y, and z coordinates for the grid
            nx, ny, nz = grid.size
            cs = world.cellsize
            idxs = _np.flip(_np.mgrid[0:nz, 0:ny, 0:nx], axis=0)  # meshgrid of indices
            x, y, z = [(grid.origin[i] + idxs[i]) * cs[i] for i in [0, 1, 2]]

            # evaluate the geometry function for each position in this meshgrid
            geometry_array = _np.vectorize(geometry, otypes=[bool])(x, y, z)

        else:
            # When here, the geometry is not None, not callable, so it should be an
            # ndarray or at least should be convertable to ndarray
            geometry_array = _np.array(geometry, dtype=bool)
            if geometry_array.shape != grid.shape:
                raise ValueError(
                    "The dimensions of the geometry do not match the dimensions "
                    + "of the grid."
                )

        self._impl = world._impl.add_ferromagnet(grid._impl, geometry_array, name)

    def __repr__(self):
        """Return Ferromagnet string representation."""
        return f"Ferromagnet(grid={self.grid}, name='{self.name}')"

    @property
    def name(self):
        """Name of the ferromagnet."""
        return self._impl.name

    @property
    def grid(self):
        """Return the underlying grid of the ferromagnet."""
        return Grid._from_impl(self._impl.system.grid)

    @property
    def cellsize(self):
        """Dimensions of the cell."""
        return self._impl.system.cellsize

    @property
    def geometry(self):
        """Geometry of the ferromagnet."""
        return self._impl.system.geometry

    @property
    def center(self):
        """Center of the ferromagnet.

        Returns
        -------
        center: tuple[float] of size 3
            xyz coordinate of the center of the ferromagnet.
        """
        return self._impl.system.center

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

    @property
    def bias_magnetic_field(self):
        """Uniform bias magnetic field which will affect a ferromagnet.

        The value should be specifed in Teslas.
        """
        return Parameter(self._impl.bias_magnetic_field)

    @bias_magnetic_field.setter
    def bias_magnetic_field(self, value):
        self.bias_magnetic_field.set(value)

    def minimize(self):
        """Minimize the total energy."""
        self._impl.minimize()

    # ----- MATERIAL PARAMETERS -----------

    @property
    def msat(self):
        """Saturation magnetization."""
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
        """Exchange constant."""
        return Parameter(self._impl.aex)

    @aex.setter
    def aex(self, value):
        self.aex.set(value)

    @property
    def ku1(self):
        """Uniaxial anisotropy parameter Ku1."""
        return Parameter(self._impl.ku1)

    @ku1.setter
    def ku1(self, value):
        self.ku1.set(value)

    @property
    def ku2(self):
        """Uniaxial anisotropy parameter Ku2."""
        return Parameter(self._impl.ku2)

    @ku2.setter
    def ku2(self, value):
        self.ku2.set(value)

    @property
    def anisU(self):
        """Uniaxial anisotropy direction (the easy axis)."""
        return Parameter(self._impl.anisU)

    @anisU.setter
    def anisU(self, value):
        self.anisU.set(value)

    @property
    def xi(self):
        """Non-adiabaticity of the spin-transfer torque."""
        return Parameter(self._impl.xi)

    @xi.setter
    def xi(self, value):
        self.xi.set(value)

    @property
    def pol(self):
        """Electrical current polarization."""
        return Parameter(self._impl.pol)

    @pol.setter
    def pol(self, value):
        self.pol.set(value)

    @property
    def jcur(self):
        """Electrical current density."""
        return Parameter(self._impl.jcur)

    @jcur.setter
    def jcur(self, value):
        self.jcur.set(value)

    @property
    def temperature(self):
        """Temperature."""
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
    def dmi_field(self):
        """Effective field of the Dzyaloshinskii-Moriya interaction.

        Returns a FieldQuantity which evaluates the effective field corresponding to the
        DMI energy density.

        Returns
        -------
        dmi_field : mumax5.FieldQuantity
        """
        return FieldQuantity(_cpp.dmi_field(self._impl))

    @property
    def dmi_energy_density(self):
        r"""Energy density related to the Dzyaloshinskii-Moriya interaction.

        Returns a FieldQuantity which evaluates the Dzyaloshinskii-Moriya energy
        density:

        .. math:: \varepsilon_{\text{DMI}} = \frac{1}{2} D_{ijk}
              \left[ m_j \frac{d}{dx_i} m_k - m_k \frac{d}{dx_i} m_j \right]

        Returns
        -------
        dmi_energy_density : mumax5.FieldQuantity
        """
        return FieldQuantity(_cpp.dmi_energy_density(self._impl))

    @property
    def dmi_energy(self):
        """Energy related to the Dzyaloshinskii-Moriya interaction.

        Returns
        -------
        dmi_energy_density : float
        """
        return ScalarQuantity(_cpp.dmi_energy(self._impl))

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
