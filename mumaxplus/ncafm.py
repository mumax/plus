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


class NCAFM(Magnet):
    """Create a non-collinear antiferromagnet instance.

    Parameters
    ----------
    world : mumaxplus.World
        World in which the non-collinear antiferromagnet lives.
    grid : mumaxplus.Grid
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

    def __init__(self, world, grid, name="", geometry=None, regions=None):
        super().__init__(world._impl.add_ncafm,
                         world, grid, name, geometry, regions)

    def __repr__(self):
        """Return non-collinear antiferromagnet string representation."""
        return f"Antiferromagnet(grid={self.grid}, name='{self.name}')"

    def __setattr__(self, name, value):
        """Set NCAFM or sublattice properties.
        
            If the NCAFM doesn't have the named attribute, then the corresponding
            attributes of all sublattices are set.
            e.g. to set the saturation magnetization of all sublattices to the
            same value, one could use:
                nc_antiferromagnet.msat = 800e3
            which is equal to
                nc_antiferromagnet.sub1.msat = 800e3
                nc_antiferromagnet.sub2.msat = 800e3
        """
        if hasattr(NCAFM, name) or name == "_impl":
            # set attribute of yourself, without causing recursion
            object.__setattr__(self, name, value)
        elif hasattr(Ferromagnet, name):
            setattr(self.sub1, name, value)
            setattr(self.sub2, name, value)
            setattr(self.sub3, name, value)
        else:
            raise AttributeError(
                r'Both NC_Antiferromagnet and Ferromagnet have no attribute "{}".'.format(name))

    @property
    def sub1(self):
        """First sublattice instance."""
        return Ferromagnet._from_impl(self._impl.sub1())
    
    @property
    def sub2(self):
        """Second sublattice instance."""
        return Ferromagnet._from_impl(self._impl.sub2())

    @property
    def sub3(self):
        """Third sublattice instance."""
        return Ferromagnet._from_impl(self._impl.sub3())
    
    @property
    def sublattices(self):
        return (self.sub1, self.sub2, self.sub3)

    @property
    def bias_magnetic_field(self):
        """Uniform bias magnetic field which will affect an nc_antiferromagnet.

        The value should be specifed in Teslas.
        """
        return self.sub1.bias_magnetic_field

    @bias_magnetic_field.setter
    def bias_magnetic_field(self, value):
        self.sub1.bias_magnetic_field.set(value)
        self.sub2.bias_magnetic_field.set(value)
        self.sub3.bias_magnetic_field.set(value)

    @property
    def enable_demag(self):
        """Enable/disable demagnetization switch of all sublattices.

        Default = True.
        """
        return self.sub1.enable_demag

    @enable_demag.setter
    def enable_demag(self, value):
        self.sub1.enable_demag = value
        self.sub2.enable_demag = value
        self.sub3.enable_demag = value

    def minimize(self, tol=1e-6, nsamples=20):
        return 0

    def relax(self, tol=1e-9):
        return 0


    # ----- MATERIAL PARAMETERS -----------

    @property
    def ncafmex_cell(self):
        """Intracell nc_antiferromagnetic exchange constant (J/m).
        This parameter plays the role of exchange constant of
        the nc_antiferromagnetic homogeneous exchange interaction
        in a single simulation cell.
        
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
        elif _np.any(self.ncafmex_cell.eval() > 0):
            warn = True
        
        if warn:
            warnings.warn("The nc_antiferromagnetic exchange constant ncafmex_cell"
                          + " is set to a positive value, instead of negative (or zero)."
                          + " Make sure this is intentional!", UserWarning)

    @property
    def ncafmex_nn(self):
        """Intercell nc_antiferromagnetic exchange constant (J/m).
        This parameter plays the role of exchange constant of
        the nc_antiferromagnetic inhomogeneous exchange interaction
        between neighbouring simulation cells.
        
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
            warnings.warn("The nc_antiferromagnetic exchange constant ncafmex_nn"
                          + " is set to a positive value, instead of negative (or zero)."
                          + " Make sure this is intentional!", UserWarning)

    @property
    def inter_ncafmex_nn(self):
        """Interregional nc_antiferromagnetic exchange constant (J/m).
        If set to zero (default), then the harmonic mean of
        the exchange constants of the two regions are used.

        When no exchange interaction between different regions
        is wanted, set `scale_ncafmex_nn` to zero.

        This parameter should be set with
        >>> magnet.inter_ncafmex_nn.set_between(region1, region2, value)

        See Also
        --------
        ncafmex_nn, inter_exchange, scale_ncafmex_nn, scale_exchange
        """
        return InterParameter(self._impl.inter_ncafmex_nn)

    @inter_ncafmex_nn.setter
    def inter_ncafmex_nn(self, value):
        if value > 0:
            warnings.warn("The nc_antiferromagnetic exchange constant inter_ncafmex_nn"
                          + " is set to a positive value, instead of negative (or zero)."
                          + " Make sure this is intentional!", UserWarning)
        self.inter_ncafmex_nn.set(value)

    @property
    def scale_ncafmex_nn(self):
        """Scaling of the nc_antiferromagnetic exchange constant between
        different regions. This factor is multiplied by the harmonic
        mean of the exchange constants of the two regions.

        If `inter_ncafmex_nn` is set to a non-zero value, then this
        overrides `scale_ncafmex_nn`, i.e. `scale_ncafmex_nn` is
        automatically set to zero when `inter_ncafmex_nn` is not.

        This parameter should be set with
        >>> magnet.scale_ncafmex_nn.set_between(region1, region2, value)

        See Also
        --------
        ncafmex_nn, inter_ncafmex_nn, inter_exchange, scale_exchange
        """
        return InterParameter(self._impl.scale_ncafmex_nn)

    @scale_ncafmex_nn.setter
    def scale_ncafmex_nn(self, value):
        self.scale_ncafmex_nn.set(value)

    @property
    def latcon(self):
        """Lattice constant (m).

        Physical lattice constant of the NC_Antiferromagnet. This doesn't break the
        micromagnetic character of the simulation package, but is only used to
        calculate the homogeneous exchange field, i.e. the nc_antiferromagnetic
        exchange interaction between spins at the same site.

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
    def dmi_tensor(self):
        """
        Get the DMI tensor of this NC_Antiferromagnet. This tensor
        describes intersublattice DMI exchange.

        Note that individual sublattices can have their own tensor
        to describe intrasublattice DMI exchange.

        Returns
        -------
        DmiTensor
            The DMI tensor of this NC_Antiferromagnet.
        
        See Also
        --------
        DmiTensor, dmi_tensors
        """
        return DmiTensor(self._impl.dmi_tensor)

    @property
    def dmi_tensors(self):
        """ Returns the DMI tensor of self, self.sub1, self.sub2 and self.sub3.

        This group can be used to set the intersublattice and all intrasublattice
        DMI tensors at the same time.

        For example, to set interfacial DMI in the whole system to the same value,
        one could use
        >>> magnet = NC_Antiferromagnet(world, grid)
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

    # ----- QUANTITIES ----------------------

    @property
    def full_magnetization(self):
        """Full non-collinear antiferromagnetic magnetization M1 + M2 + M3 (A/m).
        
        See Also
        --------
        Ferromagnet.full_magnetization
        """
        return FieldQuantity(_cpp.full_magnetization(self._impl))
'''
    @property
    def neel_vector(self):
        """Weighted dimensionless Neel vector of an antiferromagnet/ferrimagnet.
        (msat1*m1 - msat2*m2) / (msat1 + msat2)
        """
        return FieldQuantity(_cpp.neel_vector(self._impl))
    @property
    def angle_field(self):
        """Returns the deviation from the optimal angle (180°) between
        magnetization vectors in the same cell which are coupled by the
        intracell exchange interaction (rad).

        See Also
        --------
        max_intracell_angle
        afmex_cell
        """
        return FieldQuantity(_cpp.angle_field(self._impl))
    
    @property
    def max_intracell_angle(self):
        """The maximal deviation from 180° between AFM-exchange coupled magnetization
        vectors in the same simulation cell (rad).

        See Also
        --------
        angle_field
        afmex_cell
        Ferromagnet.max_angle
        """
        return ScalarQuantity(_cpp.max_intracell_angle(self._impl))

    @property
    def total_energy_density(self):
        """Total energy density of both sublattices combined (J/m³). Kinetic and
        elastic energy densities of the antiferromagnet are also included if
        elastodynamics is enabled.
        
        See Also
        --------
        total_energy
        enable_elastodynamics, elastic_energy_density, kinetic_energy_density
        """
        return FieldQuantity(_cpp.total_energy_density(self._impl))

    @property
    def total_energy(self):
        """Total energy of both sublattices combined (J). Kinetic and elastic
        energies of the antiferromagnet are also included if elastodynamics is
        enabled.
        
        See Also
        --------
        total_energy_density
        enable_elastodynamics, elastic_energy, kinetic_energy
        """
        return ScalarQuantity(_cpp.total_energy(self._impl))
'''