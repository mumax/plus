"""DmiTensor implementation."""

from .parameter import Parameter

class DmiTensorGroup:
    """Proxy class for managing multiple DmiTensor instances."""

    def __init__(self, tensors):
        self.tensors = tensors

    def __getattr__(self, name):

        if any(hasattr(tensor, name) for tensor in self.tensors):
            def method(*args, **kwargs):
                for tensor in self.tensors:
                    if hasattr(tensor, name):
                        getattr(tensor, name)(*args, **kwargs)
            return method

        raise AttributeError(f"'DmiTensor' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == "tensors":
            super().__setattr__(name, value)
        elif name in DmiTensor.__dict__:
            for tensor in self.tensors:
                setattr(tensor, name, value)
        else:
            raise AttributeError(f"'DmiTensor' object has no attribute '{name}'")

class DmiTensor:
    r"""Contains the DMI parameters of a ferromagnet.

    In mumax⁺, the Dzyaloshinskii-Moriya interaction is defined by the energy density

    .. math:: \varepsilon_{\text{DMI}} = \frac{1}{2} D_{ijk}
              \left[ m_j \partial_i m_k - m_k \partial_i m_j \right]

    with summation indices running over :math:`x`, :math:`y`, and :math:`z`. The DMI strengths
    and chiral properties are contained in the dmi tensor :math:`D_{ijk}`. This tensor is
    antisymmetric in it's magnetic indices (:math:`D_{ijk} = - D_{ikj}`) and hence can be
    fully described by only 9 elements.

    ``DmiTensor`` has 9 Parameter properties: ``xxy``, ``xxz``, ``xyz``, ``yxy``,
    ``yxz``, ``yyz``, ``zxy``, ``zxz``, and ``zyz``. These 9 parameters fully define the
    dmi tensor of a ferromagnet. The parameters can be set separately, or can be set
    through one of the following methods:

    - ``DmiTensor.set_interfacial_dmi``
    - ``DmiTensor.set_bulk_dmi``
    
    Neumann boundary conditions are determined by
    
    .. math:: 2 A \, n_i\partial_i\mathbf{m} = \mathbf{\Gamma} ,
    
    where :math:`A` is the exchange constant and

    .. math:: \Gamma_k = m_j n_i D_{ijk}

    with :math:`n_i` being the component of the surface normal in the :math:`i` th direction.

    Examples
    --------
    >>> world = World(cellsize=(1e-9,1e-9,1e-9))
    >>> magnet = Ferromagnet(world, Grid((64,64,1)))
    >>> magnet.dmi_tensor.xxz = 3e-3
    >>> magnet.dmi_tensor.yyz = 3e-3

    >>> world = World(cellsize=(1e-9,1e-9,1e-9))
    >>> magnet = Ferromagnet(world, Grid((64,64,1)))
    >>> magnet.dmi_tensor.set_interfacial_dmi(3e-3)

    See Also
    --------
    Ferromagnet.dmi_tensor
    Ferromagnet.dmi_field
    Ferromagnet.dmi_energy
    Ferromagnet.dmi_energy_density
    """

    def __init__(self, impl):
        """Create a DmiTensor from a cpp DmiTensor instance.

        .. warning: The end user should not create DmiTensor instances. Each Ferromagnet
                    already has a DmiTensor as an attribute which can be used to set
                    the DMI parameters. See ``Ferromagnet.dmi_tensor``.

        Parameters
        ----------
        impl: _mumaxpluscpp.dmi_tensor
        """
        self._impl = impl

    def make_zero(self):
        """Set all DMI parameters to zero."""
        self.xxy = 0.0
        self.xxz = 0.0
        self.xyz = 0.0
        self.yxy = 0.0
        self.yxz = 0.0
        self.yyz = 0.0
        self.zxy = 0.0
        self.zxz = 0.0
        self.zyz = 0.0

    def set_interfacial_dmi(self, value):
        """Set interfacial DMI parameters for an interface in the xy-plane.

        Using this function is equivalent to:

        >>> dmi_tensor.make_zero()
        >>> dmi_tensor.xxz = value
        >>> dmi_tensor.yyz = value

        Parameters
        ----------
        value
            The interfacial DMI strength. This value is used to set individual DMI
            parameters of the DMI tensor. The value can be anything which can be used in
            ``Parameter.set``.
        """
        self.make_zero()
        self.xxz.set(value)
        self.yyz.set(value)

    def set_bulk_dmi(self, value):
        """Set bulk DMI parameters.

        Using this method is equivalent to:

        >>> dmi_tensor.make_zero()
        >>> dmi_tensor.xyz = value
        >>> dmi_tensor.yxz = -value
        >>> dmi_tensor.zxy = value

        Parameters
        ----------
        value
            The bulk DMI strength. This value is used to set individual DMI parameters
            of the DMI tensor. The value can be anything which can be used in
            ``Parameter.set``.
        """
        self.make_zero()
        self.xyz.set(value)
        self.yxz.set(-value)
        self.zxy.set(value)

    @property
    def xxy(self):
        """Dzyaloshinskii-Moriya interaction strength parameter `D_xxy` (unit: J/m2).

        Returns
        -------
        Parameter
            DMI strength parameter `D_xxy`
        """
        return Parameter(self._impl.xxy)

    @xxy.setter
    def xxy(self, value):
        self.xxy.set(value)

    @property
    def xyz(self):
        """Dzyaloshinskii-Moriya interaction strength parameter `D_xyz` (unit: J/m2).

        Returns
        -------
        Parameter
            DMI strength parameter `D_xyz`
        """
        return Parameter(self._impl.xyz)

    @xyz.setter
    def xyz(self, value):
        self.xyz.set(value)

    @property
    def xxz(self):
        """Dzyaloshinskii-Moriya interaction strength parameter `D_xxz` (unit: J/m2).

        Returns
        -------
        Parameter
        """
        return Parameter(self._impl.xxz)

    @xxz.setter
    def xxz(self, value):
        self.xxz.set(value)

    @property
    def yxy(self):
        """Dzyaloshinskii-Moriya interaction strength parameter `D_yxy` (unit: J/m2).

        Returns
        -------
        Parameter
        """
        return Parameter(self._impl.yxy)

    @yxy.setter
    def yxy(self, value):
        self.yxy.set(value)

    @property
    def yyz(self):
        """Dzyaloshinskii-Moriya interaction strength parameter `D_yyz` (unit: J/m2).

        Returns
        -------
        Parameter
        """
        return Parameter(self._impl.yyz)

    @yyz.setter
    def yyz(self, value):
        self.yyz.set(value)

    @property
    def yxz(self):
        """Dzyaloshinskii-Moriya interaction strength parameter `D_yxz` (unit: J/m2).

        Returns
        -------
        Parameter
        """
        return Parameter(self._impl.yxz)

    @yxz.setter
    def yxz(self, value):
        self.yxz.set(value)

    @property
    def zxy(self):
        """Dzyaloshinskii-Moriya interaction strength parameter `D_zxy` (unit: J/m2).

        Returns
        -------
        Parameter
        """
        return Parameter(self._impl.zxy)

    @zxy.setter
    def zxy(self, value):
        self.zxy.set(value)

    @property
    def zyz(self):
        """Dzyaloshinskii-Moriya interaction strength parameter `D_zyz` (unit: J/m2).

        Returns
        -------
        Parameter
        """
        return Parameter(self._impl.zyz)

    @zyz.setter
    def zyz(self, value):
        self.zyz.set(value)

    @property
    def zxz(self):
        """Dzyaloshinskii-Moriya interaction strength parameter `D_zxz` (unit: J/m2).

        Returns
        -------
        Parameter
        """
        return Parameter(self._impl.zxz)

    @zxz.setter
    def zxz(self, value):
        self.zxz.set(value)
