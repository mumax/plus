"""Plotting helper functions."""

import matplotlib.pyplot as _plt
import matplotlib as _matplotlib
import numpy as _np
import math as _math
import pyvista as _pv
import warnings as _warnings

from numbers import Integral, Number
from typing import Optional, Literal
from types import MethodType
from itertools import product
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable as _make_axes_locatable
from matplotlib.transforms import Bbox, BboxTransform

import mumaxplus as _mxp

# ----- rgb -----

def hsl_to_rgb(H, S, L):
    """Convert color from HSL to RGB."""
    Hp = _np.mod(H/(_np.pi/3.0), 6.0)
    C = _np.where(L<=0.5, 2*L*S, 2*(1-L)*S)
    X = C * (1 - _np.abs(_np.mod(Hp, 2.0) - 1.0))
    m = L - C / 2.0

    # R = m + X for 1<=Hp<2 or 4<=Hp<5
    # R = m + C for 0<=Hp<1 or 5<=Hp<6
    R = m + _np.select([((1<=Hp)&(Hp<2)) | ((4<=Hp)&(Hp<5)),
                        (Hp<1) | (5<=Hp)], [X, C], 0.)
    # G = m + X for 0<=Hp<1 or 3<=Hp<4
    # G = m + C for 1<=Hp<3
    G = m + _np.select([(Hp<1) | ((3<=Hp)&(Hp<4)),
                        (1<=Hp)&(Hp<3)], [X, C], 0.)
    # B = m + X for 2<=Hp<3 or 5<=Hp<6
    # B = m + C for 3<=Hp<5
    B = m + _np.select([((2<=Hp)&(Hp<3)) | (5<=Hp),
                        (3<=Hp)&(Hp<5)], [X, C], 0.)

    # clip rgb values to be in [0,1]
    R, G, B = _np.clip(R,0.,1.), _np.clip(G,0.,1.), _np.clip(B,0.,1.)

    return R, G, B


def vector_to_rgb(x, y, z):
    """Map vector (with norm ≤ 1) to RGB.
    
    Note
    ----
    The saturation does not depend on the z-component, so the color sphere is
    continuous.
    """
    H = _np.arctan2(y, x)
    S = _np.sqrt(x ** 2 + y ** 2)  # no z so color sphere is continuous!
    L = 0.5 + 0.5 * z
    return hsl_to_rgb(H, S, L)


def get_rgb(field_quantity: _mxp.FieldQuantity|_np.ndarray,
            OoP_axis_idx: Optional[Literal[0, 1, 2]] = 2,
            layer: Optional[int] = None,
            geometry: Optional[_np.ndarray] = None) -> _np.ndarray:
    """Get rgb values of given field quantity along a given layer.

    Notes
    -----
    There is also a CUDA version of this function:
    :func:`mumaxplus.FieldQuantity.get_rgb()`, but the RGB components are on the
    first axis, like with FieldQuantity evaluations. Here, RGB values are on the
    last axis to work nicely with plotting libraries, such as imshow in matplotlib.

    The final color scheme is different from mumax³. In this case, the
    saturation does not depend on the z-component anymore, meaning the z=0 plane
    remains unchanged, but other colors will appear slightly less saturated.
    This ensures that the color sphere is continuous everywhere, particularly
    when crossing the xz- or yz-plane with a normalized length less than 1,
    where the colors will fade through gray.

    Parameters
    ----------
    field_quantity : mumaxplus.FieldQuantity or numpy.ndarray
        The vector field_quantity of which to get the rgb representation. It can
        have any number of axes, but the first axis must contain the 3 vector
        components. It needs to have the shape (3, nz, ny, nx) in order to index
        along a given layer.
    OoP_axis_idx : int, optional, default=2
        The index of the axis pointing out of plane: 0, 1 and 2 correspond to
        x, y and z respectively. The other two are aranged according to a
        right-handed coordinate system.
        If None, both OoP_axis_idx and layer are ignored and rgb values are
        calculated for all cells.
        This cannot be interpreted if field_quantity does not have 4 dimensions.
    layer : int, optional
        Layer index of the out-of-plane axis.
        Calculates rgb for all layers if None (default).
        This cannot be interpreted if field_quantity does not have 4 dimensions.
    geometry : numpy.ndarray of booleans, optional
        Cells where geometry is False are not used in normalization and are set to gray.
        The shape of the geometry should correspond to the shape of the
        field_quantity, except for the first component axis.
        The given geometry is ignored if field_quantity is a FieldQuantity.

    Returns
    -------
    rgb : ndarray
        It has shape ``(n_vertical, n_horizontal, 3)`` if OoP_axis_idx and layer
        are given, otherwize ``(*field_quantity.shape[1:], 3)``.

    See Also
    --------
    vector_to_rgb
    get_rgba
    """

    is_quantity = isinstance(field_quantity, _mxp.FieldQuantity)

    # check input
    if not is_quantity and not isinstance(field_quantity, _np.ndarray):
        raise TypeError("The first argument should be a FieldQuantity or an ndarray.")

    if (ncomp := field_quantity.shape[0]) != 3:
        raise ValueError(f"field_quantity has {ncomp} components instead of 3")

    if layer is not None and OoP_axis_idx is not None:
        if (ndim := len(field_quantity.shape)) != 4:
            raise IndexError(f"Can not take layer index {layer} of axis " +
                             f"{OoP_axis_idx} of field_quantity with {ndim} " +
                             "dimensions instead of 4.")
        if (layer_max := field_quantity.shape[-1-OoP_axis_idx]) <= layer:
            raise IndexError(f"layer {layer} is out of range of length {layer_max}")

    if not OoP_axis_idx in [0, 1, 2, None]:
        raise IndexError(f"OoP_axis_index should be 0, 1, 2 or None, not {OoP_axis_idx}.")


    # FieldQuantity in 3D or with trivial index: use CUDA
    if (is_quantity and ((layer is None or OoP_axis_idx is None)  # 3D
        or (field_quantity.shape[-1-OoP_axis_idx] == 1))):  # trivial

        # use faster CUDA get_rgb for 3D rgb
        rgb_front = field_quantity.get_rgb()

        if field_quantity.shape[-1-OoP_axis_idx] == 1:  # take trivial slice
            rgb_front = slice_field_right_handed(rgb_front, OoP_axis_idx, 0)

        # put rgb at the end for plotting
        return _np.moveaxis(rgb_front, 0, -1)

    # evaluate field if given field_quantity
    field = field_quantity.eval() if is_quantity else field_quantity.copy()

    # set to 0 outisde geometry -> gray and 0 norm
    if is_quantity: geometry = field_quantity._impl.system.geometry
    if geometry is not None: field *= geometry[None]

    # select the layer
    if layer is not None and OoP_axis_idx is not None:
        field = slice_field_right_handed(field, OoP_axis_idx, layer)

    # rescale (after layer selection) to make maximum norm 1
    field /= _np.max(_np.linalg.norm(field, axis=0)) if _np.any(field) else field

    # Create rgba image from the vector data
    rgb = _np.ones((*(field.shape[1:]), 3))  # last index for R, G and B channels
    rgb[...,0], rgb[...,1], rgb[...,2] = vector_to_rgb(*field)

    return rgb

def get_rgba(field_quantity: _mxp.FieldQuantity|_np.ndarray,
             OoP_axis_idx: Optional[Literal[0, 1, 2]] = 2,
             layer: Optional[int] = None,
             geometry: Optional[_np.ndarray] = None) -> _np.ndarray:
    """Get rgba values of given field_quantity.

    See docstring of :func:`get_rgb` for an explanation of the parameters.
    
    Additionally, the geometry is used to set alpha value to 0 where geometry is False.

    See Also
    --------
    get_rgb
    """
    rgb = get_rgb(field_quantity, OoP_axis_idx, layer, geometry)

    rgba = _np.ones((*rgb.shape[:-1], 4))
    rgba[..., :3] = rgb

    if isinstance(field_quantity, _mxp.FieldQuantity):
        geometry = field_quantity._impl.system.geometry
    if geometry is not None:
        if layer is not None and OoP_axis_idx is not None:
            geometry = slice_field_right_handed(geometry, OoP_axis_idx, layer)
        rgba[...,3] = geometry

    return rgba

# ----- end of rgb -----

def _get_axis_components(out_of_plane_axis: Literal['x', 'y', 'z']) -> tuple[int, int, int]:
    """Translates out of plane axis string to a right handed coordinate system
    (x:0, y:1, z:2), where the out of plane axis is last."""

    if out_of_plane_axis == 'x':
        return 1, 2, 0
    if out_of_plane_axis == 'y':
        return 2, 0, 1
    if out_of_plane_axis == 'z':
        return 0, 1, 2
    
    raise ValueError(f"Unknown axis \'{out_of_plane_axis}\', use \'x\', \'y\' or \'z\' instead.")


def slice_field_right_handed(field: _np.ndarray,
                             OoP_axis_idx: Literal[0, 1, 2] = 2,
                             layer: int = 0) -> _np.ndarray:
    """Return a right-handed two dimensional slice of the given field with
    shape (ncomp, n_vertical, n_horizontal) or (n_vertical, n_horizontal) by
    taking the `layer` index of the out-of-plane axis given by `OoP_axis_idx`.

    Parameters
    ----------
    field : numpy.ndarray with shape ([ncomp,] nz, ny, nx)
        Field array to slice
    OoP_axis_idx : int, default=2
        Index of the out of plane axis. 0, 1 and 2 represent x, y and z respectively.
    layer : int, default=0
        Chosen index at which to slice the out-of-plane axis.

    Returns
    -------
    field_2D : numpy.ndarray with shape ([ncomp,] n_vertical, n_horizontal)
        Right-handed two dimensional slice of given field.
    """
    slice_ = [slice(None)] * field.ndim
    slice_[-1 - OoP_axis_idx] = layer  # index correct axis at chosen layer
    field_2D = field[tuple(slice_)]

    if OoP_axis_idx == 1:  # y
        field_2D = _np.swapaxes(field_2D, -1, -2)  # ([ncomp,] nx, nz)  for right-hand axes

    return field_2D


def _quantity_2D_extent(field_quantity: _mxp.FieldQuantity|_np.ndarray,
                        hor_axis_idx: Literal[0, 1, 2] = 0,
                        vert_axis_idx: Literal[0, 1, 2] = 1) \
                            -> Optional[tuple[int, int, int, int]]:
    """If the given field_quantity has an extent, the two dimensional extent is
    given as (left, right, bottom, top), with chosen indices for the horizontal
    and vertical axes. Returns None if the extent can't be determined.
    """
    if isinstance(field_quantity, _mxp.FieldQuantity):
        qty_extent = field_quantity._impl.system.extent
        left, right = qty_extent[2*hor_axis_idx], qty_extent[2*hor_axis_idx + 1]
        bottom, top = qty_extent[2*vert_axis_idx], qty_extent[2*vert_axis_idx + 1]
        return (left, right, bottom, top)
    if isinstance(field_quantity, _np.ndarray):
        left, right = -0.5, field_quantity.shape[3 - hor_axis_idx] - 0.5
        bottom, top = -0.5, field_quantity.shape[3 - vert_axis_idx] - 0.5
        return (left, right, bottom, top)
    return None


def _get_colorbar_verticality(colorbar_kwargs: dict) -> bool:
    """Get the verticality of the colorbar depending on the given keyword
    arguments. This follows the logic of the matplotlib library.
    """
    if "location" in colorbar_kwargs.keys():  # location has priority
        if (colorbar_kwargs["location"] == "top" or
            colorbar_kwargs["location"] == "bottom"):
            return False
    elif ("orientation" in colorbar_kwargs.keys() and
        colorbar_kwargs["orientation"] == "horizontal"):
        return False

    return True  # default

# ----- downsample -----

def _get_resampled_meshgrid(old_size: tuple[int, int], new_size: tuple[int, int],
                              quantity: Optional[_mxp.FieldQuantity] = None,
                              hor_axis_idx: Literal[0, 1, 2] = 0,
                              vert_axis_idx: Literal[0, 1, 2] = 1) \
                                -> tuple[_np.ndarray, _np.ndarray]:
    """Get a 2D meshrgid of resampled coordinates, which indicate cell centers
    of a new grid overlayed on top of an old grid, with all edges aligned.

    If the `quantity` is not given, the center of the bottom left cell with
    index [0, 0] is assumed to live at coordinate (0, 0), with cell sizes of 1.
    If it is given, [0, 0] lives at the origin of the quantity's grid, with cell
    sizes corresponding to the world's cell sizes.

    Parameters
    ----------
    old_size : tuple of 2 integers
        The old 2D shape (n_horizontal, n_vertical) of an array.
    new_size : tuple of 2 integers
        The new 2D shape (n_horizontal, n_vertical) of an array.
    quantity : mumaxplus.FieldQuantity, optional
        If given, the coordinates are translated to align with the origin of its
        grid and are scaled with the world's cellsizes.
    hor_axis_idx : int, default=0
        The index of the horizontal axis (0:x, 1:y, 2:z). This is only used for
        the origin and cell size if `quantity` is given.
    vert_axis_idx : int, default=1
        The index of the horizontal axis (0:x, 1:y, 2:z). This is only used for
        the origin and cell size if `quantity` is given.
    
    Returns
    -------
    meshgrid : tuple of 2 numpy.ndarray
        The meshgrid of new coordinates.
    """
    nx_old, ny_old = old_size
    nx_new, ny_new = new_size

    # align edges of first cells, but use new spacing
    x = _np.arange(0, nx_new) * nx_old/nx_new - 0.5 + nx_old/nx_new / 2
    y = _np.arange(0, ny_new) * ny_old/ny_new - 0.5 + ny_old/ny_new / 2

    if quantity is not None:
        origin = quantity.grid.origin
        cellsize = quantity._impl.system.cellsize
        
        cx_old, cy_old = cellsize[hor_axis_idx], cellsize[vert_axis_idx]
        ox_old, oy_old = origin[hor_axis_idx], origin[vert_axis_idx]

        # translate to origin
        x += ox_old
        y += oy_old

        # transform to units (meters)
        x *= cx_old
        y *= cy_old

    return _np.meshgrid(x, y, indexing='xy')  # [y, x] indexing like mumax fields


def _get_length_fraction_inside(o_i: int, n_i: int, r: float) -> float:
    """Returns the fraction of the side length of the old small cell that is
    inside the new cell, depending on the ratio of the total number of cells.

    Parameters
    ----------
    old_idx : int or numpy.ndarray of integers
        Index of a smaller cell in a larger array.
    new_idx : int or numpy.ndarray of integers
        Index of a larger cell in a smaller array.
    ratio : float or numpy.ndarray of floats
        Ratio of the old number of cells over new number of cells. Assumed to be
        larger than or equal to 1.

    Returns
    -------
    float or numpy.ndarray of floats
        Fraction of the smaller cell with the old index inside the larger cell
        with the new index.
    """
    # old index, new index, old-new shape ratio
    n_l, n_r = n_i * r, (n_i + 1) * r  # left and right boundaries of new cell

    if n_l <= o_i and o_i + 1 <= n_r:  # fully inside new cell
        return 1.
    if n_l <= o_i < n_r:  # only left side inside new cell
        return n_r - o_i
    if n_l < o_i + 1 <= n_r:  # only right side inside new cell
        return o_i + 1 - n_l
    return 0.  # not inside

def downsample(field: _np.ndarray, new_shape: tuple, intrinsic: bool = True) -> _np.ndarray:
    """Downsample the field to a new shape. If the given and returned fields are
    imagined to be overlayed on top of one another with their edges aligned,
    then the cells of the returned array contain the average (or sum, if not
    intrinsic) of all cells they (partially) cover, weighted by the fraction of
    the covered area/volume.

    Parameters
    ----------
    field : numpy.ndarray
        The field to downsample.
    new_shape : tuple of integers
        The new shape, which must have the same number of dimensions as the
        given `field`, with smaller or equal length per dimension.
    intrinsic : bool, True
        Whether to take the average or the total sum, so whether to divide the
        resulting sum by the number of old cells contained within the new cells
        or not.
    
    Returns
    -------
    new_field : numpy.ndarray
        The downsampled field with a smaller size.
    """
    old_shape = field.shape
    ndim = field.ndim

    if ndim != len(new_shape):
        raise ValueError("New shape does not have same number of dimensions as old shape.")
    for n_old, n_new in zip(old_shape, new_shape):
        if n_new > n_old:
            raise ValueError("Can't resample to larger arrays, can only downsample.")

    new_field = _np.zeros(new_shape)
    on_shape_ratio = _np.array([old_shape[i] / new_shape[i] for i in range(ndim)])

    for idx_new in _np.ndindex(new_shape):  # loop over all new cells
        sum = 0.
        # loop over all old cells that are at least partly inside the new cell 
        old_ranges = [range(int(i_new * ratio), _math.ceil((i_new + 1) * ratio))
                      for (i_new, ratio) in zip(idx_new, on_shape_ratio)]
        for idx_old in product(*old_ranges):
            # find fraction of the length/area/volume inside
            frac = 1.0
            for i in range(ndim):
                frac *= _get_length_fraction_inside(idx_old[i], idx_new[i], on_shape_ratio[i])

            sum += frac * field[*idx_old]
        
        new_field[*idx_new] = sum

    if intrinsic: new_field /= _np.prod(on_shape_ratio)

    return new_field

# ----- end of downsample -----
# ----- SI units -----
# This code is copied from the Hotspice code written by Jonathan Maes and slightly tweaked.
# https://github.com/bvwaeyen/Hotspice/blob/main/hotspice/utils.py#L129

SIprefix_to_magnitude = {'q': -30, 'r': -27, 'y': -24, 'z': -21, 'a': -18,
                         'f': -15, 'p': -12, 'n': -9, 'µ': -6, 'm': -3, 'c': -2, 'd': -1,
                         '': 0, 'da': 1, 'h': 2, 'k': 3, 'M': 6, 'G': 9, 'T': 12,
                         'P': 15, 'E': 18, 'Z': 21, 'Y': 24, 'R': 27, 'Q': 30}

def SIprefix_to_mul(unit) -> float:
    return 10**SIprefix_to_magnitude[unit]

magnitude_to_SIprefix = {v: k for k, v in SIprefix_to_magnitude.items()}

def appropriate_SIprefix(n: float|_np.ndarray, unit_prefix='',
                         only_thousands=True) -> tuple[float, str]:
    """Converts `n` with old SI prefix `unit_prefix` to a reasonable number with
    a new SI prefix.

    Parameters
    ----------
    n : float or numpy.ndarray
        Number or array of numbers to convert.
    unit_prefix : str, default=''
        Previous SI prefix of the unit of `n`.
    only_thousands : bool, default=True
        If True (default), then centi, deci, deca and hecto are not used.

    Returns
    -------
    tuple[float, str]
        The new scalar value(s) and the new appropriate SI prefix.

    Examples
    --------
    >>> appropriate_SIprefix(0.0000238, 'm')
    (23.8, 'n')

    Converting 0.0000238 ms returns 23.8 ns.
    """
    # If `n` is an array, use the average of absolute values as representative of the scale
    value = _np.average(_np.abs(n)) if isinstance(n, _np.ndarray) else n

    if unit_prefix not in SIprefix_to_magnitude.keys():
        raise ValueError(f"'{unit_prefix}' is not a supported SI prefix.")
    
    offset_magnitude = SIprefix_to_magnitude[unit_prefix]
    # TODO: maybe 'floor' looks nicer than 'round'; less decimal numbers
    # Don't use any prefix if exactly 0
    nearest_magnitude = (round(_np.log10(abs(value))) if value != 0 else 0) + offset_magnitude
    # Make sure it is in the known range
    nearest_magnitude = _np.clip(nearest_magnitude,
                                 min(SIprefix_to_magnitude.values()),
                                 max(SIprefix_to_magnitude.values()))
    
    supported_magnitudes = magnitude_to_SIprefix.keys()
    if only_thousands: supported_magnitudes = [mag for mag in supported_magnitudes if (mag % 3) == 0]
    for supported_magnitude in sorted(supported_magnitudes):
        if supported_magnitude <= nearest_magnitude:
            used_magnitude = supported_magnitude
        else:
            break
    
    return (n/10**(used_magnitude - offset_magnitude), magnitude_to_SIprefix[used_magnitude])

# ----- end of SI units -----
# ----- plot_field -----

class UnitScalarFormatter(_matplotlib.ticker.ScalarFormatter):
    """An extension of the ScalarFormatter to take units into account.
    https://github.com/matplotlib/matplotlib/blob/v3.10.3/lib/matplotlib/ticker.py#L397
    """
    def __init__(self, SIprefix: str, unit: str, useOffset=None, useMathText=None,
                 useLocale=None, usetex=None):
        self.SIprefix = SIprefix
        self.SImultiplier = SIprefix_to_mul(SIprefix)
        self.unit = unit
        super().__init__(useOffset, useMathText, useLocale, usetex=usetex)


    def set_locs(self, locs):
        """Use rescaled locs according to preferred unit multiplier for all math,
        then save original locs for proper drawing in data coordinates."""
        rescaled_locs = [0]*len(locs)
        for i, loc in enumerate(locs):
            rescaled_locs[i] = loc / self.SImultiplier
        super().set_locs(rescaled_locs)  # rescaled processing happens here
        self.locs = locs  # but keep original locs saved for proper drawing


    def __call__(self, value, pos=None):
        """Format rescaled values on axis"""
        return super().__call__(value / self.SImultiplier, pos)

    def format_data(self, value):
        # TODO: figure out the use of this function
        return super().format_data(value / self.SImultiplier)

    def format_data_short(self, value):
        """This is exactly the same implementation as with ScalarFormatter.
        Value is only divided by self.SImultiplier at the end and a unit is added.
        """

        # docstring inherited
        if value is _np.ma.masked:
            return ""
        if isinstance(value, Integral):
            fmt = "%d"
        else:
            if getattr(self.axis, "__name__", "") in ["xaxis", "yaxis"]:
                if self.axis.__name__ == "xaxis":
                    axis_trf = self.axis.axes.get_xaxis_transform()
                    axis_inv_trf = axis_trf.inverted()
                    screen_xy = axis_trf.transform((value, 0))
                    neighbor_values = axis_inv_trf.transform(
                        screen_xy + [[-1, 0], [+1, 0]])[:, 0]
                else:  # yaxis:
                    axis_trf = self.axis.axes.get_yaxis_transform()
                    axis_inv_trf = axis_trf.inverted()
                    screen_xy = axis_trf.transform((0, value))
                    neighbor_values = axis_inv_trf.transform(
                        screen_xy + [[0, -1], [0, +1]])[:, 1]
                delta = abs(neighbor_values - value).max()
            else:
                # Rough approximation: no more than 1e4 divisions.
                a, b = self.axis.get_view_interval()
                delta = (b - a) / 1e4
            fmt = f"%-#.{_matplotlib.cbook._g_sig_digits(value, delta)}g"
        
        fmt += f" {self.SIprefix}{self.unit}"  # add unit
        value /= self.SImultiplier  # show in preferred unit
        return self._format_maybe_minus_and_locale(fmt, value)

class _Plotter:
    """This class is intended for organizing the data and parameters of
    :func:`plot_field`, but not to be instantiated or manipulated by the
    user.
    """
    def __init__(self, field_quantity,
                 out_of_plane_axis, layer, component, geometry, field,
                 file_name, show, ax, figsize, title, xlabel, ylabel,
                 imshow_symmetric_clim, imshow_kwargs,
                 enable_colorbar, colorbar_kwargs,
                 enable_quiver, arrow_size, quiver_cmap, quiver_symmetric_clim,
                 quiver_kwargs):
        """see the docstring of :func:`plot_field`."""

        # check field_quantity input
        is_quantity = isinstance(field_quantity, _mxp.FieldQuantity)
        if not is_quantity and not isinstance(field_quantity, _np.ndarray):
            raise TypeError("The first argument should be a FieldQuantity or an ndarray.")
        
        if len(field_quantity.shape) != 4:
            raise ValueError(
                "The field quantity has the wrong number of dimensions: "
                + f"{len(field_quantity.shape)} instead of 4."
            )

        if (is_quantity and field is not None and field.shape != field_quantity.shape):
            raise ValueError("The field_quantity and field need to have the same shape.")

        # components
        self.ncomp = field_quantity.shape[0]
        if component is not None:
            if component >= self.ncomp:
                raise IndexError(f"Component index {component} out of range, "
                                 + f"must be less than {self.ncomp}.")
        elif (self.ncomp != 1) and (self.ncomp != 3):
            raise ValueError(f"Cannot plot field quantity with {self.ncomp} "
                             + "components without specifying which component "
                             + "to plot.")

        if component is None and self.ncomp == 1:
            self.comp = 0  # plot the only component
        else:
            self.comp = component

        self.out_of_plane_axis, self.layer = out_of_plane_axis, layer
        self.hor_axis_idx, self.vert_axis_idx, self.OoP_axis_idx = _get_axis_components(out_of_plane_axis)

        # split types to know what we're working with
        if is_quantity:
            _field = field if field is not None else field_quantity.eval()
            self.quantity = field_quantity
        else:
            _field = field_quantity
            self.quantity = None

        # only need 2D slice of field
        self.field_2D = slice_field_right_handed(_field, self.OoP_axis_idx, self.layer)
        self.field_shape = field_quantity.shape

        # geometry
        self.geom_2D = None
        if geometry is not None:
            if geometry.shape == field_quantity.shape[1:]:
                self.geom_2D = slice_field_right_handed(geometry, self.OoP_axis_idx, self.layer)
            else:
                raise ValueError(
                    f"The shape of the given geometry {geometry.shape} does "
                    + "not match the spacial shape of the field quantity "
                    + f"{field_quantity.shape[1:]}")
        elif self.quantity is not None:
            self.geom_2D = slice_field_right_handed(self.quantity._impl.system.geometry,
                                                    self.OoP_axis_idx, self.layer)

        # file name
        self.file_name = file_name

        # show (and default)
        if show is None:
            if ax is None and file_name is None:
                self.show = True
            else:
                self.show = False
        else:
            self.show = show

        # ax: use or create
        self.rescale_figsize = False
        if ax is None:
            _, self.ax = _plt.subplots(figsize=figsize)
            if figsize is None: self.rescale_figsize = True
        else:
            self.ax = ax

        # title
        self.title = title

        # labels
        self.xlabel = xlabel
        self.ylabel = ylabel

        # colorbar
        self.enable_colorbar = enable_colorbar
        self.colorbar_kwargs = colorbar_kwargs.copy()
        self.number_of_colorbars_added = 0

        # imshow
        self.imshow_kwargs = imshow_kwargs.copy()
        im_extent = _quantity_2D_extent(field_quantity, self.hor_axis_idx, self.vert_axis_idx)
        self.imshow_kwargs.setdefault("extent", im_extent)
        self.imshow_kwargs.setdefault("origin", "lower")
        # vector image or scalar image? vector if field with 3 components, but none selected
        self.vector_image_bool = ((self.ncomp == 3) and (self.comp is None))

        # imshow symmetric clim
        if not "vmin" in self.imshow_kwargs.keys() and \
           not "vmax" in self.imshow_kwargs.keys():
            self.imshow_symmetric_clim = imshow_symmetric_clim
        else:
            self.imshow_symmetric_clim = False

        if self.imshow_symmetric_clim: self.imshow_kwargs.setdefault("cmap", "bwr")

        # with or without quiver?
        if self.ncomp == 3:  # vector
            if enable_quiver is not None:  # let user decide
                self.enable_quiver = enable_quiver
            else:  # or add arrows if no specific comp given
                self.enable_quiver = self.comp is None
        else:  # no quiver possible
            self.enable_quiver = False

        # save relevant quiver information if needed
        if self.enable_quiver:
            self.arrow_size = arrow_size
            self.arrow_size_fraction = 3/4  # make arrow smaller than max
            self.quiver_kwargs = quiver_kwargs.copy()  # leave user input alone
            self.quiver_kwargs.setdefault("pivot", "middle")

            # quiver symmetric clim
            if not "vmin" in self.quiver_kwargs.keys() and \
               not "vmax" in self.quiver_kwargs.keys():
                self.quiver_symmetric_clim = quiver_symmetric_clim
            else:
                self.quiver_symmetric_clim = False

            # quiver cmap
            if "cmap" in self.quiver_kwargs.keys():
                if quiver_cmap is None:  # use quiver_cmap, not cmap kwarg
                    self.quiver_cmap = self.quiver_kwargs.pop("cmap")
                else:  # both are specified for some reason
                    _warnings.warn("The quiver colormap is provided twice, using `quiver_cmap`.")
                    self.quiver_kwargs.pop("cmap")
                    self.quiver_cmap = quiver_cmap
            else:
                self.quiver_cmap = quiver_cmap

    def plot_image(self):
        # imshow
        if self.vector_image_bool:  # vector field
            im_rgba = get_rgba(self.field_2D, geometry=self.geom_2D,
                               OoP_axis_idx=None, layer=None)
            self.im = self.ax.imshow(im_rgba, **self.imshow_kwargs)
        else:  # show requested component
            scalar_field = self.field_2D[self.comp]
            if self.geom_2D is not None:  # mask False geometry
                scalar_field = _np.ma.array(scalar_field, mask=_np.invert(self.geom_2D))

            # make symmetric clim if not user provided
            if not "vmin" in self.imshow_kwargs.keys() and \
               not "vmax" in self.imshow_kwargs.keys() and \
               self.imshow_symmetric_clim:
                vmax = _np.max(_np.abs(scalar_field))
                self.imshow_kwargs["vmax"] = vmax
                self.imshow_kwargs["vmin"] = -vmax

            self.im = self.ax.imshow(scalar_field, **self.imshow_kwargs)
            
            # cbar
            # Name only the component if relevant. Let title display quantity name, unless empty
            if self.ncomp > 1:
                cname = f"component {self.comp}"
            elif self.quantity and (qname := self.quantity.name):
                cname = qname
            else:
                cname = ""
            self.add_cbar(self.im, name=cname)
    
    def plot_quiver(self):
        if not self.enable_quiver:
            return

        # downsample 2D field
        ncomp, ny_old, nx_old = self.field_2D.shape
        nx_new = max(int(nx_old / self.arrow_size), 1)
        ny_new = max(int(ny_old / self.arrow_size), 1)

        X, Y = _get_resampled_meshgrid((nx_old, ny_old), (nx_new, ny_new), self.quantity,
                                        self.hor_axis_idx, self.vert_axis_idx)

        sampled_field = downsample(self.field_2D, (ncomp, ny_new, nx_new))
        U, V = sampled_field[self.hor_axis_idx], sampled_field[self.vert_axis_idx]

        # scale arrows correctly, but set kwargs together
        if not "scale" in self.quiver_kwargs and not "scale_units" in self.quiver_kwargs:
            # the longest allowed vector in xy units
            max_allowed_len = self.arrow_size * self.arrow_size_fraction
            if self.quantity:
                cellsize = self.quantity._impl.system.cellsize
                max_allowed_len *= min(cellsize[self.hor_axis_idx], cellsize[self.vert_axis_idx])

            max_IP_norm = _np.max(_np.sqrt(U**2 + V**2))  # longest in-plane arrow in UV units
            self.quiver_kwargs["scale"] = max_IP_norm / max_allowed_len
            self.quiver_kwargs["scale_units"] = "xy"

        # plot requested quiver
        if isinstance(self.quiver_cmap, str) and "hsl" in self.quiver_cmap.lower():  # HSL with rgb
            q_rgb = _np.reshape(get_rgb(sampled_field, OoP_axis_idx=None, layer=None),
                                (nx_new*ny_new, 3))
            self.quiver = self.ax.quiver(X, Y, U, V, color=q_rgb, **self.quiver_kwargs)
        elif self.quiver_cmap == None:  # uniform color
            self.quiver_kwargs.setdefault("alpha", 0.4)
            self.quiver = self.ax.quiver(X, Y, U, V, **self.quiver_kwargs)
        else:  # OoP component colored
            sampled_field_OoP = sampled_field[self.OoP_axis_idx]
            vmin, vmax = None, None
            if self.quiver_symmetric_clim:
                vmax = _np.max(_np.abs(sampled_field_OoP))
                vmin = -vmax
            self.quiver_kwargs.setdefault("clim", (vmin, vmax))
            self.quiver = self.ax.quiver(X, Y, U, V, sampled_field_OoP,
                               cmap=self.quiver_cmap, **self.quiver_kwargs)
            self.add_cbar(self.quiver, name=f"Out-of-plane {self.out_of_plane_axis}-component")

    def add_cbar(self, cp, name: str = ""):
        """Adds a colorbar associated with the given plot next to `self.ax`, if
        `self.enable_colorbar` is True.
        
        If relevant, this adds appropriate unit to the label with a
        `UnitScalarFormatter` as formatter, unless "label" or "format" is
        specified by the user in `self.colorbar_kwargs`.

        A new Axes for the colorbar is created to the right of the plot with
        equal height, unless any kwargs about location, orientation or size have
        been set in `self.colorbar_kwargs`.

        Parameters
        ----------
        cp : matplotlib.cm.ScalarMappable
            The plot with which the colorbar is associated.
        
        name : str, default=""
            Name of the colorbar, which forms (part of) the label.

        Returns
        -------
        Colorbar, optional
            Returns the colorbar, if created.
        """

        if not self.enable_colorbar:
            return

        # get user kwargs, but leave alone for later use (e.g. multiple cbars)
        cbar_kwargs = self.colorbar_kwargs.copy()

        # add label and formatter if neither is user-provided
        if not any([k in cbar_kwargs.keys() for k in ["label", "format"]]):
            label = name
            if self.quantity and (unit := self.quantity.unit):
                vmin, vmax = cp.get_clim()
                _, prefix = appropriate_SIprefix(max(abs(vmin), abs(vmax)))
                label += f" ({prefix}{unit})"
                cbar_kwargs["format"] = UnitScalarFormatter(prefix, unit)
            cbar_kwargs["label"] = label

        # make cax so cbar scales with ax height, if no sizes/positions are user-provided
        if not any([k in cbar_kwargs.keys() for k in ["ax", "cax", "location",
            "orientation", "fraction", "shrink", "aspect", "pad", "anchor", "panchor"]]):
            if self.number_of_colorbars_added == 0:  # new divider if none exists
                self.divider = _make_axes_locatable(self.ax)
            # leave fixed space for ticks and label of other cbar
            pad = "5%" if self.number_of_colorbars_added == 0 else 0.8
            cbar_kwargs["cax"] = self.divider.append_axes(position="right", size="5%", pad=pad)

        self.number_of_colorbars_added += 1

        return self.ax.figure.colorbar(cp, **cbar_kwargs)

    def replace_get_cursor_data(self):
        """Replaces the `get_cursor_data` method of the AxesImage artist `self.im`
        in order to interactively show the raw data values instead of the image
        array entries. This is useful for when the data is manually converted to
        rgb(a) values.

        This is a modified version of the original
        `matplotlib,image.AxesImage.get_cursor_data`.
        https://github.com/matplotlib/matplotlib/blob/v3.10.5/lib/matplotlib/image.py#L979-L1004
        """

        xmin, xmax, ymin, ymax = self.im.get_extent()
        if self.im.origin == 'upper':
            ymin, ymax = ymax, ymin
        imin, imax, jmin, jmax = 0, self.field_2D.shape[1], 0, self.field_2D.shape[2]

        data_extent = Bbox([[xmin, ymin], [xmax, ymax]])
        array_extent = Bbox([[jmin, imin], [jmax, imax]])
        data_to_array_transform = BboxTransform(boxin=data_extent, boxout=array_extent)

        def new_get_cursor_data(im_self, event):
            # first argument is `self` when a method of AxesImage.

            trans = im_self.get_transform().inverted()  # changes with viewing window
            trans += data_to_array_transform
            point = trans.transform([event.x, event.y])  # cursor coordinates

            if any(_np.isnan(point)):
                return None
            j, i = point.astype(int)
            # Clip the coordinates at array bounds
            if ((not (0 <= i < imax) or not (0 <= j < jmax)) or
                (self.geom_2D is not None and not self.geom_2D[i, j])):
                # outside
                return None
            else:
                return self.field_2D[:, i, j]

        self.im.get_cursor_data = MethodType(new_get_cursor_data, self.im)

    def replace_format_cursor_data(self):
        """Replaces the `format_cursor_data` method of the AxesImage artist
        `self.im` to include an appropriate SI unit.

        This is based on the original `matplotlib.axes.Axes.get_cursor_data`.
        https://github.com/matplotlib/matplotlib/blob/v3.10.5/lib/matplotlib/artist.py#L1330-L1360
        """
        def new_format_cursor_data(data):
            try:
                data[0]
            except (TypeError, IndexError):
                data = [data]

            data = _np.array([item for item in data if isinstance(item, Number)])
            if len(data) == 0:
                return ""

            if append_unit:= (self.quantity and self.quantity.unit):
                data, prefix = appropriate_SIprefix(data)

            data_str = ', '.join(f'{item:0.3g}' for item in data)
            if len(data) > 1:
                data_str = "(" + data_str + ")"
            
            if append_unit:
                data_str += " " + prefix + self.quantity.unit

            return data_str

        self.im.format_cursor_data = new_format_cursor_data

    def dress_axes(self, max_width_over_height_ratio=6., max_height_over_width_ratio=3.):
        """Dress `self.ax` to make it prettier using all known information.

        Sets:
        - xlim and ylim
        - xlabel and ylabel, including units
        - aspect ratio
        - relevant title
        - get_cursor_data in order to inspect raw vector data
        - format_cursor_data for more human readable data.
        """
        
        left, right, bottom, top = self.imshow_kwargs["extent"]

        # axis limits
        self.ax.set_xlim(left, right)
        self.ax.set_ylim(bottom, top)

        # xlabel
        if self.xlabel is None:
            if self.quantity:
                x_maxabs = max(abs(left), abs(right))  # find largest number
                _, x_prefix = appropriate_SIprefix(x_maxabs)
                self.ax.set_xlabel(f"${'xyz'[self.hor_axis_idx]}$ ({x_prefix}m)")
                # axis label ticks with appropriate numbers according to prefix
                self.ax.xaxis.set_major_formatter(UnitScalarFormatter(x_prefix, "m"))
            else:
                self.ax.set_xlabel(f"${'xyz'[self.hor_axis_idx]}$ (index)")
        elif self.xlabel:
            self.ax.set_xlabel(self.xlabel)

        # ylabel
        if self.ylabel is None:
            if self.quantity:
                y_maxabs = max(abs(bottom), abs(top))
                _, y_prefix = appropriate_SIprefix(y_maxabs)
                self.ax.set_ylabel(f"${'xyz'[self.vert_axis_idx]}$ ({y_prefix}m)")
                # axis label ticks with appropriate numbers according to prefix
                self.ax.yaxis.set_major_formatter(UnitScalarFormatter(y_prefix, "m"))
            else:
                self.ax.set_ylabel(f"${'xyz'[self.vert_axis_idx]}$ (index)")
        elif self.xlabel:
            self.ax.set_ylabel(self.ylabel)

        self.ax.set_facecolor("gray")
        
        # use "equal" aspect ratio if not too rectangular
        if ((right - left) / (top - bottom) < max_width_over_height_ratio and
            (hw_ratio := (top - bottom) / (right - left)) < max_height_over_width_ratio):
            self.ax.set_aspect("equal")
            
            if self.rescale_figsize:
                fig = self.ax.figure
                w0, h0 = fig.get_size_inches()
                hw_ratio = hw_ratio ** 0.7  # get closer to square ratio

                # same area, different ratio, add space per colorbar
                inch_per_cbar = 0.8
                if _get_colorbar_verticality(self.colorbar_kwargs):
                    w1 = _np.sqrt(w0 * h0 / hw_ratio)
                    w1 += self.number_of_colorbars_added * inch_per_cbar
                    h1 = h0*w0 / w1
                else:
                    h1 = _np.sqrt(w0 * h0 * hw_ratio)
                    h1 += self.number_of_colorbars_added * inch_per_cbar
                    w1 = h0*w0 / h1

                fig.set_size_inches(w1, h1)
        else:
            self.ax.set_aspect("auto")

        # title
        # e.g.: component 2 of ferromagnet_1:magnetization in z-layer 16
        if self.title is None:  # make default title
            # component if not in colorbar and non-trivial
            component = ""
            if not self.vector_image_bool and not self.enable_colorbar and self.ncomp > 1:
                component = f"component {self.comp}"

            # name of quantity  # TODO: let user give name?
            name = self.quantity.name if self.quantity else ""

            # layer of OoP axis if non-trivial
            layer = ""
            if self.field_shape[3 - self.OoP_axis_idx] > 1:
                layer = f"${self.out_of_plane_axis}$-layer {self.layer}"

            # combine into title
            title = component + " of " + name if component and name else component + name
            title = title + " in " + layer if title and layer else title + layer

            if title:
                self.ax.set_title(title)

        elif self.title:  # user set title
            self.ax.set_title(self.title)

        # replace get_cursor_data of image when plotting vector rgb
        if self.vector_image_bool:
            self.replace_get_cursor_data()

        # replace format_cursor_data of image to add appropriate SI unit
        self.replace_format_cursor_data()

    def plot(self) -> Axes:
        """The one function to plot everything."""

        self.plot_image()

        self.plot_quiver()

        self.dress_axes()

        if self.file_name:
            self.ax.figure.savefig(self.file_name)

        if self.show:
            _plt.show()

        return self.ax


def plot_field(field_quantity: _mxp.FieldQuantity|_np.ndarray,
               out_of_plane_axis: Literal['x', 'y', 'z'] = 'z', layer: int = 0,
               component: Optional[int] = None, geometry: Optional[_np.ndarray] = None,
               field: Optional[_np.ndarray] = None,
               file_name: Optional[str] = None, show: Optional[bool] = None,
               ax: Optional[Axes] = None, figsize: Optional[tuple[float, float]] = None,
               title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
               imshow_symmetric_clim: bool = False, imshow_kwargs: dict = {},
               enable_colorbar: bool = True, colorbar_kwargs: dict = {},
               enable_quiver: bool = None, arrow_size: float = 8.,
               quiver_cmap: Optional[str]= None, quiver_symmetric_clim: bool = True,
               quiver_kwargs : dict = {}) -> Axes:
    """Plot a :func:`mumaxplus.FieldQuantity` or `numpy.ndarray`
    with 1 component as a scalar field, with 3 components as a vector field or
    plot one selected component as a a scalar field.
    Vector fields are plotted using an HSL colorscheme, with optionally added
    arrows. This differs slightly from mumax³, see :func:`get_rgb`.
    
    Parameters
    ----------
    field_quantity : mumaxplus.FieldQuantity or numpy.ndarray
        The field_quantity needs to have 4 dimensions with the shape (ncomp, nx, ny, nz).
        Additional dressing of the Axes is done if given a mumaxplus.FieldQuantity.

    out_of_plane_axis : string, default="z"
        The axis pointing out of plane: "x", "y" or "z". The other two are plotted according to a right-handed coordinate system.

        - "x": z over y
        - "y": x over z
        - "z": y over x

    layer : int, default=0
        Chosen index at which to slice the out-of-plane axis.

    component : int, optional
        The component of the field_quantity to plot as an image.
        If set to an integer, that component is plotted as a scalar field.
        If None (default), a field_quantity with

        - 1 component is plotted as a scalar field
        - 3 components is plotted as a vector field with an HSL colorscheme.
        - a different number of components can't be plotted.

    geometry : numpy.ndarray, optional
        The geometry of the field_quantity with shape (nz, ny, nx) to mask
        scalar field plots where geometry is False.

    field : numpy.ndarray, optional
        If given and if `field_quantity` is a mumaxplus.FieldQuantity, this
        field will be plotted instead of evaluating `field_quantity` (again).
        `field_quantity` will still be used for dressing the plot (extent, name,
        unit, ...).
        If `field_quantity` is a numpy.ndarray then `field` is ignored.

    file_name : string, optional
        If given, the resulting figure will be saved with the given file name.

    show : bool, optional
        Whether to call `matplotlib.pyplot.show` at the end.
        If None (default), `matplotlib.pyplot.show` will be called only if no
        `ax` and no `file_name` have been provided.

    ax : matplotlib.axes.Axes, optional
        The Axes instance to plot onto. If None (default), new Figure and Axes
        instances will be created.

    figsize : tuple[float] of size 2, optional
        The size of the figure, if a new figure has to be created.

    title : str, optional
        The title of the Axes. `None` will generate a default title, while an
        empty string won't set any title.

    xlabel, ylabel : str, optional
        The label of the x/y-axis. `None` will generate a default label, while
        an empty string won't set any label.

    imshow_symmetric_clim : bool, default=False
        Whether to map zero to the central color if a scalar field is plotted
        in the image.
        This is ignored if vmin or vmax is given in `imshow_kwargs`.
        This is best used with diverging colormaps, like "bwr".

    imshow_kwargs : dict, default={}
        Keyword arguments to pass to `matplotlib.axes.Axes.imshow`, e.g. "cmap".

    enable_colorbar : bool, default=True
        Whether to automatically add a colorbar to the figure of the Axes when relevant.
    
    colorbar_kwargs : dict, default={}
        Keyword arguments to pass to `matplotlib.figure.Figure.colorbar`.
        Relevant formatting and labeling with units is done automatically,
        unless the keyword arguments "label" or "format" are specified.

    enable_quiver : bool, optional
        Whether to plot arrows on top of the colored image. If None (default),
        arrows are only added if no specific component for the image has been
        given.
        This is only relevant for field quantities with 3 components.

    arrow_size : float, default=8
        Length of an arrow as a number of cells, so one arrow is designated to
        an area of `arrow_size` by `arrow_size`.
        This is only relevant for field quantities with 3 components.

    quiver_cmap : string, optional
        A colormap to use for the quiver. By default, no colormap is used, so
        the arrows are a solid color. If set to "HSL", the 3D vector data is
        used for an HSL colorscheme, where the x- and y-components control the
        hue and saturation and the z-component controls lightness, regardless of
        the `out_of_plane_axis`. Any matplotlib colormap can also be given to
        color the arrows according to the out-of-plane component.
        This is only relevant for field quantities with 3 components.

    quiver_symmetric_clim : bool, default=True
        Whether to map zero to the central color if the arrows are colored
        according to the out-of-plane component of the vector field.
        This is ignored if clim is given in `quiver_kwargs`.
        This is best used with diverging colormaps, like "bwr".
        This is only relevant for field quantities with 3 components.

    quiver_kwargs : dict, default={}
        Keyword arguments to pass to `matplotlib.axes.Axes.quiver`.
        This is only relevant for field quantities with 3 components.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The resulting Axes on which is plotted.

    See Also
    --------
    get_rgb, get_rgba
    """
    plotter = _Plotter(field_quantity,
                       out_of_plane_axis, layer, component, geometry, field,
                       file_name, show, ax, figsize, title, xlabel, ylabel,
                       imshow_symmetric_clim, imshow_kwargs,
                       enable_colorbar, colorbar_kwargs,
                       enable_quiver, arrow_size, quiver_cmap, quiver_symmetric_clim,
                       quiver_kwargs)
    return plotter.plot()

# ----- end of plot_field -----
# ----- inspect_field -----

def inspect_field(field_quantity: _mxp.FieldQuantity|_np.ndarray,
                  out_of_plane_axis: Literal['x', 'y', 'z'] = 'z', layer: int = 0,
                  geometry: Optional[_np.ndarray] = None,
                  field: Optional[_np.ndarray] = None,
                  file_name: Optional[str] = None, show: Optional[bool] = None,
                  figsize: Optional[tuple[float, float]] = None,
                  nrows: Optional[bool] = None, ncols: Optional[bool] = None,
                  suptitle: Optional[str] = None,
                  xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                  symmetric_clim: bool = False, imshow_kwargs: dict = {},
                  enable_colorbar: bool = True, shared_colorbar : bool = False,
                  colorbar_kwargs: dict = {}):
    """Plot every component of a :func:`mumaxplus.FieldQuantity` or
    `numpy.ndarray` as a scalar field subplot on one figure.
    
    Parameters
    ----------
    field_quantity : mumaxplus.FieldQuantity or numpy.ndarray
        The field_quantity needs to have 4 dimensions with the shape (ncomp, nx, ny, nz).
        Additional dressing of the Axes is done if given a mumaxplus.FieldQuantity.

    out_of_plane_axis : string, default="z"
        The axis pointing out of plane: "x", "y" or "z". The other two are plotted according to a right-handed coordinate system.

        - "x": z over y
        - "y": x over z
        - "z": y over x

    layer : int, default=0
        Chosen index at which to slice the out-of-plane axis.

    geometry : numpy.ndarray, optional
        The geometry of the field_quantity with shape (nz, ny, nx) to mask plots
        where geometry is False.

    field : numpy.ndarray, optional
        If given and if `field_quantity` is a mumaxplus.FieldQuantity, this
        field will be plotted instead of evaluating `field_quantity` (again).
        `field_quantity` will still be used for dressing the plot (extent, name,
        unit, ...).
        If `field_quantity` is a numpy.ndarray then `field` is ignored.

    file_name : string, optional
        If given, the resulting figure will be saved with the given file name.

    show : bool, optional
        Whether to call `matplotlib.pyplot.show` at the end.
        If None (default), `matplotlib.pyplot.show` will be called only if no
        `file_name` has been provided.

    figsize : tuple[float] of size 2, optional
        The size of the new figure created. Automatically tries to find suitable
        figsize if not given.

    nrows, ncols : int, optional
        The number of rows/columns of the subplot grid. If only one is defined,
        the other will be made large enough to accommodate all components of the
        field_quantity. If both are None (default), a suitable arrangement will
        be sought.
    
    suptitle : str, optional
        The suptitle of the Figure. `None` will generate a default title, while
        an empty string won't set any title.

    xlabel, ylabel : str, optional
        The label of the x/y-axes of all lowest/leftmost Axes respectively.
        `None` will generate a default label, while an empty string won't set
        any label.

    symmetric_clim : bool, default=False
        Whether to map zero to the central color. This is ignored if vmin or
        vmax is given in `imshow_kwargs`.
        This is best used with diverging colormaps, like "bwr".

    imshow_kwargs : dict, default={}
        Keyword arguments to pass to `matplotlib.axes.Axes.imshow`, e.g. "cmap".

    enable_colorbar : bool, default=True
        Whether to add (a) colorbar(s) to the figure of the Axes.

    shared_colorbar : bool, default=False
        Whether to share one colorbar with the color limits (vmin, vmax) between
        all plots.
    
    colorbar_kwargs : dict, default={}
        Keyword arguments to pass to `matplotlib.figure.Figure.colorbar`.
        Relevant formatting and labeling with units is done automatically,
        unless the keyword arguments "label" or "format" are specified.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axs : matplotlib.axes.Axes or array of Axes
        All the resulting Axes on which is plotted.
    """
    # --- gather variables ---
    _imshow_kwargs = imshow_kwargs.copy()

    ncomp = field_quantity.shape[0]

    is_quantity = isinstance(field_quantity, _mxp.FieldQuantity)
    if is_quantity and field is None:
        field = field_quantity.eval()  # evaluate once

    enable_individual_colorbar = enable_colorbar and not shared_colorbar
    enable_shared_colorbar = enable_colorbar and shared_colorbar

    hor_axis_idx, vert_axis_idx, OoP_axis_idx = _get_axis_components(out_of_plane_axis)
    # get typical ratio of one plot
    left, right, bottom, top = _quantity_2D_extent(field_quantity, hor_axis_idx, vert_axis_idx)
    hw_ratio = (top - bottom) / (right - left)

    # nrows and ncols
    if nrows is None and ncols is None:  # find suitable arrangement
        # find roughly appropriate rectangle that is certainly large enough
        n_s = max(1, int(_np.sqrt(ncomp)))  # small factor
        n_l = max(1, int(_np.ceil(ncomp / n_s)))  # large factor

        # choose maximally square layout
        nrows, ncols = (n_l, n_s) if hw_ratio < 1. else (n_s, n_l)

    elif nrows is None:  # make enough space
        nrows = int(_np.ceil(ncomp / ncols))
    elif ncols is None:
        ncols = int(_np.ceil(ncomp / nrows))
    else:
        nrows, ncols = int(nrows), int(ncols)
        if ncols * nrows < ncomp:  # check user input
            raise ValueError(
                f"nrows = {nrows} and ncols = {ncols} does not leave enough " + 
                f"room to plot {ncomp} component{'' if ncomp == 1 else 's'}.")

    # --- figsize ---
    if figsize is None:
        # add space per colorbar
        vertical_cbar = _get_colorbar_verticality(colorbar_kwargs)
        number_of_cbars = 0
        if enable_shared_colorbar: number_of_cbars = 1
        elif enable_individual_colorbar:
            number_of_cbars = ncols if vertical_cbar else nrows

        # roughly calculate appropriate figsize by giving each subplot an area
        w0, h0 = 4.8, 3.6
        fig_area = (ncols * nrows) * (w0 * h0)
        fig_hw_ratio = ((hw_ratio ** 0.7) * nrows / ncols)  # slightly closer to square
        inch_per_cbar = 0.8

        if vertical_cbar:
            w1 = _np.sqrt(fig_area / fig_hw_ratio)
            w1 += number_of_cbars * inch_per_cbar
            h1 = fig_area / w1
        else:
            h1 = _np.sqrt(fig_area * fig_hw_ratio)
            h1 += number_of_cbars * inch_per_cbar
            w1 = fig_area / h1

        figsize = (w1, h1)

    # fig, axs
    fig, axs = _plt.subplots(nrows=nrows, ncols=ncols,
                             sharex="all", sharey="all", figsize=figsize)
    try: flat_axs = axs.flatten()
    except: flat_axs = [axs]

    if suptitle is None:  # make default suptitle
        # e.g.: ferromagnet_1:magnetization in z-layer 16
        qname = field_quantity.name if is_quantity else ""
        # layer of OoP axis if non-trivial
        layer_str = ""
        if field.shape[3 - OoP_axis_idx] > 1:
            layer_str = f"${out_of_plane_axis}$-layer {layer}"
        # combine
        suptitle = qname + " in " + layer_str if qname and layer_str else qname + layer_str

    if suptitle:
        fig.suptitle(suptitle)

    if enable_shared_colorbar:
        # --- find vmin vmax ---
        vmin = _np.min(field)
        vmax = _np.max(field)
        if symmetric_clim:
            vmax = max(abs(vmin), abs(vmax))
            vmin = - vmax
            _imshow_kwargs.setdefault("cmap", "bwr")

        # don't overwrite user preference, unless it is None (automatic)
        for key, value in [("vmin", vmin), ("vmax", vmax)]:
            if not key in _imshow_kwargs.keys() or \
                (key in _imshow_kwargs and _imshow_kwargs[key] == None):
                _imshow_kwargs[key] = value

        # --- add one shared colorbar ---
        vmin, vmax = _imshow_kwargs["vmin"], _imshow_kwargs["vmax"]
        cbar_kwargs = colorbar_kwargs.copy()  # copy user kwargs

        # add label and formatter if neither is user-provided
        if not any([k in cbar_kwargs.keys() for k in ["label", "format"]]):
            # Name the quantity with unit if possible
            label = ""
            if is_quantity and (qname := field_quantity.name):
                label += qname
            if is_quantity and (unit := field_quantity.unit):
                _, prefix = appropriate_SIprefix(max(abs(vmin), abs(vmax)))
                label += f" ({prefix}{unit})"
                cbar_kwargs["format"] = UnitScalarFormatter(prefix, unit)
            cbar_kwargs["label"] = label

        # create custom colorbar
        norm = _matplotlib.colors.Normalize(vmin, vmax)
        cmap = _imshow_kwargs["cmap"] if "cmap" in _imshow_kwargs.keys() else None
        sm = _matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)

        fig.colorbar(sm, ax=axs, **cbar_kwargs)


    # --- plotting of individual components ---
    for component in range(ncomp):
        ax = flat_axs[component]

        xlabel_ = xlabel if component // ncols == nrows - 1 else ''  # bottom row
        ylabel_ = ylabel if component % ncols == 0 else ''  # left column

        plot_field(field_quantity, out_of_plane_axis=out_of_plane_axis,
                   layer=layer, component=component, geometry=geometry,
                   field=field, show=False, title=f"component {component}",
                   ax=ax, xlabel=xlabel_, ylabel=ylabel_,
                   imshow_symmetric_clim=symmetric_clim,
                   imshow_kwargs=_imshow_kwargs,  # modified
                   enable_colorbar=enable_individual_colorbar,
                   colorbar_kwargs=colorbar_kwargs,  # user-provided, not modified
                   enable_quiver=False)

    if file_name:
        fig.savefig(file_name)

    show_ = file_name is None if show is None else show
    if show_: _plt.show()

    return fig, axs

# ----- end of inspect_field -----

def show_regions(magnet,
                 out_of_plane_axis: Literal['x', 'y', 'z'] = 'z', layer: int = 0,
                 component: Optional[int] = None, geometry: Optional[_np.ndarray] = None,
                 field: Optional[_np.ndarray] = None,
                 file_name: Optional[str] = None, show: Optional[bool] = None,
                 ax: Optional[Axes] = None, figsize: Optional[tuple[float, float]] = None,
                 title: Optional[str] = None,
                 xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                 imshow_symmetric_clim: bool = False, imshow_kwargs: dict = {}) -> Axes:
    """Plot the boundaries between regions of the given magnet.
    For details about the function parameters, see :func:`plot_field`."""
    regions = magnet.regions
    boundaries = _np.zeros((1,) + regions.shape, dtype=int)

    boundaries[0, 1:, :, :] |= regions[1:, :, :] != regions[:-1, :, :]   # up
    boundaries[0, :, 1:, :] |= regions[:, 1:, :] != regions[:, :-1, :]   # forward
    boundaries[0, :, :, 1:] |= regions[:, :, 1:] != regions[:, :, :-1]   # right

    # Reuse _Plotter logic, but no need for quiver map or cbar
    if title is None:
        title = magnet.name + ": Region boundaries"
    plotter = _Plotter(boundaries, out_of_plane_axis, layer, component, geometry, field,
                       file_name, show, ax, figsize, title, xlabel, ylabel,
                       imshow_symmetric_clim, imshow_kwargs,
                       None, {},
                       None, None, None, None, {})
    return plotter.plot()



# ========== 3D pyvista plotting ==========

def show_magnet_geometry(magnet):
    """Show the geometry of a :func:`mumaxplus.Ferromagnet`."""
    geom = magnet.geometry

                 # [::-1] for [x,y,z] not [z,y,x] and +1 for cells, not points
    image_data = _pv.ImageData(dimensions=_np.array(geom.shape[::-1])+1,  
                 spacing=magnet.cellsize,
                 origin=_np.array(magnet.origin) - 0.5*_np.array(magnet.cellsize))
    image_data.cell_data["values"] = _np.float32(geom.flatten("C"))  # "C" because [z,y,x]
    threshed = image_data.threshold_percent(0.5)  # only show True

    plotter = _pv.Plotter()
    plotter.add_mesh(threshed, color="lightgrey",
                     show_edges=True, show_scalar_bar=False, lighting=False)
    plotter.add_title(f"{magnet.name} geometry")
    plotter.show_axes()
    plotter.view_xy()
    plotter.add_mesh(image_data.outline(), color="black", lighting=False)
    plotter.show()


def show_field_3D(quantity, cmap="HSL", enable_quiver=True, symmetric_clim=True):
    """Plot a :func:`mumaxplus.FieldQuantity` with 3 components as a vectorfield.

    Parameters
    ----------
    quantity : mumaxplus.FieldQuantity (3 components)
        The `FieldQuantity` to plot as a vectorfield.
    cmap : string, optional, default: "HSL"
        A colormap to use. By default an HSL colormap is used. This differs
        slightly from mumax³, see :func:`get_rgb`.
        Any matplotlib colormap can also be given to color the vectors according
        to their z-component. It's best to use diverging colormaps, like "bwr".
    enable_quiver : boolean, optional, default: True
        If set to True, a cone is placed at each cell indicating the direction.
        If False, colored voxels are used instead.
    symmetric_clim : bool, default=True
        Whether to have symmetric color limits if the given cmap is not "HSL".
    
    See Also
    --------
    get_rgb
    """

    if not isinstance(quantity, _mxp.FieldQuantity):
        raise TypeError("The first argument should be a FieldQuantity")

    if (quantity.ncomp != 3):
        raise ValueError("Can not create a vector field image because the field"
                         + " quantity does not have 3 components.")

    # set global theme, because individual plotter instance themes are broken
    _pv.global_theme = _pv.themes.DarkTheme()
    # the plotter
    plotter = _pv.Plotter()

    # make pyvista grid
    shape = quantity.shape[-1:0:-1]  # xyz not 3zyx
    cell_size = quantity._impl.system.cellsize
    image_data = _pv.ImageData(dimensions=_np.asarray(shape)+1,  # cells, not points
                 spacing=cell_size,
                 origin=_np.asarray(quantity._impl.system.origin)
                                    - 0.5*_np.asarray(cell_size))

    image_data.cell_data["field"] = quantity.eval().reshape((3, -1)).T  # cell data

    # don't show cells without geometry
    image_data.cell_data["geom"] = _np.float32(quantity._impl.system.geometry).flatten("C")
    threshed = image_data.threshold_percent(0.5, scalars="geom")

    if enable_quiver:  # use cones to display direction
        cres = 6  # number of vertices in cone base
        cone = _pv.Cone(center=(1/4, 0, 0), radius=0.32, height=1, resolution=cres)
        factor = min(cell_size[0:2]) if shape[2]==1 else min(cell_size)
        factor *= 0.95  # no touching
        factor /= _np.max(_np.linalg.norm(threshed["field"], axis=1))  # proper magnitude support

        quiver = threshed.glyph(orient="field", factor=factor, geom=cone)

        # color
        if "hsl" in cmap.lower():  # Use the HSL colorscheme
            # don't need to set opacity for geometry, threshold did this
            rgb = get_rgb(threshed["field"].T, OoP_axis_idx=None, layer=None, geometry=None)
            # we need to color every quiver vertex individually, each cone has cres+1
            quiver.point_data["rgb"] = _np.repeat(rgb, cres+1, axis=0)
            plotter.add_mesh(quiver, scalars="rgb", rgb=True, lighting=False)
        else:  # matplotlib colormap
            quiver.point_data["z-component"] = _np.repeat(threshed["field"][:,2], cres+1, axis=0)
            clim = None
            if symmetric_clim:
                vmax = _np.max(_np.abs(quiver["z-component"]))
                clim = (-vmax, vmax)
            plotter.add_mesh(quiver, scalars="z-component", cmap=cmap,
                             clim=clim, lighting=False)
    else:  # use colored voxels
        if "hsl" in cmap.lower():  # Use the HSL colorscheme
            # don't need to set opacity for geometry, threshold did this
            threshed.cell_data["rgb"] = get_rgb(threshed["field"].T,
                                   OoP_axis_idx=None, layer=None, geometry=None)
            plotter.add_mesh(threshed, scalars="rgb", rgb=True, lighting=False)
        else:  # matplotlib colormap
            threshed.cell_data["z-component"] = threshed["field"][:,2]
            clim = None
            if symmetric_clim:
                vmax = _np.max(_np.abs(threshed["z-component"]))
                clim = (-vmax, vmax)
            plotter.add_mesh(threshed, scalars="z-component", cmap=cmap,
                             clim=clim, lighting=False)

    # final touches
    plotter.add_mesh(image_data.outline(), color="white", lighting=False)
    plotter.add_title(quantity.name)
    plotter.show_axes()
    plotter.view_xy()
    plotter.set_background((0.3, 0.3, 0.3))  # otherwise black or white is invisible
    plotter.show()
    _pv.global_theme = _pv.themes.Theme()  # reset theme
