"""Plotting helper functions."""

import matplotlib.pyplot as _plt
import matplotlib as _matplotlib
import numpy as _np
import math as _math
from itertools import product as _product
import pyvista as _pv

from numbers import Integral
from typing import Optional, Literal
from matplotlib.axes import Axes

import mumaxplus as _mxp


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
    """Map vector (with norm ≤ 1) to RGB."""
    H = _np.arctan2(y, x)
    S = _np.sqrt(x ** 2 + y ** 2 + z ** 2)
    L = 0.5 + 0.5 * z
    return hsl_to_rgb(H, S, L)


def get_rgba(field, quantity=None, layer=None):
    """Get rgba values of given field.
    There is also a CUDA version of this function which utilizes the GPU.
    Use :func:`mumaxplus.FieldQuantity.get_rgb()`, but it has a different shape.

    Parameters
    ----------
    quantity : FieldQuantity (default None)
        Used to set alpha value to 0 where geometry is False.
    layer : int (default None)
        z-layer of which to get rgba. Calculates rgba for all layers if None.

    Returns
    -------
    rgba : ndarray
        shape (ny, nx, 4) if layer is given, otherwise (nz, ny, nx, 4).
    """
    if layer is not None:
        field = field[:, layer]  # select the layer

    # rescale to make maximum norm 1
    data = field / _np.max(_np.linalg.norm(field, axis=0)) if _np.any(field) else field

    # Create rgba image from the vector data
    rgba = _np.ones((*(data.shape[1:]), 4))  # last index for R,G,B, and alpha channel
    rgba[...,0], rgba[...,1], rgba[...,2] = vector_to_rgb(data[0], data[1], data[2])

    # Set alpha channel to one inside the geometry, and zero outside
    if quantity is not None:
        geom = quantity._impl.system.geometry
        rgba[..., 3] = geom[layer] if layer is not None else geom

    return rgba


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

def _quantity_2D_extent(fieldquantity: _mxp.FieldQuantity|_np.ndarray,
                        hor_axis_idx: int = 0, vert_axis_idx: int = 1) \
                            -> Optional[tuple[int, int, int, int]]:
    """If the given fieldquantity has an extent, the two dimensional extent is
    given as (left, right, bottom, top), with chosen indices for the horizontal
    and vertical axes. Returns None if the extent can't be determined.
    """
    if isinstance(fieldquantity, _mxp.FieldQuantity):
        qty_extent = fieldquantity._impl.system.extent
        left, right = qty_extent[2*hor_axis_idx], qty_extent[2*hor_axis_idx + 1]
        bottom, top = qty_extent[2*vert_axis_idx], qty_extent[2*vert_axis_idx + 1]
        return (left, right, bottom, top)
    if isinstance(fieldquantity, _np.ndarray):
        left, right = -0.5, fieldquantity.shape[3 - hor_axis_idx] - 0.5
        bottom, top = -0.5, fieldquantity.shape[3 - vert_axis_idx] - 0.5
        return (left, right, bottom, top)
    return None

# TODO: better name
# TODO: docstring
# TODO: vectorize?
def _get_fraction(o_i: int, n_i: int, r: float) -> float:
    # old index, new index, old-new shape ratio
    n_l, n_r = n_i * r, (n_i + 1) * r  # left and right boundaries of new cell

    if n_l <= o_i and o_i + 1 <= n_r:  # fully inside new cell
        return 1.
    if n_l <= o_i < n_r:  # only left side inside new cell
        return n_r - o_i
    if n_l < o_i + 1 <= n_r:  # only right side inside new cell
        return o_i + 1 - n_l
    return 0.  # not inside


def _get_downsampled_meshgrid(old_size: tuple[int, int], new_size: tuple[int, int],
                              quantity: Optional[_mxp.FieldQuantity] = None,
                              hor_axis_idx: int = 0, vert_axis_idx: int = 1) \
                                -> tuple[_np.ndarray, _np.ndarray]:
    # TODO: docstring
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

def downsample(field: _np.ndarray, new_size: tuple, intrinsic: bool = True) -> _np.ndarray:
    """Placeholder docstring so I don't forget
    field is an array with shape (ncomp, nz, ny, nx)
    new_size is the grid size that should be used for the result as (nx, ny, nz) (other way!)
    The edges are lined-up, old small cells are divided between the new large
    cells according to the fraction of the area/volume inside each of the
    covering large cells.
    Intrinsic means we take the average. If False, extrinsic, so everything is summed up but not divided.
    """
    # TODO: better docstring
    # TODO: explain "downsampling"
    # TODO: field is assumed ndarray. Should it also accept FieldQuantities?
    # TODO: explain the difference between intrinsic and extrinsic quantities...
    # The use will almost always be for intrinsic quantities, which need to be divided in the end (averaging)
    # TODO: can (and should) this be CUDA-fied?

    ncomp = field.shape[0]
    old_shape = field.shape[1:]
    dim = len(old_shape)
    new_shape = new_size[::-1]  # TODO: new shape should be (nx, ny, nz)? grid size or field shape?
    # TODO: check length of new_shape, check positive, check <= old, check integers
    new_field = _np.zeros((ncomp, *new_shape))

    on_shape_ratio = [old_shape[i] / new_shape[i] for i in range(dim)]

    # TODO: vectorize?
    for idx_new in _np.ndindex(new_shape):  # loop over all new cells
        sum = _np.zeros((ncomp))
        if intrinsic: denom = _np.zeros((ncomp))

        # loop over all old cells that are at least partly inside the new cell 
        old_ranges = [range(int(i_new * ratio), _math.ceil((i_new + 1) * ratio))
                      for (i_new, ratio) in zip(idx_new, on_shape_ratio)]
        for idx_old in _product(*old_ranges):

            # find fraction of the length/area/volume inside
            frac = 1.0
            for i in range(dim):
                frac *= _get_fraction(idx_old[i], idx_new[i], on_shape_ratio[i])

            f = field[:, *idx_old]
            sum += frac * f
            if intrinsic: denom += frac
        
        new_field[:, *idx_new] = sum / denom if intrinsic else sum

    return new_field

# --------------------------------------------------
# This code is copied from the Hotspice code written by Jonathan Maes and slightly tweaked.
# https://github.com/bvwaeyen/Hotspice/blob/main/hotspice/utils.py#L129

SIprefix_to_magnitude = {'q': -30, 'r': -27, 'y': -24, 'z': -21, 'a': -18,
                         'f': -15, 'p': -12, 'n': -9, 'µ': -6, 'm': -3, 'c': -2, 'd': -1,
                         '': 0, 'da': 1, 'h': 2, 'k': 3, 'M': 6, 'G': 9, 'T': 12,
                         'E': 15, 'Z': 18, 'Y': 21, 'R': 24, 'Q': 30}

def SIprefix_to_mul(unit: Literal['f', 'p', 'n', 'µ', 'm', 'c', 'd', '', 'da', 'h', 'k', 'M', 'G', 'T']) -> float:
    return 10**SIprefix_to_magnitude[unit]

magnitude_to_SIprefix = {v: k for k, v in SIprefix_to_magnitude.items()}

def appropriate_SIprefix(n: float|_np.ndarray,
                         unit_prefix: Literal['f', 'p', 'n', 'µ', 'm', 'c', 'd', '', 'da', 'h', 'k', 'M', 'G', 'T']='',
                         only_thousands=True) -> tuple[float, str]:
    """ Converts `n` (which already has SI prefix `unit_prefix` for whatever unit
        it is in) to a reasonable number with a new SI prefix. Returns a tuple
        with (the new scalar values, the new SI prefix).If `only_thousands` is
        True (default), then centi, deci, deca and hecto are not used.
        Example: converting 0.0000238 ms would be `appropriate_SIprefix(0.0000238, 'm')` -> `(23.8, 'n')`
    """
    # If `n` is an array, the median is usually representative of the scale
    value = _np.median(n) if isinstance(n, _np.ndarray) else n

    if unit_prefix not in SIprefix_to_magnitude.keys():
        raise ValueError(f"'{unit_prefix}' is not a supported SI prefix.")
    
    offset_magnitude = SIprefix_to_magnitude[unit_prefix]
    # TODO: I personally think 'floor' looks nicer than 'round'; less decimal numbers
    nearest_magnitude = (round(_np.log10(abs(value))) if value != 0 else -_np.inf) + offset_magnitude
    # Make sure it is in the known range
    nearest_magnitude = _np.clip(nearest_magnitude,
                                min(SIprefix_to_magnitude.values()),
                                max(SIprefix_to_magnitude.values()))
    
    supported_magnitudes = magnitude_to_SIprefix.keys()
    if only_thousands: supported_magnitudes = [mag for mag in supported_magnitudes if (mag % 3) == 0]
    for supported_magnitude in sorted(supported_magnitudes):
        if supported_magnitude <= nearest_magnitude:
            used_magnitude = supported_magnitude
    
    return (n/10**(used_magnitude - offset_magnitude), magnitude_to_SIprefix[used_magnitude])

# --------------------------------------------------

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
    :func:`plot_vector_field`, but not to be instantiated or manipulated by the
    user.
    """
    def __init__(self, fieldquantity, out_of_plane_axis, layer,
                 file_name, show, ax, imshow_cmap, symmetric_clim,
                 quiver, arrow_size, quiver_cmap, **quiver_kwargs):
        """see the docstring of :func:`plot_vector_field`."""

        # check fieldquantity input
        if not isinstance(fieldquantity, _mxp.FieldQuantity) and \
           not isinstance(fieldquantity, _np.ndarray):
            raise TypeError("The first argument should be a FieldQuantity or an ndarray.")
        
        if (fieldquantity.shape[0] != 3):
            raise ValueError(
                "Can not create a vector field image because the field quantity "
                + "does not have 3 components."
            )

        if len(fieldquantity.shape) != 4:
            raise ValueError(
                "The field quantity has the wrong number of dimensions: "
                + f"{len(fieldquantity.shape)} instead of 4."
            )

        self.out_of_plane_axis, self.layer = out_of_plane_axis, layer
        self.hor_axis_idx, self.vert_axis_idx, self.OoP_idx = _get_axis_components(out_of_plane_axis)

        self.file_name = file_name

        if show is None:
            if ax is None and file_name is None:
                self.show = True
            else:
                self.show = False

        if ax is None:
            _, self.ax = _plt.subplots()
        else:
            self.ax = ax

        # split types to know what we're working with
        if isinstance(fieldquantity, _mxp.FieldQuantity):
            self.field = fieldquantity.eval()
            self.quantity = fieldquantity
        else:
            self.field = _np.copy(fieldquantity)
            self.quantity = None

        self.set_field_2D()

        self.imshow_cmap = imshow_cmap
        self.symmetric_clim = symmetric_clim
        self.quiver = quiver
        self.arrow_size = arrow_size
        self.quiver_cmap = quiver_cmap
        self.quiver_kwargs = quiver_kwargs.copy()  # leave user input alone
        self.quiver_kwargs.setdefault("pivot", "middle")

        # TODO: tweak?
        self.max_width_over_height_ratio = 6
        self.max_height_over_width_ratio = 3


    def set_field_2D(self):
        # TODO: should be a function, but field_2D should not be a property
        # make field_2D with (ncomp, vert_axis, hor_axis) shape
        slice_ = [slice(None)]*4
        slice_[3 - self.OoP_idx] = self.layer  # slice correct axis at chosen layer
        self.field_2D = self.field[tuple(slice_)]
        if self.out_of_plane_axis == 'y':
            self.field_2D = _np.swapaxes(self.field_2D, 1, 2)  # (ncomp, nx, nz)  for right-hand axes

    def plot_image(self):
        # imshow
        im_extent = _quantity_2D_extent(self.quantity, self.hor_axis_idx, self.vert_axis_idx)
        if self.imshow_cmap == "mumax3":
            # TODO: update get_rgba
            im_rgba = get_rgba(self.field_2D)  # (y_axis, x_axis, rgba)
            self.ax.imshow(im_rgba, origin="lower", extent=im_extent)
        else:  # show out of plane component
            field_OoP = self.field_2D[self.OoP_idx]
            vmin, vmax = None, None
            if self.symmetric_clim:
                vmax = _np.max(_np.abs(field_OoP))
                vmin = -vmax
            self.ax.imshow(field_OoP, origin="lower", extent=im_extent,
                    cmap=self.imshow_cmap, vmin=vmin, vmax=vmax)
    
    def plot_quiver(self):
        if self.quiver:
            _, ny_old, nx_old = self.field_2D.shape
            nx_new = max(int(nx_old / self.arrow_size), 1)
            ny_new = max(int(ny_old / self.arrow_size), 1)

            X, Y = _get_downsampled_meshgrid((nx_old, ny_old), (nx_new, ny_new), self.quantity,
                                            self.hor_axis_idx, self.vert_axis_idx)

            sampled_field = downsample(self.field_2D, new_size=(nx_new, ny_new))
            U, V = sampled_field[self.hor_axis_idx], sampled_field[self.vert_axis_idx]

            if self.quiver_cmap == "mumax3":  # HSL with rgb
                q_rgba = _np.reshape(get_rgba(sampled_field), (nx_new*ny_new, 4))
                self.ax.quiver(X, Y, U, V, color=q_rgba, **self.quiver_kwargs)
            elif self.quiver_cmap == None:  # uniform color
                self.quiver_kwargs.setdefault("alpha", 0.4)
                self.ax.quiver(X, Y, U, V, **self.quiver_kwargs)
            else:  # OoP component colored
                sampled_field_OoP = sampled_field[self.OoP_idx]
                vmin, vmax = None, None
                if self.symmetric_clim:
                    vmax = _np.max(_np.abs(sampled_field_OoP))
                    vmin = -vmax
                self.quiver_kwargs.setdefault("clim", (vmin, vmax))
                self.ax.quiver(X, Y, U, V, sampled_field_OoP, cmap=self.quiver_cmap,
                               **self.quiver_kwargs)

    def dress_axes(self):
        # TODO: docstring
        # TODO: better title using name
        # TODO: geometry for filter field??
        # TODO: check validity of axis indices
        
        left, right, bottom, top = _quantity_2D_extent(
            self.quantity if self.quantity else self.field,
            self.hor_axis_idx, self.vert_axis_idx)

        # axis limits
        self.ax.set_xlim(left, right)
        self.ax.set_ylim(bottom, top)

        # axis labels
        if self.quantity:
            x_maxabs = max(abs(left), abs(right))  # find largest number
            y_maxabs = max(abs(bottom), abs(top))
            _, x_prefix = appropriate_SIprefix(x_maxabs)
            _, y_prefix = appropriate_SIprefix(y_maxabs)
            self.ax.set_xlabel(f"${'xyz'[self.hor_axis_idx]}$ ({x_prefix}m)")
            self.ax.set_ylabel(f"${'xyz'[self.vert_axis_idx]}$ ({y_prefix}m)")

            # axis label ticks with appropriate numbers according to prefix
            self.ax.xaxis.set_major_formatter(UnitScalarFormatter(x_prefix, "m"))
            self.ax.yaxis.set_major_formatter(UnitScalarFormatter(y_prefix, "m"))

            # set title to fieldquantity name
            self.ax.set_title(self.quantity.name)  # TODO: add slice, component, layer, ...
        else:
            self.ax.set_xlabel(f"${'xyz'[self.hor_axis_idx]}$ (index)")
            self.ax.set_ylabel(f"${'xyz'[self.vert_axis_idx]}$ (index)")

        self.ax.set_facecolor("gray")  # TODO: is this still relevant?
        
        # use "equal" aspect ratio if not too rectangular
        if ((right - left) / (top - bottom) < self.max_width_over_height_ratio and
            (top - bottom) / (right - left) < self.max_height_over_width_ratio):
            self.ax.set_aspect("equal")
        else:
            self.ax.set_aspect("auto")


    def plot_vector_field(self) -> Axes:
        # TODO: docstring

        self.plot_image()

        self.plot_quiver()

        self.dress_axes()
        # TODO: make a beautiful title

        if self.file_name:
            self.ax.figure.savefig(self.file_name)

        if self.show:
            _plt.show()

        return self.ax


def plot_vector_field(fieldquantity: _mxp.FieldQuantity|_np.ndarray,
                      out_of_plane_axis: str = 'z', layer: int = 0,
                      file_name: Optional[str] = None, show: Optional[bool] = None,
                      ax: Optional[Axes] = None,
                      imshow_cmap: str = "mumax3", symmetric_clim: bool = True,
                      quiver: bool = True, arrow_size: float = 16.,
                      quiver_cmap: Optional[str]= None, **quiver_kwargs) -> Axes:
    """Plot a :func:`mumaxplus.FieldQuantity` or `numpy.ndarray` with 3
    components as a vector field using the mumax³ (HSL) colorscheme or a scalar
    colormap of the out-of-plane component, with optionally added arrows.
    
    Parameters
    ----------
    fieldquantity : mumaxplus.FieldQuantity or numpy.ndarray
        The fieldquantity needs to have 4 dimensions with the shape
        (ncomp, nx, ny, nz) and with ncomp=3.
        Additional dressing of the Axes is done if given a mumaxplus.FieldQuantity.

    out_of_plane_axis : string, default="z"
        The axis pointing out of plane: "x", "y" or "z". The other two are plotted according to a right-handed coordinate system.

        - "x": z over y
        - "y": x over z
        - "z": y over x

    layer : int, default=0
        The index to take of the `out_of_plane_axis`.

    file_name : string, optional
        If given, the resulting figure will be saved with the given file name.

    show : bool, optional
        Whether to call `matplotlib.pyplot.show` at the end.
        If None (default), `matplotlib.pyplot.show` will be called only if no
        `ax` and no `file_name` have been provided.

    ax : matplotlib.axes.Axes, optional
        The Axes instance to plot onto. If None (default), new Figure and Axes
        instances will be created.

    imshow_cmap : string, default="mumax3"
        A colormap to use for the image. By default, the mumax³ (HSL)
        colorscheme is used. Any matplotlib colormap can also be given to color
        according to the out-of-plane component.

    symmetric_clim : bool, default=True
        If a matplotlib colormap is used for the out-of-plane component for
        either the image or quiver, zero is set as the central color if set to
        True (default). This is best used with diverging colormaps, like "bwr".

    quiver : bool, default=True
        Whether to plot the quiver on top of the colored image.

    arrow_size : float, default=16
        Length of an arrow as a number of cells, so one arrow is designated to
        an area of `arrow_size` by `arrow_size`.
        
    quiver_cmap : string, optional
        A colormap to use for the quiver. By default, no colormap is used, so
        the arrows are a solid color. If set to "mumax3", the 3D vector data is
        used for the mumax³ (HSL) colorscheme. Any matplotlib colormap can also
        be given to color the arrows according to the out-of-plane component.

    **quiver_kwargs
        Keyword arguments to pass to `matplotlib.pyplot.quiver`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The resulting Axes on which is plotted.
    """
    plotter = _Plotter(fieldquantity, out_of_plane_axis, layer,
                       file_name, show, ax, imshow_cmap, symmetric_clim,
                       quiver, arrow_size, quiver_cmap, **quiver_kwargs)
    return plotter.plot_vector_field()


def show_layer(quantity, component=0, layer=0):
    """Visualize a single component of a :func:`mumaxplus.FieldQuantity`."""
    if not isinstance(quantity, _mxp.FieldQuantity):
        raise TypeError("The first argument should be a FieldQuantity")

    field = quantity.eval()
    field = field[component, layer]  # select component and layer

    geometry = quantity._impl.system.geometry[layer]
    field = _np.ma.array(field, mask=_np.invert(geometry))

    cmap = _plt.get_cmap("viridis")
    cmap.set_bad(alpha=0.0)  # This will affect cells outside the mask (i.e. geometry)

    fig = _plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor("gray")
    ax.set_title(quantity.name + ", component=%d" % component)
    ax.imshow(
        field, cmap=cmap, origin="lower", extent=_quantity_2D_extent(quantity)
    )
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    _plt.show()


def show_regions(magnet, layer=0):
    """Plot the boundaries between regions of the given magnet."""
    regions_array = magnet.regions
    assert regions_array.ndim == 3, f"Expected 3D array, got {regions_array.ndim}D"

    regions = regions_array[layer]
    boundaries = _np.zeros_like(regions, dtype=bool)

    boundaries[1:, :] |= regions[1:, :] != regions[:-1, :]   # up
    boundaries[:, 1:] |= regions[:, 1:] != regions[:, :-1]   # left

    _plt.figure()
    _plt.imshow(~boundaries, cmap='gray', origin="lower", extent=_quantity_2D_extent(magnet))
    _plt.xlabel("$x$ (m)")
    _plt.ylabel("$y$ (m)")
    _plt.title(magnet.name + ":region_boundaries")
    _plt.show()


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


def show_field_3D(quantity, cmap="mumax3", quiver=True):
    """Plot a :func:`mumaxplus.FieldQuantity` with 3 components as a vectorfield.

    Parameters
    ----------
    quantity : mumaxplus.FieldQuantity (3 components)
        The `FieldQuantity` to plot as a vectorfield.
    cmap : string, optional, default: "mumax3"
        A colormap to use. By default the mumax³ colormap is used.
        Any matplotlib colormap can also be given to color the vectors according
        to their z-component. It's best to use diverging colormaps, like "bwr".
    quiver : boolean, optional, default: True
        If set to True, a cone is placed at each cell indicating the direction.
        If False, colored voxels are used instead.
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

    if quiver:  # use cones to display direction
        cres = 6  # number of vertices in cone base
        cone = _pv.Cone(center=(1/4, 0, 0), radius=0.32, height=1, resolution=cres)
        factor = min(cell_size[0:2]) if shape[2]==1 else min(cell_size)
        factor *= 0.95  # no touching

        quiver = threshed.glyph(orient="field", scale=False, factor=factor, geom=cone)

        # color
        if "mumax" in cmap.lower():  # Use the mumax³ colorscheme
            # don't need quantity to set opacity for geometry, threshold did this
            rgba = get_rgba(threshed["field"].T, quantity=None, layer=None)
            # we need to color every quiver vertex individually, each cone has cres+1
            quiver.point_data["rgba"] = _np.repeat(rgba, cres+1, axis=0)
            plotter.add_mesh(quiver, scalars="rgba", rgba=True, lighting=False)
        else:  # matplotlib colormap
            quiver.point_data["z-component"] = _np.repeat(threshed["field"][:,2], cres+1, axis=0)
            plotter.add_mesh(quiver, scalars="z-component", cmap=cmap,
                             clim=(-1,1), lighting=False)
    else:  # use colored voxels
        if "mumax" in cmap.lower():  # Use the mumax³ colorscheme
            # don't need quantity to set opacity for geometry, threshold did this
            threshed.cell_data["rgba"] = get_rgba(threshed["field"].T,
                                                  quantity=None, layer=None)
            plotter.add_mesh(threshed, scalars="rgba", rgba=True, lighting=False)
        else:  # matplotlib colormap
            threshed.cell_data["z-component"] = threshed["field"][:,2]
            plotter.add_mesh(threshed, scalars="z-component", cmap=cmap,
                             clim=(-1,1), lighting=False)

    # final touches
    plotter.add_mesh(image_data.outline(), color="white", lighting=False)
    plotter.add_title(quantity.name)
    plotter.show_axes()
    plotter.view_xy()
    plotter.set_background((0.3, 0.3, 0.3))  # otherwise black or white is invisible
    plotter.show()
    _pv.global_theme = _pv.themes.Theme()  # reset theme
