"""Plotting helper functions."""

import matplotlib.pyplot as _plt
import numpy as _np
import pyvista as _pv

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
    """Map vector (with norm<1) to RGB."""
    H = _np.arctan2(y, x)
    S = _np.sqrt(x ** 2 + y ** 2 + z ** 2)
    L = 0.5 + 0.5 * z
    return hsl_to_rgb(H, S, L)


def _quantity_img_xy_extent(quantity):
    """Return the extent for imshow images of xy cross sections..."""
    cx, cy, _ = quantity._impl.system.cellsize
    ox, oy, _ = quantity.grid.origin
    nx, ny, _ = quantity.grid.size
    extent = [
        cx * (ox - 0.5),
        cx * (ox - 0.5 + nx),
        cy * (oy - 0.5),
        cy * (oy - 0.5 + ny),
    ]
    return extent

def get_rgba(field, quantity=None, layer=None):
    """Get rgba values of given field.
    There is also a CUDA version of this function which utilizes the GPU.
    Use `quantity.get_rgb()`, but it has a different shape.

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


def show_field(quantity, layer=0):
    """Plot a mumaxplus.FieldQuantity with 3 components using the mumax³ colorscheme."""
    if not isinstance(quantity, _mxp.FieldQuantity):
        raise TypeError("The first argument should be a FieldQuantity")
    
    if (quantity.ncomp != 3):
        raise ValueError(
            "Can not create a vector field image because the field quantity "
            + "does not have 3 components."
        )

    field = quantity.eval()
    
    rgba = [get_rgba(field, quantity, layer)]
    plotter(quantity, rgba)


def plotter(quantity, rgba, name=""):
    fig = _plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(quantity.name + name)
    ax.set_facecolor("gray")
    ax.imshow(rgba[0], origin="lower", extent=_quantity_img_xy_extent(quantity))
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    _plt.show()



def show_layer(quantity, component=0, layer=0):
    """Visualize a single component of a mumaxplus.FieldQuantity."""
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
        field, cmap=cmap, origin="lower", extent=_quantity_img_xy_extent(quantity)
    )
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    _plt.show()

def show_neel_quiver(quantity, title=''):
    
    # Still needs some fudging...
    # Function uses ancient representation of Antiferromagnet.
    # This does NOT work in the current version of mumax⁺.

    field = quantity.eval()
    m1, m2 = field[0:3], field[3:6]
    mx1, my1, mz1 = m1[0][0], m1[1][0], m1[2][0]
    mx2, my2, mz2 = m2[0][0], m2[1][0], m2[2][0]

    mx = mx1 + mx2
    my = my1 + my2
    mz = mz1 + mz2

    # Normalizing to [-1, +1]
    if mx.max() > 1 or mx.min() < 1:
        mxx = 2 * (mx - mx.min()) / (mx.max() - mx.min()) - 1
    if my.max() > 1 or my.min() < 1:
        myy = 2 * (my - my.min()) / (my.max() - my.min()) - 1
    if mz.max() > 1 or mz.min() < 1:
        mzz = 2 * (mz - mz.min()) / (mz.max() - mz.min()) - 1
    
    #Plotting
    cmap = _plt.get_cmap('jet')
    norm = _plt.Normalize(mzz.min(), mzz.max())
    fig, ax = _plt.subplots()
    ax.quiver(mxx, myy)

    cax = ax.imshow(mzz, cmap=cmap, interpolation='none')
    cbar = _plt.colorbar(_plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)

    cbar.set_label('z-comp')
    if title:
        ax.set_title(title)
    _plt.show()


def show_magnet_geometry(magnet):
    """Show the geometry of a mumaxplus.ferromagnet."""
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
    """Plot a mumaxplus.FieldQuantity with 3 components as a vectorfield.

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

def show_regions(magnet, layer=0):
    """Plot the boundaries between regions of the given magnet."""
    regions_array = magnet.regions
    assert regions_array.ndim == 3, f"Expected 3D array, got {regions_array.ndim}D"

    regions = regions_array[layer]
    boundaries = _np.zeros_like(regions, dtype=bool)

    boundaries[1:, :] |= regions[1:, :] != regions[:-1, :]   # up
    boundaries[:, 1:] |= regions[:, 1:] != regions[:, :-1]   # left

    _plt.figure()
    _plt.imshow(~boundaries, cmap='gray', origin="lower", extent=_quantity_img_xy_extent(magnet))
    _plt.xlabel("$x$ (m)")
    _plt.ylabel("$y$ (m)")
    _plt.title(magnet.name + ":region_boundaries")
    _plt.show()