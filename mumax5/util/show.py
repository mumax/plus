"""Plotting helper functions."""

import matplotlib.pyplot as _plt
import numpy as _np

import mumax5 as _m5


def hsl_to_rgb(H, S, L):
    """Convert color from HSL to RGB."""
    Hp = H / (_np.pi / 3)
    if Hp < 0:
        Hp += 6.0
    elif Hp > 6.0:
        Hp -= 6.0
    if L <= 0.5:
        C = 2 * L * S
    else:
        C = 2 * (1 - L) * S

    X = C * (1 - _np.abs(_np.mod(Hp, 2.0) - 1.0))
    m = L - C / 2.0
    rgb = _np.array([m, m, m])
    if Hp >= 0 and Hp < 1:
        rgb += _np.array([C, X, 0])
    elif Hp < 2:
        rgb += _np.array([X, C, 0])
    elif Hp < 3:
        rgb += _np.array([0, C, X])
    elif Hp < 4:
        rgb += _np.array([0, X, C])
    elif Hp < 5:
        rgb += _np.array([X, 0, C])
    elif Hp < 6:
        rgb += _np.array([C, 0, X])
    else:
        rgb = _np.array([0, 0, 0])

    return rgb


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


def show_field(quantity, layer=0):
    """Plot a mumax5.FieldQuantity with 3 components using the mumax3 colorscheme."""
    if not isinstance(quantity, _m5.FieldQuantity):
        raise TypeError("The first argument should be a FieldQuantity")

    if quantity.ncomp != 3:
        raise ValueError(
            "Can not create a vector field image because the field quantity "
            + "does not have 3 components."
        )

    field = quantity.eval()
    field = field[:, layer]  # select the layer
    field /= _np.max(_np.linalg.norm(field, axis=0))  # rescale to make maximum norm 1

    # Create rgba image from the vector data
    _, ny, nx = field.shape  # image size
    rgba = _np.ones((ny, nx, 4))  # last index for R,G,B, and alpha channel
    for ix in range(nx):
        for iy in range(ny):
            rgba[iy, ix, 0:3] = vector_to_rgb(*field[:, iy, ix])

    # Set alpha channel to one inside the geometry, and zero outside
    rgba[:, :, 3] = quantity._impl.system.geometry[layer]

    fig = _plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(quantity.name)
    ax.set_facecolor("gray")
    ax.imshow(rgba, origin="lower", extent=_quantity_img_xy_extent(quantity))
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    _plt.show()


def show_layer(quantity, component=0, layer=0):
    """Visualize a single component of a mumax5.FieldQuantity."""
    f = quantity.eval()
    _plt.title(quantity.name)
    _plt.imshow(f[component, layer])
    _plt.show()
