"""Plotting helper functions."""

import matplotlib.pyplot as plt
import numpy as np


def vectorfield_to_rgb(field):
    """Map field modulus values to RGB."""
    field /= np.max(np.linalg.norm(field, axis=0))
    rgb = np.zeros((*(field[0].shape), 3))
    for ix in range(field.shape[3]):
        for iy in range(field.shape[2]):
            for iz in range(field.shape[1]):
                fx, fy, fz = field[:, iz, iy, ix]
                H = np.arctan2(fy, fx)
                S = 1.0
                L = 0.5 + 0.5 * fz
                Hp = H / (np.pi / 3)
                if Hp < 0:
                    Hp += 6.0
                elif Hp > 6.0:
                    Hp -= 6.0
                if L <= 0.5:
                    C = 2 * L * S
                else:
                    C = 2 * (1 - L) * S

                X = C * (1 - np.abs(np.mod(Hp, 2.0) - 1.0))

                m = L - C / 2.0
                rgbcell = np.array([m, m, m])
                if Hp >= 0 and Hp < 1:
                    rgbcell += np.array([C, X, 0])
                elif Hp < 2:
                    rgbcell += np.array([X, C, 0])
                elif Hp < 3:
                    rgbcell += np.array([0, C, X])
                elif Hp < 4:
                    rgbcell += np.array([0, X, C])
                elif Hp < 5:
                    rgbcell += np.array([X, 0, C])
                elif Hp < 6:
                    rgbcell += np.array([C, 0, X])
                else:
                    rgbcell = np.array([0, 0, 0])
                rgb[iz, iy, ix, :] = rgbcell
    return rgb


def show_field(quantity, layer=0):
    """Plot a mumax5.FieldQuantity with 3 components using the mumax3 colorscheme."""
    plt.title(quantity.name)
    rgb = vectorfield_to_rgb(quantity.eval())
    plt.imshow(rgb[layer])
    plt.show()


def show_layer(quantity, component=0, layer=0):
    """Visualize a single component of a mumax5.FieldQuantity."""
    f = quantity.eval()
    plt.title(quantity.name)
    plt.imshow(f[component, layer])
    plt.show()
