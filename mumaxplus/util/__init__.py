"""Utilities for mumax⁺."""

from .config import (
    antivortex,
    blochskyrmion,
    gaussian_spherical_IP,
    gaussian_spherical_OoP,
    gaussian_uniform_IP,
    neelskyrmion,
    twodomain,
    vortex,
)
from .constants import GAMMALL_DEFAULT, HBAR, KB, MU0, MUB, QE
from .formulary import *
from .mfm import MFM
from .shape import *
from .show import (
    get_rgb,
    get_rgba,
    inspect_field,
    plot_field,
    show_field_3D,
    show_magnet_geometry,
    show_regions,
)
from .voronoi import VoronoiTessellator

__all__ = [
    # constants
    "GAMMALL_DEFAULT",
    "MU0",
    "KB",
    "QE",
    "MUB",
    "HBAR",
    # config
    "twodomain",
    "vortex",
    "antivortex",
    "neelskyrmion",
    "blochskyrmion",
    "gaussian_spherical_OoP",
    "gaussian_spherical_IP",
    "gaussian_uniform_IP",
    # formulary
    "magnetostatic_energy_density",
    "Km",
    "exchange_length",
    "l_ex",
    "wall_width",
    "helical_length",
    "magnetic_hardness",
    "bulk_modulus",
    "Rayleigh_damping_coefficients",
    "Rayleigh_damping_stiffness_coefficient",
    # show
    "get_rgb",
    "get_rgba",
    "plot_field",
    "inspect_field",
    "show_magnet_geometry",
    "show_field_3D",
    "show_regions",
    # misc
    "VoronoiTessellator",
    "MFM",
]
