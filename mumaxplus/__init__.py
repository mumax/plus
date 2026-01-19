"""GPU accelerated micromagnetic simulator."""

import os


FP_PRECISION = os.environ.get("MUMAXPLUS_FP_PRECISION")
if not FP_PRECISION: FP_PRECISION = "SINGLE"
match FP_PRECISION.upper():
    case "SINGLE" | "1" | "32":
        import _mumaxpluscpp_single as _cpp
    case "DOUBLE" | "2" | "64":
        import _mumaxpluscpp_double as _cpp
    case _:
        raise RuntimeError(f"Unknown MUMAXPLUS_FP_PRECISION='{FP_PRECISION}'")

from .antiferromagnet import Antiferromagnet
from .dmitensor import DmiTensor
from .ferromagnet import Ferromagnet
from .fieldquantity import FieldQuantity
from .grid import Grid
from .interparameter import InterParameter
from .magnet import Magnet
from .ncafm import NcAfm
from .parameter import Parameter
from .poissonsystem import PoissonSystem
from .scalarquantity import ScalarQuantity
from .strayfield import StrayField
from .timesolver import TimeSolver
from .traction import BoundaryTraction
from .variable import Variable
from .world import World
from . import util

FP_PRECISION = {1: "SINGLE", 2: "DOUBLE"}.get(_cpp.FP_PRECISION, "UNKNOWN") # Use _cpp value, as that is certainly the correct one

__all__ = [
    "_cpp",
    "Antiferromagnet",
    "BoundaryTraction",
    "DmiTensor",
    "Ferromagnet",
    "FieldQuantity",
    "Grid",
    "InterParameter",
    "Magnet",
    "NcAfm",
    "Parameter",
    "ScalarQuantity",
    "StrayField",
    "TimeSolver",
    "Variable",
    "World",
    "PoissonSystem",
    "util",
    "FP_PRECISION"
]
