"""A web GUI for mumax+ (live field view, parameters, solver control)."""

from .server import GUI, serve

__all__ = ["GUI", "serve"]
__version__ = "0.1.0"
