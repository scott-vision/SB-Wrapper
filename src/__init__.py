"""Convenient access to the core SBSynergy components."""

from .microscope_client import MicroscopeClient, Point
from .SBPointFinder import PointFinder

try:  # Optional heavy dependency
    from .SBAccess import SBAccess
except Exception:  # pragma: no cover - dependency missing
    SBAccess = None  # type: ignore[assignment]

__all__ = [
    "MicroscopeClient",
    "Point",
    "PointFinder",
    "SBAccess",
]
