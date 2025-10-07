"""Compatibility wrapper around :mod:`SBAccess`."""

from __future__ import annotations

from ._base_loader import export_public, load_base_module

_SB_ACCESS = load_base_module("SBAccess")

__all__ = export_public(_SB_ACCESS, globals())

