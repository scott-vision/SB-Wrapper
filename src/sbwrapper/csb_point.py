"""Compatibility wrapper around :mod:`CSBPoint`."""

from __future__ import annotations

from ._base_loader import export_public, load_base_module

_CSB_POINT = load_base_module("CSBPoint")

__all__ = export_public(_CSB_POINT, globals())

