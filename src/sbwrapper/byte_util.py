"""Compatibility wrapper around :mod:`ByteUtil`."""

from __future__ import annotations

from ._base_loader import export_public, load_base_module

_BYTE_UTIL = load_base_module("ByteUtil")

__all__ = export_public(_BYTE_UTIL, globals())

