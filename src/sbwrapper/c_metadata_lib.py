"""Compatibility wrapper around :mod:`CMetadataLib`."""

from __future__ import annotations

from ._base_loader import export_public, load_base_module

_C_METADATA_LIB = load_base_module("CMetadataLib")

__all__ = export_public(_C_METADATA_LIB, globals())

