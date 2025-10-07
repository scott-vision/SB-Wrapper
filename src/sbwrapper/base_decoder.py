"""Compatibility wrapper around :mod:`BaseDecoder`."""

from __future__ import annotations

from ._base_loader import export_public, load_base_module

_BASE_DECODER = load_base_module("BaseDecoder")

__all__ = export_public(_BASE_DECODER, globals())

