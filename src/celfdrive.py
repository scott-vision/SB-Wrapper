from __future__ import annotations

"""CelFDrive backend helpers for montage-based detection."""

from typing import Dict

from .services.celfdrive_workflow import detect_montage as _detect_montage
from .services.microscope import MicroscopeService


def detect_montage(
    ms: MicroscopeService,
    channel: int = 0,
    z: int = 0,
    max_project: bool = False,
    use_sahi: bool = False,
) -> Dict[str, object]:
    """Compatibility wrapper delegating to :mod:`services.celfdrive_workflow`."""

    detection = _detect_montage(
        ms,
        channel=channel,
        z=z,
        max_project=max_project,
        use_sahi=use_sahi,
    )
    return detection.to_dict()


__all__ = ["detect_montage"]
