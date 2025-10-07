"""Central configuration helpers for SB Wrapper."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import yaml

# The default configuration file ships with the project root.  Deployments can
# replace the values in ``config.yaml`` with local settings.
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

# Cached configuration loaded from disk.  The file is only read once per
# process unless :func:`reload_settings` is called.
_file_settings: Optional[Mapping[str, Any]] = None

# Stack of in-memory overrides.  These are primarily intended for tests where
# we want predictable configuration without touching the on-disk file.
_override_stack: list[Mapping[str, Any]] = []


def _load_from_file(path: Path) -> Mapping[str, Any]:
    """Return the configuration mapping stored at ``path``."""

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, Mapping):
        raise ValueError("Configuration file must contain a mapping at the top level")

    return loaded


def reload_settings() -> None:
    """Clear the cached configuration so it is re-read from disk."""

    global _file_settings
    _file_settings = None


def reset_overrides() -> None:
    """Remove any active in-memory configuration overrides."""

    _override_stack.clear()


@contextmanager
def override_settings(settings: Mapping[str, Any]) -> Iterator[None]:
    """Temporarily replace configuration values within a context block."""

    _override_stack.append(settings)
    try:
        yield
    finally:
        _override_stack.pop()


def _active_settings() -> Mapping[str, Any]:
    """Return the currently active configuration mapping."""

    if _override_stack:
        return _override_stack[-1]

    global _file_settings
    if _file_settings is None:
        _file_settings = _load_from_file(_DEFAULT_CONFIG_PATH)

    return _file_settings


def get_setting(path: str, default: Optional[Any] = None) -> Any:
    """Fetch a configuration value using a dotted ``path`` notation."""

    if not path:
        return _active_settings()

    value: Any = _active_settings()
    for segment in path.split("."):
        if not isinstance(value, Mapping) or segment not in value:
            return default
        value = value[segment]
        if value is None:
            return default

    return value
