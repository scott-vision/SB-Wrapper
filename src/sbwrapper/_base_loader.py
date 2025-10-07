"""Utilities for importing the pristine SlideBook base modules."""

from __future__ import annotations

import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Iterable

_BASE_ENV_VAR = "SBWRAPPER_BASE_PATH"


class BaseModuleImportError(ImportError):
    """Raised when the Base module directory cannot be located."""


@lru_cache(maxsize=1)
def get_base_path() -> Path:
    """Return the directory containing the unmodified SlideBook modules."""

    env_value = os.getenv(_BASE_ENV_VAR)
    if env_value:
        candidate = Path(env_value).expanduser().resolve()
        if not candidate.exists():
            raise BaseModuleImportError(
                f"Base module directory {candidate!s} from {_BASE_ENV_VAR} does not exist"
            )
        if not candidate.is_dir():
            raise BaseModuleImportError(
                f"Base module path {candidate!s} from {_BASE_ENV_VAR} is not a directory"
            )
        return candidate

    package_dir = Path(__file__).resolve()
    for parent in package_dir.parents:
        candidate = parent / "Base"
        if candidate.is_dir():
            return candidate.resolve()

    raise BaseModuleImportError(
        "Unable to locate the Base module directory. Set SBWRAPPER_BASE_PATH to the correct location."
    )


def _ensure_base_on_sys_path(base_path: Path) -> None:
    base_str = str(base_path)
    if base_str not in sys.path:
        sys.path.insert(0, base_str)


@lru_cache(maxsize=None)
def load_base_module(name: str) -> ModuleType:
    """Import and return a module from the Base directory."""

    base_path = get_base_path()
    _ensure_base_on_sys_path(base_path)

    module = importlib.import_module(name)
    module_path = getattr(module, "__file__", None)
    if module_path is None or not Path(module_path).resolve().is_file():
        raise BaseModuleImportError(
            f"Loaded module {name} does not appear to originate from the Base directory"
        )

    resolved_module_path = Path(module_path).resolve()
    try:
        resolved_module_path.relative_to(base_path)
    except ValueError as exc:  # pragma: no cover - defensive
        raise BaseModuleImportError(
            f"Module {name} was imported from {resolved_module_path}, which is outside of {base_path}"
        ) from exc

    return module


def export_public(module: ModuleType, target_globals: dict[str, object]) -> list[str]:
    """Copy public attributes from *module* into *target_globals*."""

    exported_names: Iterable[str]
    exported_names = getattr(module, "__all__", None) or [
        name
        for name in dir(module)
        if not name.startswith("_") and not isinstance(getattr(module, name), ModuleType)
    ]

    copied: list[str] = []
    for name in exported_names:
        target_globals[name] = getattr(module, name)
        copied.append(name)
    return copied


__all__ = [
    "BaseModuleImportError",
    "export_public",
    "get_base_path",
    "load_base_module",
]

