import logging
from typing import Dict, List, Optional, Tuple, Union

from .SBAccess import MicroscopeHardwareComponent, MicroscopeStates


class ObjectiveManager:
    """High-level helper for reading and setting microscope objectives.

    Parameters
    ----------
    sb_access:
        An :class:`SBAccess` instance providing access to microscope
        functions like ``GetObjectives`` and ``FocusWindowScopeSelectObjective``.
    ensure_connection:
        Callable returning ``True`` when connected; checked before hardware
        operations.
    logger:
        Optional logger instance.
    """

    def __init__(self, sb_access, ensure_connection=lambda: True, logger: Optional[logging.Logger] = None):
        self.sb = sb_access
        self.ensure_connection = ensure_connection
        self.log = logger or logging.getLogger(__name__)
        self._name_to_pos: Dict[str, int] = {}
        self._pos_to_name: Dict[int, str] = {}

    # -------------------- public API --------------------

    def get_current_objective(self) -> Optional[Tuple[str, int]]:
        """Return ``(objective_name, turret_position)`` if determinable."""
        if not self._check():
            return None

        self._refresh_cache_safe()

        try:
            name = self.sb.GetMicroscopeState(MicroscopeStates.CurrentObjective)
            if isinstance(name, str):
                name = name.strip()
                if name:
                    pos = self._name_to_pos.get(name)
                    if pos is not None:
                        return (name, pos)
                    self._refresh_cache_safe()
                    pos = self._name_to_pos.get(name)
                    if pos is not None:
                        return (name, pos)
                    self.log.debug(
                        "Objective name '%s' not in cache; falling back to turret position.",
                        name,
                    )
        except Exception as e:  # pragma: no cover - debug logging only
            self.log.debug("Could not read MicroscopeStates.CurrentObjective: %s", e)

        try:
            pos_raw = self.sb.GetHardwareComponentPosition(
                MicroscopeHardwareComponent.ObjectiveTurret
            )
            pos = self._extract_int(pos_raw)
            if pos is not None:
                self._refresh_cache_safe()
                name = self._pos_to_name.get(pos)
                return (name if name else f"Unknown@{pos}", pos)
        except Exception as e:
            self.log.warning("Could not read objective turret position: %s", e)

        return None

    def set_objective(self, objective: Union[str, int]) -> bool:
        """Set the objective by name or turret position using SlideBook's
        ``FocusWindowScopeSelectObjective`` command."""
        if not self._check():
            return False

        try:
            if isinstance(objective, str):
                name = objective.strip()
            elif isinstance(objective, int):
                self._refresh_cache_safe()
                name = self._pos_to_name.get(objective)
                if name is None:
                    self._refresh_cache_safe(force=True)
                    name = self._pos_to_name.get(objective)
                if name is None:
                    self.log.error(
                        "Objective position '%s' not found among available objectives: %s",
                        objective,
                        sorted(self._pos_to_name.keys()),
                    )
                    return False
            else:
                self.log.error("Unsupported objective specifier: %r", objective)
                return False

            success = self.sb.FocusWindowScopeSelectObjective(name)
            if success:
                self.log.info("Objective set to '%s'", name)
            else:
                self.log.warning("Failed to set objective to '%s'", name)
            return bool(success)
        except Exception as e:
            self.log.error("Error setting objective '%s': %s", objective, e)
            return False

    def list_objectives(self) -> List[Tuple[str, int]]:
        """Return a list of available objectives as ``(name, turret_position)``
        pairs using ``GetObjectives``."""
        if not self._check():
            return []
        self._refresh_cache_safe(force=True)
        return sorted(self._name_to_pos.items(), key=lambda kv: kv[1])

    # -------------------- internals --------------------

    def _check(self) -> bool:
        try:
            ok = bool(self.ensure_connection())
            if not ok:
                self.log.debug("Connection check failed.")
            return ok
        except Exception as e:
            self.log.error("ensure_connection raised: %s", e)
            return False

    def _refresh_cache_safe(self, force: bool = False) -> None:
        """Refresh internal caches from hardware."""
        if not force and self._name_to_pos and self._pos_to_name:
            return
        try:
            objectives = self.sb.GetObjectives()
            name_to_pos: Dict[str, int] = {}
            pos_to_name: Dict[int, str] = {}
            for lens in objectives:
                try:
                    name = getattr(lens, "mName", None)
                    pos = getattr(lens, "mTurretPosition", None)
                    if name is None or pos is None:
                        continue
                    pos = int(pos)
                    name_to_pos[str(name)] = pos
                    pos_to_name[pos] = str(name)
                except Exception as inner_e:  # pragma: no cover - debug only
                    self.log.debug("Skipping objective due to parse error: %s", inner_e)
            if name_to_pos:
                self._name_to_pos = name_to_pos
                self._pos_to_name = pos_to_name
            else:
                self.log.warning("Objective cache refresh produced no entries.")
        except Exception as e:
            self.log.error("Failed to refresh objective list: %s", e)

    @staticmethod
    def _extract_int(value) -> Optional[int]:
        """Extract an ``int`` from SBAccess responses."""
        try:
            if isinstance(value, tuple) and len(value) >= 1:
                value = value[0]
            if hasattr(value, "item"):
                value = value.item()
            return int(value)
        except Exception:
            return None
