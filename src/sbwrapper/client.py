"""High-level helpers built on top of :class:`sbwrapper.sb_access.SBAccess`."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Mapping, Sequence, Tuple, Union

from .sb_access import MicroscopeStates, SBAccess

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Point:
    """Simple container describing a focus point in microns."""

    x: float
    y: float
    z: float
    aux_z: float | None = None
    use_aux: bool | None = None


@dataclass(frozen=True)
class StagePosition:
    """Represents the physical location of the microscope stage."""

    x: float
    y: float
    z: float
    aux_z: float | None = None


class MicroscopeClient:
    """User-friendly wrapper exposing common :class:`SBAccess` workflows."""

    DEFAULT_CHUNK_SIZE = 128

    def __init__(
        self,
        access: SBAccess,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        logger: logging.Logger | None = None,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self._access = access
        self._chunk_size = chunk_size
        self._logger = logger or LOGGER

    # ------------------------------------------------------------------
    # Point management helpers
    def clear_points(self) -> None:
        """Remove all stored XYZ points on the instrument."""

        self._logger.debug("Clearing XYZ points")
        self._access.ClearXYZPoints()

    def add_point(
        self,
        x: float,
        y: float,
        z: float,
        *,
        aux_z: float | None = None,
        use_aux: bool | None = None,
    ) -> None:
        """Add a single XYZ point to the queue."""

        aux_value = float(aux_z) if aux_z is not None else 0.0
        is_aux = bool(use_aux if use_aux is not None else aux_z is not None)
        self._logger.debug(
            "Adding XYZ point x=%s y=%s z=%s aux_z=%s is_aux=%s",
            x,
            y,
            z,
            aux_value,
            is_aux,
        )
        self._access.AddXYZPoint(float(x), float(y), float(z), aux_value, int(is_aux))

    def add_points(
        self,
        points: Iterable[Point | Mapping[str, object] | Sequence[object]],
        *,
        clear_existing: bool = False,
        chunk_size: int | None = None,
    ) -> None:
        """Upload multiple XYZ points to the microscope in batches."""

        if clear_existing:
            self.clear_points()

        size = self._chunk_size if chunk_size is None else chunk_size
        if size < 1:
            raise ValueError("chunk_size must be >= 1")

        batch: List[Tuple[float, float, float, float, bool]] = []
        for normalized in self._iter_normalized_points(points):
            batch.append(normalized)
            if len(batch) >= size:
                self._flush_point_batch(batch)
                batch.clear()
        if batch:
            self._flush_point_batch(batch)

    def get_points(self) -> list[str]:
        """Return the raw point list maintained by SlideBook."""

        points = self._access.GetXYZPointList()
        if not points:
            return []
        if len(points) == 1 and points[0].strip().lower() == "empty":
            return []
        return [entry for entry in points if entry]

    # ------------------------------------------------------------------
    # Microscope state helpers
    def get_microscope_state(
        self, state: MicroscopeStates | str
    ) -> Union[str, float, Tuple[float, float], bool]:
        """Fetch and coerce a microscope state into a convenient type."""

        if isinstance(state, str):
            try:
                state_enum = MicroscopeStates[state]
            except KeyError as exc:
                raise ValueError(f"Unknown microscope state: {state!r}") from exc
        else:
            state_enum = state

        result = self._access.GetMicroscopeState(state_enum)
        if state_enum in {
            MicroscopeStates.CurrentFLshutter,
            MicroscopeStates.CurrentBFshutter,
        }:
            return bool(result)
        if state_enum == MicroscopeStates.CurrentXYstagePosition:
            x, y = result
            return float(x), float(y)
        if state_enum in {
            MicroscopeStates.CurrentZstagePosition,
            MicroscopeStates.CurrentAltZstagePosition,
            MicroscopeStates.CurrentMagnification,
            MicroscopeStates.CurrentLaserPower,
            MicroscopeStates.CurrentNDPrimary,
            MicroscopeStates.CurrentNDAux,
            MicroscopeStates.CurrentLampVoltage,
        }:
            return float(result)
        return result

    def get_stage_position(self, include_aux: bool = False) -> StagePosition:
        """Return the current XY(Z) position of the microscope stage."""

        x, y = self.get_microscope_state(MicroscopeStates.CurrentXYstagePosition)
        z = self.get_microscope_state(MicroscopeStates.CurrentZstagePosition)
        aux_z: float | None = None
        if include_aux:
            try:
                aux_value = self.get_microscope_state(
                    MicroscopeStates.CurrentAltZstagePosition
                )
            except Exception:
                self._logger.debug("Auxiliary Z stage state unavailable", exc_info=True)
            else:
                aux_z = float(aux_value)
        return StagePosition(float(x), float(y), float(z), aux_z)

    def set_stage_position(
        self,
        *,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
        aux_z: float | None = None,
        relative: bool = False,
    ) -> None:
        """Move the microscope stage to the requested position."""

        if relative:
            self._apply_stage_delta(x, y, z, aux_z)
            return

        current = self.get_stage_position(include_aux=aux_z is not None)
        dx = None if x is None else float(x) - current.x
        dy = None if y is None else float(y) - current.y
        dz = None if z is None else float(z) - current.z
        daux = None if aux_z is None else float(aux_z) - (current.aux_z or 0.0)
        self._apply_stage_delta(dx, dy, dz, daux)

    # ------------------------------------------------------------------
    # Internal helpers
    def _iter_normalized_points(
        self, points: Iterable[Point | Mapping[str, object] | Sequence[object]]
    ) -> Iterator[Tuple[float, float, float, float, bool]]:
        for point in points:
            if isinstance(point, Point):
                yield self._normalize_point(
                    point.x, point.y, point.z, point.aux_z, point.use_aux
                )
                continue
            if isinstance(point, Mapping):
                x = point["x"]
                y = point["y"]
                z = point["z"]
                aux_z = point.get("aux_z")
                use_aux = point.get("use_aux")
                use_aux = point.get("is_aux_z", use_aux)  # compatibility alias
                yield self._normalize_point(x, y, z, aux_z, use_aux)
                continue
            if hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
                aux_z = getattr(point, "aux_z", None)
                use_aux = getattr(point, "use_aux", None)
                yield self._normalize_point(point.x, point.y, point.z, aux_z, use_aux)
                continue
            if isinstance(point, Sequence):
                seq = list(point)
                if len(seq) < 3:
                    raise ValueError("Point sequences must contain at least x, y, z")
                x, y, z = seq[:3]
                aux_z = seq[3] if len(seq) > 3 else None
                use_aux = seq[4] if len(seq) > 4 else None
                yield self._normalize_point(x, y, z, aux_z, use_aux)
                continue
            raise TypeError(f"Unsupported point representation: {point!r}")

    def _normalize_point(
        self,
        x: object,
        y: object,
        z: object,
        aux_z: object | None,
        use_aux: object | None,
    ) -> Tuple[float, float, float, float, bool]:
        fx = float(x)
        fy = float(y)
        fz = float(z)
        if aux_z is None:
            faux = 0.0
            is_aux = bool(use_aux) if use_aux is not None else False
        else:
            faux = float(aux_z)
            is_aux = bool(use_aux) if use_aux is not None else True
        return fx, fy, fz, faux, is_aux

    def _flush_point_batch(self, batch: Sequence[Tuple[float, float, float, float, bool]]):
        self._logger.debug("Uploading %s XYZ points", len(batch))
        for x, y, z, aux_z, is_aux in batch:
            self._access.AddXYZPoint(x, y, z, aux_z, int(is_aux))

    def _apply_stage_delta(
        self,
        dx: float | None,
        dy: float | None,
        dz: float | None,
        daux: float | None,
    ) -> None:
        if dx is not None and dx != 0:
            self._logger.debug("Moving stage X by %s", dx)
            self._access.FocusWindowMainMoveX(float(dx))
        if dy is not None and dy != 0:
            self._logger.debug("Moving stage Y by %s", dy)
            self._access.FocusWindowMainMoveY(float(dy))
        if dz is not None and dz != 0:
            self._logger.debug("Moving stage Z by %s", dz)
            self._access.FocusWindowMainMoveZPrimary(float(dz))
        if daux is not None and daux != 0:
            self._logger.debug("Moving auxiliary stage Z by %s", daux)
            self._access.FocusWindowMainMoveZAuxilary(float(daux))

