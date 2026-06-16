from __future__ import annotations

from typing import Any, Mapping


class TelemetrySchemaError(RuntimeError):
    """Raised when a scientific gate cannot be evaluated from telemetry."""


def read_required_int(
    row: Mapping[str, Any],
    *,
    canonical: str,
    legacy_aliases: list[str] | None = None,
) -> int:
    """Read an integer telemetry field without silently converting absence to failure.

    The canonical schema wins. Legacy aliases are only for compatibility with
    older artifacts. If none exists, the run is infrastructure-invalid rather
    than a scientific negative.
    """
    keys = [canonical, *(legacy_aliases or [])]
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return int(float(value))
            except (TypeError, ValueError) as exc:
                raise TelemetrySchemaError(f"{key} is not an integer: {value!r}") from exc
    raise TelemetrySchemaError(f"missing required telemetry field: {canonical}")
