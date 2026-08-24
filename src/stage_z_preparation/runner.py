"""Future execution entry-point guards and engineering-canary scaffolds."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .contract import ExecutionAuthorization, StageZHold, require_execution_authorized


def run_authorized_callback(
    *,
    authorization: ExecutionAuthorization,
    parent_key: str,
    phase: str,
    callback: Callable[[], Any],
) -> Any:
    """Guard before invoking any future model/simulator callback.

    The callback is deliberately opaque here.  Static tests can prove it is
    not touched while Z0R2 is HOLD; future runtime code must still bind its
    model and environment implementations behind this guard.
    """

    require_execution_authorized(authorization, parent_key=parent_key, phase=phase)
    return callback()


def require_engineering_canary(identity: str, frozen_scientific_parents: Iterable[str]) -> None:
    if not identity or identity in set(frozen_scientific_parents):
        raise StageZHold("ENGINEERING_CANARY_OVERLAPS_SCIENTIFIC_PANEL")
    if "/task_" not in identity or "/state_" not in identity:
        raise StageZHold("ENGINEERING_CANARY_IDENTITY_INVALID")


def reject_real_runtime_in_preparation(*, model_loader: Callable[[], Any] | None = None) -> None:
    """Explicit preparation-only boundary; real loaders are never called here."""

    if model_loader is not None:
        raise StageZHold("REAL_MODEL_LOADER_FORBIDDEN_IN_STATIC_PREPARATION")


__all__ = ["reject_real_runtime_in_preparation", "require_engineering_canary", "run_authorized_callback"]
