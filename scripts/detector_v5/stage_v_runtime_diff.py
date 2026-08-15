"""Deterministic, read-only field diff for frozen runtime evidence."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.stage_v_canonical_execution_core import canonical_sha256, canonical_value  # noqa: E402


_MISSING = object()
_CONTEXT_FIELDS = (
    "parent_key", "probe_id", "branch_id", "snapshot_manifest_sha256",
    "snapshot_source_commit", "snapshot_source_tree", "current_runtime_commit",
    "current_runtime_tree", "runtime_worktree", "exact_plan_manifest_sha256",
    "runtime_provenance_receipt_sha256", "closure_report_sha256",
)


def _type_name(value: Any) -> str:
    if value is _MISSING:
        return "<missing>"
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _shape(value: Any) -> list[int] | None:
    if value is _MISSING:
        return None
    if isinstance(value, Mapping):
        kind = value.get("kind")
        if kind == "array" and isinstance(value.get("shape"), list):
            return [int(item) for item in value["shape"]]
        if kind == "image" and isinstance(value.get("size"), list):
            return [int(item) for item in value["size"]]
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [len(value)]
    return None


def _digest(value: Any) -> str | None:
    if value is _MISSING:
        return None
    try:
        return canonical_sha256(canonical_value(value))
    except Exception:
        return None


def _canonical(value: Any) -> Any:
    try:
        return canonical_value(value)
    except Exception:
        return {"kind": "unavailable", "type": _type_name(value)}


def _row(path: str, expected: Any, actual: Any, context: Mapping[str, Any]) -> dict[str, Any]:
    canonical_expected = None if expected is _MISSING else _canonical(expected)
    canonical_actual = None if actual is _MISSING else _canonical(actual)
    result = {key: context.get(key) for key in _CONTEXT_FIELDS}
    result.update({
        "canonical_path": path,
        "expected_type": _type_name(expected),
        "actual_type": _type_name(actual),
        "expected_shape": _shape(canonical_expected),
        "actual_shape": _shape(canonical_actual),
        "expected_sha256": _digest(expected),
        "actual_sha256": _digest(actual),
    })
    return result


def _compare(path: str, expected: Any, actual: Any, context: Mapping[str, Any], rows: list[dict[str, Any]]) -> None:
    if expected is _MISSING or actual is _MISSING:
        rows.append(_row(path, expected, actual, context))
        return
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in sorted(set(expected) | set(actual), key=str):
            child = f"{path}.{key}"
            _compare(child, expected.get(key, _MISSING), actual.get(key, _MISSING), context, rows)
        return
    if (isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray))
            and isinstance(actual, Sequence) and not isinstance(actual, (str, bytes, bytearray))):
        if len(expected) != len(actual):
            rows.append(_row(path, expected, actual, context))
        for index in range(max(len(expected), len(actual))):
            _compare(f"{path}[{index}]", expected[index] if index < len(expected) else _MISSING,
                     actual[index] if index < len(actual) else _MISSING, context, rows)
        return
    try:
        equal = canonical_value(expected) == canonical_value(actual)
    except Exception:
        equal = False
    if not equal:
        rows.append(_row(path, expected, actual, context))


def diff(expected: Any, actual: Any, *, context: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return stable field-level mismatches; never infer a causal explanation."""
    bound = {key: (context or {}).get(key) for key in _CONTEXT_FIELDS}
    rows: list[dict[str, Any]] = []
    _compare("$", expected, actual, bound, rows)
    return rows


runtime_diff = diff
