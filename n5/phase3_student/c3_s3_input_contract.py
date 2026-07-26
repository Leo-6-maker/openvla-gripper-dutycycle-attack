"""Fail-closed input and per-step geometry contract for C3-S3.

This module deliberately does not discover roots.  A root becomes readable only
when it is named, sealed, and explicitly present in the versioned allowlist.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


GEOMETRY_SCHEMA = "C3_S3_EPISODE_GEOMETRY_V1"
INDEPENDENT_REFERENCE = "INDEPENDENT_WORLD_REFERENCE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_allowlist(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "C3_S3_ALLOWED_INPUTS_V1":
        raise ValueError("wrong C3-S3 input allowlist schema")
    if data.get("protected_semantics_read") is not False:
        raise ValueError("allowlist must be protected-read false")
    return data


def _resolved_without_symlink(path: Path) -> Path:
    absolute = Path(path).absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink path component rejected: {current}")
    if not absolute.exists():
        raise FileNotFoundError(absolute)
    return absolute.resolve(strict=True)


def _is_descendant(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _deny_reason(path: Path, allowlist: Mapping[str, Any]) -> str | None:
    for entry in allowlist.get("denied_roots", []):
        denied = Path(str(entry["path"])).resolve(strict=False)
        if _is_descendant(path, denied):
            return str(entry.get("reason", entry.get("name", "denied root")))
    return None


def _root_entry(path: Path, allowlist: Mapping[str, Any], *, episode: bool) -> Mapping[str, Any] | None:
    key = "allowed_episode_geometry_roots" if episode else "allowed_roots"
    for entry in allowlist.get(key, []):
        root = Path(str(entry["path"])).resolve(strict=False)
        if _is_descendant(path, root):
            return entry
    return None


def require_allowed_path(path: Path, allowlist: Mapping[str, Any], *, episode: bool = False, regular: bool = True) -> Tuple[Path, Mapping[str, Any]]:
    resolved = _resolved_without_symlink(path)
    reason = _deny_reason(resolved, allowlist)
    if reason:
        raise ValueError(f"explicitly denied input: {resolved}: {reason}")
    entry = _root_entry(resolved, allowlist, episode=episode)
    if entry is None:
        raise ValueError(f"input is not in the explicit allowlist: {resolved}")
    if regular and not resolved.is_file():
        raise ValueError(f"input is not a regular file: {resolved}")
    return resolved, entry


def verify_manifest_binding(root: Path, entry: Mapping[str, Any]) -> Dict[str, Any]:
    manifest = root / str(entry["manifest_path"])
    resolved = _resolved_without_symlink(manifest)
    if not resolved.is_file():
        raise ValueError(f"manifest is not a regular file: {manifest}")
    actual = sha256_file(resolved)
    if actual != entry.get("manifest_sha256"):
        raise ValueError(f"manifest SHA mismatch: expected {entry.get('manifest_sha256')}, got {actual}")
    return {"root": str(root), "manifest": str(resolved), "manifest_sha256": actual, "entry": dict(entry)}


def load_jsonl_exact(path: Path, *, episode_id: str, step_count: int, role: str, identity: Mapping[str, Any] | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[int] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if row.get("episode_id") != episode_id:
            raise ValueError(f"{role} episode mismatch at line {line_no}")
        for field, expected in (identity or {}).items():
            if row.get(field) != expected:
                raise ValueError(f"{role} identity mismatch for {field} at line {line_no}")
        step = row.get("step")
        if type(step) is not int or step < 0 or step >= step_count or step in seen:
            raise ValueError(f"{role} duplicate/missing-invalid step at line {line_no}: {step!r}")
        seen.add(step)
        rows.append(row)
    expected = set(range(step_count))
    if seen != expected:
        raise ValueError(f"{role} exact step join failed: expected {step_count}, got {len(seen)}")
    return [next(row for row in rows if row["step"] == step) for step in range(step_count)]


def _pose(value: Mapping[str, Any], name: str) -> Dict[str, List[float]]:
    pose = value.get(name)
    if not isinstance(pose, Mapping) or len(pose.get("pos", [])) != 3 or len(pose.get("quat", [])) != 4:
        raise ValueError(f"missing or malformed {name}")
    return {"pos": [float(x) for x in pose["pos"]], "quat": [float(x) for x in pose["quat"]]}


def _quat_normalize(q: Sequence[float]) -> Tuple[float, float, float, float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in q))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("invalid quaternion")
    return tuple(float(x) / norm for x in q)  # type: ignore[return-value]


def _quat_mul(left: Sequence[float], right: Sequence[float]) -> Tuple[float, float, float, float]:
    w1, x1, y1, z1 = _quat_normalize(left)
    w2, x2, y2, z2 = _quat_normalize(right)
    return _quat_normalize((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ))


def _quat_inverse(q: Sequence[float]) -> Tuple[float, float, float, float]:
    w, x, y, z = _quat_normalize(q)
    return (w, -x, -y, -z)


def _rotate(q: Sequence[float], vector: Sequence[float]) -> Tuple[float, float, float]:
    w, x, y, z = _quat_normalize(q)
    vx, vy, vz = [float(x) for x in vector]
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + y * tz - z * ty, vy + w * ty + z * tx - x * tz, vz + w * tz + x * ty - y * tx)


def compose_pose(parent: Mapping[str, Sequence[float]], local: Mapping[str, Sequence[float]]) -> Dict[str, List[float]]:
    rotated = _rotate(parent["quat"], local["pos"])
    return {"pos": [float(parent["pos"][i]) + rotated[i] for i in range(3)], "quat": list(_quat_mul(parent["quat"], local["quat"]))}


def rotation_geodesic_error(predicted: Sequence[float], reference: Sequence[float]) -> float:
    left = _quat_normalize(predicted)
    right = _quat_normalize(reference)
    dot = max(-1.0, min(1.0, abs(sum(a * b for a, b in zip(left, right)))))
    return 2.0 * math.acos(dot)


def position_error(predicted: Sequence[float], reference: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(predicted, reference)))


def p99(values: Sequence[float]) -> Dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"value": None, "count": 0, "quantile": 0.99, "method": "linear_interpolation_n_minus_1"}
    index = (len(ordered) - 1) * 0.99
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    value = ordered[lower] if lower == upper else ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    return {"value": value, "count": len(ordered), "quantile": 0.99, "method": "linear_interpolation_n_minus_1"}


def audit_episode_geometry(entry: Mapping[str, Any], source_rows: Sequence[Mapping[str, Any]], reference_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    episode_id = str(entry["episode_id"])
    step_count = int(entry["step_count"])
    if len(source_rows) != step_count or len(reference_rows) != step_count:
        raise ValueError("source/reference step count mismatch")
    reference_by_step = {int(row["step"]): row for row in reference_rows}
    position_static: List[float] = []
    rotation_static: List[float] = []
    position_dynamic: List[float] = []
    rotation_dynamic: List[float] = []
    unknown = 0
    compared = 0
    for source_row in source_rows:
        step = int(source_row["step"])
        ref_entities = {str(item["entity_id"]): item for item in reference_by_step[step].get("entities", [])}
        if len(ref_entities) != len(reference_by_step[step].get("entities", [])):
            raise ValueError(f"duplicate reference entity at step {step}")
        source_entities = source_row.get("entities", [])
        source_by_id = {str(item["entity_id"]): item for item in source_entities}
        if len(source_by_id) != len(source_entities) or set(source_by_id) != set(ref_entities):
            raise ValueError(f"exact entity join failed at step {step}")
        for entity_id, source_entity in source_by_id.items():
            reference_entity = ref_entities[entity_id]
            if source_entity.get("status") == "UNKNOWN_ARTICULATED":
                unknown += 1
                continue
            reconstruction = source_entity.get("reconstruction")
            if not isinstance(reconstruction, Mapping) or reconstruction.get("kind") not in {"STATIC", "DYNAMIC"}:
                raise ValueError(f"unsupported reconstruction kind at {episode_id}:{step}:{entity_id}")
            predicted = compose_pose(_pose(reconstruction, "parent_world_pose"), _pose(reconstruction, "local_pose"))
            reference = _pose(reference_entity, "world_pose")
            pos = position_error(predicted["pos"], reference["pos"])
            rot = rotation_geodesic_error(predicted["quat"], reference["quat"])
            compared += 1
            if reconstruction["kind"] == "STATIC":
                position_static.append(pos)
                rotation_static.append(rot)
            else:
                position_dynamic.append(pos)
                rotation_dynamic.append(rot)
    return {
        "episode_id": episode_id,
        "task_key": entry["task_key"],
        "step_count": step_count,
        "compared_pose_count": compared,
        "unknown_articulated_count": unknown,
        "static_position_max_error_m": max(position_static, default=None),
        "static_rotation_max_error_rad": max(rotation_static, default=None),
        "dynamic_position_p99": p99(position_dynamic),
        "dynamic_rotation_p99": p99(rotation_dynamic),
        "static_position_count": len(position_static),
        "static_rotation_count": len(rotation_static),
        "dynamic_position_count": len(position_dynamic),
        "dynamic_rotation_count": len(rotation_dynamic),
        "static_position_errors_m": position_static,
        "static_rotation_errors_rad": rotation_static,
        "dynamic_position_errors_m": position_dynamic,
        "dynamic_rotation_errors_rad": rotation_dynamic,
    }


def iter_manifest_paths(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key.endswith("_path") or key == "path":
                if isinstance(item, str):
                    yield item
            yield from iter_manifest_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_manifest_paths(item)
