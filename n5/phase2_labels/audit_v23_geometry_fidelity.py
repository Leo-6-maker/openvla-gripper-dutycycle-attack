"""Read-only T1D-R2B source/reference audit.

This audit deliberately does not infer a full simulator state from the
partial privileged sidecar.  It distinguishes action replay repeatability
from an independent per-step MuJoCo reference and fails closed when the
latter is absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


class FidelityHold(RuntimeError):
    pass


FORBIDDEN = ("cal", "check", "g10", "t2r")
FULL_STATE_FIELDS = frozenset({
    "sim_state", "full_sim_state", "mujoco_state", "mj_state",
    "full_qpos", "mujoco_qpos", "qpos", "qvel",
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def reject_path(path: Path) -> None:
    lowered = str(path).lower()
    if any(token in lowered for token in FORBIDDEN):
        raise FidelityHold(f"forbidden/protected path: {path}")


def verify_root(root: Path) -> dict[str, Any]:
    reject_path(root)
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    if not root.is_dir() or not sums.is_file() or not sidecar.is_file():
        raise FidelityHold(f"unsealed root: {root}")
    side = sidecar.read_text(encoding="utf-8").strip().split()
    if len(side) != 2 or side[1] != "SHA256SUMS" or side[0] != sha256_file(sums):
        raise FidelityHold(f"root sidecar mismatch: {root}")
    expected: dict[str, str] = {}
    for raw in sums.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        digest, name = raw.split(None, 1)
        name = name.strip()
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            raise FidelityHold(f"unsafe sealed path: {name}")
        target = root / relative
        if target.is_symlink() or not target.is_file() or sha256_file(target) != digest:
            raise FidelityHold(f"sealed file mismatch: {target}")
        expected[relative.as_posix()] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
        and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    }
    if set(expected) != actual:
        raise FidelityHold(f"sealed file closure mismatch: {root}")
    return {"path": str(root), "sha256sums_sha256": sha256_file(sums), "file_count": len(expected)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise FidelityHold(f"non-object JSONL row: {path}:{line_no}")
            rows.append(row)
    return rows


def source_row_audit(record: Mapping[str, Any]) -> dict[str, Any]:
    source_root = Path(str(record["source_episode_root"])).resolve()
    reject_path(source_root)
    files = {str(item["name"]): item for item in record.get("source_files", []) if isinstance(item, Mapping)}
    sidecar_spec = files.get("privileged_teacher_sidecar.jsonl")
    if sidecar_spec is None:
        raise FidelityHold(f"sidecar binding missing: {record.get('episode_id')}")
    sidecar = source_root / "privileged_teacher_sidecar.jsonl"
    if sidecar.is_symlink() or not sidecar.is_file() or sha256_file(sidecar) != sidecar_spec.get("sha256"):
        raise FidelityHold(f"sidecar seal mismatch: {sidecar}")
    rows = read_jsonl(sidecar)
    if [row.get("step") for row in rows] != list(range(len(rows))):
        raise FidelityHold(f"non-contiguous source steps: {record.get('episode_id')}")
    field_counts = Counter()
    dimensions = Counter()
    for row in rows:
        field_counts.update(row.keys())
        if isinstance(row.get("object_state"), list):
            dimensions[len(row["object_state"])] += 1
    present_full_state = sorted(FULL_STATE_FIELDS & set(field_counts))
    return {
        "episode_id": str(record["episode_id"]),
        "step_count": len(rows),
        "field_presence": {field: int(field_counts[field] > 0) for field in sorted(field_counts)},
        "object_state_dimensions": dict(sorted(dimensions.items())),
        "has_object_state": bool(field_counts["object_state"]),
        "complete_sim_state_fields": present_full_state,
        "complete_sim_state_available": bool(present_full_state),
    }


def write_sealed(root_parent: Path, root_name: str, report: Mapping[str, Any]) -> dict[str, Any]:
    root_parent.mkdir(parents=True, exist_ok=True)
    final = root_parent / root_name
    if final.exists() or final.is_symlink():
        raise FidelityHold(f"output already exists: {final}")
    staging = root_parent / f".staging_{root_name}_{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        (staging / "FIDELITY_AUDIT.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        files = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
        (staging / "SHA256SUMS").write_text(
            "".join(f"{sha256_file(staging / name)}  {name}\n" for name in files), encoding="utf-8"
        )
        sums_sha = sha256_file(staging / "SHA256SUMS")
        (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
        os.rename(staging, final)
        return {"status": report["status"], "root": str(final), "sha256sums_sha256": sums_sha}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run(pilot_root: Path, geometry_root: Path, output_parent: Path, output_name: str) -> dict[str, Any]:
    pilot_seal = verify_root(pilot_root)
    geometry_seal = verify_root(geometry_root)
    manifest = load_json(pilot_root / "PILOT_INPUT_MANIFEST.json")
    records = manifest.get("records")
    if manifest.get("episode_count") != 40 or not isinstance(records, list) or len(records) != 40:
        raise FidelityHold("pilot manifest is not the frozen 40-episode set")
    rows = [source_row_audit(record) for record in records]
    if len({row["episode_id"] for row in rows}) != 40:
        raise FidelityHold("pilot identity closure failed")
    total_steps = sum(row["step_count"] for row in rows)
    complete = [row for row in rows if row["complete_sim_state_available"]]
    runtime_audit_path = geometry_root / "runtime_audit.json"
    replay_audit = load_json(runtime_audit_path) if runtime_audit_path.is_file() else {}
    report = {
        "schema": "C3_T1D_R2B_GEOMETRY_FIDELITY_AUDIT_V1",
        "status": "HOLD_REFERENCE_MISSING" if len(complete) != len(rows) else "PASS",
        "episode_count": len(rows),
        "step_count": total_steps,
        "source_sim_state": {
            "episodes_with_complete_per_step_sim_state": len(complete),
            "episodes_missing_complete_per_step_sim_state": len(rows) - len(complete),
            "required_contract": sorted(FULL_STATE_FIELDS),
            "partial_object_state_present_episodes": sum(row["has_object_state"] for row in rows),
            "reason_if_missing": "object_state is partial and no sealed full qpos/qvel/sim_state or target-site state is present",
        },
        "reference": {
            "independent_per_step_reference_available": len(complete) == len(rows),
            "source_mode": "SEALED_SIDECAR_METADATA_ONLY",
            "action_replay_is_not_accuracy_reference": True,
        },
        "action_replay_diagnostic": {
            "available": bool(replay_audit),
            "qpos_sidecar_max_abs_error": replay_audit.get("qpos_sidecar_max_abs_error"),
            "eef_sidecar_max_abs_error": replay_audit.get("eef_sidecar_max_abs_error"),
            "object_position_error": None,
            "target_position_error": None,
            "quaternion_geodesic_error": None,
            "classification_flip_counts": None,
        },
        "bindings": {"pilot_root": pilot_seal, "geometry_root": geometry_seal},
        "protected_payload_read": False,
        "model_inference": False,
        "student_training": False,
        "rollout": False,
        "attack": False,
        "episode_rows": rows,
    }
    return write_sealed(output_parent, output_name, report)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--geometry-root", required=True, type=Path)
    parser.add_argument("--output-parent", required=True, type=Path)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.pilot_root, args.geometry_root, args.output_parent, args.output_name), sort_keys=True))
    except Exception as exc:
        print(json.dumps({"status": "HOLD_REFERENCE_MISSING", "reason": f"{type(exc).__name__}:{exc}",
                          "protected_payload_read": False, "model_inference": False,
                          "student_training": False, "rollout": False, "attack": False}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
