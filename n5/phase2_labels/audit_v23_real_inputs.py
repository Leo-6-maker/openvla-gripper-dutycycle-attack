"""Read-only capability census for the frozen 40-episode V23 pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
FORBIDDEN = {"task_success", "terminal", "reward", "outcome", "attack"}
SIM_KEYS = {"sim_state", "sim_qpos", "sim_qvel", "mj_qpos", "mj_qvel", "mujoco_state", "physics_state"}
INIT_KEYS = {"init_state", "initial_state", "reset_state", "sim_state_initial"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)


def finite_vector(value: Any, width: int | None = None) -> bool:
    if not isinstance(value, list) or (width is not None and len(value) != width):
        return False
    try:
        return all(math.isfinite(float(item)) for item in value)
    except (TypeError, ValueError, OverflowError):
        return False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row {path}:{line_number}")
            rows.append(row)
    return rows


def nested_has(data: Any, names: set[str]) -> bool:
    if isinstance(data, Mapping):
        if any(name in data and data[name] is not None for name in names):
            return True
        return any(nested_has(value, names) for value in data.values())
    if isinstance(data, list):
        return any(nested_has(value, names) for value in data)
    return False


def audit_record(record: Mapping[str, Any]) -> dict[str, Any]:
    identity = str(record["episode_id"])
    root = Path(str(record["source_episode_root"])).resolve(strict=True)
    expected_files = {item["name"]: item for item in record["source_files"]}
    files: dict[str, Path] = {}
    source_sha_pass = True
    for name, expected in expected_files.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing/non-regular source file: {path}")
        files[name] = path
        source_sha_pass &= path.stat().st_size == expected["size_bytes"]
        source_sha_pass &= sha256_file(path) == expected["sha256"]
    metadata = strict_json(files["episode_metadata.json"])
    steps = load_jsonl(files["step_records.jsonl"])
    sidecar = load_jsonl(files["privileged_teacher_sidecar.jsonl"])
    expected_steps = set(range(len(sidecar)))
    step_ids = {row.get("step") for row in sidecar}
    step_closure = step_ids == expected_steps and len(sidecar) == len(steps)
    sidecar_fields = set().union(*(row.keys() for row in sidecar)) if sidecar else set()
    step_fields = set().union(*(row.keys() for row in steps)) if steps else set()
    sidecar_sim = sum(any(key in row for key in SIM_KEYS) for row in sidecar)
    sidecar_object = sum(finite_vector(row.get("object_state")) for row in sidecar)
    named_object = sum(any(key in row for key in ("object_pose", "object_poses", "object_world_pose")) for row in sidecar)
    named_target = sum(any(key in row for key in ("target_pose", "target_poses", "target_world_pose")) for row in sidecar)
    eef = sum(finite_vector(row.get("robot0_eef_pos"), 3) and finite_vector(row.get("robot0_eef_quat"), 4) for row in sidecar)
    qpos = sum(finite_vector(row.get("robot0_gripper_qpos"), 2) for row in sidecar)
    contacts = sum(isinstance(row.get("mujoco_contact_pairs"), list) for row in sidecar)
    actions = sum(any(key in row for key in ("applied_action_7d", "action_raw", "clean_action_raw_7d")) for row in steps)
    init_available = nested_has(metadata, INIT_KEYS) or any(nested_has(row, INIT_KEYS) for row in sidecar[:1])
    model_binding = any(key in metadata for key in ("checkpoint_config_sha256", "checkpoint_tree_sha256", "collector_script_sha256"))
    bddl_binding = any(key in metadata for key in ("bddl_sha256", "bddl_path", "task_definition_sha256", "model_xml_sha256"))
    forbidden = sorted((FORBIDDEN & (sidecar_fields | step_fields)))
    lengths = Counter(len(row.get("object_state", [])) for row in sidecar if isinstance(row.get("object_state"), list))
    return {
        "episode_id": identity,
        "suite": record["suite"],
        "task_id": record["task_id"],
        "state_id": record["state_id"],
        "step_count": len(sidecar),
        "source_sha_pass": source_sha_pass,
        "step_closure": step_closure,
        "sim_state_steps": sidecar_sim,
        "object_state_steps": sidecar_object,
        "named_object_pose_steps": named_object,
        "named_target_pose_steps": named_target,
        "eef_pose_steps": eef,
        "gripper_qpos_steps": qpos,
        "contact_pair_steps": contacts,
        "applied_action_steps": actions,
        "init_state_available": init_available,
        "model_binding_available": model_binding,
        "bddl_binding_available": bddl_binding,
        "object_state_dimensions": dict(sorted(lengths.items())),
        "forbidden_payload_fields": forbidden,
        "replay_mode": "DIRECT_STATE" if sidecar_sim == len(sidecar) else (
            "DETERMINISTIC_REPLAY_CANDIDATE" if init_available and actions == len(steps) and model_binding and bddl_binding else "HOLD"
        ),
    }


def build(manifest_path: Path, output_parent: Path, output_name: str) -> dict[str, Any]:
    manifest = strict_json(manifest_path)
    if manifest.get("schema") != "V23_DEV_PILOT_V1" or manifest.get("episode_count") != 40:
        raise ValueError("frozen 40-episode V23 pilot manifest required")
    rows = [audit_record(record) for record in manifest["records"]]
    if len(rows) != 40 or len({row["episode_id"] for row in rows}) != 40:
        raise ValueError("pilot identity closure failed")
    if any(not row["source_sha_pass"] or not row["step_closure"] for row in rows):
        raise ValueError("source SHA or step closure failed")
    mode_counts = Counter(row["replay_mode"] for row in rows)
    suite_counts = defaultdict(Counter)
    for row in rows:
        suite_counts[row["suite"]][row["replay_mode"]] += 1
    capability = "DIRECT_STATE" if mode_counts == Counter({"DIRECT_STATE": 40}) else (
        "REPLAY" if mode_counts["HOLD"] == 0 and mode_counts["DETERMINISTIC_REPLAY_CANDIDATE"] == 40 else "HOLD_GEOMETRY_SOURCE_INSUFFICIENT"
    )
    summary = {
        "schema": "C3_T1B_R1A_REAL_INPUT_CAPABILITY_AUDIT_V1",
        "status": "PASS" if capability != "HOLD_GEOMETRY_SOURCE_INSUFFICIENT" else "HOLD_GEOMETRY_SOURCE_INSUFFICIENT",
        "capability_decision": capability,
        "pilot_manifest_path": str(manifest_path.resolve()),
        "pilot_manifest_sha256": sha256_file(manifest_path),
        "episode_count": len(rows),
        "step_count_total": sum(row["step_count"] for row in rows),
        "mode_counts": dict(mode_counts),
        "suite_counts": {suite: dict(counts) for suite, counts in sorted(suite_counts.items())},
        "field_step_counts": {
            "sim_state": sum(row["sim_state_steps"] for row in rows),
            "object_state": sum(row["object_state_steps"] for row in rows),
            "named_object_pose": sum(row["named_object_pose_steps"] for row in rows),
            "named_target_pose": sum(row["named_target_pose_steps"] for row in rows),
            "eef_pose_quaternion": sum(row["eef_pose_steps"] for row in rows),
            "gripper_qpos": sum(row["gripper_qpos_steps"] for row in rows),
            "contact_pairs": sum(row["contact_pair_steps"] for row in rows),
            "applied_action": sum(row["applied_action_steps"] for row in rows),
        },
        "forbidden_payload_fields": sorted({field for row in rows for field in row["forbidden_payload_fields"]}),
        "rows": rows,
        "protected_payload_read": False,
        "model_inference": False,
        "rollout": False,
        "attack": False,
    }
    output_parent.mkdir(parents=True, exist_ok=True)
    final = output_parent / output_name
    if final.exists():
        raise FileExistsError(f"refusing to overwrite {final}")
    staging = output_parent / f".staging_{output_name}_{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        (staging / "INPUT_CAPABILITY_AUDIT.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (staging / "INPUT_CAPABILITY_AUDIT.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = list(rows[0])
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        names = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
        (staging / "SHA256SUMS").write_text("\n".join(f"{sha256_file(staging / name)}  {name}" for name in names) + "\n", encoding="utf-8")
        sums_sha = sha256_file(staging / "SHA256SUMS")
        (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
        os.rename(staging, final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"root": str(final), "status": summary["status"], "capability_decision": capability, "sha256sums_sha256": sums_sha}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-parent", required=True, type=Path)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.output_parent, args.output_name), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
