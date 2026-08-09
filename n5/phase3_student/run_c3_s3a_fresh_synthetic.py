"""C3-S3A fresh synthetic geometry fixture and numerical replay.

This path is deliberately independent of Clean2000 and H0.3 episode roots.
It creates a small sealed geometry-only fixture, audits it through the existing
fail-closed C3-S3 contract, and produces sealed smoke/run/comparison receipts.
No OpenVLA, policy, Teacher, Student, rollout, or attack code is imported.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import c3_s3_geometry_observability as c3  # noqa: E402
import c3_s3_input_contract as contract  # noqa: E402


SCHEMA = "C3_S3_GEOMFIT_SYNTHETIC_V1"
GATE_SPLIT_SCHEMA = "C3_S3A_D0_GATE_SPLIT_V1"
EXPECTED = {"STATIC": 11, "DYNAMIC": 31, "ARTICULATED": 2}
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
TASKS = tuple(f"{suite}/task_{idx:02d}" for suite in SUITES for idx in range(10))


def _sha256(path: Path) -> str:
    return contract.sha256_file(path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def build_relation_plan() -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for index in range(44):
        task_key = TASKS[index % len(TASKS)]
        relation_index = index // len(TASKS)
        category = "STATIC" if index < 11 else ("DYNAMIC" if index < 42 else "ARTICULATED")
        trajectory = {
            "parent_base_pos": [0.15 + index * 0.002, -0.10 + index * 0.001, 0.40],
            "parent_velocity": [0.0, 0.0, 0.0],
            "parent_yaw_start": 0.0,
            "parent_yaw_rate": 0.0,
            "local_pos": [0.01, -0.002, 0.03 + index * 0.0001],
            "local_step_delta": [0.01, -0.002, 0.0],
            "local_yaw": 0.0,
        }
        if category == "DYNAMIC":
            trajectory.update({
                "parent_base_pos": [0.20 + index * 0.001, -0.08, 0.42],
                "parent_velocity": [0.001, 0.0005, 0.0002],
                "parent_yaw_rate": 0.004,
                "local_pos": [0.025, 0.005, 0.04 + index * 0.0001],
                "local_step_delta": [0.0, 0.0, 0.0],
                "local_yaw": -0.02,
            })
        elif category == "ARTICULATED":
            trajectory.update({
                "parent_base_pos": [0.30 + index * 0.001, -0.12, 0.45],
                "parent_velocity": [0.0007, 0.001, 0.0003],
                "parent_yaw_rate": 0.006,
                "local_pos": [0.018, -0.012, 0.035],
                "local_step_delta": [0.0, 0.0, 0.0],
                "local_yaw": 0.015,
                "qpos_start": -0.4,
                "qpos_rate": 0.008,
            })
        row = {
            "relation_id": f"relation_{index:02d}",
            "task_key": task_key,
            "relation_index": relation_index,
            "category": category,
            "predicate": "In" if category == "STATIC" else ("On" if category == "DYNAMIC" else "Stack"),
            "entity_id": f"synthetic_entity_{index:02d}",
            "step_count": 10 if category == "STATIC" else 100,
            "trajectory": trajectory,
        }
        if category == "ARTICULATED":
            row["joint_chain"] = {
                "kind": "ARTICULATED_JOINT_CHAIN",
                "joint_name": f"synthetic_hinge_{index:02d}",
                "qpos_index": 0,
                "axis": [0.0, 0.0, 1.0],
                "limits": [-0.8, 0.8],
                "ancestor_chain": ["world", "parent", "entity"],
            }
        plan.append(row)
    return plan


def _source_quat_z(angle: float) -> List[float]:
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


def _source_pose(pos: Sequence[float], quat: Sequence[float]) -> Dict[str, List[float]]:
    return {"pos": [float(x) for x in pos], "quat": [float(x) for x in quat]}


def _source_parameters(rel: Mapping[str, Any], step: int) -> tuple[Dict[str, List[float]], Dict[str, List[float]], float | None]:
    index = int(rel["relation_id"].rsplit("_", 1)[1])
    category = str(rel["category"])
    trajectory = rel["trajectory"]
    base = trajectory["parent_base_pos"]
    velocity = trajectory["parent_velocity"]
    parent_angle = float(trajectory["parent_yaw_start"]) + float(trajectory["parent_yaw_rate"]) * step
    parent = _source_pose(
        [float(base[i]) + float(velocity[i]) * step for i in range(3)],
        _source_quat_z(parent_angle),
    )
    local_base = trajectory["local_pos"]
    local_delta = trajectory["local_step_delta"]
    local_angle = float(trajectory["local_yaw"])
    qpos = None
    if category == "STATIC":
        local_angle += 0.01 * step
        local_pos = [float(local_base[i]) + float(local_delta[i]) * step for i in range(3)]
        local_pos[1] = float(local_base[1]) - 0.002 * (step % 3)
    elif category == "ARTICULATED":
        qpos = float(trajectory["qpos_start"]) + float(trajectory["qpos_rate"]) * step
        local_angle += qpos
        local_pos = list(map(float, local_base))
    else:
        local_pos = list(map(float, local_base))
    return parent, _source_pose(local_pos, _source_quat_z(local_angle)), qpos


def _reference_quat_z(angle: float) -> List[float]:
    half = float(angle) * 0.5
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _reference_world_pose(rel: Mapping[str, Any], step: int) -> Dict[str, List[float]]:
    try:
        import mujoco  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised on the official server
        raise RuntimeError("C3-S3A requires the official MuJoCo runtime for the reference chain") from exc
    trajectory = rel["trajectory"]
    base = trajectory["parent_base_pos"]
    velocity = trajectory["parent_velocity"]
    parent_pos = [float(base[i]) + float(velocity[i]) * step for i in range(3)]
    parent_angle = float(trajectory["parent_yaw_start"]) + float(trajectory["parent_yaw_rate"]) * step
    local_pos = list(map(float, trajectory["local_pos"]))
    local_angle = float(trajectory["local_yaw"])
    chain = rel.get("joint_chain")
    articulated = str(rel["category"]) == "ARTICULATED"
    if str(rel["category"]) == "STATIC":
        local_pos[0] += float(trajectory["local_step_delta"][0]) * step
        local_pos[1] = float(trajectory["local_pos"][1]) - 0.002 * (step % 3)
        local_angle += 0.01 * step
    if articulated:
        if not isinstance(chain, Mapping):
            raise ValueError("articulated relation has no joint chain")
        qpos = float(trajectory["qpos_start"]) + float(trajectory["qpos_rate"]) * step
        if not float(chain["limits"][0]) <= qpos <= float(chain["limits"][1]):
            raise ValueError("reference qpos is outside the frozen joint limits")
        joint_xml = (
            f"<joint name='{chain['joint_name']}' type='hinge' axis='{' '.join(map(str, chain['axis']))}' "
            f"range='{' '.join(map(str, chain['limits']))}' limited='true'/>"
        )
        entity_xml = f"<body name='entity' pos='{' '.join(map(str, local_pos))}' quat='{' '.join(map(str, _reference_quat_z(local_angle)))}'><inertial pos='0 0 0' mass='0.01' diaginertia='0.0001 0.0001 0.0001'/>{joint_xml}</body>"
    else:
        entity_xml = f"<body name='entity' pos='{' '.join(map(str, local_pos))}' quat='{' '.join(map(str, _reference_quat_z(local_angle)))}'/>"
    model = mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody><body name='parent' pos='{' '.join(map(str, parent_pos))}' quat='{' '.join(map(str, _reference_quat_z(parent_angle)))}'>{entity_xml}</body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    entity_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "entity")
    if articulated:
        data.qpos[int(chain["qpos_index"])] = qpos
    mujoco.mj_forward(model, data)
    return {"pos": [float(x) for x in data.xpos[entity_id]], "quat": [float(x) for x in data.xquat[entity_id]]}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish once; silently clobbering evidence is forbidden."""
    if os.name != "posix":
        raise RuntimeError("strict no-replace publication is unavailable on this platform")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(str(source)), -100, os.fsencode(str(target)), 1)
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(target)
        raise OSError(error, os.strerror(error), str(target))


def _seal(staging: Path, final: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    if final.exists():
        raise FileExistsError(final)
    _json(staging / "MANIFEST.json", manifest)
    files = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
    manifest["payload_files"] = files
    _json(staging / "MANIFEST.json", manifest)
    files = sorted(path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file())
    lines = [f"{_sha256(staging / rel)}  {rel}" for rel in files if rel not in {"SHA256SUMS", "SHA256SUMS.sha256"}]
    (staging / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sums_sha = _sha256(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    _rename_noreplace(staging, final)
    return {"sha256sums_sha256": sums_sha, "file_count": len(lines), "root": str(final)}


def _sealed_entry(root: Path) -> Dict[str, Any]:
    manifest = root / "MANIFEST.json"
    sums = root / "SHA256SUMS"
    return {
        "manifest_path": "MANIFEST.json",
        "manifest_sha256": _sha256(manifest),
        "root_sha256s_sha256": _sha256(sums),
    }


def verify_sealed_output(root: Path) -> Dict[str, Any]:
    return contract.verify_manifest_binding(root.resolve(), _sealed_entry(root.resolve()))


def generate_dataset(root: Path, allowlist_path: Path, gate_split_path: Path, code_commit: str, code_tree: str) -> Dict[str, Any]:
    root = root.resolve()
    allowlist_path = allowlist_path.resolve()
    if root.exists():
        raise FileExistsError(f"synthetic dataset root already exists: {root}")
    plan = build_relation_plan()
    if {key: sum(row["category"] == key for row in plan) for key in EXPECTED} != EXPECTED:
        raise ValueError("synthetic relation category plan mismatch")
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".staging_{root.name}_{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        source_dir = staging / "source"
        reference_dir = staging / "reference"
        source_dir.mkdir()
        reference_dir.mkdir()
        entries: List[Dict[str, Any]] = []
        source_code_sha = _sha256_bytes(b"c3_s3a_source_reconstruction_v2")
        reference_code_sha = _sha256_bytes(b"c3_s3a_reference_mujoco_joint_chain_v2")
        for rel in plan:
            episode_id = f"c3s3a_{rel['relation_id']}"
            source_rows: List[Dict[str, Any]] = []
            reference_rows: List[Dict[str, Any]] = []
            for step in range(rel["step_count"]):
                parent_pose, local_pose, qpos = _source_parameters(rel, step)
                world_pose = _reference_world_pose(rel, step)
                identity = {
                    "episode_id": episode_id,
                    "task_key": rel["task_key"],
                    "suite": rel["task_key"].split("/", 1)[0],
                    "task_idx": int(rel["task_key"].rsplit("_", 1)[1]),
                    "state_id": 1000 + int(rel["relation_id"].rsplit("_", 1)[1]),
                    "init_seed": 910000 + int(rel["relation_id"].rsplit("_", 1)[1]),
                    "step": step,
                }
                source_rows.append({**identity, "entities": [{
                    "entity_id": rel["entity_id"],
                    "status": "OBSERVED",
                    "reconstruction": {
                        "kind": rel["category"],
                        "parent_world_pose": parent_pose,
                        "local_pose": local_pose,
                        **({"articulated_joint_chain": {**rel["joint_chain"], "qpos": qpos}} if rel["category"] == "ARTICULATED" else {}),
                    },
                }]})
                reference_rows.append({**identity, "entities": [{
                    "entity_id": rel["entity_id"],
                    "world_pose": world_pose,
                }]})
            source_file = source_dir / f"{episode_id}.jsonl"
            reference_file = reference_dir / f"{episode_id}.jsonl"
            _write_jsonl(source_file, source_rows)
            _write_jsonl(reference_file, reference_rows)
            final_source = root / "source" / source_file.name
            final_reference = root / "reference" / reference_file.name
            entries.append({
                "schema": contract.GEOMETRY_SCHEMA,
                "episode_id": episode_id,
                "task_key": rel["task_key"],
                "suite": rel["task_key"].split("/", 1)[0],
                "task_idx": int(rel["task_key"].rsplit("_", 1)[1]),
                "state_id": 1000 + int(rel["relation_id"].rsplit("_", 1)[1]),
                "init_seed": 910000 + int(rel["relation_id"].rsplit("_", 1)[1]),
                "relation_index": rel["relation_index"],
                "relation_id": rel["relation_id"],
                "category": rel["category"],
                "step_count": rel["step_count"],
                "reference_chain": contract.INDEPENDENT_REFERENCE,
                **({"articulated_joint_chain": rel["joint_chain"]} if rel["category"] == "ARTICULATED" else {}),
                "source_telemetry": {
                    "path": str(final_source),
                    "sha256": _sha256(source_file),
                    "computation_chain_id": "c3s3a-source-chain-v1",
                    "method": "OBSERVABLE_LOCAL_POSE_RECONSTRUCTION",
                    "code_sha256": source_code_sha,
                    "code_snapshot_commit": code_commit,
                    "code_snapshot_tree": code_tree,
                },
                "reference_telemetry": {
                    "path": str(final_reference),
                    "sha256": _sha256(reference_file),
                    "computation_chain_id": "c3s3a-reference-mujoco-chain-v1",
                    "method": "DIRECT_MUJOCO_WORLD_POSE_SYNTHETIC",
                    "code_sha256": reference_code_sha,
                    "code_snapshot_commit": code_commit,
                    "code_snapshot_tree": code_tree,
                },
            })
        supported = [{
            "task_key": rel["task_key"],
            "relation_index": rel["relation_index"],
            "classification": {
                "STATIC": "STATIC_FIXTURE",
                "DYNAMIC": "DYNAMIC_RECONSTRUCTABLE",
                "ARTICULATED": "ARTICULATED_RECONSTRUCTABLE",
            }[rel["category"]],
            "category": rel["category"],
        } for rel in plan]
        dataset_manifest = {
            "schema": SCHEMA,
            "status": "SEALED_SYNTHETIC_FIXTURE",
            "gate": "C3-S3A",
            "source_artifact_mutation": 0,
            "protected_reads": [],
            "protected_input_roots": [],
            "clean2000_payload_read": False,
            "model_inference": False,
            "teacher_labeling": False,
            "student_training": False,
            "rollout": False,
            "attack": False,
            "gate_split_path": str(gate_split_path.resolve()),
            "gate_split_sha256": _sha256(gate_split_path),
            "reference_engine": "MuJoCo synthetic world-body pose chain",
            "source_chain": "OBSERVABLE_LOCAL_POSE_RECONSTRUCTION",
            "reference_chain": "DIRECT_MUJOCO_WORLD_POSE_SYNTHETIC",
            "code_snapshot_commit": code_commit,
            "code_snapshot_tree": code_tree,
            "source_code_sha256": source_code_sha,
            "reference_code_sha256": reference_code_sha,
            "supported_relations": supported,
            "relation_category_counts": EXPECTED,
            "episodes": entries,
        }
        dataset_manifest_path = staging / "dataset_manifest.json"
        _json(dataset_manifest_path, dataset_manifest)
        final_manifest = root / "dataset_manifest.json"
        final_allowlist = allowlist_path
        seal = _seal(staging, root, {
            "schema": SCHEMA,
            "gate": "C3-S3A",
            "dataset_manifest_path": "dataset_manifest.json",
            "dataset_manifest_sha256": _sha256(dataset_manifest_path),
            "protected_reads": [],
            "source_artifact_mutation": 0,
            "reference_engine": "MuJoCo synthetic world-body pose chain",
            "code_snapshot_commit": code_commit,
            "code_snapshot_tree": code_tree,
        })
        allowlist = {
            "schema": "C3_S3_ALLOWED_INPUTS_V1",
            "scope": "C3_S3A_SYNTHETIC_ONLY",
            "purpose": "C3_S3A_SYNTHETIC_GEOMETRY_NUMERICAL_VALIDITY",
            "protected_semantics_read": False,
            "code_snapshot_commit": code_commit,
            "code_snapshot_tree": code_tree,
            "allowed_roots": [],
            "allowed_episode_geometry_roots": [{
                "name": "c3_s3a_synthetic",
                "path": str(root),
                "manifest_path": "dataset_manifest.json",
                "manifest_sha256": _sha256(root / "dataset_manifest.json"),
                "root_sha256s_sha256": seal["sha256sums_sha256"],
                "purpose": "C3_S3A_SYNTHETIC_GEOMETRY_NUMERICAL_VALIDITY",
            }],
            "denied_roots": [],
            "denied_purposes": ["FIT_DEV", "CAL", "CHECK", "G10", "T2R-D", "CLEAN2000_PAYLOAD", "ATTACK"],
        }
        if final_allowlist.exists():
            raise FileExistsError(f"allowlist already exists: {final_allowlist}")
        _json(final_allowlist, allowlist)
        return {
            "root": str(root),
            "allowlist": str(final_allowlist),
            "dataset_manifest": str(final_manifest),
            "dataset_manifest_sha256": _sha256(final_manifest),
            "root_sha256s_sha256": seal["sha256sums_sha256"],
            "relation_category_counts": EXPECTED,
            "episode_count": len(entries),
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_fixture(root: Path, allowlist_path: Path) -> tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    allowlist = contract.load_allowlist(allowlist_path)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or len(entries) != 44:
        raise ValueError("synthetic fixture episode closure is not 44")
    if not all(isinstance(manifest.get(key), str) and manifest[key] for key in ("code_snapshot_commit", "code_snapshot_tree")):
        raise ValueError("synthetic fixture code snapshot binding is incomplete")
    audited, inventory = c3.audit_episode_manifest(root / "dataset_manifest.json", allowlist)
    if inventory["status"] != "PASS" or inventory["episode_count"] != 44:
        raise ValueError("synthetic fixture audit did not close")
    by_id = {row["episode_id"]: row for row in entries}
    for row in audited:
        row["category"] = by_id[row["episode_id"]]["category"]
        row["relation_id"] = by_id[row["episode_id"]]["relation_id"]
        row["step_count_expected"] = by_id[row["episode_id"]]["step_count"]
    return manifest, inventory, audited


def _validate_dataset_binding(dataset_root: Path, allowlist_path: Path, binding: Mapping[str, Any], manifest: Mapping[str, Any]) -> Dict[str, Any]:
    allowlist_path = allowlist_path.resolve()
    if not allowlist_path.is_file() or _sha256(allowlist_path) != binding.get("allowlist_sha256"):
        raise ValueError("allowlist path or SHA binding mismatch")
    allowlist = contract.load_allowlist(allowlist_path)
    if binding.get("dataset_root") != str(dataset_root.resolve()):
        raise ValueError("dataset root binding mismatch")
    if binding.get("dataset_root_sha256s_sha256") != _sha256(dataset_root / "SHA256SUMS"):
        raise ValueError("dataset root SHA binding mismatch")
    manifest_path, entry = contract.require_allowed_path(dataset_root / "dataset_manifest.json", allowlist, episode=True)
    sealed = contract.verify_manifest_binding(dataset_root, entry)
    if binding.get("dataset_manifest_sha256") != sealed["manifest_sha256"]:
        raise ValueError("dataset manifest binding mismatch")
    denied = [str(item.get("path", "")) for item in allowlist.get("denied_roots", [])]
    if any(token in str(dataset_root).lower() for token in c3.PROTECTED_TOKENS) or any(
        contract._is_descendant(dataset_root.resolve(), Path(path).resolve(strict=False)) for path in denied if path
    ):
        raise ValueError("dataset root is protected or denied")
    if manifest.get("code_snapshot_commit") != binding.get("code_snapshot_commit") or manifest.get("code_snapshot_tree") != binding.get("code_snapshot_tree"):
        raise ValueError("dataset/code snapshot binding mismatch")
    if allowlist.get("code_snapshot_commit") != manifest.get("code_snapshot_commit") or allowlist.get("code_snapshot_tree") != manifest.get("code_snapshot_tree"):
        raise ValueError("allowlist/code snapshot binding mismatch")
    if binding.get("protected_reads") != [] or binding.get("clean2000_payload_read") is not False:
        raise ValueError("protected input read declared in binding")
    return {"allowlist_sha256": _sha256(allowlist_path), "manifest_sha256": manifest_path and sealed["manifest_sha256"], "root_sha256s_sha256": sealed["sha256sums_sha256"]}


def _category_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for category in EXPECTED:
        selected = [row for row in rows if row.get("category") == category]
        result[category] = {
            "relation_count": len(selected),
            "min_step_count": min((int(row["step_count"]) for row in selected), default=0),
            "step_count": sum(int(row["step_count"]) for row in selected),
            "unknown_articulated_count": sum(int(row.get("unknown_articulated_count", 0)) for row in selected),
            "static_position_denominator": sum(int(row.get("static_position_count", 0)) for row in selected),
            "dynamic_position_denominator": sum(int(row.get("dynamic_position_count", 0)) for row in selected),
        }
    return result


def _expected_relation_rows(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [dict(row) for row in manifest["supported_relations"]]


def _canonical_result(manifest: Mapping[str, Any], inventory: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], run_id: str) -> Dict[str, Any]:
    expected = _expected_relation_rows(manifest)
    task_rows = [{**row, "classification": row["classification"]} for row in expected]
    coverage = c3.relation_coverage(task_rows, rows)
    metrics = c3.aggregate_replay_metrics(rows)
    category_metrics = _category_metrics(rows)
    numerical_gate = c3.evaluate_numerical_gate(metrics, coverage, replay_evidence=True, unknown_articulated_count=metrics["unknown_articulated_count"])
    if metrics["unknown_articulated_count"] != 0:
        raise ValueError("synthetic articulated rows unexpectedly unknown")
    canonical = {
        "schema": SCHEMA,
        "code_snapshot_commit": manifest.get("code_snapshot_commit"),
        "code_snapshot_tree": manifest.get("code_snapshot_tree"),
        "dataset_manifest_sha256": inventory["manifest_sha256"],
        "dataset_root_sha256s_sha256": inventory["root_sha256s_sha256"],
        "supported_relations": expected,
        "coverage": coverage,
        "metrics": metrics,
        "category_metrics": category_metrics,
        "numerical_thresholds": c3.NUMERICAL_THRESHOLDS,
        "numerical_gate": numerical_gate,
    }
    return {
        "canonical": canonical,
        "canonical_digest": _sha256_bytes(_canonical(canonical)),
        "coverage": coverage,
        "metrics": metrics,
        "category_metrics": category_metrics,
        "numerical_gate": numerical_gate,
        "run_id": run_id,
    }


def _write_result(out: Path, payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], binding: Mapping[str, Any], mode: str, category: str | None = None) -> Dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"output root already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".staging_{out.name}_{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        selected = [row for row in rows if category is None or row.get("category") == category]
        summary = {
            "schema": SCHEMA,
            "mode": mode,
            "smoke_category": category,
            "status": "PASS" if payload["numerical_gate"]["status"] == "PASS" else "HOLD",
            "canonical_digest": payload["canonical_digest"],
            "coverage": payload["coverage"],
            "metrics": payload["metrics"],
            "category_metrics": payload["category_metrics"],
            "numerical_thresholds": c3.NUMERICAL_THRESHOLDS,
            "numerical_gate": payload["numerical_gate"],
            "selected_row_count": len(selected),
            "source_mutation": 0,
            "protected_reads": [],
            "validated_roots": [binding["dataset_root"]],
            "purpose": "C3_S3A_SYNTHETIC_GEOMETRY_NUMERICAL_VALIDITY",
            "code_snapshot_commit": binding["code_snapshot_commit"],
            "code_snapshot_tree": binding["code_snapshot_tree"],
            "model_inference": False,
            "teacher_labeling": False,
            "student_training": False,
            "rollout": False,
            "attack": False,
        }
        _json(staging / "input_binding.json", dict(binding))
        _json(staging / "canonical_payload.json", payload["canonical"])
        _json(staging / "coverage.json", payload["coverage"])
        _json(staging / "category_metrics.json", payload["category_metrics"])
        _json(staging / "summary.json", summary)
        with (staging / "episode_observability_rows.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in selected:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        seal = _seal(staging, out, {
            "schema": SCHEMA,
            "mode": mode,
            "smoke_category": category,
            "canonical_digest": payload["canonical_digest"],
            "input_binding": dict(binding),
            "protected_reads": [],
            "source_mutation": 0,
            "model_inference": False,
            "rollout": False,
            "attack": False,
        })
        verify_sealed_output(out)
        return {"root": str(out), "status": summary["status"], "canonical_digest": payload["canonical_digest"], "sha256sums_sha256": seal["sha256sums_sha256"], "selected_row_count": len(selected)}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_audit(root: Path, allowlist_path: Path, out: Path, run_id: str, mode: str, category: str | None = None) -> Dict[str, Any]:
    manifest, inventory, rows = _load_fixture(root.resolve(), allowlist_path.resolve())
    payload = _canonical_result(manifest, inventory, rows, run_id)
    binding = {
        "dataset_root": str(root.resolve()),
        "dataset_manifest_sha256": inventory["manifest_sha256"],
        "dataset_root_sha256s_sha256": inventory["root_sha256s_sha256"],
        "allowlist_path": str(allowlist_path.resolve()),
        "allowlist_sha256": _sha256(allowlist_path.resolve()),
        "protected_reads": [],
        "clean2000_payload_read": False,
        "model_inference": False,
        "reference_engine": "MuJoCo synthetic world-body pose chain",
        "code_snapshot_commit": manifest.get("code_snapshot_commit"),
        "code_snapshot_tree": manifest.get("code_snapshot_tree"),
        "run_id": run_id,
    }
    if mode == "smoke":
        if category not in EXPECTED:
            raise ValueError(f"smoke category must be one of {sorted(EXPECTED)}")
        cat = payload["category_metrics"][category]
        minimum = 10 if category == "STATIC" else 100
        if cat["relation_count"] != EXPECTED[category] or cat["min_step_count"] < minimum:
            raise ValueError(f"{category} smoke coverage/minimum failed: {cat}")
        if category == "ARTICULATED" and cat["unknown_articulated_count"] != 0:
            raise ValueError("articulated smoke has unknown rows")
    return _write_result(out.resolve(), payload, rows, binding, mode, category)


def apply_source_fault(source_rows: Sequence[Mapping[str, Any]], fault: str) -> List[Dict[str, Any]]:
    mutated = json.loads(json.dumps(list(source_rows)))
    if fault not in {"translation", "rotation", "local-transform", "qpos", "joint-axis"}:
        raise ValueError(f"unknown source fault: {fault}")
    row = mutated[0]
    reconstruction = row["entities"][0]["reconstruction"]
    if fault == "translation":
        reconstruction["parent_world_pose"]["pos"][0] += 0.02
    elif fault == "rotation":
        reconstruction["parent_world_pose"]["quat"] = _source_quat_z(0.15)
    elif fault == "local-transform":
        reconstruction["local_pose"]["pos"][2] += 0.02
    elif fault == "qpos":
        reconstruction["articulated_joint_chain"]["qpos"] = 9.0
    else:
        reconstruction["articulated_joint_chain"]["axis"] = [1.0, 0.0, 0.0]
    return mutated


def run_negative_control(root: Path, allowlist_path: Path, out: Path, fault: str) -> Dict[str, Any]:
    if out.exists():
        raise FileExistsError(out)
    manifest, inventory, _ = _load_fixture(root.resolve(), allowlist_path.resolve())
    category = "ARTICULATED" if fault in {"qpos", "joint-axis"} else "STATIC"
    entry = next(row for row in manifest["episodes"] if row["category"] == category)
    source_path = Path(entry["source_telemetry"]["path"]).resolve()
    reference_path = Path(entry["reference_telemetry"]["path"]).resolve()
    identity = {key: entry[key] for key in ("task_key", "suite", "task_idx", "state_id", "init_seed")}
    source_rows = contract.load_jsonl_exact(source_path, episode_id=entry["episode_id"], step_count=entry["step_count"], role="source", identity=identity)
    reference_rows = contract.load_jsonl_exact(reference_path, episode_id=entry["episode_id"], step_count=entry["step_count"], role="reference", identity=identity)
    mutated = apply_source_fault(source_rows, fault)
    observed_error = None
    result = None
    try:
        result = contract.audit_episode_geometry(entry, mutated, reference_rows)
        if category == "STATIC":
            limits = c3.NUMERICAL_THRESHOLDS
            if float(result["static_position_max_error_m"] or 0.0) > limits["static_position_max_error_m"] or float(result["static_rotation_max_error_rad"] or 0.0) > limits["static_rotation_max_error_rad"]:
                observed_error = "frozen static numerical threshold violation"
            else:
                raise ValueError("source fault did not violate the frozen static threshold")
        else:
            raise ValueError("articulated source fault was accepted")
    except Exception as exc:
        observed_error = observed_error or str(exc)
    if not observed_error:
        raise ValueError(f"negative control did not fail: {fault}")
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".staging_{out.name}_{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        summary = {
            "schema": "C3_S3A_NEGATIVE_CONTROL_V1",
            "status": "PASS_NEGATIVE_CONTROL",
            "fault": fault,
            "source_only_mutation": True,
            "injected_gate_status": "FAIL",
            "expected_rejection": True,
            "observed_error": observed_error,
            "selected_episode_id": entry["episode_id"],
            "category": category,
            "input_root": str(root.resolve()),
            "input_root_sha256s_sha256": inventory["root_sha256s_sha256"],
            "reference_rows_unchanged_sha256": _sha256_bytes(_canonical(reference_rows)),
            "protected_reads": [],
            "clean2000_payload_read": False,
            "model_inference": False,
            "rollout": False,
            "attack": False,
        }
        _json(staging / "summary.json", summary)
        _json(staging / "fault_binding.json", {"fault": fault, "source_only_mutation": True, "input_root": str(root.resolve()), "selected_episode_id": entry["episode_id"]})
        _seal(staging, out, summary)
        verify_sealed_output(out)
        return {"root": str(out), "status": summary["status"], "fault": fault}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def compare_runs(run_a: Path, run_b: Path, out: Path) -> Dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"comparison root already exists: {out}")
    checked = []
    for path in (run_a, run_b):
        root = path.resolve()
        verify_sealed_output(root)
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        binding = json.loads((root / "input_binding.json").read_text(encoding="utf-8"))
        dataset_root = Path(binding["dataset_root"]).resolve()
        manifest = json.loads((dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))
        _validate_dataset_binding(dataset_root, Path(binding["allowlist_path"]), binding, manifest)
        _load_fixture(dataset_root, Path(binding["allowlist_path"]))
        if summary.get("status") != "PASS":
            raise ValueError(f"run is not PASS: {root}")
        if summary.get("protected_reads") != [] or summary.get("source_mutation") != 0:
            raise ValueError("run declares protected reads or source mutation")
        if summary.get("code_snapshot_commit") != manifest.get("code_snapshot_commit") or summary.get("code_snapshot_tree") != manifest.get("code_snapshot_tree"):
            raise ValueError("run code snapshot does not match dataset")
        coverage = summary.get("coverage", {})
        if not (
            coverage.get("expected_supported_relation_count") == 44
            and coverage.get("covered_supported_relation_count") == 44
            and coverage.get("coverage_complete") is True
            and coverage.get("missing_supported_relations") == []
            and coverage.get("extra_relation_rows") == []
            and len(coverage.get("relation_denominators", [])) == 44
        ):
            raise ValueError("run relation coverage is not exact 44/44")
        for denominator in coverage["relation_denominators"]:
            if int(denominator.get("step_denominator", 0)) <= 0 or int(denominator.get("compared_pose_denominator", 0)) <= 0:
                raise ValueError("run has an empty relation denominator")
        metrics = summary.get("metrics", {})
        if any(int(metrics.get(key, 0) or 0) <= 0 for key in ("static_position_denominator", "static_rotation_denominator", "dynamic_position_p99_denominator", "dynamic_rotation_p99_denominator")):
            raise ValueError("run has an empty numerical denominator")
        category_metrics = summary.get("category_metrics", {})
        if int(category_metrics.get("STATIC", {}).get("static_position_denominator", 0)) <= 0 or int(category_metrics.get("DYNAMIC", {}).get("dynamic_position_denominator", 0)) <= 0 or int(category_metrics.get("ARTICULATED", {}).get("dynamic_position_denominator", 0)) <= 0:
            raise ValueError("category numerical denominators are empty")
        gate = summary.get("numerical_gate", {})
        if gate.get("status") != "PASS" or gate.get("threshold_violations") != [] or summary.get("numerical_thresholds") != c3.NUMERICAL_THRESHOLDS:
            raise ValueError("run numerical gate is not a clean PASS")
        checked.append({"root": root, "summary": summary, "binding": binding, "manifest": manifest})
    summaries = [item["summary"] for item in checked]
    bindings = [item["binding"] for item in checked]
    if any(summary.get("status") != "PASS" for summary in summaries):
        raise ValueError("run A/B must both be PASS")
    if summaries[0].get("canonical_digest") != summaries[1].get("canonical_digest"):
        raise ValueError("run A/B canonical digest mismatch")
    if summaries[0].get("coverage") != summaries[1].get("coverage"):
        raise ValueError("run A/B coverage mismatch")
    if summaries[0].get("metrics") != summaries[1].get("metrics"):
        raise ValueError("run A/B metrics mismatch")
    for field in ("dataset_root", "dataset_manifest_sha256", "dataset_root_sha256s_sha256", "allowlist_sha256", "code_snapshot_commit", "code_snapshot_tree"):
        if bindings[0].get(field) != bindings[1].get(field):
            raise ValueError(f"run A/B binding mismatch: {field}")
    parent = out.resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".staging_{out.name}_{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        comparison = {
            "schema": "C3_S3A_RUN_COMPARISON_V1",
            "status": "PASS",
            "run_A_root": str(run_a.resolve()),
            "run_B_root": str(run_b.resolve()),
            "run_A_sha256s_sha256": _sha256(run_a.resolve() / "SHA256SUMS"),
            "run_B_sha256s_sha256": _sha256(run_b.resolve() / "SHA256SUMS"),
            "canonical_digest": summaries[0]["canonical_digest"],
            "canonical_identical": True,
            "coverage_identical": True,
            "metrics_identical": True,
            "code_snapshot_commit": bindings[0]["code_snapshot_commit"],
            "code_snapshot_tree": bindings[0]["code_snapshot_tree"],
            "allowlist_sha256": bindings[0]["allowlist_sha256"],
            "protected_reads": [],
            "source_mutation": 0,
            "model_inference": False,
            "rollout": False,
            "attack": False,
        }
        _json(staging / "comparison.json", comparison)
        _json(staging / "summary.json", comparison)
        seal = _seal(staging, out.resolve(), comparison)
        verify_sealed_output(out.resolve())
        return {"root": str(out.resolve()), "status": "PASS", "canonical_digest": comparison["canonical_digest"], "sha256sums_sha256": seal["sha256sums_sha256"]}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("generate", "smoke", "run", "negative", "compare"), required=True)
    parser.add_argument("--root")
    parser.add_argument("--allowlist")
    parser.add_argument("--out")
    parser.add_argument("--run-id")
    parser.add_argument("--category", choices=tuple(EXPECTED))
    parser.add_argument("--run-a")
    parser.add_argument("--run-b")
    parser.add_argument("--fault", choices=("translation", "rotation", "local-transform", "qpos", "joint-axis"))
    parser.add_argument("--code-commit")
    parser.add_argument("--code-tree")
    parser.add_argument("--gate-split", default=str(HERE.parents[2] / "configs" / "C3_S3A_D0_GATE_SPLIT_V1.json"))
    args = parser.parse_args()
    try:
        if args.mode == "generate":
            if not all((args.root, args.allowlist, args.code_commit, args.code_tree)):
                raise ValueError("generate requires --root, --allowlist, --code-commit and --code-tree")
            result = generate_dataset(Path(args.root), Path(args.allowlist), Path(args.gate_split), args.code_commit, args.code_tree)
        elif args.mode in {"smoke", "run"}:
            if not all((args.root, args.allowlist, args.out, args.run_id)):
                raise ValueError(f"{args.mode} requires --root --allowlist --out --run-id")
            result = run_audit(Path(args.root), Path(args.allowlist), Path(args.out), args.run_id, args.mode, args.category)
        elif args.mode == "negative":
            if not all((args.root, args.allowlist, args.out, args.fault)):
                raise ValueError("negative requires --root --allowlist --out --fault")
            result = run_negative_control(Path(args.root), Path(args.allowlist), Path(args.out), args.fault)
        else:
            if not all((args.run_a, args.run_b, args.out)):
                raise ValueError("compare requires --run-a --run-b --out")
            result = compare_runs(Path(args.run_a), Path(args.run_b), Path(args.out))
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"C3-S3A HOLD: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
