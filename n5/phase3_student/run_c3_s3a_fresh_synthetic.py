"""C3-S3A fresh synthetic geometry fixture and numerical replay.

This path is deliberately independent of Clean2000 and H0.3 episode roots.
It creates a small sealed geometry-only fixture, audits it through the existing
fail-closed C3-S3 contract, and produces sealed smoke/run/comparison receipts.
No OpenVLA, policy, Teacher, Student, rollout, or attack code is imported.
"""
from __future__ import annotations

import argparse
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
        plan.append({
            "relation_id": f"relation_{index:02d}",
            "task_key": task_key,
            "relation_index": relation_index,
            "category": category,
            "predicate": "In" if category == "STATIC" else ("On" if category == "DYNAMIC" else "Stack"),
            "entity_id": f"synthetic_entity_{index:02d}",
            "step_count": 10 if category == "STATIC" else 100,
        })
    return plan


def _quat_z(angle: float) -> List[float]:
    return [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)]


def _pose(pos: Sequence[float], quat: Sequence[float]) -> Dict[str, List[float]]:
    return {"pos": [float(x) for x in pos], "quat": [float(x) for x in quat]}


def _step_poses(rel: Mapping[str, Any], step: int) -> tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    index = int(rel["relation_id"].rsplit("_", 1)[1])
    category = str(rel["category"])
    if category == "STATIC":
        parent = _pose([0.15 + index * 0.002, -0.10 + index * 0.001, 0.40], [1.0, 0.0, 0.0, 0.0])
        local = _pose([0.01 * step, -0.002 * (step % 3), 0.03 + index * 0.0001], _quat_z(0.01 * step))
    elif category == "DYNAMIC":
        parent = _pose([0.20 + index * 0.001 + 0.001 * step, -0.08 + 0.0005 * step, 0.42 + 0.0002 * step], _quat_z(0.004 * step))
        local = _pose([0.025, 0.005, 0.04 + index * 0.0001], _quat_z(-0.02))
    else:
        parent = _pose([0.30 + index * 0.001 + 0.0007 * step, -0.12 + 0.001 * math.sin(step / 9.0), 0.45 + 0.0003 * step], _quat_z(0.006 * step))
        local = _pose([0.018, -0.012, 0.035], _quat_z(0.015))
    return parent, local


def _mujoco_reference(parent: Mapping[str, Sequence[float]], local: Mapping[str, Sequence[float]]) -> Dict[str, List[float]]:
    try:
        import mujoco  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised on the official server
        raise RuntimeError("C3-S3A requires the official MuJoCo runtime for the reference chain") from exc
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='parent'><body name='entity'/></body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    parent_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "parent")
    entity_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "entity")
    model.body_pos[parent_id] = parent["pos"]
    model.body_quat[parent_id] = parent["quat"]
    model.body_pos[entity_id] = local["pos"]
    model.body_quat[entity_id] = local["quat"]
    mujoco.mj_forward(model, data)
    return {"pos": [float(x) for x in data.xpos[entity_id]], "quat": [float(x) for x in data.xquat[entity_id]]}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


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
    os.replace(staging, final)
    return {"sha256sums_sha256": sums_sha, "file_count": len(lines), "root": str(final)}


def generate_dataset(root: Path, allowlist_path: Path, gate_split_path: Path) -> Dict[str, Any]:
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
        source_code_sha = _sha256_bytes(b"c3_s3a_source_observable_local_pose_reconstruction_v1")
        reference_code_sha = _sha256_bytes(b"c3_s3a_reference_direct_mujoco_world_pose_v1")
        for rel in plan:
            episode_id = f"c3s3a_{rel['relation_id']}"
            source_rows: List[Dict[str, Any]] = []
            reference_rows: List[Dict[str, Any]] = []
            for step in range(rel["step_count"]):
                parent_pose, local_pose = _step_poses(rel, step)
                world_pose = _mujoco_reference(parent_pose, local_pose)
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
                        "kind": "STATIC" if rel["category"] == "STATIC" else "DYNAMIC",
                        "parent_world_pose": parent_pose,
                        "local_pose": local_pose,
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
                "source_telemetry": {
                    "path": str(final_source),
                    "sha256": _sha256(source_file),
                    "computation_chain_id": "c3s3a-source-chain-v1",
                    "method": "OBSERVABLE_LOCAL_POSE_RECONSTRUCTION",
                    "code_sha256": source_code_sha,
                },
                "reference_telemetry": {
                    "path": str(final_reference),
                    "sha256": _sha256(reference_file),
                    "computation_chain_id": "c3s3a-reference-mujoco-chain-v1",
                    "method": "DIRECT_MUJOCO_WORLD_POSE_SYNTHETIC",
                    "code_sha256": reference_code_sha,
                },
            })
        supported = [{
            "task_key": rel["task_key"],
            "relation_index": rel["relation_index"],
            "classification": "STATIC_FIXTURE" if rel["category"] == "STATIC" else "DYNAMIC_RECONSTRUCTABLE",
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
        })
        allowlist = {
            "schema": "C3_S3_ALLOWED_INPUTS_V1",
            "scope": "C3_S3A_SYNTHETIC_ONLY",
            "purpose": "C3_S3A_SYNTHETIC_GEOMETRY_NUMERICAL_VALIDITY",
            "protected_semantics_read": False,
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
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        raise


def _load_fixture(root: Path, allowlist_path: Path) -> tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    allowlist = contract.load_allowlist(allowlist_path)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or len(entries) != 44:
        raise ValueError("synthetic fixture episode closure is not 44")
    audited, inventory = c3.audit_episode_manifest(root / "dataset_manifest.json", allowlist)
    if inventory["status"] != "PASS" or inventory["episode_count"] != 44:
        raise ValueError("synthetic fixture audit did not close")
    by_id = {row["episode_id"]: row for row in entries}
    for row in audited:
        row["category"] = by_id[row["episode_id"]]["category"]
        row["relation_id"] = by_id[row["episode_id"]]["relation_id"]
        row["step_count_expected"] = by_id[row["episode_id"]]["step_count"]
    return manifest, inventory, audited


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
            "numerical_gate": payload["numerical_gate"],
            "selected_row_count": len(selected),
            "source_mutation": 0,
            "protected_reads": [],
            "validated_roots": [binding["dataset_root"]],
            "purpose": "C3_S3A_SYNTHETIC_GEOMETRY_NUMERICAL_VALIDITY",
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
        return {"root": str(out), "status": summary["status"], "canonical_digest": payload["canonical_digest"], "sha256sums_sha256": seal["sha256sums_sha256"], "selected_row_count": len(selected)}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
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


def compare_runs(run_a: Path, run_b: Path, out: Path) -> Dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"comparison root already exists: {out}")
    summaries = []
    for path in (run_a, run_b):
        c3.verify_sha256sums(path.resolve())
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        summaries.append(summary)
    if any(summary.get("status") != "PASS" for summary in summaries):
        raise ValueError("run A/B must both be PASS")
    if summaries[0].get("canonical_digest") != summaries[1].get("canonical_digest"):
        raise ValueError("run A/B canonical digest mismatch")
    if summaries[0].get("coverage") != summaries[1].get("coverage"):
        raise ValueError("run A/B coverage mismatch")
    if summaries[0].get("metrics") != summaries[1].get("metrics"):
        raise ValueError("run A/B metrics mismatch")
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
            "protected_reads": [],
            "source_mutation": 0,
            "model_inference": False,
            "rollout": False,
            "attack": False,
        }
        _json(staging / "comparison.json", comparison)
        _json(staging / "summary.json", comparison)
        seal = _seal(staging, out.resolve(), comparison)
        return {"root": str(out.resolve()), "status": "PASS", "canonical_digest": comparison["canonical_digest"], "sha256sums_sha256": seal["sha256sums_sha256"]}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("generate", "smoke", "run", "compare"), required=True)
    parser.add_argument("--root")
    parser.add_argument("--allowlist")
    parser.add_argument("--out")
    parser.add_argument("--run-id")
    parser.add_argument("--category", choices=tuple(EXPECTED))
    parser.add_argument("--run-a")
    parser.add_argument("--run-b")
    parser.add_argument("--gate-split", default=str(HERE.parents[2] / "configs" / "C3_S3A_D0_GATE_SPLIT_V1.json"))
    args = parser.parse_args()
    try:
        if args.mode == "generate":
            if not all((args.root, args.allowlist)):
                raise ValueError("generate requires --root and --allowlist")
            result = generate_dataset(Path(args.root), Path(args.allowlist), Path(args.gate_split))
        elif args.mode in {"smoke", "run"}:
            if not all((args.root, args.allowlist, args.out, args.run_id)):
                raise ValueError(f"{args.mode} requires --root --allowlist --out --run-id")
            result = run_audit(Path(args.root), Path(args.allowlist), Path(args.out), args.run_id, args.mode, args.category)
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
