"""C3-G-DEV Stage 3 synthetic tri-state replay.

Consumes only the sealed C3-S3A synthetic geometry root and emits sealed,
geometry-only predicate cases. No task outcomes, Teacher labels, or models are
read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import c3_s3_geometry_observability as c3  # noqa: E402
import c3_s3_input_contract as input_contract  # noqa: E402
from c3_g_predicate_evaluator import CONTRACT_PATH, evaluate_case, load_contract  # noqa: E402
from run_c3_s3a_fresh_synthetic import _canonical, _seal, verify_sealed_output  # noqa: E402


SCHEMA = "C3_G_PREDICATE_STAGE3_SYNTHETIC_V1"
CASE_KINDS = ("TRUE", "FALSE", "BOUNDARY", "IDENTITY_MISMATCH", "POSE_HARD_NEGATIVE", "UNKNOWN")
EXPECTED_CATEGORIES = {"STATIC": 11, "DYNAMIC": 31, "ARTICULATED": 2}


def _sha256(path: Path) -> str:
    return input_contract.sha256_file(path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _quat_mul(left: Sequence[float], right: Sequence[float]) -> list[float]:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def _rotate(quaternion: Sequence[float], vector: Sequence[float]) -> list[float]:
    w, x, y, z = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx]


def _compose(parent: Mapping[str, Sequence[float]], local_pos: Sequence[float]) -> dict[str, list[float]]:
    rotated = _rotate(parent["quat"], local_pos)
    return {"pos": [float(parent["pos"][i]) + rotated[i] for i in range(3)], "quat": [float(x) for x in parent["quat"]]}


def _target_role(predicate: str) -> str:
    return "REGION_TARGET" if predicate == "In" else "OBJECT_TARGET"


def build_predicate_cases(manifest: Mapping[str, Any], reference_poses: Mapping[str, Mapping[str, Any]], contract: Mapping[str, Any]) -> List[Dict[str, Any]]:
    tolerance = float(contract["tolerance"]["position_m"])
    cases: List[Dict[str, Any]] = []
    for entry in manifest["episodes"]:
        predicate = {"STATIC": "In", "DYNAMIC": "On", "ARTICULATED": "Stack"}[entry["category"]]
        target_pose = reference_poses[entry["episode_id"]]
        target_extents = [1.0, 1.0, 1.0]
        object_extents = [0.1, 0.1, 0.1]
        target_role = _target_role(predicate)
        true_local = [0.0, 0.0, 0.0] if predicate == "In" else [0.0, 0.0, 1.1]
        false_local = [1.2, 0.0, 0.0] if predicate == "In" else [0.0, 0.0, 1.2]
        boundary_local = [0.9 + tolerance, 0.0, 0.0] if predicate == "In" else [0.0, 0.0, 1.1 + tolerance]
        hard_local = [0.9 + 2.0 * tolerance, 0.0, 0.0] if predicate == "In" else [0.9 + 2.0 * tolerance, 0.0, 1.1]
        local_by_kind = {
            "TRUE": true_local,
            "FALSE": false_local,
            "BOUNDARY": boundary_local,
            "IDENTITY_MISMATCH": true_local,
            "POSE_HARD_NEGATIVE": hard_local,
            "UNKNOWN": true_local,
        }
        expected_value = {
            "TRUE": "TRUE",
            "FALSE": "FALSE",
            "BOUNDARY": "TRUE",
            "IDENTITY_MISMATCH": "UNKNOWN",
            "POSE_HARD_NEGATIVE": "FALSE",
            "UNKNOWN": "UNKNOWN",
        }
        for kind in CASE_KINDS:
            object_pose = _compose(target_pose, local_by_kind[kind])
            target_case_role = "UNKNOWN_ROLE" if kind == "UNKNOWN" else target_role
            case = {
                "episode_id": entry["episode_id"],
                "step": 0,
                "relation_id": entry["relation_id"],
                "predicate": predicate,
                "case_kind": kind,
                "object": {"id": f"{entry['episode_id']}:object", "role": "MANIPULATED_OBJECT", "pose": object_pose, "half_extents": object_extents},
                "target": {"id": f"{entry['episode_id']}:target", "role": target_case_role, "pose": target_pose, "half_extents": target_extents, "stackable": predicate == "Stack"},
                "expected_identity": {"episode_id": entry["episode_id"], "step": 0, "object_id": f"{entry['episode_id']}:object", "target_id": f"{entry['episode_id']}:target"},
                "expected_value": expected_value[kind],
            }
            if kind == "IDENTITY_MISMATCH":
                case["expected_identity"]["step"] = 1
            cases.append(case)
    return cases


def _load_inputs(dataset_root: Path, allowlist_path: Path, reference_run_a: Path, reference_run_b: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    allowlist = input_contract.load_allowlist(allowlist_path.resolve())
    manifest_path = dataset_root.resolve() / "dataset_manifest.json"
    audited, inventory = c3.audit_episode_manifest(manifest_path, allowlist)
    if inventory["status"] != "PASS" or len(audited) != 44:
        raise ValueError("C3-S3A synthetic input is not a sealed 44-episode root")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_contract.verify_manifest_binding(dataset_root.resolve(), next(item for item in allowlist["allowed_episode_geometry_roots"] if item["path"] == str(dataset_root.resolve())))
    reference_poses: dict[str, Any] = {}
    for entry in manifest["episodes"]:
        identity = {key: entry[key] for key in ("task_key", "suite", "task_idx", "state_id", "init_seed")}
        path = Path(entry["reference_telemetry"]["path"])
        rows = input_contract.load_jsonl_exact(path, episode_id=entry["episode_id"], step_count=entry["step_count"], role="reference", identity=identity)
        reference_poses[entry["episode_id"]] = rows[0]["entities"][0]["world_pose"]
    for external in (reference_run_a.resolve(), reference_run_b.resolve()):
        manifest_file = external / "MANIFEST.json"
        input_contract.verify_manifest_binding(external, {"manifest_path": "MANIFEST.json", "manifest_sha256": _sha256(manifest_file), "root_sha256s_sha256": _sha256(external / "SHA256SUMS")})
    return manifest, reference_poses, {
        "dataset_manifest_sha256": _sha256(manifest_path),
        "dataset_root_sha256s_sha256": _sha256(dataset_root / "SHA256SUMS"),
        "allowlist_sha256": _sha256(allowlist_path.resolve()),
        "reference_run_a": str(reference_run_a.resolve()),
        "reference_run_b": str(reference_run_b.resolve()),
        "reference_run_a_sha256s_sha256": _sha256(reference_run_a / "SHA256SUMS"),
        "reference_run_b_sha256s_sha256": _sha256(reference_run_b / "SHA256SUMS"),
        "code_snapshot_commit": manifest.get("code_snapshot_commit"),
        "code_snapshot_tree": manifest.get("code_snapshot_tree"),
    }


def _evaluate(cases: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    for case in cases:
        observed = evaluate_case(case, contract)
        records.append({"episode_id": case["episode_id"], "relation_id": case["relation_id"], "step": case["step"], "predicate": case["predicate"], "case_kind": case["case_kind"], "expected_value": case["expected_value"], "observed_value": observed["value"], "reason": observed["reason"], "pass": observed["value"] == case["expected_value"]})
    relation_counts: dict[str, int] = {}
    for record in records:
        relation_counts[record["relation_id"]] = relation_counts.get(record["relation_id"], 0) + 1
    summary = {
        "status": "PASS" if len(records) == 44 * 6 and all(record["pass"] for record in records) and set(relation_counts.values()) == {6} else "FAIL",
        "record_count": len(records),
        "relation_count": len(relation_counts),
        "case_kinds": {kind: sum(record["case_kind"] == kind for record in records) for kind in CASE_KINDS},
        "failed_records": [record for record in records if not record["pass"]],
        "protected_reads": [],
        "task_success_read": False,
        "reward_read": False,
        "teacher_read": False,
        "outcome_read": False,
        "model_inference": False,
        "training": False,
        "rollout": False,
        "attack": False,
    }
    return records, summary


def run_evaluation(dataset_root: Path, allowlist_path: Path, reference_run_a: Path, reference_run_b: Path, out: Path, run_id: str) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(out)
    manifest, reference_poses, binding = _load_inputs(dataset_root, allowlist_path, reference_run_a, reference_run_b)
    frozen = load_contract()
    cases = build_predicate_cases(manifest, reference_poses, frozen)
    records, summary = _evaluate(cases, frozen)
    canonical = {"schema": SCHEMA, "predicate_contract_sha256": _sha256(CONTRACT_PATH), "dataset_manifest_sha256": binding["dataset_manifest_sha256"], "records": records, "summary": {key: value for key, value in summary.items() if key != "failed_records"}}
    payload = {"canonical": canonical, "canonical_digest": _sha256_bytes(_canonical(canonical))}
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".staging_{out.name}_{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        _json(staging / "input_binding.json", {**binding, "predicate_contract_path": str(CONTRACT_PATH), "predicate_contract_sha256": _sha256(CONTRACT_PATH), "run_id": run_id, "protected_reads": []})
        _json(staging / "canonical_payload.json", payload["canonical"])
        _json(staging / "summary.json", {**summary, "schema": SCHEMA, "canonical_digest": payload["canonical_digest"], "run_id": run_id})
        with (staging / "predicate_records.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        seal = _seal(staging, out, {"schema": SCHEMA, "canonical_digest": payload["canonical_digest"], "input_binding": {**binding, "predicate_contract_sha256": _sha256(CONTRACT_PATH)}, "protected_reads": []})
        verify_sealed_output(out)
        return {"root": str(out), "status": summary["status"], "canonical_digest": payload["canonical_digest"], "sha256sums_sha256": seal["sha256sums_sha256"]}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def compare_evaluations(run_a: Path, run_b: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(out)
    summaries = []
    for path in (run_a.resolve(), run_b.resolve()):
        verify_sealed_output(path)
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "PASS" or summary.get("record_count") != 264 or summary.get("relation_count") != 44 or summary.get("case_kinds") != {kind: 44 for kind in CASE_KINDS} or summary.get("protected_reads") != []:
            raise ValueError("stage 3 run is not a complete PASS")
        summaries.append(summary)
    if summaries[0]["canonical_digest"] != summaries[1]["canonical_digest"]:
        raise ValueError("stage 3 A/B canonical mismatch")
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".staging_{out.name}_{uuid.uuid4().hex}"
    staging.mkdir()
    comparison = {"schema": "C3_G_STAGE3_COMPARISON_V1", "status": "PASS", "run_A": str(run_a.resolve()), "run_B": str(run_b.resolve()), "canonical_digest": summaries[0]["canonical_digest"], "record_count": 264, "relation_count": 44, "case_kinds": {kind: 44 for kind in CASE_KINDS}, "protected_reads": [], "model_inference": False, "training": False, "rollout": False, "attack": False}
    try:
        _json(staging / "comparison.json", comparison)
        _json(staging / "summary.json", comparison)
        seal = _seal(staging, out, comparison)
        verify_sealed_output(out)
        return {"root": str(out), "status": "PASS", "canonical_digest": comparison["canonical_digest"], "sha256sums_sha256": seal["sha256sums_sha256"]}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("run", "compare"), required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--allowlist")
    parser.add_argument("--reference-run-a")
    parser.add_argument("--reference-run-b")
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--run-a")
    parser.add_argument("--run-b")
    args = parser.parse_args()
    try:
        if args.mode == "run":
            if not all((args.dataset_root, args.allowlist, args.reference_run_a, args.reference_run_b, args.run_id)):
                raise ValueError("stage 3 run requires dataset, allowlist, reference A/B and run id")
            result = run_evaluation(Path(args.dataset_root), Path(args.allowlist), Path(args.reference_run_a), Path(args.reference_run_b), Path(args.out), args.run_id)
        else:
            if not all((args.run_a, args.run_b)):
                raise ValueError("stage 3 compare requires run A/B")
            result = compare_evaluations(Path(args.run_a), Path(args.run_b), Path(args.out))
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"C3-G Stage 3 HOLD: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
