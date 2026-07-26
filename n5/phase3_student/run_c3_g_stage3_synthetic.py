"""Sealed, independently consumable C3-G-DEV Stage 3 synthetic replay.

The old Stage 3 root is historical and intentionally untouched.  This R1
path persists every case input separately from its result and binds both by a
content hash, so a reviewer can recompute the tri-state without importing the
producer's case generator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import c3_s3_geometry_observability as c3  # noqa: E402
import c3_s3_input_contract as input_contract  # noqa: E402
from c3_g_predicate_evaluator import CONTRACT_PATH, evaluate_case, load_contract  # noqa: E402
from run_c3_s3a_fresh_synthetic import _seal, verify_sealed_output  # noqa: E402


SCHEMA = "C3_G_PREDICATE_STAGE3_SYNTHETIC_V2"
CASE_KINDS = ("TRUE", "FALSE", "BOUNDARY", "IDENTITY_MISMATCH", "POSE_HARD_NEGATIVE", "UNKNOWN")
EXPECTED_CASES = 264
EXPECTED_RELATIONS = 44
REQUIRED_INPUT_FIELDS = {
    "case_id", "task_key", "relation_id", "relation_index", "predicate", "case_kind", "expected_tri_state",
    "object_id", "object_role", "object_world_position", "object_world_quaternion_wxyz", "object_half_extents", "object_geometry_source",
    "target_id", "target_role", "target_world_position", "target_world_quaternion_wxyz", "target_half_extents", "target_geometry_source",
    "state_namespace", "episode_id", "step", "contract_sha256", "evaluator_source_sha256", "target_pose_source_digest",
}


def _sha256(path: Path) -> str:
    return input_contract.sha256_file(path)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_json_value(value: Any) -> Any:
    """Canonicalize non-finite values without emitting invalid JSON."""
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "NaN"}
        if math.isinf(value):
            return {"__float__": "Infinity" if value > 0 else "-Infinity"}
        return value
    if isinstance(value, Mapping):
        return {str(key): _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json_value(item) for item in value]
    return value


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_safe_json_value(value), allow_nan=False, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _canonical(value: Any) -> bytes:
    return json.dumps(_safe_json_value(value), allow_nan=False, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _strict_loads(raw: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-canonical JSON constant: {value}")
    return json.loads(raw, parse_constant=reject_constant)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = _strict_loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_no} is not an object: {path}")
        rows.append(value)
    return rows


def _hash_without(row: Mapping[str, Any], field: str) -> str:
    value = dict(row)
    value.pop(field, None)
    return _sha256_bytes(_canonical(value))


def _target_role(predicate: str) -> str:
    return "REGION_TARGET" if predicate == "In" else "OBJECT_TARGET"


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


def _reference_info(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    if isinstance(value.get("pose"), Mapping):
        pose = value["pose"]
        digest = str(value.get("source_digest") or _sha256_bytes(_canonical(pose)))
        return pose, digest
    return value, _sha256_bytes(_canonical(value))


def build_predicate_cases(manifest: Mapping[str, Any], reference_poses: Mapping[str, Mapping[str, Any]], contract: Mapping[str, Any]) -> List[Dict[str, Any]]:
    containment_margin = float(contract["tolerance"]["containment_margin_m"])
    contract_sha = _sha256(CONTRACT_PATH)
    evaluator_sha = _sha256(HERE)
    cases: List[Dict[str, Any]] = []
    for entry in manifest["episodes"]:
        predicate = {"STATIC": "In", "DYNAMIC": "On", "ARTICULATED": "Stack"}[entry["category"]]
        target_pose, target_digest = _reference_info(reference_poses[entry["episode_id"]])
        target_extents = [1.0, 1.0, 1.0]
        object_extents = [0.1, 0.1, 0.1]
        target_role = _target_role(predicate)
        true_local = [0.0, 0.0, 0.0] if predicate == "In" else [0.0, 0.0, 1.1]
        false_local = [1.2, 0.0, 0.0] if predicate == "In" else [0.0, 0.0, 1.2]
        boundary_local = [0.9 + containment_margin, 0.0, 0.0] if predicate == "In" else [0.0, 0.0, 1.1 + containment_margin]
        hard_local = [0.9 + 2.0 * containment_margin, 0.0, 0.0] if predicate == "In" else [0.9 + 2.0 * containment_margin, 0.0, 1.1]
        local_by_kind = {"TRUE": true_local, "FALSE": false_local, "BOUNDARY": boundary_local, "IDENTITY_MISMATCH": true_local, "POSE_HARD_NEGATIVE": hard_local, "UNKNOWN": true_local}
        expected = {"TRUE": "TRUE", "FALSE": "FALSE", "BOUNDARY": "TRUE", "IDENTITY_MISMATCH": "UNKNOWN", "POSE_HARD_NEGATIVE": "FALSE", "UNKNOWN": "UNKNOWN"}
        relation_index = int(entry.get("relation_index", 0))
        task_key = str(entry.get("task_key", "synthetic/unknown"))
        for kind in CASE_KINDS:
            object_id = f"{entry['episode_id']}:object"
            target_id = f"{entry['episode_id']}:target"
            object_pose = _compose(target_pose, local_by_kind[kind])
            target_case_role = "UNKNOWN_ROLE" if kind == "UNKNOWN" else target_role
            case_id = f"{task_key}|{entry['relation_id']}|{kind}"
            case: Dict[str, Any] = {
                "case_id": case_id,
                "task_key": task_key,
                "relation_id": entry["relation_id"],
                "relation_index": relation_index,
                "predicate": predicate,
                "case_kind": kind,
                "expected_tri_state": expected[kind],
                "expected_value": expected[kind],
                "object_id": object_id,
                "object_role": "MANIPULATED_OBJECT",
                "object_world_position": list(object_pose["pos"]),
                "object_world_quaternion_wxyz": list(object_pose["quat"]),
                "object_half_extents": object_extents,
                "object_geometry_source": {"kind": "SYNTHETIC_SOURCE_RECONSTRUCTION", "computation_chain_id": "C3_S3A_SOURCE_RECONSTRUCTION_V2"},
                "target_id": target_id,
                "target_role": target_case_role,
                "target_world_position": list(target_pose["pos"]),
                "target_world_quaternion_wxyz": list(target_pose["quat"]),
                "target_half_extents": target_extents,
                "target_geometry_source": {"kind": "INDEPENDENT_MUJOCO_WORLD_REFERENCE", "computation_chain_id": "C3_S3A_REFERENCE_MUJOCO_JOINT_CHAIN_V2", "sha256": target_digest},
                "state_namespace": "C3_S3A_SYNTHETIC_V1",
                "episode_id": entry["episode_id"],
                "step": 0,
                "contract_sha256": contract_sha,
                "evaluator_source_sha256": evaluator_sha,
                "target_pose_source_digest": target_digest,
                "reconstruction_kind": "ARTICULATED_JOINT_CHAIN" if entry["category"] == "ARTICULATED" else entry["category"],
                "joint_qpos_binding": entry.get("joint_chain"),
                "object": {"id": object_id, "role": "MANIPULATED_OBJECT", "pose": object_pose, "half_extents": object_extents},
                "target": {"id": target_id, "role": target_case_role, "pose": target_pose, "half_extents": target_extents, "stackable": predicate == "Stack"},
                "expected_identity": {"episode_id": entry["episode_id"], "step": 0, "object_id": object_id, "target_id": target_id},
            }
            if kind == "IDENTITY_MISMATCH":
                case["expected_identity"]["step"] = 1
            case["case_input_sha256"] = _hash_without(case, "case_input_sha256")
            cases.append(case)
    return cases


def _load_inputs(dataset_root: Path, allowlist_path: Path, reference_run_a: Path, reference_run_b: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    allowlist = input_contract.load_allowlist(allowlist_path.resolve())
    manifest_path = dataset_root.resolve() / "dataset_manifest.json"
    audited, inventory = c3.audit_episode_manifest(manifest_path, allowlist)
    if inventory["status"] != "PASS" or len(audited) != 44:
        raise ValueError("C3-S3A synthetic input is not a sealed 44-episode root")
    manifest = _strict_loads(manifest_path.read_text(encoding="utf-8"))
    input_contract.verify_manifest_binding(dataset_root.resolve(), next(item for item in allowlist["allowed_episode_geometry_roots"] if item["path"] == str(dataset_root.resolve())))
    reference_poses: dict[str, Any] = {}
    for entry in manifest["episodes"]:
        identity = {key: entry[key] for key in ("task_key", "suite", "task_idx", "state_id", "init_seed")}
        path = Path(entry["reference_telemetry"]["path"])
        rows = input_contract.load_jsonl_exact(path, episode_id=entry["episode_id"], step_count=entry["step_count"], role="reference", identity=identity)
        reference_poses[entry["episode_id"]] = {"pose": rows[0]["entities"][0]["world_pose"], "source_digest": entry["reference_telemetry"]["sha256"]}
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


def _case_input(case: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(case)
    value.pop("case_input_sha256", None)
    value["case_input_sha256"] = _hash_without(value, "case_input_sha256")
    if not REQUIRED_INPUT_FIELDS.issubset(value):
        raise ValueError(f"case input missing fields: {sorted(REQUIRED_INPUT_FIELDS - set(value))}")
    return value


def _record(case: Mapping[str, Any], observed: Mapping[str, Any], case_input_sha256: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "case_id": case["case_id"],
        "episode_id": case["episode_id"],
        "step": case["step"],
        "task_key": case["task_key"],
        "relation_id": case["relation_id"],
        "relation_index": case["relation_index"],
        "predicate": case["predicate"],
        "case_kind": case["case_kind"],
        "expected_tri_state": case["expected_tri_state"],
        "observed_tri_state": observed["value"],
        "reason_code": observed["reason"],
        "raw_measurements": observed.get("raw_measurements", {}),
        "case_input_sha256": case_input_sha256,
        "pass": observed["value"] == case["expected_tri_state"],
    }
    value["record_sha256"] = _hash_without(value, "record_sha256")
    return value


def _counts(records: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in records).items()))


def _evaluate(cases: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inputs = [_case_input(case) for case in cases]
    records = [_record(case, evaluate_case(case, contract), case["case_input_sha256"]) for case in inputs]
    relation_counts = _counts(records, "relation_id")
    duplicate_case_ids = sorted(case_id for case_id, count in Counter(row["case_id"] for row in inputs).items() if count > 1)
    expected_state_counts = _counts(inputs, "expected_tri_state")
    summary: dict[str, Any] = {
        "status": "PASS" if len(inputs) == EXPECTED_CASES and len(relation_counts) == EXPECTED_RELATIONS and set(relation_counts.values()) == {6} and all(row["pass"] for row in records) and not duplicate_case_ids else "FAIL",
        "case_input_count": len(inputs),
        "predicate_record_count": len(records),
        "record_count": len(records),
        "relation_count": len(relation_counts),
        "per_relation_count": relation_counts,
        "per_predicate_count": _counts(records, "predicate"),
        "per_case_kind_count": _counts(records, "case_kind"),
        "case_kinds": {kind: sum(row["case_kind"] == kind for row in records) for kind in CASE_KINDS},
        "per_expected_state_count": expected_state_counts,
        "per_observed_state_count": _counts(records, "observed_tri_state"),
        "per_reason_code_count": _counts(records, "reason_code"),
        "duplicate_case_ids": duplicate_case_ids,
        "failed_records": [row["case_id"] for row in records if not row["pass"]],
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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_safe_json_value(row), allow_nan=False, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def _stream_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_canonical(list(rows)))


def run_evaluation(dataset_root: Path, allowlist_path: Path, reference_run_a: Path, reference_run_b: Path, out: Path, run_id: str) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(out)
    manifest, reference_poses, binding = _load_inputs(dataset_root, allowlist_path, reference_run_a, reference_run_b)
    frozen = load_contract()
    cases = build_predicate_cases(manifest, reference_poses, frozen)
    inputs = [_case_input(case) for case in cases]
    records, summary = _evaluate(inputs, frozen)
    input_digest = _stream_digest(inputs)
    output_digest = _stream_digest(records)
    canonical = {
        "schema": SCHEMA,
        "predicate_contract_sha256": _sha256(CONTRACT_PATH),
        "evaluator_source_sha256": _sha256(HERE),
        "dataset_manifest_sha256": binding["dataset_manifest_sha256"],
        "case_inputs": inputs,
        "predicate_records": records,
        "summary": {key: value for key, value in summary.items() if key != "failed_records"},
        "input_canonical_digest": input_digest,
        "output_canonical_digest": output_digest,
    }
    canonical_digest = _sha256_bytes(_canonical(canonical))
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".staging_{out.name}_{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        _write_jsonl(staging / "case_inputs.jsonl", inputs)
        _write_jsonl(staging / "predicate_records.jsonl", records)
        summary.update({"schema": SCHEMA, "canonical_digest": canonical_digest, "input_canonical_digest": input_digest, "output_canonical_digest": output_digest, "run_id": run_id, "input_output_sha_mismatches": []})
        _json(staging / "input_binding.json", {**binding, "predicate_contract_path": str(CONTRACT_PATH), "predicate_contract_sha256": _sha256(CONTRACT_PATH), "evaluator_source_path": str(HERE), "evaluator_source_sha256": _sha256(HERE), "run_id": run_id, "protected_reads": [], "tolerance_provenance": frozen["tolerance_provenance"]})
        _json(staging / "canonical_payload.json", canonical)
        _json(staging / "summary.json", summary)
        seal = _seal(staging, out, {"schema": SCHEMA, "canonical_digest": canonical_digest, "input_canonical_digest": input_digest, "output_canonical_digest": output_digest, "input_binding": {**binding, "predicate_contract_sha256": _sha256(CONTRACT_PATH), "evaluator_source_sha256": _sha256(HERE)}, "protected_reads": []})
        verify_sealed_output(out)
        return {"root": str(out), "status": summary["status"], "canonical_digest": canonical_digest, "sha256sums_sha256": seal["sha256sums_sha256"]}
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _verify_run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    verify_sealed_output(root)
    inputs = _read_jsonl(root / "case_inputs.jsonl")
    records = _read_jsonl(root / "predicate_records.jsonl")
    if len(inputs) != EXPECTED_CASES or len(records) != EXPECTED_CASES:
        raise ValueError("case input/output count is not 264")
    input_ids = [str(row.get("case_id")) for row in inputs]
    output_ids = [str(row.get("case_id")) for row in records]
    duplicates = sorted(case_id for case_id, count in Counter(input_ids + output_ids).items() if count > 2)
    if len(set(input_ids)) != EXPECTED_CASES or len(set(output_ids)) != EXPECTED_CASES:
        raise ValueError("duplicate case IDs in Stage 3 streams")
    if set(input_ids) != set(output_ids):
        raise ValueError("missing or extra case IDs between streams")
    by_id = {row["case_id"]: row for row in inputs}
    mismatches: list[str] = []
    for row in inputs:
        if set(REQUIRED_INPUT_FIELDS) - set(row):
            raise ValueError(f"case input missing required field: {row.get('case_id')}")
        if row.get("case_input_sha256") != _hash_without(row, "case_input_sha256"):
            mismatches.append(f"input:{row['case_id']}")
    for row in records:
        if row.get("record_sha256") != _hash_without(row, "record_sha256"):
            mismatches.append(f"output:{row['case_id']}")
        if row.get("case_input_sha256") != by_id[row["case_id"]].get("case_input_sha256"):
            mismatches.append(f"binding:{row['case_id']}")
    if mismatches:
        raise ValueError(f"input/output SHA binding mismatch: {mismatches[:5]}")
    summary = _strict_loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise ValueError("Stage 3 root status is not PASS")
    if summary.get("case_input_count") != EXPECTED_CASES or summary.get("predicate_record_count") != EXPECTED_CASES or summary.get("relation_count") != EXPECTED_RELATIONS:
        raise ValueError("Stage 3 summary closure is incomplete")
    relation_counts = _counts(records, "relation_id")
    if set(relation_counts.values()) != {6}:
        raise ValueError("each relation must have six cases")
    if summary.get("per_relation_count") != relation_counts or summary.get("per_case_kind_count") != _counts(records, "case_kind"):
        raise ValueError("Stage 3 summary counts do not match streams")
    input_digest = _stream_digest(inputs)
    output_digest = _stream_digest(records)
    if summary.get("input_canonical_digest") != input_digest or summary.get("output_canonical_digest") != output_digest:
        raise ValueError("Stage 3 stream digest mismatch")
    return {"root": root, "inputs": inputs, "records": records, "summary": summary, "input_ids": set(input_ids), "output_ids": set(output_ids), "input_canonical_digest": input_digest, "output_canonical_digest": output_digest, "duplicate_case_ids": duplicates, "input_output_sha_mismatches": mismatches}


def compare_evaluations(run_a: Path, run_b: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(out)
    checked = [_verify_run(run_a), _verify_run(run_b)]
    if checked[0]["input_ids"] != checked[1]["input_ids"] or checked[0]["output_ids"] != checked[1]["output_ids"]:
        raise ValueError("A/B case ID closure mismatch")
    if checked[0]["input_canonical_digest"] != checked[1]["input_canonical_digest"]:
        raise ValueError("A/B input canonical digest mismatch")
    if checked[0]["output_canonical_digest"] != checked[1]["output_canonical_digest"]:
        raise ValueError("A/B output canonical digest mismatch")
    summaries = [item["summary"] for item in checked]
    if summaries[0].get("canonical_digest") != summaries[1].get("canonical_digest"):
        raise ValueError("A/B canonical digest mismatch")
    relation_counts = _counts(checked[0]["records"], "relation_id")
    case_kind_counts = _counts(checked[0]["records"], "case_kind")
    comparison = {
        "schema": "C3_G_STAGE3_COMPARISON_V2",
        "status": "PASS",
        "run_A": str(Path(run_a).resolve()),
        "run_B": str(Path(run_b).resolve()),
        "case_input_count": EXPECTED_CASES,
        "predicate_record_count": EXPECTED_CASES,
        "relation_count": EXPECTED_RELATIONS,
        "per_relation_count": relation_counts,
        "per_predicate_count": _counts(checked[0]["records"], "predicate"),
        "per_case_kind_count": case_kind_counts,
        "per_expected_state_count": _counts(checked[0]["inputs"], "expected_tri_state"),
        "per_observed_state_count": _counts(checked[0]["records"], "observed_tri_state"),
        "per_reason_code_count": _counts(checked[0]["records"], "reason_code"),
        "missing_input_ids": [],
        "missing_output_ids": [],
        "extra_case_ids": [],
        "duplicate_case_ids": [],
        "input_output_sha_mismatches": [],
        "a_b_input_canonical_digest": checked[0]["input_canonical_digest"],
        "a_b_output_canonical_digest": checked[0]["output_canonical_digest"],
        "canonical_digest": summaries[0]["canonical_digest"],
        "canonical_identical": True,
        "protected_reads": [],
        "source_mutation": 0,
        "model_inference": False,
        "teacher_labeling": False,
        "student_training": False,
        "rollout": False,
        "attack": False,
    }
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".staging_{out.name}_{uuid.uuid4().hex}"
    staging.mkdir()
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
        print(json.dumps(result, allow_nan=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"C3-G Stage 3 HOLD: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
