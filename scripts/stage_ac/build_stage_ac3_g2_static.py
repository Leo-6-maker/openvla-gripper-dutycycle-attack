#!/usr/bin/env python3
"""Reconcile the append-only AC3 G2 branch evidence without new execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
SUITES = ("libero_10", "libero_object", "libero_spatial")
CONDITIONS = (("CLEAN_REFERENCE", 0), ("OPEN_T3", 3), ("OPEN_T5", 5), ("OPEN_T10", 10))
CONDITION_NAMES = {name for name, _dose in CONDITIONS}
PASS = "PASS"
INVALID = "ENGINEERING_INVALID_OR_HORIZON_CENSORED"
TERMINAL = {PASS, INVALID}
GATE = "STAGE_AC_AC3_AC4_AC5_TREATMENT_NAIVE_MULTI_MODEL_PHYSICAL_REPLICATION_PROGRAM_V1"
FORBIDDEN_COUNTERS = (
    "pgd_calls",
    "attacked_env_steps",
    "protected_reads",
    "eval160_reads",
    "ac2_exposure",
    "attack_outcome_reads",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical(value))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = path.read_bytes()
    return json.loads(data.decode("utf-8")), {"path": str(path), "bytes": len(data), "sha256": sha256_bytes(data)}


def file_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": sha256_bytes(data)}


def write_new(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    require(not path.exists(), f"AC3_G2_APPEND_ONLY_OUTPUT_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return {"path": str(path), "bytes": len(data), "sha256": sha256_bytes(data)}


def under(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"AC3_G2_{label}_PATH_ESCAPE:{path}") from exc


def load_branch_files(directory: Path, pattern: str, sidecars: bool = False) -> dict[str, tuple[Path, dict[str, Any], dict[str, Any]]]:
    require(directory.is_dir(), f"AC3_G2_EVIDENCE_DIR_MISSING:{directory}")
    result: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    for path in sorted(directory.glob(pattern)):
        data, record = read_json(path)
        branch_id = str(data.get("branch_id"))
        require(branch_id.startswith("AC3-"), f"AC3_G2_BRANCH_ID_MISSING:{path}")
        if sidecars:
            expected = path.name[len("FAILURE_") : -len("_V7.json")]
            require(path.name.startswith("FAILURE_AC3-") and path.name.endswith("_V7.json") and expected == branch_id, f"AC3_G2_SIDECAR_FILENAME_MISMATCH:{path}")
        else:
            require(path.stem == branch_id, f"AC3_G2_RECEIPT_FILENAME_MISMATCH:{path}")
        require(branch_id not in result, f"AC3_G2_DUPLICATE_BRANCH_RECORD:{branch_id}:{directory}")
        result[branch_id] = (path, data, record)
    return result


def check_runtime_authority(root: Path, path: Path) -> dict[str, Any]:
    authority, record = read_json(path)
    require(authority.get("schema") == "STAGE_AC_AC3Q_RUNTIME_SOURCE_AUTHORITY_V9", "AC3_G2_RUNTIME_AUTHORITY_SCHEMA_INVALID")
    require(authority.get("status") == "STAGE_AC_AC3Q_RUNTIME_SOURCE_AUTHORITY_FROZEN", "AC3_G2_RUNTIME_AUTHORITY_NOT_FROZEN")
    for entry in authority.get("runtime_files", {}).values():
        runtime_path = root / str(entry["path"])
        require(runtime_path.is_file(), f"AC3_G2_RUNTIME_FILE_MISSING:{runtime_path}")
        data = runtime_path.read_bytes()
        require(len(data) == int(entry["bytes"]) and sha256_bytes(data) == str(entry["sha256"]), f"AC3_G2_RUNTIME_FILE_MISMATCH:{runtime_path}")
    return record


def check_video(video: Any, evidence_root: Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if video is None:
        return None
    require(isinstance(video, dict), "AC3_G2_VIDEO_METADATA_INVALID")
    path = Path(str(video.get("path"))).resolve()
    under(path, evidence_root / "videos", "VIDEO")
    require(path.is_file(), f"AC3_G2_VIDEO_MISSING:{path}")
    key = str(path)
    if key not in cache:
        cache[key] = file_record(path)
    actual = cache[key]
    require(actual["bytes"] == int(video.get("bytes")), f"AC3_G2_VIDEO_BYTES_MISMATCH:{path}")
    require(actual["sha256"] == str(video.get("sha256")), f"AC3_G2_VIDEO_SHA_MISMATCH:{path}")
    require(int(video.get("fps", 0)) == 10 and int(video.get("frames", 0)) > 0, f"AC3_G2_VIDEO_METADATA_INVALID:{path}")
    return {**actual, "fps": int(video["fps"]), "frames": int(video["frames"])}


def validate_action(row: dict[str, Any], branch_id: str, step: int) -> None:
    require(int(row.get("step", -1)) == step, f"AC3_G2_ROW_STEP_INVALID:{branch_id}:{step}")
    for field in ("raw_policy_action", "normalized_action", "env_action"):
        action = row.get(field)
        require(isinstance(action, list) and len(action) == 7, f"AC3_G2_ACTION_DIM_INVALID:{branch_id}:{step}:{field}")
        require(all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in action), f"AC3_G2_ACTION_NONFINITE:{branch_id}:{step}:{field}")
    require(float(row.get("arm_delta_linf", 1.0)) <= 1e-7, f"AC3_G2_ROW_ARM_DRIFT:{branch_id}:{step}")


def validate_pass(job: dict[str, Any], data: dict[str, Any], evidence_root: Path, video_cache: dict[str, dict[str, Any]], sample_by_branch: dict[str, dict[str, Any]]) -> dict[str, Any]:
    branch_id = str(job["branch_id"])
    require(data.get("schema") == "STAGE_AC_AC3_BRANCH_RECEIPT_V1", f"AC3_G2_PASS_SCHEMA_INVALID:{branch_id}")
    require(data.get("gate") == GATE, f"AC3_G2_PASS_GATE_INVALID:{branch_id}")
    for key in ("cell_id", "parent_exposure_class", "scientific_claim"):
        require(data.get(key) == job.get(key), f"AC3_G2_PASS_BINDING_INVALID:{branch_id}:{key}")
    require(data.get("claim_boundary") == "AC3 primary physical branch execution; no promotion until G5", f"AC3_G2_PASS_CLAIM_BOUNDARY_INVALID:{branch_id}")
    for key in ("state_restore_exact", "causal_input_binding_pass", "control_action_reference_exact", "arm_preserved", "exact_open_delivery", "queue_reset_verified", "telemetry_aligned", "treatment_compliant"):
        require(data.get(key) is True, f"AC3_G2_PASS_INVARIANT_INVALID:{branch_id}:{key}")
    require(float(data.get("max_arm_delta_linf", 1.0)) <= 1e-7, f"AC3_G2_PASS_ARM_DRIFT:{branch_id}")
    require(int(data.get("anchor_step", -1)) == int(job["selected_anchor"]["step"]), f"AC3_G2_PASS_ANCHOR_STEP:{branch_id}")
    require(str(data.get("anchor_state_sha256")) == str(job["selected_anchor"]["boundary_state_sha256"]), f"AC3_G2_PASS_ANCHOR_STATE:{branch_id}")
    require(str(data.get("source_clean_trajectory_digest")) == str(job["selected_anchor"]["source_clean_trajectory_digest"]), f"AC3_G2_PASS_TRAJECTORY_DIGEST:{branch_id}")
    rows = data.get("rows")
    condition = str(job["condition"])
    dose = int(job["dose"])
    expected_rows = 20 if condition == "CLEAN_REFERENCE" else 10 + dose
    require(isinstance(rows, list) and len(rows) == expected_rows, f"AC3_G2_PASS_ROWS_INVALID:{branch_id}:{len(rows) if isinstance(rows, list) else None}:{expected_rows}")
    anchor = int(job["selected_anchor"]["step"])
    for offset, row in enumerate(rows):
        require(isinstance(row, dict), f"AC3_G2_PASS_ROW_NOT_OBJECT:{branch_id}:{offset}")
        validate_action(row, branch_id, anchor + offset)
    expected_open = 0 if condition == "CLEAN_REFERENCE" else dose
    require(int(data.get("open_intervention_steps", -1)) == expected_open, f"AC3_G2_PASS_OPEN_COUNT:{branch_id}")
    compliance = data.get("treatment_compliance") or {}
    require(bool(compliance.get("command_delivery_valid")) and int(compliance.get("delivered_open_steps", -1)) == expected_open, f"AC3_G2_PASS_TREATMENT_COMPLIANCE:{branch_id}")
    require(data.get("queue_boundary_steps") == data.get("expected_queue_boundary_steps"), f"AC3_G2_PASS_QUEUE_BOUNDARY:{branch_id}")
    treatment = data.get("treatment_receipts") or []
    require(isinstance(treatment, list) and len(treatment) == expected_open, f"AC3_G2_PASS_TREATMENT_RECEIPTS:{branch_id}")
    for item in treatment:
        action = item.get("env_action")
        require(isinstance(action, list) and len(action) == 7 and float(action[-1]) == -1.0, f"AC3_G2_PASS_NATIVE_OPEN:{branch_id}")
        require(float(item.get("arm_delta_linf", 1.0)) <= 1e-7, f"AC3_G2_PASS_TREATMENT_ARM_DRIFT:{branch_id}")
    video = check_video(data.get("video"), evidence_root, video_cache)
    if branch_id in sample_by_branch:
        require(video is not None, f"AC3_G2_BLIND_SAMPLE_PASS_VIDEO_MISSING:{branch_id}")
        require(data.get("blinded_video_id") == sample_by_branch[branch_id]["blinded_video_id"], f"AC3_G2_BLIND_ID_MISMATCH:{branch_id}")
    return {
        "status": PASS,
        "rows": len(rows),
        "expected_open_steps": expected_open,
        "video": video,
        "physical_class": data.get("physical_class"),
        "v_phys_label": data.get("v_phys_label"),
        "runtime_counters": dict(sorted((str(k), int(v)) for k, v in (data.get("runtime_counters") or {}).items())),
    }


def validate_invalid(job: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    branch_id = str(job["branch_id"])
    require(data.get("status") == INVALID, f"AC3_G2_INVALID_STATUS:{branch_id}")
    error = data.get("error") or {}
    require(isinstance(error, dict) and error.get("message"), f"AC3_G2_INVALID_ERROR_MISSING:{branch_id}")
    require(data.get("next_legal_action") == "STOP_FOR_PI", f"AC3_G2_INVALID_NEXT_ACTION:{branch_id}")
    counters = data.get("runtime_counters") or {}
    for key in FORBIDDEN_COUNTERS:
        require(int(counters.get(key, 0)) == 0, f"AC3_G2_INVALID_FORBIDDEN_COUNTER:{branch_id}:{key}")
    return {"status": INVALID, "error": error, "runtime_counters": dict(sorted((str(k), int(v)) for k, v in counters.items()))}


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    evidence_root = args.evidence_root.resolve()
    receipts_root = evidence_root / "receipts"
    recovery_root = evidence_root / "recovered_receipts_v7"
    g0, g0_record = read_json(args.g0_root)
    g1, g1_record = read_json(args.g1_root)
    manifest, manifest_record = read_json(args.manifest)
    blind_sample, blind_record = read_json(args.blind_sample)
    runtime_record = check_runtime_authority(root, args.runtime_authority)
    require(g0.get("status") == "STAGE_AC_AC3_PRELAUNCH_AUTHORITY_FROZEN_CONTINUE", "AC3_G2_G0_NOT_FROZEN")
    require(g0.get("gate") == GATE, "AC3_G2_G0_GATE_INVALID")
    require(g1.get("status") == "STAGE_AC_AC3Q_G1_ENGINEERING_QUALIFICATION_PASS_STOP_FOR_PI", "AC3_G2_G1_NOT_PASS")
    require(manifest.get("schema") == "STAGE_AC_AC3_G0_LAUNCH_MANIFEST_V1" and len(manifest.get("branches", [])) == 384, "AC3_G2_MANIFEST_INVALID")
    require(sha256_file(args.manifest) == g0["artifacts"]["launch_manifest"]["sha256"], "AC3_G2_MANIFEST_ROOT_SHA")
    require(blind_sample.get("schema") == "STAGE_AC_AC4_BLIND_AUDIT_SAMPLE_V1" and len(blind_sample.get("sample", [])) == 96, "AC3_G2_BLIND_SAMPLE_INVALID")

    jobs = {str(job["branch_id"]): job for job in manifest["branches"]}
    require(len(jobs) == 384, "AC3_G2_MANIFEST_BRANCH_DUPLICATE")
    for job in jobs.values():
        require(job.get("model_family") in MODELS and job.get("suite") in SUITES, f"AC3_G2_MANIFEST_MODEL_SUITE_INVALID:{job.get('branch_id')}")
        require(job.get("condition") in CONDITION_NAMES, f"AC3_G2_MANIFEST_CONDITION_INVALID:{job.get('branch_id')}")
    parent_conditions = defaultdict(set)
    for job in jobs.values():
        parent_conditions[(job["model_family"], job["suite"], job["canonical_parent_key"])].add(job["condition"])
    require(all(conditions == CONDITION_NAMES for conditions in parent_conditions.values()), "AC3_G2_MANIFEST_FOUR_CONDITIONS_INVALID")

    sample_by_branch = {str(row["branch_id"]): row for row in blind_sample["sample"]}
    require(len(sample_by_branch) == 96 and set(sample_by_branch).issubset(jobs), "AC3_G2_BLIND_SAMPLE_BINDING_INVALID")
    require(len({str(row["blinded_video_id"]) for row in blind_sample["sample"]}) == 96, "AC3_G2_BLIND_IDS_DUPLICATE")

    base = load_branch_files(receipts_root, "AC3-*.json")
    recovery = load_branch_files(recovery_root, "AC3-*.json")
    sidecars = load_branch_files(receipts_root, "FAILURE_AC3-*_V7.json", sidecars=True)
    require(set(base) == set(jobs), f"AC3_G2_BASE_MANIFEST_SET_INVALID:{len(base)}")
    require(set(recovery).issubset(jobs) and set(sidecars).issubset(jobs), "AC3_G2_APPEND_ONLY_SET_INVALID")
    for branch_id in recovery:
        require(base[branch_id][1].get("status") == "RUNNING", f"AC3_G2_RECOVERY_BASE_NOT_RUNNING:{branch_id}")
        require(recovery[branch_id][1].get("status") == PASS, f"AC3_G2_RECOVERY_NOT_PASS:{branch_id}")
        require((recovery[branch_id][1].get("receipt_recovery") or {}).get("status") == "RECOVERED_FROM_COMPLETE_RUNNING_RECEIPT", f"AC3_G2_RECOVERY_PROVENANCE_INVALID:{branch_id}")
    for branch_id in sidecars:
        require(base[branch_id][1].get("status") == "RUNNING", f"AC3_G2_SIDECAR_BASE_NOT_RUNNING:{branch_id}")
        require(sidecars[branch_id][1].get("status") == INVALID, f"AC3_G2_SIDECAR_NOT_INVALID:{branch_id}")

    source_cache: dict[str, dict[str, Any]] = {}
    video_cache: dict[str, dict[str, Any]] = {}
    authoritative: list[dict[str, Any]] = []
    aggregate = Counter()
    status_counts = Counter()
    authority_counts = Counter()
    shard_counts = Counter()
    invalid_rows: list[dict[str, Any]] = []
    referenced_video_paths: set[str] = set()

    for branch_id, job in sorted(jobs.items()):
        base_path, base_data, base_record = base[branch_id]
        selected_kind = "base_receipt"
        selected_path, selected_data, selected_record = base_path, base_data, base_record
        history: dict[str, Any] = {"base_receipt": base_record}
        if branch_id in recovery:
            selected_kind = "recovered_receipt"
            selected_path, selected_data, selected_record = recovery[branch_id]
            history["recovery_receipt"] = recovery[branch_id][2]
        if branch_id in sidecars:
            selected_kind = "failure_sidecar"
            selected_path, selected_data, selected_record = sidecars[branch_id]
            history["failure_sidecar"] = sidecars[branch_id][2]
        status = selected_data.get("status")
        require(status in TERMINAL, f"AC3_G2_NONTERMINAL_BRANCH:{branch_id}:{status}")
        if base_data.get("status") == "RUNNING":
            require(branch_id in recovery or branch_id in sidecars, f"AC3_G2_UNRESOLVED_RUNNING:{branch_id}")
        for key in ("branch_id", "model_family", "suite", "canonical_parent_key", "condition", "dose"):
            require(selected_data.get(key) == job.get(key), f"AC3_G2_BINDING_INVALID:{branch_id}:{key}")
        require(str(selected_data.get("source_receipt_path")) == str(job["source_receipt"]["path"]), f"AC3_G2_SOURCE_PATH_INVALID:{branch_id}")
        require(str(selected_data.get("source_receipt_sha256")) == str(job["source_receipt"]["sha256"]), f"AC3_G2_SOURCE_SHA_INVALID:{branch_id}")
        source_path = Path(str(job["source_receipt"]["path"])).resolve()
        if str(source_path) not in source_cache:
            require(source_path.is_file(), f"AC3_G2_SOURCE_RECEIPT_MISSING:{source_path}")
            source_data = source_path.read_bytes()
            require(len(source_data) == int(job["source_receipt"]["bytes"]), f"AC3_G2_SOURCE_RECEIPT_BYTES:{source_path}")
            require(sha256_bytes(source_data) == str(job["source_receipt"]["sha256"]), f"AC3_G2_SOURCE_RECEIPT_SHA:{source_path}")
            source_cache[str(source_path)] = {"path": str(source_path), "bytes": len(source_data), "sha256": sha256_bytes(source_data)}
        detail = validate_pass(job, selected_data, evidence_root, video_cache, sample_by_branch) if status == PASS else validate_invalid(job, selected_data)
        if detail.get("video"):
            referenced_video_paths.add(str(detail["video"]["path"]))
        counters = detail.get("runtime_counters", {})
        for key, value in counters.items():
            require(int(value) >= 0, f"AC3_G2_COUNTER_NEGATIVE:{branch_id}:{key}")
            aggregate[key] += int(value)
        for key in FORBIDDEN_COUNTERS:
            require(int(counters.get(key, 0)) == 0, f"AC3_G2_FORBIDDEN_COUNTER_NONZERO:{branch_id}:{key}")
        status_counts[status] += 1
        authority_counts[selected_kind] += 1
        shard_counts[(job["model_family"], job["suite"], status)] += 1
        row = {
            "branch_id": branch_id,
            "model_family": job["model_family"],
            "suite": job["suite"],
            "canonical_parent_key": job["canonical_parent_key"],
            "cell_id": job["cell_id"],
            "condition": job["condition"],
            "dose": int(job["dose"]),
            "parent_exposure_class": job["parent_exposure_class"],
            "status": status,
            "authority_source": selected_kind,
            "receipt": selected_record,
            "history": history,
            "source_receipt": source_cache[str(source_path)],
            "anchor_step": int(job["selected_anchor"]["step"]),
            "anchor_state_sha256": str(job["selected_anchor"]["boundary_state_sha256"]),
            "source_clean_trajectory_digest": str(job["selected_anchor"]["source_clean_trajectory_digest"]),
            "validation": detail,
        }
        if status == INVALID:
            invalid_rows.append({"branch_id": branch_id, "model_family": job["model_family"], "suite": job["suite"], "canonical_parent_key": job["canonical_parent_key"], "condition": job["condition"], "dose": int(job["dose"]), "authority_source": selected_kind, "receipt": selected_record, "history": history, "error": detail["error"]})
        authoritative.append(row)

    require(len(authoritative) == 384 and len({row["branch_id"] for row in authoritative}) == 384, "AC3_G2_AUTHORITATIVE_BRANCH_COUNT_INVALID")
    sample_missing: list[dict[str, Any]] = []
    for branch_id, sample in sorted(sample_by_branch.items()):
        row = next(item for item in authoritative if item["branch_id"] == branch_id)
        video = row["validation"].get("video")
        if video is None:
            sample_missing.append({"branch_id": branch_id, "blinded_video_id": sample["blinded_video_id"], "model_family": row["model_family"], "suite": row["suite"], "condition": row["condition"], "status": row["status"]})
    require(not [row for row in sample_missing if row["status"] == PASS], "AC3_G2_BLIND_SAMPLE_PASS_VIDEO_GAP")
    all_video_files = sorted((evidence_root / "videos").glob("*.mp4")) if (evidence_root / "videos").is_dir() else []
    unreferenced_videos = [str(path) for path in all_video_files if str(path.resolve()) not in referenced_video_paths]
    video_sha_groups: dict[str, list[str]] = defaultdict(list)
    for path, record in video_cache.items():
        video_sha_groups[record["sha256"]].append(path)
    duplicate_video_sha_groups = {sha: sorted(paths) for sha, paths in video_sha_groups.items() if len(paths) > 1}

    attempt_ledger = []
    for kind, records in (("base_receipt", base), ("recovered_receipt", recovery), ("failure_sidecar", sidecars)):
        for branch_id, (path, data, record) in sorted(records.items()):
            attempt_ledger.append({"branch_id": branch_id, "source": kind, "record": record, "status": data.get("status"), "condition": data.get("condition"), "model_family": data.get("model_family"), "suite": data.get("suite"), "error": data.get("error"), "receipt_recovery": data.get("receipt_recovery")})
    worker_summaries = []
    for path in sorted(receipts_root.glob("WORKER_*.json")):
        data, record = read_json(path)
        worker_summaries.append({"record": record, "status": data.get("status"), "model_family": data.get("model_family"), "suite": data.get("suite"), "jobs_completed": data.get("jobs_completed"), "jobs_new": data.get("jobs_new"), "jobs_existing": data.get("jobs_existing")})
    log_records = [file_record(path) for path in sorted((evidence_root / "logs").glob("*.log"))] if (evidence_root / "logs").is_dir() else []

    terminal_status = "STAGE_AC_AC3_G2_COMPLETE_BRANCH_EXECUTION_PASS_STOP_FOR_PI" if not invalid_rows else "STAGE_AC_AC3_G2_ENGINEERING_OR_HORIZON_HOLD_STOP_FOR_PI"
    next_action = "EXECUTE_G3_STATIC_ANALYSIS" if not invalid_rows else "STOP_FOR_PI_G2_ENGINEERING_OR_HORIZON_REVIEW"
    index = {
        "schema": "STAGE_AC_AC3_G2_BRANCH_RECEIPT_INDEX_V1",
        "status": "PASS_AC3_G2_COMPLETE_BRANCH_INDEX" if not invalid_rows else "HOLD_AC3_G2_ENGINEERING_OR_HORIZON",
        "gate": GATE,
        "claim_boundary": "AC3 G2 frozen model-suite branch execution reconciliation only; no G3 statistics or promotion",
        "evidence_root": str(evidence_root),
        "g0_root": g0_record,
        "g1_root": g1_record,
        "runtime_authority": runtime_record,
        "manifest": manifest_record,
        "blind_sample": blind_record,
        "counts": {
            "manifest_branches": len(jobs),
            "authoritative_terminal_branches": len(authoritative),
            "pass_branches": status_counts[PASS],
            "invalid_or_horizon_censored_branches": status_counts[INVALID],
            "base_receipts": len(base),
            "recovered_receipts": len(recovery),
            "failure_sidecars": len(sidecars),
            "verified_source_receipts": len(source_cache),
            "verified_videos": len(video_cache),
            "blind_sample_expected": len(sample_by_branch),
            "blind_sample_videos_present": len(sample_by_branch) - len(sample_missing),
            "blind_sample_videos_missing": len(sample_missing),
            "unreferenced_video_files": len(unreferenced_videos),
            "duplicate_video_sha_groups": len(duplicate_video_sha_groups),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "authority_source_counts": dict(sorted(authority_counts.items())),
        "shard_status_counts": {"|".join(key): value for key, value in sorted(shard_counts.items())},
        "aggregate_runtime_counters": dict(sorted(aggregate.items())),
        "scientific_firewall": {key: aggregate.get(key, 0) for key in FORBIDDEN_COUNTERS},
        "rows": authoritative,
        "invalid_or_horizon_censored": invalid_rows,
        "blind_sample_missing_videos": sample_missing,
        "video_sha_duplicate_groups": duplicate_video_sha_groups,
        "unreferenced_video_files": unreferenced_videos,
        "attempt_ledger": attempt_ledger,
        "worker_summaries": worker_summaries,
        "log_records": log_records,
        "next_legal_action": next_action,
    }
    terminal = {
        "schema": "STAGE_AC_AC3_G2_TERMINAL_V1",
        "status": terminal_status,
        "gate": GATE,
        "claim_boundary": index["claim_boundary"],
        "execution_summary": {
            "manifest_branches": len(jobs),
            "authoritative_terminal_branches": len(authoritative),
            "pass_branches": status_counts[PASS],
            "invalid_or_horizon_censored_branches": status_counts[INVALID],
            "recovered_clean_receipts": len(recovery),
            "failure_sidecars": len(sidecars),
        },
        "model_suite_status": {f"{family}|{suite}": {status: count for (f, s, status), count in shard_counts.items() if f == family and s == suite} for family in MODELS for suite in SUITES},
        "blind_sample": {"expected": len(sample_by_branch), "videos_present": len(sample_by_branch) - len(sample_missing), "videos_missing": len(sample_missing), "missing_due_to_invalid_branches": len([row for row in sample_missing if row["status"] == INVALID])},
        "aggregate_runtime_counters": dict(sorted(aggregate.items())),
        "scientific_firewall": {key: aggregate.get(key, 0) for key in FORBIDDEN_COUNTERS},
        "g3_statistics_authorized": False,
        "g4_blind_review_authorized": False,
        "invalid_branch_ids": [row["branch_id"] for row in invalid_rows],
        "next_legal_action": next_action,
    }
    outputs: dict[str, dict[str, Any]] = {}
    outputs["receipt_index"] = write_new(args.output_dir / "STAGE_AC_AC3_G2_BRANCH_RECEIPT_INDEX_V1.json", index)
    outputs["terminal"] = write_new(args.output_dir / "STAGE_AC_AC3_G2_TERMINAL_V1.json", terminal)
    root_payload = {
        "gate": GATE,
        "g0_root": g0_record,
        "g1_root": g1_record,
        "runtime_authority": runtime_record,
        "manifest": manifest_record,
        "blind_sample": blind_record,
        "receipt_index": outputs["receipt_index"],
        "terminal": outputs["terminal"],
        "counts": index["counts"],
        "aggregate_runtime_counters": index["aggregate_runtime_counters"],
        "scientific_firewall": terminal["scientific_firewall"],
        "historical_attempts_preserved": True,
        "source_and_video_bytes_verified": True,
    }
    root = {
        "schema": "STAGE_AC_AC3_G2_ROOT_SEAL_V1",
        "status": terminal_status,
        "root_payload": root_payload,
        "root_payload_sha256": canonical_hash(root_payload),
        "artifacts": outputs,
        "claim_boundary": terminal["claim_boundary"],
        "next_legal_action": next_action,
    }
    outputs["root"] = write_new(args.output_dir / "STAGE_AC_AC3_G2_ROOT_SEAL_V1.json", root)
    return {"status": terminal_status, "counts": index["counts"], "outputs": outputs, "root_payload_sha256": root["root_payload_sha256"]}


def self_test() -> None:
    assert len(MODELS) == 3 and len(SUITES) == 3
    assert CONDITION_NAMES == {"CLEAN_REFERENCE", "OPEN_T3", "OPEN_T5", "OPEN_T10"}
    assert len(FORBIDDEN_COUNTERS) == 6
    print(json.dumps({"status": "AC3_G2_STATIC_SELF_TEST_PASS", "manifest_branches": 384, "four_conditions_per_parent": True}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--g0-root", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G0_ROOT_SEAL_V1.json")
    parser.add_argument("--g1-root", type=Path, default=ROOT / "reports/STAGE_AC_AC3Q_G1_ROOT_SEAL_V1.json")
    parser.add_argument("--runtime-authority", type=Path, default=ROOT / "reports/STAGE_AC_AC3Q_RUNTIME_SOURCE_AUTHORITY_V9.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "reports/STAGE_AC_AC3_G0_LAUNCH_MANIFEST_V1.json")
    parser.add_argument("--blind-sample", type=Path, default=ROOT / "reports/STAGE_AC_AC4_BLIND_AUDIT_SAMPLE_V1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.evidence_root is None:
        parser.error("--evidence-root is required unless --self-test is used")
    print(json.dumps(build(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
