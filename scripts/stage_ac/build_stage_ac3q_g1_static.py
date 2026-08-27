#!/usr/bin/env python3
"""Seal the AC3Q G1 consumed-only engineering qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
CANARIES = (
    "libero_10/task_04/state_20",
    "libero_object/task_02/state_42",
    "libero_spatial/task_05/state_34",
)
GATE = "STAGE_AC_AC3_AC4_AC5_TREATMENT_NAIVE_MULTI_MODEL_PHYSICAL_REPLICATION_PROGRAM_V1"
PASS_CELL = "PASS_AC3Q_ENGINEERING_BRANCH_CELL"
PASS_BRANCH = "PASS_AC3Q_ENGINEERING_BRANCH"
FAILURE_STATUSES = {"AC3Q_ENGINEERING_HOLD_RUNTIME_ERROR", "AC3Q_ENGINEERING_HOLD_BRANCH_INVALID", "AC3Q_ENGINEERING_HOLD_NO_POINT"}
REQUIRED_CONDITIONS = {"CLEAN_REFERENCE", "OPEN_T3", "OPEN_T5", "OPEN_T10"}
FORBIDDEN_COUNTERS = (
    "pgd_calls",
    "attacked_env_steps",
    "v_phys_reads",
    "eval160_reads",
    "protected_reads",
    "scientific_parent_exposure",
    "ac2_exposure",
    "attack_outcome_reads",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_record(path: Path, original: str | None = None) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": original or str(path), "bytes": len(data), "sha256": digest(data)}


def read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = path.read_bytes()
    return json.loads(data.decode("utf-8")), {"path": str(path), "bytes": len(data), "sha256": digest(data)}


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return digest(canonical(value))


def write_new(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise RuntimeError(f"AC3Q_G1_APPEND_ONLY_OUTPUT_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    return {"path": str(path), "bytes": len(data), "sha256": digest(data)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def relative_to(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"AC3Q_G1_PATH_OUTSIDE_EVIDENCE_ROOT:{path}") from exc


def compact_branch(branch: dict[str, Any]) -> dict[str, Any]:
    dose = branch.get("dose")
    return {
        "route": branch.get("route"),
        "condition": branch.get("condition"),
        "dose": dose,
        "status": branch.get("status"),
        "state_restore_exact": branch.get("state_restore_exact"),
        "exact_open_delivery": branch.get("exact_open_delivery"),
        "arm_preserved": branch.get("arm_preserved"),
        "max_arm_delta_linf": branch.get("max_arm_delta_linf"),
        "queue_reset_verified": branch.get("queue_reset_verified"),
        "telemetry_aligned": branch.get("telemetry_aligned"),
        "open_intervention_steps": branch.get("open_intervention_steps"),
        "required_horizon_steps": branch.get("required_horizon_steps"),
        "trace_digest": branch.get("trace_digest"),
    }


def check_branch(branch: dict[str, Any], evidence_root: Path, family: str) -> dict[str, Any]:
    require(branch.get("status") == PASS_BRANCH, f"AC3Q_G1_BRANCH_NOT_PASS:{family}:{branch.get('route')}")
    condition = str(branch.get("condition"))
    require(condition in REQUIRED_CONDITIONS, f"AC3Q_G1_UNEXPECTED_BRANCH_CONDITION:{family}:{condition}")
    require(branch.get("state_restore_exact") is True, f"AC3Q_G1_STATE_RESTORE_INVALID:{family}:{condition}")
    require(branch.get("exact_open_delivery") is True, f"AC3Q_G1_OPEN_DELIVERY_INVALID:{family}:{condition}")
    require(branch.get("arm_preserved") is True, f"AC3Q_G1_ARM_NOT_PRESERVED:{family}:{condition}")
    require(float(branch.get("max_arm_delta_linf", 1.0)) <= 1e-7, f"AC3Q_G1_ARM_DELTA_INVALID:{family}:{condition}")
    require(branch.get("queue_reset_verified") is True, f"AC3Q_G1_QUEUE_RESET_INVALID:{family}:{condition}")
    require(branch.get("telemetry_aligned") is True, f"AC3Q_G1_TELEMETRY_INVALID:{family}:{condition}")
    rows = branch.get("rows")
    require(isinstance(rows, list) and rows, f"AC3Q_G1_ROWS_MISSING:{family}:{condition}")
    for row in rows:
        action = row.get("action")
        require(isinstance(action, list) and len(action) == 7, f"AC3Q_G1_ACTION_DIM_INVALID:{family}:{condition}")
    dose = branch.get("dose")
    expected_open_steps = 0 if condition == "CLEAN_REFERENCE" else int(dose)
    require(int(branch.get("open_intervention_steps", -1)) == expected_open_steps, f"AC3Q_G1_OPEN_STEP_COUNT_INVALID:{family}:{condition}")
    if condition != "CLEAN_REFERENCE":
        receipts = branch.get("action_receipts")
        require(isinstance(receipts, list) and len(receipts) == expected_open_steps, f"AC3Q_G1_ACTION_RECEIPTS_INVALID:{family}:{condition}")
        for item in receipts:
            action = item.get("env_action")
            require(isinstance(action, list) and len(action) == 7 and float(action[-1]) == -1.0, f"AC3Q_G1_NATIVE_OPEN_INVALID:{family}:{condition}")
            require(float(item.get("arm_delta_linf", 1.0)) <= 1e-7, f"AC3Q_G1_INTERVENTION_ARM_DRIFT:{family}:{condition}")
    video = branch.get("video")
    require(isinstance(video, dict), f"AC3Q_G1_VIDEO_METADATA_MISSING:{family}:{condition}")
    video_path = Path(str(video.get("path")))
    relative_to(video_path, evidence_root)
    require(video_path.is_file(), f"AC3Q_G1_VIDEO_MISSING:{video_path}")
    record = file_record(video_path, str(video_path))
    require(record["bytes"] == int(video.get("bytes")), f"AC3Q_G1_VIDEO_BYTES_MISMATCH:{video_path}")
    require(record["sha256"] == str(video.get("sha256")), f"AC3Q_G1_VIDEO_SHA_MISMATCH:{video_path}")
    require(int(video.get("frames", 0)) > 0 and int(video.get("fps", 0)) == 10, f"AC3Q_G1_VIDEO_METADATA_INVALID:{video_path}")
    return {"branch": compact_branch(branch), "video": record}


def select_receipts(receipts_root: Path) -> tuple[dict[tuple[str, str], tuple[Path, dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    parsed: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for path in sorted(receipts_root.glob("*.json")):
        data, record = read_json(path)
        parsed.append((path, data, record))
    selected: dict[tuple[str, str], tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    history: list[dict[str, Any]] = []
    for family in MODELS:
        for parent in CANARIES:
            matches = [(p, d, r) for p, d, r in parsed if d.get("model_family") == family and d.get("canonical_parent_key") == parent]
            require(matches, f"AC3Q_G1_RECEIPT_MISSING:{family}:{parent}")
            passing = [(p, d, r) for p, d, r in matches if d.get("status") == PASS_CELL]
            require(len(passing) == 1, f"AC3Q_G1_PASS_RECEIPT_AMBIGUOUS:{family}:{parent}:{len(passing)}")
            selected[(family, parent)] = passing[0]
            for p, d, r in matches:
                if p != passing[0][0]:
                    history.append({"model_family": family, "canonical_parent_key": parent, "status": d.get("status"), "receipt": r, "error": d.get("error")})
    return selected, history


def build(args: argparse.Namespace) -> dict[str, Any]:
    evidence_root = args.evidence_root.resolve()
    receipts_root = evidence_root / "receipts"
    require(receipts_root.is_dir(), f"AC3Q_G1_RECEIPTS_ROOT_MISSING:{receipts_root}")
    g0_root, g0_record = read_json(args.g0_root)
    require(g0_root.get("status") == "STAGE_AC_AC3_PRELAUNCH_AUTHORITY_FROZEN_CONTINUE", "AC3Q_G1_G0_NOT_FROZEN")
    require(g0_root.get("gate") == GATE, "AC3Q_G1_GATE_MISMATCH")
    selected, history = select_receipts(receipts_root)
    rows: list[dict[str, Any]] = []
    all_videos: list[dict[str, Any]] = []
    aggregate: dict[str, int] = {}
    for (family, parent), (path, receipt, receipt_record) in sorted(selected.items()):
        require(receipt.get("schema") == "STAGE_AC_AC3Q_ENGINEERING_CANARY_RECEIPT_V1", f"AC3Q_G1_SCHEMA_INVALID:{path}")
        require(receipt.get("gate") == GATE, f"AC3Q_G1_RECEIPT_GATE_INVALID:{path}")
        require(receipt.get("permanent_exclusion") is True and receipt.get("scientific_use") is False, f"AC3Q_G1_EXCLUSION_INVALID:{path}")
        require(receipt.get("scientific_claim") == "NONE_ENGINEERING_ONLY", f"AC3Q_G1_CLAIM_BOUNDARY_INVALID:{path}")
        require(receipt.get("canonical_parent_key") == parent and receipt.get("model_family") == family, f"AC3Q_G1_BINDING_INVALID:{path}")
        require(receipt.get("branch_count", 0) >= 4, f"AC3Q_G1_ROUTE_COUNT_INVALID:{path}")
        branches = receipt.get("branches")
        require(isinstance(branches, list), f"AC3Q_G1_BRANCHES_MISSING:{path}")
        conditions = {str(b.get("condition")) for b in branches}
        require(REQUIRED_CONDITIONS.issubset(conditions), f"AC3Q_G1_REQUIRED_ROUTES_MISSING:{family}:{parent}:{conditions}")
        branch_checks = [check_branch(branch, evidence_root, family) for branch in branches]
        all_videos.extend(item["video"] for item in branch_checks)
        clean_replay = receipt.get("clean_replay")
        require(isinstance(clean_replay, dict) and clean_replay.get("status") == "PASS_AC3Q_CLEAN_REPLAY_DETERMINISTIC" and clean_replay.get("trace_equal") is True, f"AC3Q_G1_CLEAN_REPLAY_INVALID:{path}")
        counters = receipt.get("runtime_counters") or {}
        for key, value in counters.items():
            aggregate[key] = aggregate.get(key, 0) + int(value)
        for key in FORBIDDEN_COUNTERS:
            require(int(counters.get(key, 0)) == 0, f"AC3Q_G1_FORBIDDEN_COUNTER_NONZERO:{path}:{key}")
        require(int(counters.get("model_inference_calls", 0)) > 0 and int(counters.get("env_step_calls", 0)) > 0, f"AC3Q_G1_RUNTIME_COUNTERS_EMPTY:{path}")
        gpu = receipt.get("gpu_admission_snapshot") or {}
        require(int(gpu.get("free_memory_mib", 0)) > 20480, f"AC3Q_G1_GPU_ADMISSION_INVALID:{path}")
        rows.append({
            "model_family": family,
            "canonical_parent_key": parent,
            "receipt": receipt_record,
            "gpu_admission": {k: gpu.get(k) for k in ("index", "free_memory_mib", "used_memory_mib", "utilization_gpu_percent")},
            "branch_count": receipt.get("branch_count"),
            "branch_pass_count": receipt.get("branch_pass_count"),
            "clean_replay_status": clean_replay.get("status"),
            "branches": [item["branch"] for item in branch_checks],
            "videos": [item["video"] for item in branch_checks],
            "runtime_counters": dict(sorted((str(k), int(v)) for k, v in counters.items())),
            "clean_trajectory_digest": receipt.get("clean_trajectory_digest"),
        })
    require(len(rows) == 9 and len(all_videos) == 45, f"AC3Q_G1_COMPLETENESS_INVALID:{len(rows)}:{len(all_videos)}")
    require(len({(r["model_family"], r["canonical_parent_key"]) for r in rows}) == 9, "AC3Q_G1_DUPLICATE_CELL")
    video_sha_groups: dict[str, list[str]] = {}
    for video in all_videos:
        video_sha_groups.setdefault(str(video["sha256"]), []).append(str(video["path"]))
    duplicate_video_sha_groups = {
        sha: paths for sha, paths in video_sha_groups.items() if len(paths) > 1
    }
    index = {
        "schema": "STAGE_AC_AC3Q_ENGINEERING_CANARY_RECEIPT_INDEX_V1",
        "status": "STAGE_AC_AC3Q_G1_ENGINEERING_QUALIFICATION_PASS",
        "gate": GATE,
        "claim_boundary": "consumed-only engineering qualification; no scientific outcome",
        "evidence_root": str(evidence_root),
        "g0_root": g0_record,
        "current_pass_cells": rows,
        "superseded_attempts": history,
        "counts": {"expected_cells": 9, "pass_cells": len(rows), "verified_branch_routes": sum(len(r["branches"]) for r in rows), "verified_videos": len(all_videos), "duplicate_video_sha_groups": len(duplicate_video_sha_groups), "receipt_files_seen": len(list(receipts_root.glob("*.json")))},
        "duplicate_video_sha_groups": duplicate_video_sha_groups,
        "aggregate_runtime_counters": dict(sorted(aggregate.items())),
    }
    terminal = {
        "schema": "STAGE_AC_AC3Q_ENGINEERING_CANARY_TERMINAL_V1",
        "status": "STAGE_AC_AC3Q_G1_ENGINEERING_QUALIFICATION_PASS_STOP_FOR_PI",
        "gate": GATE,
        "claim_boundary": "G1 consumed-only engineering qualification; no AC3 scientific result",
        "qualification": "9/9 model-canary cells passed required clean/T3/T5/T10 routes; 45 branch videos and exact receipts verified",
        "model_cell_counts": {family: sum(r["model_family"] == family for r in rows) for family in MODELS},
        "aggregate_runtime_counters": dict(sorted(aggregate.items())),
        "superseded_engineering_invalid_attempts": len(history),
        "scientific_firewall": {key: aggregate.get(key, 0) for key in FORBIDDEN_COUNTERS},
        "next_legal_action": "EXECUTE_G2_384_SCIENTIFIC_BRANCHES_AFTER_SOURCE_AUTHORITY_REVIEW",
    }
    outputs: dict[str, dict[str, Any]] = {}
    outputs["receipt_index"] = write_new(args.output_dir / "STAGE_AC_AC3Q_ENGINEERING_CANARY_RECEIPT_INDEX_V1.json", index)
    outputs["terminal"] = write_new(args.output_dir / "STAGE_AC_AC3Q_ENGINEERING_CANARY_TERMINAL_V1.json", terminal)
    source = None
    if args.runtime_authority:
        source, source_record = read_json(args.runtime_authority)
        require(source.get("status") == "STAGE_AC_AC3Q_RUNTIME_SOURCE_AUTHORITY_FROZEN", "AC3Q_G1_RUNTIME_AUTHORITY_INVALID")
        outputs["runtime_authority"] = source_record
    root_payload = {
        "gate": GATE,
        "g0_root": g0_record,
        "runtime_authority": outputs.get("runtime_authority"),
        "receipt_index": outputs["receipt_index"],
        "terminal": outputs["terminal"],
        "counts": index["counts"],
        "aggregate_runtime_counters": index["aggregate_runtime_counters"],
        "scientific_firewall": terminal["scientific_firewall"],
        "historical_attempts_preserved": True,
    }
    root = {
        "schema": "STAGE_AC_AC3Q_G1_ROOT_SEAL_V1",
        "status": "STAGE_AC_AC3Q_G1_ENGINEERING_QUALIFICATION_PASS_STOP_FOR_PI",
        "root_payload": root_payload,
        "root_payload_sha256": canonical_hash(root_payload),
        "artifacts": outputs,
        "claim_boundary": terminal["claim_boundary"],
        "next_legal_action": terminal["next_legal_action"],
    }
    outputs["root"] = write_new(args.output_dir / "STAGE_AC_AC3Q_G1_ROOT_SEAL_V1.json", root)
    return {"status": root["status"], "cells": len(rows), "branches": len(all_videos), "outputs": outputs, "root_payload_sha256": root["root_payload_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--g0-root", type=Path)
    parser.add_argument("--runtime-authority", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.self_test:
        assert len(MODELS) == 3 and len(CANARIES) == 3
        assert REQUIRED_CONDITIONS == {"CLEAN_REFERENCE", "OPEN_T3", "OPEN_T5", "OPEN_T10"}
        print(json.dumps({"status": "AC3Q_G1_STATIC_SELF_TEST_PASS", "cells": 9, "required_routes": 4}, sort_keys=True))
        return 0
    for name in ("evidence_root", "g0_root", "runtime_authority", "output_dir"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required unless --self-test is used")
    print(json.dumps(build(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
