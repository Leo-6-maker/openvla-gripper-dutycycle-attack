#!/usr/bin/env python3
"""Build the AC0 consumed-only calibration receipt index and root seal."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/STAGE_AC_AC0_CONSTRUCT_VALIDATION_PROTOCOL_V1.json"
SOURCE_V1 = ROOT / "reports/STAGE_AC_AC0_RUNTIME_SOURCE_AUTHORITY_V1.json"
SOURCE_V2 = ROOT / "reports/STAGE_AC_AC0_RUNTIME_SOURCE_AUTHORITY_V2.json"
PLAN = ROOT / "reports/STAGE_AA_AA2R2_ENGINEERING_CANARY_PLAN_V1.json"
SHARED_PANEL = ROOT / "reports/STAGE_Z_Z0R1_SHARED_36_IDENTITY_PANEL_V1.json"
FAMILIES = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
HORIZONS = {"libero_10": 520, "libero_object": 280, "libero_spatial": 220}
FORBIDDEN = (
    "open_intervention_steps",
    "pgd_calls",
    "attacked_env_steps",
    "aa_v_phys_reads",
    "v_phys_reads",
    "protected_reads",
    "scientific_parent_exposure",
    "aa2_exposure",
    "task_success_reads",
    "attack_outcome_reads",
    "eval160_reads",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def receipt_candidates(evidence_root: Path, cell_id: str) -> list[Path]:
    return [evidence_root / "receipts" / f"{cell_id}.json", evidence_root / "receipts" / f"{cell_id}-R1.json"]


def all_row_controls(clean: dict[str, Any]) -> Counter[str]:
    # Importing the same pure evaluator keeps the audit tied to the sealed AC0 implementation.
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from stage_ac.eligibility_v2 import classify_calibration_control

    rows = clean["eligibility_rows"]
    baseline = clean.get("baseline_z_m")
    return Counter(classify_calibration_control(rows, step, baseline) for step in range(len(rows)))


def audit_receipt(evidence_root: Path, cell: dict[str, Any]) -> dict[str, Any]:
    cell_id = str(cell["cell_id"])
    candidates = [path for path in receipt_candidates(evidence_root, cell_id) if path.is_file()]
    require(candidates, f"AC0_RECEIPT_MISSING:{cell_id}")
    parsed = [(path, load_json(path)) for path in candidates]
    passes = [(path, value) for path, value in parsed if value.get("status") == "PASS_AC0_CALIBRATION_CELL"]
    require(len(passes) == 1, f"AC0_PASS_RECEIPT_COUNT:{cell_id}:{len(passes)}")
    receipt_path, receipt = passes[0]
    for key in ("model_family", "canonical_parent_key", "suite", "task_idx", "state_id", "seed"):
        require(receipt.get(key) == cell.get(key), f"AC0_RECEIPT_BINDING:{cell_id}:{key}")
    require(receipt.get("permanent_exclusion") is True and receipt.get("scientific_use") is False, f"AC0_EXCLUSION:{cell_id}")
    require(receipt.get("fresh_scientific_exposure") == 0, f"AC0_FRESH_EXPOSURE:{cell_id}")

    counters = receipt.get("runtime_counters", {})
    clean = receipt.get("clean", {})
    steps = int(clean.get("steps_captured", -1))
    require(clean.get("status") == "AC0_CLEAN_CAPTURE_COMPLETE", f"AC0_CLEAN_STATUS:{cell_id}")
    require(steps >= 20, f"AC0_STEP_COUNT:{cell_id}")
    require(len(clean.get("rows", [])) == steps, f"AC0_ROW_COUNT:{cell_id}")
    require(len(clean.get("actions", [])) == steps, f"AC0_ACTION_COUNT:{cell_id}")
    require(len(clean.get("eligibility_rows", [])) == steps, f"AC0_ELIGIBILITY_ROW_COUNT:{cell_id}")
    require(clean.get("telemetry_valid_rows") == steps, f"AC0_TELEMETRY_VALIDITY:{cell_id}")
    require(len(clean.get("boundary_states", {})) == sum(bool(row.get("boundary")) for row in clean["actions"]), f"AC0_BOUNDARY_COUNT:{cell_id}")
    require(bool(clean.get("clean_trajectory_digest")), f"AC0_TRAJECTORY_DIGEST:{cell_id}")
    if clean.get("complete_trajectory") is False:
        terminal_rows = [row for row in clean["rows"] if row.get("terminal_after") is True]
        require(len(terminal_rows) == 1 and terminal_rows[-1] is clean["rows"][-1], f"AC0_TERMINAL_PLACEMENT:{cell_id}")
    else:
        require(not any(row.get("terminal_after") is True for row in clean["rows"][:-1]), f"AC0_EARLY_TERMINAL:{cell_id}")

    require(counters.get("env_step_calls") == steps + 10, f"AC0_ENV_STEP_COUNT:{cell_id}")
    require(counters.get("physical_telemetry_reads") == steps * 2, f"AC0_TELEMETRY_READ_COUNT:{cell_id}")
    require(counters.get("model_inference_calls", 0) > 0, f"AC0_INFERENCE_COUNT:{cell_id}")
    require(all(counters.get(key, 0) == 0 for key in FORBIDDEN), f"AC0_FIREWALL:{cell_id}")

    for row, action in zip(clean["rows"], clean["actions"]):
        require(len(row.get("raw_action_7d", [])) == 7 and len(row.get("env_action_7d", [])) == 7, f"AC0_ACTION_DIM:{cell_id}")
        require(len(action.get("raw", [])) == 7 and len(action.get("final", [])) == 7, f"AC0_ACTION_RECORD_DIM:{cell_id}")
    audits = receipt.get("action_pair_audit", [])
    require(audits, f"AC0_ACTION_AUDIT_MISSING:{cell_id}")
    require(all(audit.get("semantics", {}).get("accepted") is True for audit in audits), f"AC0_ACTION_AUDIT_REJECTED:{cell_id}")

    video = clean.get("video", {})
    video_path = Path(str(video.get("path", "")))
    require(video_path.is_file(), f"AC0_VIDEO_MISSING:{cell_id}")
    require(video.get("bytes") == video_path.stat().st_size and video.get("sha256") == sha256_file(video_path), f"AC0_VIDEO_HASH:{cell_id}")
    require(video.get("fps") == 10 and video.get("frames") == steps and video.get("width") == 256 and video.get("height") == 256, f"AC0_VIDEO_METADATA:{cell_id}")

    controls = all_row_controls(clean)
    critical = clean["eligibility_diagnostics"]["critical"]
    strict = critical["STRICT_NO_FLICKER"]["candidates"]
    flicker = critical["ONE_ROW_FLICKER"]["candidates"]
    strict_ranks = {row["selection_rank_sha256"] for row in strict}
    flicker_ranks = {row["selection_rank_sha256"] for row in flicker}
    require(strict_ranks <= flicker_ranks, f"AC0_FLICKER_ORDER:{cell_id}")
    require(all(int(row["metrics"].get("contact_false_rows", 1)) == 0 for row in strict), f"AC0_POSITIVE_FLICKER:{cell_id}")

    superseded = []
    for path, value in parsed:
        if path != receipt_path:
            superseded.append({"receipt": artifact(path), "status": value.get("status"), "error": value.get("error"), "runtime_counters": value.get("runtime_counters", {})})
    return {
        "cell_id": cell_id,
        "model_family": cell["model_family"],
        "canonical_parent_key": cell["canonical_parent_key"],
        "suite": cell["suite"],
        "seed": cell["seed"],
        "status": receipt["status"],
        "receipt": artifact(receipt_path),
        "video": artifact(video_path),
        "log": artifact(evidence_root / "logs" / f"{receipt_path.stem}.log") if (evidence_root / "logs" / f"{receipt_path.stem}.log").is_file() else None,
        "superseded_attempts": superseded,
        "steps_captured": steps,
        "complete_trajectory": clean.get("complete_trajectory"),
        "boundary_count": len(clean.get("boundary_states", {})),
        "model_inference_calls": counters.get("model_inference_calls"),
        "env_step_calls": counters.get("env_step_calls"),
        "physical_telemetry_reads": counters.get("physical_telemetry_reads"),
        "action_pair_audit_count": len(audits),
        "clean_trajectory_digest": clean.get("clean_trajectory_digest"),
        "critical_candidates": {key: len(value.get("candidates", [])) for key, value in critical.items()},
        "critical_reason_counts": {key: value.get("reason_counts", {}) for key, value in critical.items()},
        "control_labels_all_rows": dict(sorted(controls.items())),
        "scientific_firewall": {key: counters.get(key, 0) for key in FORBIDDEN},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    protocol = load_json(PROTOCOL)
    plan = load_json(PLAN)
    source_v2 = load_json(SOURCE_V2)
    require(protocol.get("status") == "STAGE_AC_AC0_PROVISIONAL_ENGINEERING_CALIBRATION_ONLY", "AC0_PROTOCOL_STATUS")
    require(protocol.get("fresh_science_authorized") is False, "AC0_FRESH_SCIENCE_FIREWALL")
    require(len(plan.get("canaries", [])) == 9 and plan.get("cell_count") == 9, "AC0_PLAN_COUNT")
    require(source_v2.get("status") == "STAGE_AC_AC0_RUNTIME_SOURCE_AUTHORITY_FROZEN", "AC0_SOURCE_AUTHORITY")
    require(sha256_file(ROOT / "scripts/stage_ac/run_stage_ac0_calibration.py") == source_v2["new_runtime_files"]["runner"]["sha256"], "AC0_SOURCE_RUNNER_HASH")
    require(sha256_file(PROTOCOL) == source_v2["new_runtime_files"]["protocol"]["sha256"], "AC0_SOURCE_PROTOCOL_HASH")
    require(sha256_file(PLAN) == source_v2["input_authorities"]["canary_plan"]["sha256"], "AC0_SOURCE_PLAN_HASH")
    require(sha256_file(SHARED_PANEL) == source_v2["input_authorities"]["stage_z_shared_identity_panel"]["sha256"], "AC0_SOURCE_PANEL_HASH")

    records = [audit_receipt(args.evidence_root, cell) for cell in plan["canaries"]]
    positive_by_model = Counter()
    support_by_model = Counter()
    controls_by_model: dict[str, Counter[str]] = {family: Counter() for family in FAMILIES}
    for record in records:
        family = str(record["model_family"])
        positive_by_model[family] += int(record["critical_candidates"]["STRICT_NO_FLICKER"])
        for label, count in record["control_labels_all_rows"].items():
            controls_by_model[family][label] += int(count)
        support_by_model[family] += int(record["control_labels_all_rows"].get("SUPPORT_CONTACT", 0))
    require(all(positive_by_model[family] > 0 for family in FAMILIES), "AC0_POSITIVE_CONTROL_MODEL_BALANCE")
    require(all(support_by_model[family] > 0 for family in FAMILIES), "AC0_SUPPORT_CONTROL_MODEL_BALANCE")

    aggregate = Counter()
    for record in records:
        for key in ("model_inference_calls", "env_step_calls", "physical_telemetry_reads", "action_pair_audit_count"):
            aggregate[key] += int(record[key] or 0)
        for key in FORBIDDEN:
            aggregate[key] += int(record["scientific_firewall"][key])

    control_summary = {
        "positive_control_definition": "strict no-flicker critical candidate with privileged 3-row contact, local 20-row continuation, support false, distance/lift/relative-drift gates",
        "negative_control_definition": "privileged support-contact rows; unknown rows are not controls",
        "positive_candidates_by_model": dict(sorted(positive_by_model.items())),
        "support_negative_rows_by_model": dict(sorted(support_by_model.items())),
        "all_control_labels_by_model": {family: dict(sorted(counts.items())) for family, counts in controls_by_model.items()},
        "pre_contact_rows_total": sum(counts["PRE_CONTACT"] for counts in controls_by_model.values()),
        "intended_release_or_drop_rows_total": sum(counts["INTENDED_RELEASE_OR_DROP"] for counts in controls_by_model.values()),
        "unknown_rows_excluded_from_controls": sum(counts["UNKNOWN_CONTROL"] for counts in controls_by_model.values()),
        "flicker_variant_selected": "STRICT_NO_FLICKER",
        "flicker_selection_reason": "It is the least permissive frozen variant and accepts the observed positive candidate windows; support-contact rows remain ineligible under both variants.",
        "scope_limit": "No clean-only consumed canary produced an intended-release/drop control. AC0 therefore validates established-grasp eligibility and support-contact exclusion only; it does not claim release/drop endpoint validity.",
    }
    terminal_status = "STAGE_AC_AC0_CONSTRUCT_VALIDATION_PASS_STOP_FOR_PI"
    terminal = {
        "schema": "STAGE_AC_AC0_CONSTRUCT_VALIDATION_TERMINAL_V1",
        "status": terminal_status,
        "gate": protocol["gate"],
        "claim_boundary": "Consumed-only engineering calibration; no fresh Stage-AC identity, scientific denominator, treatment, endpoint outcome, or promotion claim.",
        "source_authority": artifact(SOURCE_V2),
        "superseded_source_authority": artifact(SOURCE_V1),
        "protocol": artifact(PROTOCOL),
        "canary_plan": artifact(PLAN),
        "evidence_root": str(args.evidence_root),
        "cell_count": len(records),
        "records": records,
        "control_validation": control_summary,
        "aggregate_runtime_counters": dict(sorted(aggregate.items())),
        "scientific_firewall": {key: aggregate[key] for key in FORBIDDEN},
        "ac1_authorized": False,
        "ac2_authorized": False,
        "ac3_authorized": False,
        "next_legal_action": "STOP_FOR_PI",
    }
    index = {
        "schema": "STAGE_AC_AC0_CALIBRATION_RECEIPT_INDEX_V1",
        "status": terminal_status,
        "gate": protocol["gate"],
        "receipt_count": len(records),
        "receipts": records,
        "terminal_sha256_after_write": None,
    }
    terminal_path = ROOT / "reports/STAGE_AC_AC0_CONSTRUCT_VALIDATION_TERMINAL_V1.json"
    index_path = ROOT / "reports/STAGE_AC_AC0_CALIBRATION_RECEIPT_INDEX_V1.json"
    root_path = ROOT / "reports/STAGE_AC_AC0_ROOT_SEAL_V1.json"
    if args.write:
        require(not any(path.exists() for path in (terminal_path, index_path, root_path, root_path.with_suffix(".sha256"))), "AC0_APPEND_ONLY_ARTIFACT_EXISTS")
        write_json(terminal_path, terminal)
        index["terminal_sha256_after_write"] = sha256_file(terminal_path)
        write_json(index_path, index)
        root = {
            "schema": "STAGE_AC_AC0_ROOT_SEAL_V1",
            "status": terminal_status,
            "gate": protocol["gate"],
            "terminal": artifact(terminal_path),
            "receipt_index": artifact(index_path),
            "source_authority": artifact(SOURCE_V2),
            "superseded_source_authority": artifact(SOURCE_V1),
            "protocol": artifact(PROTOCOL),
            "canary_plan": artifact(PLAN),
            "receipt_count": len(records),
            "records": records,
            "control_validation": control_summary,
            "aggregate_runtime_counters": dict(sorted(aggregate.items())),
            "scientific_firewall": {key: aggregate[key] for key in FORBIDDEN},
            "ac1_authorized": False,
            "ac2_authorized": False,
            "ac3_authorized": False,
            "next_legal_action": "STOP_FOR_PI",
        }
        root["root_payload_sha256"] = hashlib.sha256(canonical(root)).hexdigest()
        write_json(root_path, root)
        root_path.with_suffix(".sha256").write_text(f"{sha256_file(root_path)}  {root_path.name}\n", encoding="utf-8")
    print(json.dumps({"status": terminal_status, "receipt_count": len(records), "write": args.write}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
