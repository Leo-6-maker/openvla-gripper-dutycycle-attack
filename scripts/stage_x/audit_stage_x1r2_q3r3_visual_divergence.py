#!/usr/bin/env python3
"""CPU-only characterization of sealed Q3R2 visual divergence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


VISUAL_FIELDS = ("raw_agentview_sha256", "processor_pixel_values_sha256")
CONTROL_FIELDS = ("action_env_7d", "robot0_eef_pos", "robot0_gripper_qpos", "direct_generated_token_ids")
RAW_PAYLOAD_KEYS = {"raw_agentview", "raw_agentview_bytes", "agentview_image", "processor_pixel_values", "pixel_values"}
RAW_FRAME_SUFFIXES = {".png", ".jpg", ".jpeg", ".npy", ".npz", ".mp4", ".mkv", ".webm"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_file(spec: dict[str, Any]) -> Path:
    path = Path(spec["path"])
    if not path.is_file():
        raise SystemExit(f"MISSING_INPUT:{path}")
    observed = sha256_file(path)
    if observed != spec["sha256"]:
        raise SystemExit(f"INPUT_SHA256_MISMATCH:{path}:{observed}:{spec['sha256']}")
    return path


def first_divergence(rows0: list[dict[str, Any]], rows1: list[dict[str, Any]]) -> dict[str, Any] | None:
    for index, (row0, row1) in enumerate(zip(rows0, rows1)):
        fields = sorted({*row0.keys(), *row1.keys()})
        changed = [field for field in fields if row0.get(field) != row1.get(field)]
        if changed:
            return {
                "step": row0.get("step", index),
                "changed_fields": changed,
                "visual_fields_changed": [field for field in VISUAL_FIELDS if row0.get(field) != row1.get(field)],
                "nonvisual_fields_changed": [field for field in changed if field not in VISUAL_FIELDS],
                "control_fields_equal": {field: row0.get(field) == row1.get(field) for field in CONTROL_FIELDS},
            }
    return None


def visual_divergence(rows0: list[dict[str, Any]], rows1: list[dict[str, Any]]) -> dict[str, Any]:
    first = None
    for index, (row0, row1) in enumerate(zip(rows0, rows1)):
        changed = [field for field in VISUAL_FIELDS if row0.get(field) != row1.get(field)]
        if changed:
            first = {
                "step": row0.get("step", index),
                "changed_fields": changed,
                "control_fields_equal": {field: row0.get(field) == row1.get(field) for field in CONTROL_FIELDS},
            }
            break
    raw_payload_keys = sorted(set().union(*(set(row) & RAW_PAYLOAD_KEYS for row in rows0 + rows1)))
    return {
        "first_visual_hash_divergence": first,
        "raw_payload_keys_present": raw_payload_keys,
        "raw_frame_numeric_metrics": {
            "changed_pixel_count": None,
            "changed_pixel_fraction": None,
            "per_channel_max_abs_diff": None,
            "per_channel_mean_abs_diff": None,
            "rmse": None,
            "processor_space_linf": None,
            "status": "NOT_AVAILABLE_HASH_ONLY_TELEMETRY"
        },
        "preprocess_localization": "NOT_IDENTIFIABLE_FROM_HASH_ONLY_TELEMETRY"
    }


def protected_zero(report: dict[str, Any]) -> bool:
    boundary = report.get("protected_boundary", {})
    return all(int(boundary.get(key, 0)) == 0 for key in ("pgd_calls", "physical_interventions", "vphys_reads", "attack_outcome_reads", "attacked_env_steps", "protected_reads")) and boundary.get("eval160") == "UNREAD" and boundary.get("protected_evaluation") == "UNREAD"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--branch-config", type=Path, required=True)
    parser.add_argument("--runtime-surface", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    branch = load_json(args.branch_config)
    surface = load_json(args.runtime_surface)
    if branch.get("status") != "FROZEN_PROSPECTIVE_BRANCH_ESTIMAND_ENGINEERING_ONLY":
        raise SystemExit("BRANCH_ESTIMAND_NOT_FROZEN")
    if branch["supersedes"]["q3r2_c_status"] != "OWNER_REVIEW_Q3R2_CLEAN_PREFIX_DETERMINISM_NOT_ESTABLISHED":
        raise SystemExit("Q3R2_HOLD_NOT_PRESERVED")

    suites: dict[str, Any] = {}
    for suite, spec in manifest["suite_inputs"].items():
        report_path = verify_file(spec["report"])
        report = load_json(report_path)
        if not protected_zero(report):
            raise SystemExit(f"PROTECTED_COUNTER_NONZERO:{suite}")
        result: dict[str, Any] = {
            "report_path": str(report_path),
            "report_sha256": spec["report"]["sha256"],
            "sealed_status": report.get("status"),
            "selected_parent": report.get("selected_parent_key"),
            "comparison": report.get("comparison"),
        }
        if "repeat_0" in spec and "repeat_1" in spec:
            path0 = verify_file(spec["repeat_0"])
            path1 = verify_file(spec["repeat_1"])
            rows0, rows1 = load_jsonl(path0), load_jsonl(path1)
            result["telemetry"] = {"repeat_0_sha256": spec["repeat_0"]["sha256"], "repeat_1_sha256": spec["repeat_1"]["sha256"], "rows": [len(rows0), len(rows1)]}
            result["first_any_divergence"] = first_divergence(rows0, rows1)
            result["visual_characterization"] = visual_divergence(rows0, rows1)
        else:
            result["visual_characterization"] = {"status": "SEALED_REPORT_ONLY", "numeric_metrics": "NOT_AVAILABLE"}
        suites[suite] = result

    inventory_root = Path(manifest["server_output_root"])
    inventory = sorted(str(path.relative_to(inventory_root)) for path in inventory_root.rglob("*") if path.is_file() and path.suffix.lower() in RAW_FRAME_SUFFIXES)
    result = {
        "schema": "STAGE_X_X1R2_Q3R3_VISUAL_DIVERGENCE_AUDIT_V1",
        "status": "STAGE_X1R2_Q3R3_BRANCH_ESTIMAND_FREEZE_PASS",
        "scope": "static/CPU/offline only; no model inference, simulator construction, env.step, PGD, physical intervention, V_phys, Eval160, or protected read",
        "q3r2_hold_preserved": True,
        "suite_results": suites,
        "raw_frame_inventory": {"root": str(inventory_root), "matching_files": inventory, "count": len(inventory), "numeric_metrics_available": bool(inventory)},
        "runtime_surface": {"status": surface.get("status"), "provenance_gaps": surface.get("provenance_gaps", [])},
        "branch_estimand": {"config": str(args.branch_config), "status": branch.get("status"), "reference_clean_once": branch["reference_clean"]["per_parent_count"] == 1, "prebranch_model_forbidden": branch["branch_prefix"]["openvla_before_branch_forbidden"] and branch["branch_prefix"]["student_before_branch_forbidden"]},
        "protected_boundary": {"model_inference_calls": 0, "env_step_calls": 0, "pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "protected_reads": 0, "eval160": "UNREAD", "protected_evaluation": "UNREAD"},
        "next_gate": "STAGE_X1R2_Q3R3_BRANCH_RUNNER_STATIC_PASS"
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "raw_frame_count": len(inventory), "suites": sorted(suites)}, sort_keys=True))


if __name__ == "__main__":
    main()
