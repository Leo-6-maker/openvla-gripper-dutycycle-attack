#!/usr/bin/env python3
"""Offline, fail-closed synthesis for the sealed Stage-Z Z3-C receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from stage_z_preparation.z3_contract import (
    ARM_TOLERANCE,
    DOSES,
    H_PHYS,
    NATIVE_OPEN,
    NATIVE_OPEN_RAW,
    physical_class,
    physical_label,
)


MODEL_DIR = {
    "M0_OPENVLA": "M0_OPENVLA",
    "M1_OPENVLA_OFT": "M1_OPENVLA_OFT",
    "M2_PI05_LIBERO": "M2_PI05_LIBERO",
}
ARMS = (
    "CLEAN_BRANCH_CRITICAL",
    "COMMAND_OPEN_T3_CRITICAL",
    "COMMAND_OPEN_T5_CRITICAL",
    "COMMAND_OPEN_T10_CRITICAL",
    "COMMAND_OPEN_T5_NONCRITICAL_CONTROL",
)
DOSE_BY_ARM = {
    "CLEAN_BRANCH_CRITICAL": 0,
    "COMMAND_OPEN_T3_CRITICAL": 3,
    "COMMAND_OPEN_T5_CRITICAL": 5,
    "COMMAND_OPEN_T10_CRITICAL": 10,
    "COMMAND_OPEN_T5_NONCRITICAL_CONTROL": 5,
}
VALID_LABELS = {"V_PHYS", "NO_PHYSICAL_VULNERABILITY"}
FORBIDDEN_COUNTERS = ("protected_reads", "eval160_reads", "pgd_calls", "attack_outcome_reads")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(ok: bool, message: str, failures: list[str]) -> None:
    if not ok:
        failures.append(message)


def vec(value: Any, n: int = 7) -> tuple[float, ...] | None:
    if not isinstance(value, list) or len(value) != n:
        return None
    try:
        result = tuple(float(x) for x in value)
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(x) for x in result) else None


def same(left: Any, right: Any, tol: float = 0.0) -> bool:
    a, b = vec(left), vec(right)
    return a is not None and b is not None and all(abs(x - y) <= tol for x, y in zip(a, b))


def position(row: dict[str, Any]) -> tuple[float, float, float] | None:
    value = row.get("post_object_position")
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        result = tuple(float(x) for x in value)
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(x) for x in result) else None


def bootstrap(values: list[float], seed: int, replicates: int = 2000) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "lower_2_5": None, "upper_97_5": None, "replicates": replicates}
    rng = random.Random(seed)
    estimates = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(replicates)]
    estimates.sort()
    lo = estimates[max(0, int(0.025 * len(estimates)) - 1)]
    hi = estimates[min(len(estimates) - 1, int(0.975 * len(estimates)))]
    return {"n": len(values), "mean": sum(values) / len(values), "lower_2_5": lo, "upper_97_5": hi, "replicates": replicates}


def receipt_path(root: Path, job: dict[str, Any]) -> Path:
    return root / "stage_z_z3c_outputs_v1" / MODEL_DIR[job["model_family"]] / job["suite"] / f"{job['branch_id']}.json"


def validate_branch(job: dict[str, Any], branch: dict[str, Any], failures: list[str]) -> None:
    bid = job["branch_id"]
    for key in ("branch_id", "model_family", "suite", "canonical_parent_key", "arm", "duration", "anchor_class", "anchor_step", "anchor_state_sha256", "source_receipt_path", "source_receipt_sha256"):
        require(branch.get(key) == job.get(key), f"{bid}:FIELD_MISMATCH:{key}", failures)
    require(branch.get("status") == "PASS", f"{bid}:STATUS", failures)
    require(branch.get("model_inference") is False, f"{bid}:MODEL_INFERENCE", failures)
    require(branch.get("state_restore_exact") is True, f"{bid}:STATE_RESTORE", failures)
    require(branch.get("causal_input_binding_pass") is True, f"{bid}:CAUSAL_BINDING", failures)
    require(branch.get("control_action_reference_exact") is True, f"{bid}:CONTROL_REPLAY", failures)
    counters = branch.get("runtime_counters", {})
    for key in FORBIDDEN_COUNTERS:
        require(counters.get(key, 0) == 0, f"{bid}:FORBIDDEN_COUNTER:{key}", failures)
    rows = branch.get("rows", [])
    duration = DOSE_BY_ARM[job["arm"]]
    required = 20 if duration == 0 else duration + H_PHYS
    require(len(rows) == required, f"{bid}:ROWS:{len(rows)}:{required}", failures)
    require(branch.get("available_horizon_steps") == required, f"{bid}:HORIZON", failures)
    require(counters.get("env_step_calls") == required, f"{bid}:ENV_COUNTER", failures)
    require(counters.get("physical_telemetry_reads") == required, f"{bid}:TELEMETRY_COUNTER", failures)
    require(all(row.get("post_contact_telemetry_valid") is True for row in rows), f"{bid}:TELEMETRY_INVALID", failures)
    for relative, row in enumerate(rows):
        require(row.get("relative_step") == relative, f"{bid}:ROW_ORDER:{relative}", failures)
        require(vec(row.get("raw_policy_action")) is not None, f"{bid}:RAW_ACTION:{relative}", failures)
        require(vec(row.get("env_action")) is not None, f"{bid}:ENV_ACTION:{relative}", failures)
        require(float(row.get("arm_delta_linf", math.inf)) <= ARM_TOLERANCE, f"{bid}:ARM_DRIFT:{relative}", failures)
        forced = duration > 0 and relative < duration and job["arm"] != "CLEAN_BRANCH_CRITICAL"
        if forced:
            raw, normalized, final = vec(row["raw_policy_action"]), vec(row["normalized_action"]), vec(row["env_action"])
            require(abs(raw[-1] - NATIVE_OPEN_RAW[job["model_family"]]) <= ARM_TOLERANCE, f"{bid}:RAW_OPEN:{relative}", failures)
            require(abs(normalized[-1] - NATIVE_OPEN_RAW[job["model_family"]]) <= ARM_TOLERANCE, f"{bid}:NORMALIZED_OPEN:{relative}", failures)
            require(abs(final[-1] - NATIVE_OPEN) <= ARM_TOLERANCE, f"{bid}:FINAL_OPEN:{relative}", failures)
        else:
            require(same(row.get("raw_policy_action"), row.get("reference_raw_action")), f"{bid}:RAW_REPLAY:{relative}", failures)
            require(same(row.get("env_action"), row.get("reference_env_action")), f"{bid}:ENV_REPLAY:{relative}", failures)
    if duration == 0:
        require(branch.get("treatment_receipts") == [], f"{bid}:CLEAN_TREATMENT_RECEIPTS", failures)
    else:
        treatments = branch.get("treatment_receipts", [])
        require(len(treatments) == duration, f"{bid}:TREATMENT_COUNT", failures)
        require(branch.get("treatment_compliant") is True, f"{bid}:TREATMENT_COMPLIANCE", failures)
        require(branch.get("treatment_compliance", {}).get("delivered_open_steps") == duration, f"{bid}:OPEN_COUNT", failures)
        for row in treatments:
            require(abs(float(row.get("arm_delta_linf", math.inf))) <= ARM_TOLERANCE, f"{bid}:TREATMENT_ARM_DRIFT", failures)
            require(abs(float(row["raw_policy_action"][-1]) - NATIVE_OPEN_RAW[job["model_family"]]) <= ARM_TOLERANCE, f"{bid}:TREATMENT_RAW_OPEN", failures)
            require(abs(float(row["normalized_action"][-1]) - NATIVE_OPEN_RAW[job["model_family"]]) <= ARM_TOLERANCE, f"{bid}:TREATMENT_NORMALIZED_OPEN", failures)
            require(abs(float(row["env_action"][-1]) - NATIVE_OPEN) <= ARM_TOLERANCE, f"{bid}:TREATMENT_FINAL_OPEN", failures)


def branch_metrics(branch: dict[str, Any], clean: dict[str, Any] | None) -> dict[str, Any]:
    rows = branch["rows"]
    aperture = [float(row["post_aperture"]) - float(row["pre_aperture"]) for row in branch.get("treatment_receipts", []) if row.get("pre_aperture") is not None and row.get("post_aperture") is not None]
    displacement: list[float] = []
    if clean is not None:
        for row, ref in zip(rows, clean["rows"]):
            a, b = position(row), position(ref)
            if a is not None and b is not None:
                displacement.append(math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3))))
    return {
        "aperture_excess_max": max(aperture) if aperture else None,
        "object_displacement_max_m": max(displacement) if displacement else None,
        "physical_class": branch.get("physical_class"),
        "v_phys_label": branch.get("v_phys_label"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--launch-plan", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--z3r1-terminal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    def rooted(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()
    manifest_path, protocol_path = rooted(args.manifest), rooted(args.protocol)
    launch_path, reconciliation_path, z3r1_path = rooted(args.launch_plan), rooted(args.reconciliation), rooted(args.z3r1_terminal)
    output = rooted(args.output_dir)
    manifest, protocol = load(manifest_path), load(protocol_path)
    launch, reconciliation, z3r1 = load(launch_path), load(reconciliation_path), load(z3r1_path)
    failures: list[str] = []
    require(manifest.get("status") == "STAGE_Z_Z3_EXECUTION_MANIFEST_FROZEN_NOT_EXECUTED", "MANIFEST_STATUS", failures)
    require(protocol.get("status") == "STAGE_Z_Z3_SOURCE_AUTHORITY_FROZEN", "PROTOCOL_STATUS", failures)
    require(protocol.get("population", {}).get("fixed_matrix_branches") == 460, "POPULATION_BRANCHES", failures)
    require(reconciliation.get("status") == "STAGE_Z_Z3_EXECUTION_MANIFEST_RECONCILED_TO_V2_PASS", "RECONCILIATION_STATUS", failures)
    require(z3r1.get("status") == "STAGE_Z_Z3R1_SENTINEL_RECOVERY_PASS_STOP_FOR_PI", "Z3R1_STATUS", failures)

    jobs = manifest["jobs"]
    require(len(jobs) == 460, f"MANIFEST_JOB_COUNT:{len(jobs)}", failures)
    expected = {job["branch_id"]: job for job in jobs}
    require(len(expected) == len(jobs), "DUPLICATE_BRANCH_ID", failures)
    branches: dict[str, dict[str, Any]] = {}
    index: list[dict[str, Any]] = []
    for job in jobs:
        path = receipt_path(root, job)
        require(path.is_file(), f"MISSING_RECEIPT:{job['branch_id']}", failures)
        if not path.is_file():
            continue
        branch = load(path)
        branches[job["branch_id"]] = branch
        validate_branch(job, branch, failures)
        index.append({
            "branch_id": job["branch_id"],
            "model_family": job["model_family"],
            "suite": job["suite"],
            "canonical_parent_key": job["canonical_parent_key"],
            "arm": job["arm"],
            "duration": job["duration"],
            "anchor_class": job["anchor_class"],
            "anchor_step": job["anchor_step"],
            "anchor_state_sha256": job["anchor_state_sha256"],
            "receipt_path": str(path.relative_to(root)).replace("\\", "/"),
            "receipt_sha256": sha(path),
            "bytes": path.stat().st_size,
            "status": branch.get("status"),
        })

    parent_groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for job in jobs:
        branch = branches.get(job["branch_id"])
        if branch is not None:
            parent_groups[(job["model_family"], job["suite"], job["canonical_parent_key"])][job["arm"]] = branch
    expected_arm_set = set(ARMS)
    parent_rows: list[dict[str, Any]] = []
    for (model, suite, parent), arms in sorted(parent_groups.items()):
        require(set(arms) == expected_arm_set, f"PARENT_ARM_SET:{model}:{suite}:{parent}", failures)
        clean = arms.get("CLEAN_BRANCH_CRITICAL")
        if clean is None:
            continue
        clean_class = physical_class(clean, 20)
        require(clean.get("physical_class") == clean_class, f"CLEAN_CLASS_MISMATCH:{model}:{suite}:{parent}", failures)
        doses: dict[str, Any] = {}
        for arm in ("COMMAND_OPEN_T3_CRITICAL", "COMMAND_OPEN_T5_CRITICAL", "COMMAND_OPEN_T10_CRITICAL"):
            treatment = arms.get(arm)
            dose = DOSE_BY_ARM[arm]
            if treatment is None:
                continue
            expected_class = physical_class(treatment, dose + H_PHYS, clean)
            expected_label = physical_label(clean, treatment, dose, model)
            require(treatment.get("physical_class") == expected_class, f"TREATMENT_CLASS_MISMATCH:{model}:{suite}:{parent}:{arm}", failures)
            require(treatment.get("v_phys_label") == expected_label, f"VPHYS_LABEL_MISMATCH:{model}:{suite}:{parent}:{arm}", failures)
            doses[str(dose)] = {"label": expected_label, **branch_metrics(treatment, clean)}
        control = arms.get("COMMAND_OPEN_T5_NONCRITICAL_CONTROL")
        if control is not None:
            expected_control = physical_class(control, 15)
            require(control.get("physical_class") == expected_control, f"CONTROL_CLASS_MISMATCH:{model}:{suite}:{parent}", failures)
            control_metrics = branch_metrics(control, None)
            control_metrics["physical_class"] = expected_control
        else:
            control_metrics = {"physical_class": None}
        parent_rows.append({"model_family": model, "suite": suite, "canonical_parent_key": parent, "clean_physical_class": clean_class, "critical": doses, "noncritical_t5_control": control_metrics})

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_suite: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in parent_rows:
        by_model[row["model_family"]].append(row)
        by_suite[(row["model_family"], row["suite"])].append(row)

    seed = int(sha(protocol_path)[:8], 16)
    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"parents": len(rows), "dose": {}, "noncritical_t5_control": {}}
        for dose in DOSES:
            labels = [row["critical"].get(str(dose), {}).get("label") for row in rows]
            counts = dict(Counter(labels))
            valid = [1.0 if label == "V_PHYS" else 0.0 for label in labels if label in VALID_LABELS]
            result["dose"][str(dose)] = {"label_counts": counts, "valid_parents": len(valid), "invalid_or_abstain": len(labels) - len(valid), "v_phys_parents": int(sum(valid)), "v_phys_rate_all_parents": (sum(valid) / len(rows) if rows else None), "v_phys_rate_valid_parents": (sum(valid) / len(valid) if valid else None), "parent_bootstrap": bootstrap(valid, seed + dose)}
        result["noncritical_t5_control"] = dict(Counter(row["noncritical_t5_control"].get("physical_class") for row in rows))
        return result

    model_summary = {model: aggregate(rows) for model, rows in sorted(by_model.items())}
    suite_summary = {f"{model}/{suite}": aggregate(rows) for (model, suite), rows in sorted(by_suite.items())}
    counters = Counter()
    for branch in branches.values():
        for key, value in branch.get("runtime_counters", {}).items():
            counters[key] += int(value or 0)
    video_files = list(root.glob("stage_z_z3c_outputs_v1/*/*/manual_videos/*.mp4"))
    ordered_receipt_digest = hashlib.sha256("".join(f"{row['branch_id']}:{row['receipt_sha256']}\n" for row in index).encode()).hexdigest()
    index_doc = {"schema": "STAGE_Z_Z3C_BRANCH_RECEIPT_INDEX_V1", "status": "PASS_Z3C_460_BRANCH_INDEX", "ordered_manifest_branch_count": len(index), "ordered_receipt_digest_sha256": ordered_receipt_digest, "rows": index}
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "STAGE_Z_Z3C_BRANCH_RECEIPT_INDEX_V1.json"
    index_path.write_text(json.dumps(index_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    synthesis = {
        "schema": "STAGE_Z_Z3C_TERMINAL_SYNTHESIS_V1",
        "status": "PASS_Z3C_FIXED_MATRIX_COMPLETE" if not failures else "HOLD_Z3C_TERMINAL_VALIDATION_FAILURE",
        "authority": {"protocol_sha256": sha(protocol_path), "manifest_sha256": sha(manifest_path), "launch_plan_sha256": sha(launch_path), "reconciliation_sha256": sha(reconciliation_path), "z3r1_terminal_sha256": sha(z3r1_path), "ordered_receipt_digest_sha256": ordered_receipt_digest},
        "execution": {"expected_branches": 460, "receipt_count": len(branches), "pass_receipts": sum(branch.get("status") == "PASS" for branch in branches.values()), "failure_count": len(failures), "model_parent_count": len(parent_rows), "model_parent_by_model": {model: len(rows) for model, rows in sorted(by_model.items())}, "branch_counters": dict(counters), "forbidden_reads_zero": all(counters[key] == 0 for key in FORBIDDEN_COUNTERS), "model_inference_zero": all(branch.get("model_inference") is False for branch in branches.values())},
        "physical_results": {"primary_unit": "MODEL_PARENT", "model_summary": model_summary, "model_suite_summary": suite_summary, "parent_rows": parent_rows, "dose_order": list(DOSES), "physical_contract": {"h_phys": H_PHYS, "native_open": NATIVE_OPEN, "arm_tolerance": ARM_TOLERANCE, "v_phys_label": "V_PHYS", "invalid_is_abstain": True}},
        "manual_audit": {"outcome_blind": True, "human_labels_pending": True, "video_file_count": len(video_files), "video_bytes": sum(path.stat().st_size for path in video_files), "max_videos": 120, "all_branch_video": False},
        "claim_boundary": "Sealed command-OPEN physical counterfactual only; official policy inference, visual PGD, task success, protected/Eval160, F1, and BRIDGE are not measured here.",
        "next_legal_action": "STOP_FOR_PI_NO_Z4_OR_BRIDGE",
        "validation_failures": failures,
    }
    synthesis_path = output / "STAGE_Z_Z3C_TERMINAL_SYNTHESIS_V1.json"
    synthesis_path.write_text(json.dumps(synthesis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    root_seal = {
        "schema": "STAGE_Z_Z3C_ROOT_SEAL_V1",
        "status": synthesis["status"],
        "terminal_synthesis": {"path": str(synthesis_path.relative_to(root)).replace("\\", "/"), "sha256": sha(synthesis_path)},
        "branch_index": {"path": str(index_path.relative_to(root)).replace("\\", "/"), "sha256": sha(index_path)},
        "analysis_script": {"path": "scripts/stage_z/build_stage_z_z3c_terminal.py", "sha256": sha(Path(__file__))},
        "source_bindings": synthesis["authority"],
        "execution": synthesis["execution"],
        "resource_snapshot": {"filesystem": "/mnt/sdc", "free_bytes_at_seal": shutil.disk_usage("/mnt/sdc").free, "min_free_margin_bytes": 5368709120},
        "claim_boundary": synthesis["claim_boundary"],
        "next_legal_action": synthesis["next_legal_action"],
        "validation_failures": failures,
    }
    root_path = output / "STAGE_Z_Z3C_ROOT_SEAL_V1.json"
    root_path.write_text(json.dumps(root_seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(json.dumps({"status": root_seal["status"], "failures": failures[:20]}, sort_keys=True))
    print(json.dumps({"status": synthesis["status"], "branches": len(branches), "parents": len(parent_rows), "root_seal": str(root_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
