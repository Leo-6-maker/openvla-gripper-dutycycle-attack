#!/usr/bin/env python3
"""Independent closure, provenance, and Gate D/T auditor for Stage 3A."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CONDITIONS = (
    "CLEAN_SHADOW",
    "RANDOM_NORM_MATCHED_E006_S20_SHADOW",
    "PGD_E003_S20_SHADOW",
    "PGD_E006_S20_SHADOW",
    "ORACLE_OPEN_SHADOW",
)
ARM_BY_CONDITION = {
    "CLEAN_SHADOW": "CLEAN",
    "RANDOM_NORM_MATCHED_E006_S20_SHADOW": "RAND_T10",
    "PGD_E003_S20_SHADOW": "TRUE_T10",
    "PGD_E006_S20_SHADOW": "TRUE_T10",
    "ORACLE_OPEN_SHADOW": "COMMAND_OPEN_ORACLE",
}
EXPECTED_FORMAL_SEEDS = {2026080501, 2026080502, 2026080503}
STAGE2_COMMIT = "f14415b104501df2a5cf7b35e8965966866eea9e"
STAGE2_TREE = "6a982a474b9fe788e8466f2a7f144b45d6a1cd89"
CHECKPOINT_SHA = "ce7f03088d84a796d38fbdc107cea7f21bdb4808e35f7dc754e1b52e48bce1d4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{__import__('os').getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_steps(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def assert_finite_vector(value: Any, size: int, label: str) -> None:
    if not isinstance(value, list) or len(value) != size or not all(math.isfinite(float(item)) for item in value):
        raise RuntimeError(f"{label} is not a finite vector of dimension {size}")


def audit_job(job: dict[str, Any], transfer: dict[str, Any], config_index: dict[str, Any]) -> dict[str, Any]:
    root = Path(job["job_root"])
    arm = ARM_BY_CONDITION[job["condition"]]
    manifest = read_json(root / "run_manifest.json")
    summary = read_json(root / "smoke_summary.json")
    result = read_json(root / arm / "result.json")
    complete = read_json(root / arm / "COMPLETE.json")
    trace = read_json(root / arm / "evaluation_detector_trace.json")
    steps = load_steps(root / arm / "steps.jsonl")
    if manifest.get("scientific_role") != "STAGE3A_SHADOW_ONLY" or not manifest.get("formal_matrix_execution"):
        raise RuntimeError(f"job is not a Stage3A formal artifact: {job['job_id']}")
    if summary.get("valid") is not True or result.get("status") != "PASS" or complete.get("status") != "PASS":
        raise RuntimeError(f"job result is not PASS: {job['job_id']}")
    if complete.get("result_sha256") != sha256_file(root / arm / "result.json"):
        raise RuntimeError(f"COMPLETE result seal mismatch: {job['job_id']}")
    if manifest.get("stage3a_condition") != job["condition"] or manifest.get("suite") != job["suite"] or manifest.get("task_index") != job["task_index"] or manifest.get("seed") != job["seed"]:
        raise RuntimeError(f"job identity binding mismatch: {job['job_id']}")
    if manifest.get("stage3a_source_commit") != STAGE2_COMMIT or manifest.get("stage3a_source_tree") != STAGE2_TREE:
        raise RuntimeError(f"job Stage 2 source binding mismatch: {job['job_id']}")
    if manifest.get("stage3a_checkpoint_sha256") != transfer.get("checkpoint_sha256") or manifest.get("stage3a_scheduler_freeze_sha256") != transfer.get("scheduler_freeze_sha256"):
        raise RuntimeError(f"job checkpoint/scheduler binding mismatch: {job['job_id']}")
    if manifest.get("guard_intervention_count") != 0 or manifest.get("eval160_reads") != 0 or manifest.get("protected_eval_reads") != 0:
        raise RuntimeError(f"job crossed forbidden boundary: {job['job_id']}")
    if result.get("attack_trigger_source") != "N4 first emit" or result.get("guard_intervention_count") != 0:
        raise RuntimeError(f"job trigger/guard contract mismatch: {job['job_id']}")
    state = manifest.get("state_identity", {})
    if state.get("initial_state_sha256") != job["initial_state_sha256"]:
        raise RuntimeError(f"job init state binding mismatch: {job['job_id']}")
    config_path = str(manifest.get("config_path", ""))
    expected_rel = config_index["formal_conditions"][job["condition"]]["config_path"]
    if not config_path.replace("\\", "/").endswith(expected_rel):
        raise RuntimeError(f"job config path mismatch: {job['job_id']}")
    config_record = next(row for row in config_index["files"] if row["path"] == expected_rel)
    if manifest.get("config_sha256") != config_record["sha256"]:
        raise RuntimeError(f"job config SHA mismatch: {job['job_id']}")
    if not isinstance(trace, list) or not trace:
        raise RuntimeError(f"missing evaluation detector trace: {job['job_id']}")
    nonwait = [row for row in steps if not row.get("is_wait_step")]
    if len(trace) != len(nonwait):
        raise RuntimeError(f"trace/step closure mismatch: {job['job_id']}")
    emissions = []
    actives = []
    max_logit = -math.inf
    attack_window_logits = []
    active_steps = 0
    for index, (shadow, step) in enumerate(zip(trace, nonwait)):
        if shadow.get("episode_id") != job["episode_id"] or shadow.get("policy_step") != index:
            raise RuntimeError(f"shadow identity/step mismatch: {job['job_id']} step={index}")
        if shadow.get("input_action_source") != "clean_policy_action_before_attack":
            raise RuntimeError(f"shadow action source is not clean pre-attack: {job['job_id']}")
        if any(shadow.get(key) is not False for key in ("evaluation_detector_affects_action", "evaluation_detector_affects_timing", "evaluation_detector_affects_termination")):
            raise RuntimeError(f"shadow detector has a control hook: {job['job_id']}")
        assert_finite_vector(shadow.get("raw_action"), 7, f"{job['job_id']} raw_action")
        assert_finite_vector(shadow.get("env_action"), 7, f"{job['job_id']} env_action")
        assert_finite_vector(shadow.get("features_25d"), 25, f"{job['job_id']} features")
        if not math.isfinite(float(shadow.get("logit"))):
            raise RuntimeError(f"non-finite shadow logit: {job['job_id']}")
        if shadow["raw_action"] != step.get("clean_raw_action") or shadow["env_action"] != step.get("clean_env_action"):
            raise RuntimeError(f"shadow action differs from clean policy action: {job['job_id']} step={index}")
        if step.get("attack_planned") and step.get("attack_executed"):
            attack_window_logits.append(float(shadow["logit"]))
        max_logit = max(max_logit, float(shadow["logit"]))
        active_steps += int(bool(shadow.get("latched_active")))
        if shadow.get("emission"):
            emissions.append(index)
        if shadow.get("latched_active"):
            actives.append(index)
    if result.get("evaluation_detector_first_emission") != (emissions[0] if emissions else None) or result.get("evaluation_detector_first_active") != (actives[0] if actives else None):
        raise RuntimeError(f"first detector state mismatch: {job['job_id']}")
    if job["condition"] == "CLEAN_SHADOW":
        for row in nonwait:
            if row.get("clean_env_action") != row.get("final_env_action"):
                raise RuntimeError(f"CLEAN action was modified: {job['job_id']}")
        if result.get("attack_executed_frames") != 0:
            raise RuntimeError(f"CLEAN has attack frames: {job['job_id']}")
    harmful = [
        int(row["policy_step"])
        for row in nonwait
        if row.get("attack_executed") and float(row.get("final_env_gripper", 1.0)) <= -0.5
    ]
    first_harmful = harmful[0] if harmful else None
    if result.get("first_harmful_open") != first_harmful:
        raise RuntimeError(f"harmful-open definition mismatch: {job['job_id']}")
    if job["condition"] == "ORACLE_OPEN_SHADOW" and result.get("attack_executed_frames", 0) > 0:
        if any(float(row["final_env_action"][-1]) != -1.0 for row in nonwait if row.get("attack_executed")):
            raise RuntimeError(f"oracle did not produce canonical open: {job['job_id']}")
    return {
        **job,
        "triggered": bool(emissions),
        "first_emission": emissions[0] if emissions else None,
        "first_active": actives[0] if actives else None,
        "first_harmful_open": first_harmful,
        "timely": first_harmful is not None and actives and actives[0] <= first_harmful,
        "max_logit": max_logit,
        "mean_attack_window_logit": statistics.mean(attack_window_logits) if attack_window_logits else None,
        "active_step_fraction": active_steps / max(len(trace), 1),
        "task_success": bool(result.get("task_success")),
    }


def write_seal(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            continue
        rows.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")
    sums_sha = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return {"file_count": len(rows), "sha256sums_sha256": sums_sha}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    manifest = read_json(root / "STAGE3A_SHADOW_RUN_MANIFEST.json")
    transfer = read_json(root / "FINAL_CHECKPOINT_TRANSFER_AUDIT.json")
    config_index = read_json(root / "STAGE3A_ATTACK_CONFIG_INDEX.json")
    task_manifest = read_json(root / "STAGE3A_TASK_MANIFEST.json")
    jobs = manifest.get("jobs", [])
    diagnostic = bool(manifest.get("diagnostic"))
    expected_count = 4 if diagnostic else 60
    audit: dict[str, Any] = {
        "schema": "D8_STAGE3A_SHADOW_AUDIT_V1",
        "status": "RUNNING",
        "planned": len(jobs),
        "expected_planned": expected_count,
        "completed": 0,
        "invalid": 0,
        "failed_protocol_jobs": 0,
        "duplicate_identities": 0,
        "missing_identities": 0,
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0 if diagnostic else 60,
    }
    try:
        if len(jobs) != expected_count or manifest.get("max_workers_per_physical_gpu") != 1:
            raise RuntimeError("planned matrix or worker contract mismatch")
        if transfer.get("status") != "PASS" or transfer.get("checkpoint_sha256") != CHECKPOINT_SHA:
            raise RuntimeError("transfer audit is not PASS")
        if task_manifest.get("status") != "FROZEN":
            raise RuntimeError("task manifest is not frozen")
        identities = [(job.get("condition"), job.get("canonical_parent_key"), job.get("seed")) for job in jobs]
        audit["duplicate_identities"] = len(identities) - len(set(identities))
        if audit["duplicate_identities"]:
            raise RuntimeError("duplicate Stage3A identity")
        if not diagnostic:
            expected = {(condition, task["canonical_parent_key"], seed) for condition in CONDITIONS for task in task_manifest["tasks"] for seed in EXPECTED_FORMAL_SEEDS}
            audit["missing_identities"] = len(expected - set(identities))
            if audit["missing_identities"]:
                raise RuntimeError("missing Stage3A identities")
        rows = [audit_job(job, transfer, config_index) for job in jobs]
        audit["completed"] = len(rows)
        audit["invalid"] = 0
        if not all(job.get("status") == "COMPLETED" for job in jobs):
            raise RuntimeError("dispatcher manifest is not closed COMPLETED")
        if diagnostic:
            audit["status"] = "DIAGNOSTIC_PASS"
            audit["rows"] = rows
            atomic_json(root / "STAGE3A_SHADOW_AUDIT.json", audit)
            atomic_json(root / "STAGE3A_SHADOW_SUMMARY.json", {"status": "DIAGNOSTIC_PASS", "rows": rows, "planned": 4, "completed": 4})
            write_seal(root)
            return 0

        by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_condition[row["condition"]].append(row)
        condition_totals = {
            condition: {
                "n": len(by_condition[condition]),
                "trigger_episodes": sum(row["triggered"] for row in by_condition[condition]),
                "trigger_rate": sum(row["triggered"] for row in by_condition[condition]) / 12,
                "task_success": sum(row["task_success"] for row in by_condition[condition]),
                "max_logit": max(row["max_logit"] for row in by_condition[condition]),
                "mean_attack_window_logit": statistics.mean([row["mean_attack_window_logit"] for row in by_condition[condition] if row["mean_attack_window_logit"] is not None]) if any(row["mean_attack_window_logit"] is not None for row in by_condition[condition]) else None,
                "active_step_fraction_mean": statistics.mean(row["active_step_fraction"] for row in by_condition[condition]),
            }
            for condition in CONDITIONS
        }
        d = {
            "CLEAN_SHADOW": condition_totals["CLEAN_SHADOW"]["trigger_episodes"] <= 1,
            "RANDOM_NORM_MATCHED_E006_S20_SHADOW": condition_totals["RANDOM_NORM_MATCHED_E006_S20_SHADOW"]["trigger_episodes"] <= 2,
            "PGD_E003_S20_SHADOW": condition_totals["PGD_E003_S20_SHADOW"]["trigger_episodes"] >= 6,
            "PGD_E006_S20_SHADOW": condition_totals["PGD_E006_S20_SHADOW"]["trigger_episodes"] >= 9,
            "ORACLE_OPEN_SHADOW": condition_totals["ORACLE_OPEN_SHADOW"]["trigger_episodes"] >= 10,
        }
        gap = condition_totals["PGD_E006_S20_SHADOW"]["trigger_rate"] - max(
            condition_totals["CLEAN_SHADOW"]["trigger_rate"],
            condition_totals["RANDOM_NORM_MATCHED_E006_S20_SHADOW"]["trigger_rate"],
        )
        d["pgd_e006_selectivity_gap_ge_0.50"] = gap >= 0.50
        d_pass = all(d.values())
        c4 = by_condition["PGD_E006_S20_SHADOW"]
        harmful_rows = [row for row in c4 if row["first_harmful_open"] is not None]
        timely_rows = [row for row in harmful_rows if row["first_active"] is not None and row["first_active"] <= row["first_harmful_open"]]
        after_count = sum(row["first_active"] is not None and row["first_active"] > row["first_harmful_open"] for row in harmful_rows)
        same_count = sum(row["first_active"] is not None and row["first_active"] == row["first_harmful_open"] for row in harmful_rows)
        before_count = sum(row["first_active"] is not None and row["first_active"] < row["first_harmful_open"] for row in harmful_rows)
        lead_times = [row["first_harmful_open"] - row["first_active"] for row in harmful_rows if row["first_active"] is not None]
        t_pass = bool(harmful_rows) and len(timely_rows) / len(harmful_rows) >= 0.75
        summary = {
            "schema": "D8_STAGE3A_SHADOW_SUMMARY_V1",
            "status": "CLOSED",
            "planned": 60,
            "completed": len(rows),
            "invalid": 0,
            "failed_protocol_jobs": 0,
            "duplicate_identities": 0,
            "missing_identities": 0,
            "condition_totals": condition_totals,
            "selectivity_gap": gap,
            "paired_task_seed_rows": rows,
            "eval160_reads": 0,
            "protected_eval_reads": 0,
            "attack_rollouts": 60,
        }
        selectivity = {
            "schema": "D8_STAGE3A_SELECTIVITY_GATE_V1",
            "gate": "D",
            "verdict": "PASS" if d_pass else "FAIL",
            "criteria": d,
            "condition_totals": condition_totals,
            "selectivity_gap": gap,
        }
        timing = {
            "schema": "D8_STAGE3A_TIMING_GATE_V1",
            "gate": "T",
            "verdict": "PASS" if t_pass else "FAIL",
            "detected_pgd_e006_episodes_with_harmful_open": len(harmful_rows),
            "timely_count": len(timely_rows),
            "timely_fraction": len(timely_rows) / len(harmful_rows) if harmful_rows else None,
            "before_open_count": before_count,
            "same_step_count": same_count,
            "after_open_count": after_count,
            "no_harmful_open_count": len(c4) - len(harmful_rows),
            "median_detector_lead_time": statistics.median(lead_times) if lead_times else None,
            "lead_time_definition": "first_harmful_open_policy_step - first_evaluation_detector_active_policy_step",
        }
        decision = "STAGE3A_SELECTIVE_AND_TIMELY" if d_pass and t_pass else "STAGE3A_SELECTIVE_BUT_LATE" if d_pass else "STAGE3A_SCIENTIFIC_SELECTIVITY_FAIL"
        audit.update({"status": "PASS", "rows": rows, "condition_totals": condition_totals, "selectivity_gate": selectivity, "timing_gate": timing, "decision": decision})
        atomic_json(root / "STAGE3A_SHADOW_SUMMARY.json", summary)
        atomic_json(root / "STAGE3A_SELECTIVITY_GATE.json", selectivity)
        atomic_json(root / "STAGE3A_TIMING_GATE.json", timing)
        atomic_json(root / "STAGE3A_DECISION.json", {"decision": decision, "stage3b_authorized": bool(d_pass and t_pass), "stage4s_shadow_authorized": bool(d_pass)})
        atomic_json(root / "STAGE3A_SHADOW_AUDIT.json", audit)
        write_seal(root)
        return 0
    except Exception as exc:
        audit.update({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        atomic_json(root / "STAGE3A_SHADOW_AUDIT.json", audit)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
