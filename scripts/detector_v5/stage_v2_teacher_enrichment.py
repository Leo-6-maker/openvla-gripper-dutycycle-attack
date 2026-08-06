"""Read-only Stage V2 Teacher enrichment analysis.

The formal path consumes only a sealed Stage V root.  Diagnostic mode is
explicitly non-formal and is useful for schema/statistics tests only.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Iterable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.monitoring.audit_stage_v_closure import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    verify_sha_manifest,
)


CONFIG_SCHEMA = "STAGE_V2_TEACHER_ENRICHMENT_CONFIG_V2"
REPORT_SCHEMA = "STAGE_V2_TEACHER_ENRICHMENT_REPORT_V2"
ROWS_SCHEMA = "STAGE_V2_TEACHER_ENRICHMENT_ROWS_V2"
OPEN_ARMS = ("OPEN_T3", "OPEN_T5", "OPEN_T10")
PRIMARY_ARM = "OPEN_T10"
AUXILIARY_ARMS = ("OPEN_T3", "OPEN_T5")
GROUPS = ("teacher_corridor", "background_random", "safe_release_support")
LABELS = ("local_vulnerability", "task_vulnerability")
TRUE_VALUES = {"1", "true", "yes", "y", "positive", "member", "in", "present"}
FALSE_VALUES = {"0", "false", "no", "n", "negative", "nonmember", "out", "absent"}


class StageV2PreconditionError(RuntimeError):
    pass


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite(item) for item in value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    return True


def _int(value: Any) -> int | None:
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _nested(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _first(values: Iterable[Any]) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _value(source: Mapping[str, Any], aliases: Iterable[str]) -> Any:
    for alias in aliases:
        value = _nested(source, alias)
        if value is not None:
            return value
    return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        value = value.strip().lower()
        if value in TRUE_VALUES:
            return True
        if value in FALSE_VALUES:
            return False
    return None


def _feature(sources: Iterable[Mapping[str, Any]], aliases: Iterable[str]) -> bool | None:
    for source in sources:
        value = _value(source, aliases)
        result = _as_bool(value)
        if result is not None:
            return result
    return None


def _sources(row: Mapping[str, Any], parent: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values: list[Mapping[str, Any]] = [row]
    for source in (row.get("candidate_state"), row.get("teacher"), row.get("metadata"), parent):
        if isinstance(source, Mapping):
            values.append(source)
            for nested in (source.get("candidate_state"), source.get("teacher"), source.get("metadata")):
                if isinstance(nested, Mapping):
                    values.append(nested)
    return values


def _group(row: Mapping[str, Any], parent: Mapping[str, Any]) -> tuple[str | None, str | None, dict[str, bool | None]]:
    sources = _sources(row, parent)
    corridor = _feature(
        sources,
        ("teacher_corridor_membership", "in_teacher_corridor", "teacher_corridor", "corridor_membership"),
    )
    background = _feature(
        sources,
        ("background_random_membership", "background_membership", "random_membership", "is_background", "is_random"),
    )
    safe_release = _feature(
        sources,
        ("safe_release_support_membership", "safe_release_membership", "support_membership", "is_safe_release", "is_support"),
    )
    flags = {"teacher_corridor": corridor, "background_random": background, "safe_release_support": safe_release}
    true_groups = [name for name, value in flags.items() if value is True]
    if len(true_groups) == 1:
        return true_groups[0], None, flags
    if len(true_groups) > 1:
        return None, "AMBIGUOUS_GROUP_MEMBERSHIP", flags
    if all(value is False for value in flags.values()):
        return None, "UNASSIGNED_GROUP", flags
    return None, "UNKNOWN_GROUP_MEMBERSHIP", flags


def _label(row: Mapping[str, Any], name: str) -> bool | None:
    comparison = row.get("comparison")
    if not isinstance(comparison, Mapping):
        return None
    return _as_bool(comparison.get(name))


def load_observations(stage_v_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    invalid_rows = 0
    parent_count = 0
    branch_count = 0
    for parent_result_path in sorted(stage_v_root.rglob("PARENT_RESULT.json")):
        if "MONITOR" in parent_result_path.relative_to(stage_v_root).parts:
            continue
        parent = load_json(parent_result_path)
        if not isinstance(parent, Mapping):
            invalid_rows += 1
            continue
        parent_count += 1
        parent_key = str(parent.get("canonical_parent_key", ""))
        branch_file = parent_result_path.parent / "COUNTERFACTUAL_BRANCHES.jsonl"
        if not branch_file.is_file():
            invalid_rows += 1
            continue
        try:
            handle = branch_file.open(encoding="utf-8")
        except OSError:
            invalid_rows += 1
            continue
        with handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                branch_count += 1
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    invalid_rows += 1
                    continue
                if not isinstance(row, Mapping) or str(row.get("arm", "")) not in OPEN_ARMS:
                    continue
                arm = str(row.get("arm"))
                probe_step = _int(_first((row.get("probe_step"), row.get("candidate_step"), row.get("step"))))
                group, group_reason, flags = _group(row, parent)
                sources = _sources(row, parent)
                role = _first(_value(source, ("teacher_role", "role")) for source in sources)
                phase = _first(_value(source, ("teacher_phase", "phase")) for source in sources)
                corridor_start = _int(_first(_value(source, ("teacher_corridor_start_step", "corridor_start_step")) for source in sources))
                timing_offset = None if probe_step is None or corridor_start is None else probe_step - corridor_start
                observations.append(
                    {
                        "row_id": f"{parent_key}:{probe_step if probe_step is not None else line_number}:{arm}",
                        "canonical_parent_key": parent_key,
                        "suite": str(parent.get("suite", row.get("suite", "UNKNOWN"))),
                        "task_index": parent.get("task_index"),
                        "state_index": parent.get("state_index"),
                        "candidate_step": probe_step,
                        "arm": arm,
                        "teacher_role": role,
                        "teacher_phase": phase,
                        "group": group,
                        "group_reason": group_reason,
                        "group_flags": flags,
                        "local_vulnerability": _label(row, "local_vulnerability"),
                        "task_vulnerability": _label(row, "task_vulnerability"),
                        "corridor_start_step": corridor_start,
                        "timing_offset": timing_offset,
                    }
                )
    return observations, {"parent_count": parent_count, "branch_count": branch_count, "invalid_branch_rows": invalid_rows}


def _wilson(positive: int, denominator: int) -> list[float] | None:
    if denominator <= 0:
        return None
    z = 1.959963984540054
    p = positive / denominator
    denominator_term = 1.0 + z * z / denominator
    centre = (p + z * z / (2 * denominator)) / denominator_term
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * denominator)) / denominator) / denominator_term
    return [max(0.0, centre - spread), min(1.0, centre + spread)]


def _rate(positive: int, denominator: int) -> dict[str, Any]:
    if denominator <= 0:
        return {"numerator": positive, "denominator": denominator, "rate": None, "ci95": None, "status": "UNAVAILABLE_ZERO_DENOMINATOR"}
    return {
        "numerator": positive,
        "denominator": denominator,
        "rate": positive / denominator,
        "ci95": _wilson(positive, denominator),
        "status": "AVAILABLE",
    }


def _primary_candidate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collapse repeated arms to one identity per parent/candidate under OPEN_T10."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("arm") != PRIMARY_ARM:
            continue
        parent = str(row.get("canonical_parent_key", ""))
        step = row.get("candidate_step")
        identity = f"{parent}:{step}"
        groups.setdefault(identity, []).append(row)
    collapsed: list[dict[str, Any]] = []
    conflicts = 0
    for identity, items in sorted(groups.items()):
        first = dict(items[0])
        labels = {item.get("local_vulnerability") for item in items if isinstance(item.get("local_vulnerability"), bool)}
        task_labels = {item.get("task_vulnerability") for item in items if isinstance(item.get("task_vulnerability"), bool)}
        if len(labels) > 1 or len(task_labels) > 1:
            conflicts += 1
        first["row_id"] = identity
        if labels:
            first["local_vulnerability"] = True if True in labels else False
        if task_labels:
            first["task_vulnerability"] = True if True in task_labels else False
        collapsed.append(first)
    return collapsed, {
        "primary_arm": PRIMARY_ARM,
        "candidate_state_count": len(collapsed),
        "duplicate_source_rows": sum(max(0, len(items) - 1) for items in groups.values()),
        "identity_conflicts": conflicts,
        "auxiliary_arms": list(AUXILIARY_ARMS),
    }


def _counts(rows: list[Mapping[str, Any]], label: str) -> dict[str, int]:
    teacher = [row for row in rows if row.get("group") == "teacher_corridor" and isinstance(row.get(label), bool)]
    baseline = [row for row in rows if row.get("group") == "background_random" and isinstance(row.get(label), bool)]
    return {
        "teacher_positive": sum(row[label] is True for row in teacher),
        "teacher_total": len(teacher),
        "background_positive": sum(row[label] is True for row in baseline),
        "background_total": len(baseline),
    }


def _fisher_two_sided(a: int, b: int, c: int, d: int) -> float | None:
    total = a + b + c + d
    if total <= 0:
        return None
    from math import comb

    def probability(x: int) -> float:
        return comb(a + b, x) * comb(c + d, a + b - x) / comb(total, a + c)

    lo = max(0, (a + b) - (b + d))
    hi = min(a + b, a + c)
    observed = probability(a)
    return min(1.0, sum(probability(x) for x in range(lo, hi + 1) if probability(x) <= observed + 1e-15))


def _corrected_enrichment(counts: Mapping[str, int]) -> float | None:
    if counts["teacher_total"] <= 0 or counts["background_total"] <= 0:
        return None
    teacher_rate = (counts["teacher_positive"] + 0.5) / (counts["teacher_total"] + 1.0)
    background_rate = (counts["background_positive"] + 0.5) / (counts["background_total"] + 1.0)
    return teacher_rate / background_rate


def _bootstrap_cluster(rows: list[dict[str, Any]], label: str, repetitions: int, seed: int) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        clusters.setdefault(str(row.get("canonical_parent_key", row.get("row_id", "UNKNOWN"))), []).append(row)
    if not clusters or repetitions <= 0:
        return {"repetitions": 0, "seed": seed, "ci95": None, "status": "UNAVAILABLE_ZERO_DENOMINATOR"}
    keys = sorted(clusters)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(repetitions):
        sample = [item for key in (rng.choice(keys) for _ in keys) for item in clusters[key]]
        value = _corrected_enrichment(_counts(sample, label))
        if value is not None and math.isfinite(value):
            values.append(value)
    if not values:
        return {"repetitions": repetitions, "seed": seed, "ci95": None, "status": "UNAVAILABLE_ZERO_DENOMINATOR"}
    values.sort()
    return {
        "repetitions": repetitions, "seed": seed,
        "ci95": [_percentile(values, 0.025), _percentile(values, 0.975)],
        "status": "AVAILABLE",
    }


def _percentile(values: list[float], q: float) -> float:
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return float(values[index])


def _metric(rows: list[Mapping[str, Any]], label: str) -> dict[str, Any]:
    known = [row for row in rows if row.get("group") in GROUPS and isinstance(row.get(label), bool)]
    unknown = len(rows) - len(known)
    teacher = [row for row in known if row.get("group") == "teacher_corridor"]
    baseline = [row for row in known if row.get("group") == "background_random"]
    support = [row for row in known if row.get("group") == "safe_release_support"]
    all_positive = sum(row[label] is True for row in known)
    all_negative = sum(row[label] is False for row in known)
    teacher_positive = sum(row[label] is True for row in teacher)
    baseline_positive = sum(row[label] is True for row in baseline)
    base_rate = _rate(baseline_positive, len(baseline))
    teacher_rate = _rate(teacher_positive, len(teacher))
    ratio = None
    ratio_status = "AVAILABLE"
    if base_rate["rate"] is None or teacher_rate["rate"] is None:
        ratio_status = "UNAVAILABLE_ZERO_DENOMINATOR"
    elif base_rate["rate"] <= 0:
        ratio_status = "INFINITE_ZERO_BASELINE" if teacher_rate["rate"] > 0 else "UNAVAILABLE_ZERO_BASELINE_RATE"
    else:
        ratio = teacher_rate["rate"] / base_rate["rate"]
    counts = _counts(rows, label)
    corrected_ratio = _corrected_enrichment(counts)
    fisher = _fisher_two_sided(
        counts["teacher_positive"], counts["teacher_total"] - counts["teacher_positive"],
        counts["background_positive"], counts["background_total"] - counts["background_positive"],
    )
    precision = _rate(teacher_positive, len(teacher))
    recall = _rate(teacher_positive, all_positive)
    tn = counts["background_total"] - counts["background_positive"]
    specificity = _rate(tn, tn + counts["background_positive"])
    return {
        "label": label,
        "rows": len(rows),
        "unknown_or_abstain": unknown,
        "teacher_corridor": teacher_rate,
        "background_random": base_rate,
        "safe_release_support": _rate(sum(row[label] is True for row in support), len(support)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "enrichment": {
            "ratio": ratio, "status": ratio_status,
            "corrected_ratio": corrected_ratio,
            "corrected_status": "AVAILABLE" if corrected_ratio is not None else "UNAVAILABLE_ZERO_DENOMINATOR",
            "haldane__anscombe": True,
            "fisher_exact_two_sided_p": fisher,
        },
        "contingency": counts,
        "risk_difference": None if teacher_rate["rate"] is None or base_rate["rate"] is None else teacher_rate["rate"] - base_rate["rate"],
        "finite": finite({"teacher": teacher_rate, "baseline": base_rate, "ratio": ratio}),
    }


def _mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def compute_report(
    rows: list[dict[str, Any]],
    *,
    config: Mapping[str, Any],
    execution_class: str,
    binding: Mapping[str, Any],
    input_summary: Mapping[str, Any],
) -> dict[str, Any]:
    primary_rows, primary_summary = _primary_candidate_rows(rows)
    analysis_rows = rows if execution_class != "FORMAL" else primary_rows
    local = _metric(analysis_rows, "local_vulnerability")
    task = _metric(analysis_rows, "task_vulnerability")
    suite_breakdown: dict[str, Any] = {}
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        item = _metric([row for row in analysis_rows if row.get("suite") == suite], "local_vulnerability")
        suite_breakdown[suite] = {
            "enrichment": item["enrichment"]["ratio"],
            "corrected_enrichment": item["enrichment"]["corrected_ratio"],
            "enrichment_status": item["enrichment"]["status"],
            "risk_difference": item["risk_difference"],
            "teacher_rate": item["teacher_corridor"],
            "background_rate": item["background_random"],
            "unknown_or_abstain": item["unknown_or_abstain"],
        }
    per_arm = {arm: _metric([row for row in rows if row.get("arm") == arm], "local_vulnerability") for arm in OPEN_ARMS}
    timing = [int(row["timing_offset"]) for row in analysis_rows if isinstance(row.get("timing_offset"), int)]
    dose_response = {
        arm: {
            "teacher_corridor_local_rate": per_arm[arm]["teacher_corridor"],
            "background_random_local_rate": per_arm[arm]["background_random"],
        }
        for arm in OPEN_ARMS
    }
    local_ratio = local["enrichment"]["corrected_ratio"]
    local_recall = local["recall"]["rate"]
    conditions = {
        "corrected_local_vulnerability_enrichment_ge_3": isinstance(local_ratio, (int, float)) and local_ratio >= 3.0,
        "local_vulnerability_recall_ge_0_60": isinstance(local_recall, (int, float)) and local_recall >= 0.60,
        "all_suite_local_enrichment_gt_1": all(
            isinstance(item["risk_difference"], (int, float)) and item["risk_difference"] > 0 for item in suite_breakdown.values()
        ),
        "all_denominators_available": all(
            item["corrected_enrichment"] is not None for item in suite_breakdown.values()
        ) and local["enrichment"]["corrected_ratio"] is not None,
        "all_metrics_finite": finite({"local": local, "task": task, "suites": suite_breakdown, "arms": per_arm}),
        "identity_closure_pass": input_summary.get("invalid_branch_rows", 0) == 0 and primary_summary["identity_conflicts"] == 0,
    }
    # Diagnostic canaries exercise schema/recompute paths only; they are never a
    # scientific gate and therefore do not promote themselves to a formal PASS.
    gate_pass = all(conditions.values()) if execution_class == "FORMAL" else True
    report = {
        "schema": REPORT_SCHEMA,
        "status": "STAGE_V2_TEACHER_PROPOSAL_PASS" if gate_pass else "STAGE_V2_TEACHER_PROPOSAL_FAIL",
        "execution_class": execution_class,
        "formal": execution_class == "FORMAL",
        "for_gate": execution_class == "FORMAL",
        "read_only": True,
        "observation_unit": "unique candidate state under OPEN_T10",
        "primary_arm": PRIMARY_ARM,
        "auxiliary_arms": list(AUXILIARY_ARMS),
        "source_binding": dict(binding),
        "input_summary": dict(input_summary),
        "primary_summary": primary_summary,
        "primary_observation_count": len(analysis_rows),
        "local_vulnerability_enrichment": local_ratio,
        "local_vulnerability_recall": local_recall,
        "local_vulnerability": local,
        "task_vulnerability": task,
        "suite_breakdown": suite_breakdown,
        "per_arm": per_arm,
        "timing_offset": {"count": len(timing), "mean": _mean(timing), "unknown_count": len(analysis_rows) - len(timing)},
        "t3_t5_t10_dose_response": dose_response,
        "bootstrap": {
            "local_vulnerability": _bootstrap_cluster(
                analysis_rows, "local_vulnerability", int(config.get("bootstrap", {}).get("repetitions", 10000)),
                int(config.get("bootstrap", {}).get("seed", 2026080601)),
            ),
            "task_vulnerability": _bootstrap_cluster(
                analysis_rows, "task_vulnerability", int(config.get("bootstrap", {}).get("repetitions", 10000)),
                int(config.get("bootstrap", {}).get("seed", 2026080601)),
            ),
        },
        "gate": {"status": "PASS" if gate_pass else "FAIL", "conditions": conditions, "thresholds": {"local_enrichment": 3.0, "local_recall": 0.60, "suite_enrichment": 1.0}},
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "attack_rollouts": 0,
        "vis_rollouts": 0,
        "pgd_rollouts": 0,
        "generated_utc": utc_now(),
    }
    return report


def _active_stage_v_process(stage_v_root: Path) -> bool:
    try:
        output = subprocess.run(["ps", "-eo", "args="], check=False, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    root = str(stage_v_root)
    producer_tokens = ("run_stage_v_counterfactual", "run_stage_v_local_supervisor", "launch_stage_v_map")
    return any(root in line and any(token in line for token in producer_tokens) for line in output.splitlines())


def formal_binding(
    stage_v_root: Path,
    *,
    expected_source_commit: str,
    expected_source_tree: str,
    expected_parent_manifest_sha256: str | None = None,
    expected_run_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    closure_path = stage_v_root / "STAGE_V_CLOSURE_RECEIPT.json"
    closure = load_json(closure_path)
    if not isinstance(closure, Mapping) or closure.get("status") != "STAGE_V_FORMAL_MAP_CLOSED":
        raise StageV2PreconditionError("STAGE_V_CLOSURE_RECEIPT_NOT_PASS")
    dispatcher = load_json(stage_v_root / "DISPATCHER_COMPLETE.json")
    supervisor = load_json(stage_v_root / "SUPERVISOR_COMPLETE.json")
    auditor = load_json(stage_v_root / "STAGE_V_COUNTERFACTUAL_AUDIT.json")
    if not isinstance(dispatcher, Mapping) or dispatcher.get("status") != "PASS":
        raise StageV2PreconditionError("STAGE_V_DISPATCHER_COMPLETE_NOT_PASS")
    if not isinstance(supervisor, Mapping) or supervisor.get("status") != "PASS":
        raise StageV2PreconditionError("STAGE_V_SUPERVISOR_COMPLETE_NOT_PASS")
    if not isinstance(auditor, Mapping) or auditor.get("verdict") != "PASS":
        raise StageV2PreconditionError("STAGE_V_INDEPENDENT_AUDIT_NOT_PASS")
    seal_ok, seal_errors, _ = verify_sha_manifest(stage_v_root)
    if not seal_ok:
        raise StageV2PreconditionError("STAGE_V_ROOT_SEAL_FAIL:" + ";".join(seal_errors))
    receipt_root_seal = closure.get("root_seal")
    actual_root_seal = sha256_file(stage_v_root / "SHA256SUMS")
    if receipt_root_seal != actual_root_seal:
        raise StageV2PreconditionError("STAGE_V_ROOT_SEAL_BINDING_MISMATCH")
    if closure.get("source_commit") != expected_source_commit or closure.get("source_tree") != expected_source_tree:
        raise StageV2PreconditionError("STAGE_V_SOURCE_BINDING_MISMATCH")
    accepted = closure.get("accepted_parents", closure.get("accepted_parent_results"))
    if _int(closure.get("planned_parents")) != 40 or _int(closure.get("completed_parents")) != 40 or _int(accepted) != 40:
        raise StageV2PreconditionError("STAGE_V_CLOSURE_PARENT_COUNT_FAIL")
    if _active_stage_v_process(stage_v_root):
        raise StageV2PreconditionError("STAGE_V_RESIDUAL_PROCESS")
    run_manifest_path = stage_v_root / "RUN_MANIFEST.json"
    run_manifest = load_json(run_manifest_path)
    if not isinstance(run_manifest, Mapping):
        raise StageV2PreconditionError("STAGE_V_RUN_MANIFEST_INVALID")
    for key in ("eval160_reads", "protected_eval_reads", "attack_rollouts"):
        if _int(run_manifest.get(key)) != 0:
            raise StageV2PreconditionError(f"STAGE_V_BOUNDARY_NONZERO:{key}")
    run_hash = sha256_file(run_manifest_path)
    if expected_run_manifest_sha256 and run_hash != expected_run_manifest_sha256:
        raise StageV2PreconditionError("STAGE_V_RUN_MANIFEST_SHA_MISMATCH")
    parent_hash = str(closure.get("manifest_sha256", ""))
    if expected_parent_manifest_sha256 and parent_hash != expected_parent_manifest_sha256:
        raise StageV2PreconditionError("STAGE_V_PARENT_MANIFEST_SHA_MISMATCH")
    return {
        "stage_v_root": str(stage_v_root.resolve()),
        "stage_v_closure_receipt_sha256": sha256_file(closure_path),
        "stage_v_root_seal_sha256": actual_root_seal,
        "source_commit": expected_source_commit,
        "source_tree": expected_source_tree,
        "parent_manifest_sha256": parent_hash,
        "run_manifest_sha256": run_hash,
    }


def diagnostic_binding(stage_v_root: Path, expected_source_commit: str, expected_source_tree: str) -> dict[str, Any]:
    if not (stage_v_root / "DIAGNOSTIC_CANARY_ONLY").is_file():
        raise StageV2PreconditionError("DIAGNOSTIC_CANARY_MARKER_MISSING")
    return {
        "stage_v_root": str(stage_v_root.resolve()),
        "stage_v_closure_receipt_sha256": None,
        "stage_v_root_seal_sha256": None,
        "source_commit": expected_source_commit,
        "source_tree": expected_source_tree,
        "parent_manifest_sha256": None,
        "run_manifest_sha256": sha256_file(stage_v_root / "RUN_MANIFEST.json") if (stage_v_root / "RUN_MANIFEST.json").is_file() else None,
    }


def run_analysis(
    stage_v_root: Path,
    output_root: Path,
    config_path: Path,
    *,
    expected_source_commit: str,
    expected_source_tree: str,
    expected_parent_manifest_sha256: str | None = None,
    expected_run_manifest_sha256: str | None = None,
    diagnostic_canary: bool = False,
) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("",):
        raise StageV2PreconditionError("CUDA_VISIBLE_DEVICES_MUST_BE_EMPTY_OR_UNSET")
    config = load_json(config_path)
    if not isinstance(config, Mapping) or config.get("schema") != CONFIG_SCHEMA:
        raise StageV2PreconditionError("STAGE_V2_CONFIG_INVALID")
    binding = (
        diagnostic_binding(stage_v_root, expected_source_commit, expected_source_tree)
        if diagnostic_canary
        else formal_binding(
            stage_v_root,
            expected_source_commit=expected_source_commit,
            expected_source_tree=expected_source_tree,
            expected_parent_manifest_sha256=expected_parent_manifest_sha256,
            expected_run_manifest_sha256=expected_run_manifest_sha256,
        )
    )
    if output_root.exists():
        raise StageV2PreconditionError("OUTPUT_ROOT_ALREADY_EXISTS")
    output_root.mkdir(parents=True)
    input_summary = {"execution_class": "DIAGNOSTIC_CANARY_ONLY" if diagnostic_canary else "FORMAL"}
    rows, observed = load_observations(stage_v_root)
    input_summary.update(observed)
    receipt = {
        "schema": "STAGE_V2_INPUT_RECEIPT_V1",
        "execution_class": input_summary["execution_class"],
        "read_only": True,
        "for_gate": not diagnostic_canary,
        "source_binding": binding,
        "config_sha256": sha256_file(config_path),
        "stage_v2_runner_sha256": sha256_file(Path(os.environ.get("STAGE_V2_RUNNER_PATH", str(Path(__file__).resolve())))),
        "generated_utc": utc_now(),
    }
    atomic_write_json(output_root / "STAGE_V2_INPUT_RECEIPT.json", receipt)
    atomic_write_json(output_root / "STAGE_V2_CONFIG.json", config)
    rows_path = output_root / "STAGE_V2_TEACHER_ENRICHMENT_ROWS.jsonl"
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    report = compute_report(
        rows,
        config=config,
        execution_class=input_summary["execution_class"],
        binding=binding,
        input_summary=input_summary,
    )
    atomic_write_json(output_root / "STAGE_V2_TEACHER_ENRICHMENT_REPORT.json", report)
    atomic_write_json(
        output_root / "STAGE_V2_PRODUCER_COMPLETE.json",
        {"schema": "STAGE_V2_PRODUCER_COMPLETE_V1", "status": "PASS", "report_sha256": sha256_file(output_root / "STAGE_V2_TEACHER_ENRICHMENT_REPORT.json"), "completed_utc": utc_now()},
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-v-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--expected-parent-manifest-sha256")
    parser.add_argument("--expected-run-manifest-sha256")
    parser.add_argument("--diagnostic-canary", action="store_true")
    parser.add_argument("--run-independent-audit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_analysis(
            args.stage_v_root.resolve(),
            args.output_root.resolve(),
            args.config.resolve(),
            expected_source_commit=args.expected_source_commit,
            expected_source_tree=args.expected_source_tree,
            expected_parent_manifest_sha256=args.expected_parent_manifest_sha256,
            expected_run_manifest_sha256=args.expected_run_manifest_sha256,
            diagnostic_canary=args.diagnostic_canary,
        )
        if args.run_independent_audit:
            auditor = SCRIPT_DIR / "audit_stage_v2_teacher_enrichment.py"
            command = [
                sys.executable,
                str(auditor),
                "--stage-v-root",
                str(args.stage_v_root.resolve()),
                "--v2-root",
                str(args.output_root.resolve()),
                "--config",
                str(args.config.resolve()),
                "--expected-source-commit",
                args.expected_source_commit,
                "--expected-source-tree",
                args.expected_source_tree,
            ]
            if args.diagnostic_canary:
                command.append("--diagnostic-canary")
            audit = subprocess.run(command, check=False)
            if audit.returncode != 0:
                return audit.returncode
        print(json.dumps({"status": report["status"], "execution_class": report["execution_class"]}, sort_keys=True))
        return 0
    except (OSError, StageV2PreconditionError) as exc:
        print(f"STAGE_V2_PRECONDITION_FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
