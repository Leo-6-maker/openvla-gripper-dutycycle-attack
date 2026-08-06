"""Independent read-only audit for the Stage V2 enrichment artifact."""
from __future__ import annotations

import argparse
import datetime as _datetime
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.monitoring.audit_stage_v_closure import atomic_write_json, sha256_file, verify_sha_manifest
try:
    from .stage_v2_teacher_enrichment import (
        CONFIG_SCHEMA,
        REPORT_SCHEMA,
        StageV2PreconditionError,
        canonical_json,
        diagnostic_binding,
        formal_binding,
        load_json,
    )
except ImportError:  # pragma: no cover - direct script execution on server.
    from stage_v2_teacher_enrichment import (
        CONFIG_SCHEMA,
        REPORT_SCHEMA,
        StageV2PreconditionError,
        canonical_json,
        diagnostic_binding,
        formal_binding,
        load_json,
    )


SCHEMA = "STAGE_V2_INDEPENDENT_AUDIT_V1"
OPEN_ARMS = ("OPEN_T3", "OPEN_T5", "OPEN_T10")
PRIMARY_ARM = "OPEN_T10"
AUXILIARY_ARMS = ("OPEN_T3", "OPEN_T5")
GROUPS = ("teacher_corridor", "background_random", "safe_release_support")
TRUE_VALUES = {"1", "true", "yes", "y", "positive", "member", "in", "present"}
FALSE_VALUES = {"0", "false", "no", "n", "negative", "nonmember", "out", "absent"}


# Intentional duplication: this is the independent audit boundary.  Calling
# the producer's parser/statistics would turn a producer bug into an auditor
# PASS.  Keep the implementation boring and structurally parallel.
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
        lowered = value.strip().lower()
        if lowered in TRUE_VALUES:
            return True
        if lowered in FALSE_VALUES:
            return False
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


def _feature(sources: Iterable[Mapping[str, Any]], aliases: Iterable[str]) -> bool | None:
    for source in sources:
        result = _as_bool(_value(source, aliases))
        if result is not None:
            return result
    return None


def _group(row: Mapping[str, Any], parent: Mapping[str, Any]) -> tuple[str | None, str | None, dict[str, bool | None]]:
    sources = _sources(row, parent)
    flags = {
        "teacher_corridor": _feature(sources, ("teacher_corridor_membership", "in_teacher_corridor", "teacher_corridor", "corridor_membership")),
        "background_random": _feature(sources, ("background_random_membership", "background_membership", "random_membership", "is_background", "is_random")),
        "safe_release_support": _feature(sources, ("safe_release_support_membership", "safe_release_membership", "support_membership", "is_safe_release", "is_support")),
    }
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
    return _as_bool(comparison.get(name)) if isinstance(comparison, Mapping) else None


def _int(value: Any) -> int | None:
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_observations(stage_v_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    invalid_rows = parent_count = branch_count = 0
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
            lines = branch_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            invalid_rows += 1
            continue
        for line_number, line in enumerate(lines, 1):
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
            arm = str(row["arm"])
            probe_step = _int(_first(row.get(name) for name in ("probe_step", "candidate_step", "step")))
            group, group_reason, flags = _group(row, parent)
            sources = _sources(row, parent)
            role = _first(_value(source, ("teacher_role", "role")) for source in sources)
            phase = _first(_value(source, ("teacher_phase", "phase")) for source in sources)
            corridor_start = _int(_first(_value(source, ("teacher_corridor_start_step", "corridor_start_step")) for source in sources))
            observations.append({
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
                "timing_offset": None if probe_step is None or corridor_start is None else probe_step - corridor_start,
            })
    return observations, {"parent_count": parent_count, "branch_count": branch_count, "invalid_branch_rows": invalid_rows}


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return True


def _primary(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("arm") != PRIMARY_ARM:
            continue
        identity = f"{row.get('canonical_parent_key')}:{row.get('candidate_step')}"
        grouped.setdefault(identity, []).append(row)
    collapsed: list[dict[str, Any]] = []
    conflicts = 0
    for identity, items in sorted(grouped.items()):
        first = dict(items[0])
        local = {item.get("local_vulnerability") for item in items if isinstance(item.get("local_vulnerability"), bool)}
        task = {item.get("task_vulnerability") for item in items if isinstance(item.get("task_vulnerability"), bool)}
        conflicts += int(len(local) > 1 or len(task) > 1)
        first["row_id"] = identity
        if local:
            first["local_vulnerability"] = True if True in local else False
        if task:
            first["task_vulnerability"] = True if True in task else False
        collapsed.append(first)
    return collapsed, {"primary_arm": PRIMARY_ARM, "candidate_state_count": len(collapsed), "duplicate_source_rows": sum(max(0, len(items) - 1) for items in grouped.values()), "identity_conflicts": conflicts, "auxiliary_arms": list(AUXILIARY_ARMS)}


def _wilson(positive: int, denominator: int) -> list[float] | None:
    if denominator <= 0:
        return None
    z = 1.959963984540054
    p = positive / denominator
    term = 1.0 + z * z / denominator
    centre = (p + z * z / (2 * denominator)) / term
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * denominator)) / denominator) / term
    return [max(0.0, centre - spread), min(1.0, centre + spread)]


def _rate(positive: int, denominator: int) -> dict[str, Any]:
    if denominator <= 0:
        return {"numerator": positive, "denominator": denominator, "rate": None, "ci95": None, "status": "UNAVAILABLE_ZERO_DENOMINATOR"}
    return {"numerator": positive, "denominator": denominator, "rate": positive / denominator, "ci95": _wilson(positive, denominator), "status": "AVAILABLE"}


def _counts(rows: list[Mapping[str, Any]], label: str) -> dict[str, int]:
    teacher = [row for row in rows if row.get("group") == "teacher_corridor" and isinstance(row.get(label), bool)]
    baseline = [row for row in rows if row.get("group") == "background_random" and isinstance(row.get(label), bool)]
    return {"teacher_positive": sum(row[label] is True for row in teacher), "teacher_total": len(teacher), "background_positive": sum(row[label] is True for row in baseline), "background_total": len(baseline)}


def _fisher(a: int, b: int, c: int, d: int) -> float | None:
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


def _corrected(counts: Mapping[str, int]) -> float | None:
    if counts["teacher_total"] <= 0 or counts["background_total"] <= 0:
        return None
    return ((counts["teacher_positive"] + 0.5) / (counts["teacher_total"] + 1.0)) / ((counts["background_positive"] + 0.5) / (counts["background_total"] + 1.0))


def _bootstrap(rows: list[dict[str, Any]], label: str, repetitions: int, seed: int) -> dict[str, Any]:
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
        value = _corrected(_counts(sample, label))
        if value is not None and math.isfinite(value):
            values.append(value)
    if not values:
        return {"repetitions": repetitions, "seed": seed, "ci95": None, "status": "UNAVAILABLE_ZERO_DENOMINATOR"}
    values.sort()
    def percentile(q: float) -> float:
        return float(values[min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))])
    return {"repetitions": repetitions, "seed": seed, "ci95": [percentile(0.025), percentile(0.975)], "status": "AVAILABLE"}


def _metric(rows: list[Mapping[str, Any]], label: str) -> dict[str, Any]:
    known = [row for row in rows if row.get("group") in GROUPS and isinstance(row.get(label), bool)]
    teacher = [row for row in known if row.get("group") == "teacher_corridor"]
    baseline = [row for row in known if row.get("group") == "background_random"]
    support = [row for row in known if row.get("group") == "safe_release_support"]
    all_positive = sum(row[label] is True for row in known)
    counts = _counts(list(rows), label)
    teacher_positive = sum(row[label] is True for row in teacher)
    baseline_positive = sum(row[label] is True for row in baseline)
    teacher_rate = _rate(teacher_positive, len(teacher))
    baseline_rate = _rate(baseline_positive, len(baseline))
    ratio = None
    ratio_status = "AVAILABLE"
    if teacher_rate["rate"] is None or baseline_rate["rate"] is None:
        ratio_status = "UNAVAILABLE_ZERO_DENOMINATOR"
    elif baseline_rate["rate"] <= 0:
        ratio_status = "INFINITE_ZERO_BASELINE" if teacher_rate["rate"] > 0 else "UNAVAILABLE_ZERO_BASELINE_RATE"
    else:
        ratio = teacher_rate["rate"] / baseline_rate["rate"]
    tn = counts["background_total"] - counts["background_positive"]
    return {
        "label": label, "rows": len(rows), "unknown_or_abstain": len(rows) - len(known),
        "teacher_corridor": teacher_rate, "background_random": baseline_rate,
        "safe_release_support": _rate(sum(row[label] is True for row in support), len(support)),
        "precision": _rate(teacher_positive, len(teacher)), "recall": _rate(teacher_positive, all_positive),
        "specificity": _rate(tn, tn + counts["background_positive"]),
        "enrichment": {"ratio": ratio, "status": ratio_status, "corrected_ratio": _corrected(counts), "corrected_status": "AVAILABLE" if _corrected(counts) is not None else "UNAVAILABLE_ZERO_DENOMINATOR", "haldane__anscombe": True, "fisher_exact_two_sided_p": _fisher(counts["teacher_positive"], counts["teacher_total"] - counts["teacher_positive"], counts["background_positive"], counts["background_total"] - counts["background_positive"])},
        "contingency": counts,
        "risk_difference": None if teacher_rate["rate"] is None or baseline_rate["rate"] is None else teacher_rate["rate"] - baseline_rate["rate"],
        "finite": _finite({"teacher": teacher_rate, "baseline": baseline_rate, "ratio": ratio}),
    }


def _recompute(rows: list[dict[str, Any]], *, config: Mapping[str, Any], execution_class: str,
               binding: Mapping[str, Any], input_summary: Mapping[str, Any]) -> dict[str, Any]:
    primary_rows, primary_summary = _primary(rows)
    analysis_rows = rows if execution_class != "FORMAL" else primary_rows
    local = _metric(analysis_rows, "local_vulnerability")
    task = _metric(analysis_rows, "task_vulnerability")
    suites: dict[str, Any] = {}
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        item = _metric([row for row in analysis_rows if row.get("suite") == suite], "local_vulnerability")
        suites[suite] = {"enrichment": item["enrichment"]["ratio"], "corrected_enrichment": item["enrichment"]["corrected_ratio"], "enrichment_status": item["enrichment"]["status"], "risk_difference": item["risk_difference"], "teacher_rate": item["teacher_corridor"], "background_rate": item["background_random"], "unknown_or_abstain": item["unknown_or_abstain"]}
    per_arm = {arm: _metric([row for row in rows if row.get("arm") == arm], "local_vulnerability") for arm in OPEN_ARMS}
    timing = [int(row["timing_offset"]) for row in analysis_rows if isinstance(row.get("timing_offset"), int)]
    dose_response = {arm: {"teacher_corridor_local_rate": per_arm[arm]["teacher_corridor"], "background_random_local_rate": per_arm[arm]["background_random"]} for arm in OPEN_ARMS}
    local_ratio = local["enrichment"]["corrected_ratio"]
    local_recall = local["recall"]["rate"]
    conditions = {
        "corrected_local_vulnerability_enrichment_ge_3": isinstance(local_ratio, (int, float)) and local_ratio >= 3.0,
        "local_vulnerability_recall_ge_0_60": isinstance(local_recall, (int, float)) and local_recall >= 0.60,
        "all_suite_local_enrichment_gt_1": all(isinstance(item["risk_difference"], (int, float)) and item["risk_difference"] > 0 for item in suites.values()),
        "all_denominators_available": all(item["corrected_enrichment"] is not None for item in suites.values()) and local_ratio is not None,
        "all_metrics_finite": _finite({"local": local, "task": task, "suites": suites, "arms": per_arm}),
        "identity_closure_pass": input_summary.get("invalid_branch_rows", 0) == 0 and primary_summary["identity_conflicts"] == 0,
    }
    gate_pass = all(conditions.values()) if execution_class == "FORMAL" else True
    bootstrap_config = config.get("bootstrap", {}) if isinstance(config.get("bootstrap"), Mapping) else {}
    repetitions = int(bootstrap_config.get("repetitions", 10000))
    seed = int(bootstrap_config.get("seed", 2026080701))
    return {
        "schema": REPORT_SCHEMA,
        "status": "STAGE_V2_TEACHER_PROPOSAL_PASS" if gate_pass else "STAGE_V2_TEACHER_PROPOSAL_FAIL",
        "execution_class": execution_class, "formal": execution_class == "FORMAL", "for_gate": execution_class == "FORMAL", "read_only": True,
        "observation_unit": "unique candidate state under OPEN_T10", "primary_arm": PRIMARY_ARM, "auxiliary_arms": list(AUXILIARY_ARMS),
        "source_binding": dict(binding), "input_summary": dict(input_summary), "primary_summary": primary_summary,
        "primary_observation_count": len(analysis_rows), "local_vulnerability_enrichment": local_ratio, "local_vulnerability_recall": local_recall,
        "local_vulnerability": local, "task_vulnerability": task, "suite_breakdown": suites, "per_arm": per_arm,
        "timing_offset": {"count": len(timing), "mean": sum(timing) / len(timing) if timing else None, "unknown_count": len(analysis_rows) - len(timing)},
        "t3_t5_t10_dose_response": dose_response,
        "bootstrap": {"local_vulnerability": _bootstrap(analysis_rows, "local_vulnerability", repetitions, seed), "task_vulnerability": _bootstrap(analysis_rows, "task_vulnerability", repetitions, seed)},
        "gate": {"status": "PASS" if gate_pass else "FAIL", "conditions": conditions, "thresholds": {"local_enrichment": 3.0, "local_recall": 0.60, "suite_enrichment": 1.0}},
        "eval160_reads": 0, "protected_eval_reads": 0, "attack_rollouts": 0, "vis_rollouts": 0, "pgd_rollouts": 0,
        "generated_utc": utc_now(),
    }


def _write_v2_seal(root: Path) -> dict[str, Any]:
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            continue
        entries.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n")
    sums = root / "SHA256SUMS"
    sums.write_text("".join(entries), encoding="utf-8")
    with sums.open("r+b") as handle:
        import os

        os.fsync(handle.fileno())
    sidecar = root / "SHA256SUMS.sha256"
    sidecar.write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")
    with sidecar.open("r+b") as handle:
        import os

        os.fsync(handle.fileno())
    return {"files": len(entries), "sha256sums_sha256": sha256_file(sums)}


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _report_core(value: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {"generated_utc"}
    return {key: item for key, item in value.items() if key not in ignored}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row is not an object")
                rows.append(value)
    return rows


def audit(
    stage_v_root: Path,
    v2_root: Path,
    config_path: Path,
    *,
    expected_source_commit: str,
    expected_source_tree: str,
    diagnostic_canary: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    if (v2_root / "SHA256SUMS").is_file() or (v2_root / "SHA256SUMS.sha256").is_file():
        seal_ok, seal_errors, _ = verify_sha_manifest(v2_root)
        if not seal_ok:
            errors.extend(f"v2_{item}" for item in seal_errors)
    config = load_json(config_path)
    if not isinstance(config, Mapping) or config.get("schema") != CONFIG_SCHEMA:
        errors.append("config_invalid")
        config = {}
    try:
        binding = (
            diagnostic_binding(stage_v_root, expected_source_commit, expected_source_tree)
            if diagnostic_canary
            else formal_binding(stage_v_root, expected_source_commit=expected_source_commit, expected_source_tree=expected_source_tree)
        )
    except StageV2PreconditionError as exc:
        binding = {}
        errors.append(str(exc))
    input_receipt = load_json(v2_root / "STAGE_V2_INPUT_RECEIPT.json")
    report = load_json(v2_root / "STAGE_V2_TEACHER_ENRICHMENT_REPORT.json")
    if not isinstance(input_receipt, Mapping):
        errors.append("input_receipt_missing_or_invalid")
    elif input_receipt.get("source_binding") != binding:
        errors.append("input_binding_mismatch")
    if not isinstance(report, Mapping) or report.get("schema") != REPORT_SCHEMA:
        errors.append("producer_report_missing_or_invalid")
        report = {}
    if not _finite(report):
        errors.append("producer_report_non_finite")
    try:
        observed, observed_summary = _read_observations(stage_v_root)
        stored_rows = _read_rows(v2_root / "STAGE_V2_TEACHER_ENRICHMENT_ROWS.jsonl")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        observed, observed_summary, stored_rows = [], {}, []
        errors.append(f"rows_unreadable:{exc}")
    if [canonical_json(row) for row in observed] != [canonical_json(row) for row in stored_rows]:
        errors.append("producer_rows_do_not_match_sealed_input")
    recomputed = _recompute(
        observed,
        config=config,
        execution_class="DIAGNOSTIC_CANARY_ONLY" if diagnostic_canary else "FORMAL",
        binding=binding,
        input_summary={
            **observed_summary,
            "execution_class": "DIAGNOSTIC_CANARY_ONLY" if diagnostic_canary else "FORMAL",
        },
    )
    if _report_core(recomputed) != _report_core(report):
        errors.append("independent_recompute_disagrees")
    for key in ("eval160_reads", "protected_eval_reads", "attack_rollouts", "vis_rollouts", "pgd_rollouts"):
        if report.get(key, 0) != 0:
            errors.append(f"boundary_nonzero:{key}")
    root_seal_status = "NOT_REQUIRED_DIAGNOSTIC"
    if not diagnostic_canary:
        root_seal_ok, root_seal_errors, _ = verify_sha_manifest(stage_v_root)
        root_seal_status = "PASS" if root_seal_ok else "FAIL"
        errors.extend(root_seal_errors)
    audit_report = {
        "schema": SCHEMA,
        "verdict": "PASS" if not errors else "FAIL",
        "execution_class": "DIAGNOSTIC_CANARY_ONLY" if diagnostic_canary else "FORMAL",
        "formal": not diagnostic_canary,
        "for_gate": not diagnostic_canary,
        "stage_v_root": str(stage_v_root),
        "v2_root": str(v2_root),
        "source_binding": binding,
        "root_seal_status": root_seal_status,
        "producer_report_sha256": sha256_file(v2_root / "STAGE_V2_TEACHER_ENRICHMENT_REPORT.json") if (v2_root / "STAGE_V2_TEACHER_ENRICHMENT_REPORT.json").is_file() else None,
        "recomputed_status": recomputed.get("status"),
        "observed_row_count": len(observed),
        "stored_row_count": len(stored_rows),
        "errors": sorted(set(errors)),
        "audited_utc": utc_now(),
    }
    atomic_write_json(v2_root / "STAGE_V2_INDEPENDENT_AUDIT.json", audit_report)
    _write_v2_seal(v2_root)
    return audit_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-v-root", required=True, type=Path)
    parser.add_argument("--v2-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--diagnostic-canary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit(
            args.stage_v_root.resolve(),
            args.v2_root.resolve(),
            args.config.resolve(),
            expected_source_commit=args.expected_source_commit,
            expected_source_tree=args.expected_source_tree,
            diagnostic_canary=args.diagnostic_canary,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"STAGE_V2_AUDIT_FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"verdict": report["verdict"], "errors": report["errors"]}, sort_keys=True))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
