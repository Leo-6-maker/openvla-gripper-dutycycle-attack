#!/usr/bin/env python3
"""Analyze frozen Stage V/VI-B2 mechanism telemetry without running an environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from audit_stage_x0_mediator_availability import (
    ARMS,
    COUNTERS,
    DOSES,
    audit_labels,
    branch_object,
    branch_paths,
    collect_protected,
    first_value,
    identity,
    load_json,
    load_jsonl,
    sha256_file,
)


def git_value(worktree: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(worktree), *args], text=True).strip()


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def vector(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(finite(item) for item in value)


def mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def bootstrap_parent_mean(values_by_parent: dict[str, list[float]]) -> dict[str, Any]:
    parent_values = {key: mean(values) for key, values in values_by_parent.items() if values}
    parent_values = {key: float(value) for key, value in parent_values.items() if value is not None}
    if not parent_values:
        return {"parent_count": 0, "mean": None, "ci95": None, "replicates": 2000, "seed": 20260817}
    keys = sorted(parent_values)
    observed = float(sum(parent_values.values()) / len(parent_values))
    rng = random.Random(20260817)
    draws = [
        sum(parent_values[rng.choice(keys)] for _ in keys) / len(keys)
        for _ in range(2000)
    ]
    return {
        "parent_count": len(keys),
        "mean": observed,
        "ci95": [quantile(draws, 0.025), quantile(draws, 0.975)],
        "replicates": 2000,
        "seed": 20260817,
    }


def label_value(label: dict[str, Any]) -> int | None:
    if label.get("binary_label_consumable") is not True:
        return None
    if label.get("control_valid") is not True or label.get("treatment_valid") is not True:
        return None
    if label.get("censoring_class") not in (None, "NONE"):
        return None
    if label.get("label_class") == "V_PHYS":
        return 1
    if label.get("label_class") == "NO_PHYSICAL_VULNERABILITY":
        return 0
    return None


def control_failure(label: dict[str, Any]) -> int | None:
    if label.get("control_valid") is not True:
        return None
    value = label.get("control_physical_class")
    if value == "NO_PHYSICAL_FAILURE":
        return 0
    if isinstance(value, str) and value:
        return 1
    return None


def branch_rows(branch: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = branch.get("rows")
    if not isinstance(rows, list):
        return {}
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("relative_step"), int):
            return {}
        step = int(row["relative_step"])
        if step in output:
            raise ValueError(f"duplicate relative_step {step}")
        output[step] = row
    return output


def treatment_fraction(branch: dict[str, Any]) -> float | None:
    compliance = branch.get("treatment_compliance")
    if not isinstance(compliance, dict) or compliance.get("command_delivery_valid") is not True:
        return None
    delivered = compliance.get("delivered_open_steps")
    expected = compliance.get("expected_open_steps")
    if not finite(delivered) or not finite(expected) or float(expected) <= 0:
        return None
    return float(delivered) / float(expected)


def mechanism_record(
    stage: str,
    parent: str,
    probe_id: str,
    probe_step: int,
    arms: dict[str, dict[str, Any]],
    labels: dict[tuple[str, str, str, int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    control = branch_rows(arms["CONTROL"])
    result: list[dict[str, Any]] = []
    for dose in DOSES:
        treatment = branch_rows(arms[dose])
        common = sorted(set(control).intersection(treatment))
        label = labels.get((stage, parent, probe_id, probe_step, dose), {})
        endpoint = label_value(label) if label else None
        control_y = control_failure(label) if label else None

        m1 = treatment_fraction(arms[dose])
        m2_values: list[float] = []
        m3_valid = True
        m3_loss = False
        m4_values: list[float] = []
        for step in common:
            control_row = control[step]
            treatment_row = treatment[step]
            if finite(control_row.get("gripper_aperture")) and finite(treatment_row.get("gripper_aperture")):
                m2_values.append(float(treatment_row["gripper_aperture"]) - float(control_row["gripper_aperture"]))
            if (
                control_row.get("post_contact_telemetry_valid") is not True
                or treatment_row.get("post_contact_telemetry_valid") is not True
                or not isinstance(control_row.get("post_object_gripper_contact"), bool)
                or not isinstance(treatment_row.get("post_object_gripper_contact"), bool)
                or not isinstance(control_row.get("post_object_support_contact"), bool)
                or not isinstance(treatment_row.get("post_object_support_contact"), bool)
            ):
                m3_valid = False
            else:
                m3_loss = m3_loss or (
                    (control_row["post_object_gripper_contact"] and not treatment_row["post_object_gripper_contact"])
                    or (control_row["post_object_support_contact"] and not treatment_row["post_object_support_contact"])
                )
            if vector(control_row.get("post_object_position")) and vector(treatment_row.get("post_object_position")):
                m4_values.append(
                    math.sqrt(sum(
                        (float(treatment_row["post_object_position"][index]) - float(control_row["post_object_position"][index])) ** 2
                        for index in range(3)
                    ))
                )

        result.append({
            "stage": stage,
            "canonical_parent_key": parent,
            "parent_unit": f"{stage}|{parent}",
            "probe_id": probe_id,
            "probe_step": probe_step,
            "dose": dose,
            "endpoint_v_phys": endpoint,
            "control_failure": control_y,
            "censoring_class": label.get("censoring_class") if label else "LABEL_ABSENT",
            "treatment_compliant": label.get("treatment_compliant") if label else None,
            "overlap_steps": len(common),
            "m1_commanded_open_fraction": m1,
            "m2_aperture_excess_auc_vs_control": sum(m2_values) if m2_values else None,
            "m2_aperture_overlap_mean_delta": mean(m2_values),
            "m3_any_contact_loss": m3_loss if m3_valid and common else None,
            "m4_max_object_displacement": max(m4_values) if m4_values else None,
        })
    return result


def load_raw_groups(protocol: dict[str, Any]) -> tuple[dict[tuple[str, str, str, int], dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    stage_v_paths, stage_vi_paths = branch_paths(protocol)
    groups: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    protected: list[dict[str, Any]] = []
    for stage, paths in (("STAGE_V", stage_v_paths), ("STAGE_VI_B2", stage_vi_paths)):
        for path in sorted(paths):
            for record in load_jsonl(path):
                branch = branch_object(record)
                parent, probe_id, probe_step = identity(record, branch, path)
                arm = first_value(record, branch, "arm")
                if arm not in ARMS:
                    raise ValueError(f"invalid arm {arm!r}")
                key = (stage, parent, probe_id, int(probe_step))
                if arm in groups[key]:
                    raise ValueError(f"duplicate arm {key} {arm}")
                groups[key][str(arm)] = branch
                collect_protected(record, protected)
    return groups, protected


def load_labels(protocol: dict[str, Any]) -> tuple[dict[tuple[str, str, str, int, str], dict[str, Any]], list[dict[str, Any]]]:
    labels: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    protected: list[dict[str, Any]] = []
    for stage, spec in (("STAGE_V", protocol["inputs"]["stage_v_labels"]), ("STAGE_VI_B2", protocol["inputs"]["stage_vi_b2_labels"])):
        path = Path(spec["path"])
        for label in load_jsonl(path):
            key = (stage, str(label["canonical_parent_key"]), str(label["probe_id"]), int(label["probe_step"]), str(label["dose"]))
            if key in labels:
                raise ValueError(f"duplicate label {key}")
            labels[key] = label
            collect_protected(label, protected)
    return labels, protected


def summarize_numeric(records: list[dict[str, Any]], dose: str, key: str) -> dict[str, Any]:
    eligible = [record for record in records if record["dose"] == dose and finite(record.get(key))]
    values = [float(record[key]) for record in eligible]
    parent_values: dict[str, list[float]] = defaultdict(list)
    for record in eligible:
        parent_values[record["parent_unit"]].append(float(record[key]))
    summary = bootstrap_parent_mean(parent_values)
    summary.update({"dose": dose, "metric": key, "row_count": len(values), "raw_mean": mean(values), "raw_min": min(values) if values else None, "raw_max": max(values) if values else None})
    return summary


def summarize_binary(records: list[dict[str, Any]], dose: str, key: str) -> dict[str, Any]:
    eligible = [record for record in records if record["dose"] == dose and isinstance(record.get(key), bool)]
    values = [1.0 if record[key] else 0.0 for record in eligible]
    parent_values: dict[str, list[float]] = defaultdict(list)
    for record in eligible:
        parent_values[record["parent_unit"]].append(1.0 if record[key] else 0.0)
    summary = bootstrap_parent_mean(parent_values)
    summary.update({"dose": dose, "metric": key, "row_count": len(values), "positive_count": int(sum(values)), "raw_rate": mean(values)})
    return summary


def summarize_endpoint(records: list[dict[str, Any]], dose: str) -> dict[str, Any]:
    eligible = [record for record in records if record["dose"] == dose and record.get("endpoint_v_phys") in (0, 1)]
    parent_values: dict[str, list[float]] = defaultdict(list)
    for record in eligible:
        parent_values[record["parent_unit"]].append(float(record["endpoint_v_phys"]))
    summary = bootstrap_parent_mean(parent_values)
    summary.update({
        "dose": dose,
        "metric": "V_phys",
        "row_count": len(eligible),
        "positive_count": sum(int(record["endpoint_v_phys"]) for record in eligible),
        "abstain_or_unknown_count": sum(1 for record in records if record["dose"] == dose) - len(eligible),
        "raw_rate": mean([float(record["endpoint_v_phys"]) for record in eligible]),
    })
    return summary


def paired_control(records: list[dict[str, Any]], dose: str) -> dict[str, Any]:
    eligible = [record for record in records if record["dose"] == dose and record.get("endpoint_v_phys") in (0, 1) and record.get("control_failure") in (0, 1)]
    cells = Counter((int(record["control_failure"]), int(record["endpoint_v_phys"])) for record in eligible)
    control_rate = mean([float(record["control_failure"]) for record in eligible])
    treatment_rate = mean([float(record["endpoint_v_phys"]) for record in eligible])
    discordant_control0_treatment1 = cells[(0, 1)]
    discordant_control1_treatment0 = cells[(1, 0)]
    odds_ratio = None
    if discordant_control1_treatment0 > 0:
        odds_ratio = float(discordant_control0_treatment1 / discordant_control1_treatment0)
    return {
        "dose": dose,
        "pair_count": len(eligible),
        "cells_control_failure_by_vphys": {f"{control}:{treatment}": cells[(control, treatment)] for control in (0, 1) for treatment in (0, 1)},
        "control_failure_rate": control_rate,
        "vphys_rate": treatment_rate,
        "paired_risk_difference_vphys_minus_control_failure": None if control_rate is None or treatment_rate is None else treatment_rate - control_rate,
        "discordant_control0_treatment1": discordant_control0_treatment1,
        "discordant_control1_treatment0": discordant_control1_treatment0,
        "mcnemar_identifiable": bool(discordant_control0_treatment1 or discordant_control1_treatment0),
        "odds_ratio_control0_treatment1_over_control1_treatment0": odds_ratio,
    }


def dose_pairwise(records: list[dict[str, Any]], low: str, high: str) -> dict[str, Any]:
    by_id = {(record["stage"], record["canonical_parent_key"], record["probe_id"], record["probe_step"]): record for record in records}
    low_records = {key: record for key, record in by_id.items() if record["dose"] == low}
    high_records = {key: record for key, record in by_id.items() if record["dose"] == high}
    diffs: list[float] = []
    parent_values: dict[str, list[float]] = defaultdict(list)
    for key in sorted(set(low_records).intersection(high_records)):
        low_y = low_records[key].get("endpoint_v_phys")
        high_y = high_records[key].get("endpoint_v_phys")
        if low_y in (0, 1) and high_y in (0, 1):
            difference = float(high_y - low_y)
            diffs.append(difference)
            parent_values[high_records[key]["parent_unit"]].append(difference)
    summary = bootstrap_parent_mean(parent_values)
    summary.update({"low_dose": low, "high_dose": high, "pair_count": len(diffs), "raw_mean_difference_high_minus_low": mean(diffs)})
    return summary


def monotonicity(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, int], dict[str, int]] = defaultdict(dict)
    for record in records:
        if record.get("endpoint_v_phys") in (0, 1):
            groups[(record["stage"], record["canonical_parent_key"], record["probe_id"], record["probe_step"])][record["dose"]] = int(record["endpoint_v_phys"])
    patterns: Counter[str] = Counter()
    by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    for key, values in groups.items():
        if all(dose in values for dose in DOSES):
            pattern = "".join(str(values[dose]) for dose in DOSES)
            patterns[pattern] += 1
            by_stage[key[0]][pattern] += 1
    monotone = sum(patterns[pattern] for pattern in ("000", "001", "011", "111"))
    return {
        "complete_three_dose_probe_count": int(sum(patterns.values())),
        "patterns": dict(sorted(patterns.items())),
        "monotone_count": int(monotone),
        "nonmonotone_count": int(sum(patterns.values()) - monotone),
        "by_stage": {stage: dict(sorted(counts.items())) for stage, counts in sorted(by_stage.items())},
    }


def entropy(values: list[int]) -> float | None:
    if not values:
        return None
    counts = Counter(values)
    total = len(values)
    return float(-sum((count / total) * math.log2(count / total) for count in counts.values()))


def temporal_heterogeneity(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("endpoint_v_phys") in (0, 1):
            groups[(record["stage"], record["canonical_parent_key"], record["dose"])].append(record)
    parent_rows: list[dict[str, Any]] = []
    for (stage, parent, dose), group in sorted(groups.items()):
        ordered = [int(record["endpoint_v_phys"]) for record in sorted(group, key=lambda item: (int(item["probe_step"]), str(item["probe_id"]))) ]
        transitions = sum(left != right for left, right in zip(ordered, ordered[1:]))
        longest = 0
        current = 0
        for value in ordered:
            current = current + 1 if value == 1 else 0
            longest = max(longest, current)
        positives = [index for index, value in enumerate(ordered) if value == 1]
        parent_rows.append({
            "stage": stage,
            "canonical_parent_key": parent,
            "dose": dose,
            "observed_probe_count": len(ordered),
            "vulnerable_fraction": mean([float(value) for value in ordered]),
            "transition_count": transitions,
            "longest_vulnerable_run": longest,
            "entropy": entropy(ordered),
            "first_vulnerable_index": positives[0] if positives else None,
            "last_vulnerable_index": positives[-1] if positives else None,
        })
    summary: dict[str, Any] = {"parent_dose_count": len(parent_rows), "rows": parent_rows, "macro": {}}
    for dose in DOSES:
        subset = [row for row in parent_rows if row["dose"] == dose]
        summary["macro"][dose] = {
            metric: mean([float(row[metric]) for row in subset if row[metric] is not None])
            for metric in ("observed_probe_count", "vulnerable_fraction", "transition_count", "longest_vulnerable_run", "entropy")
        }
    return summary


def outcome_association(records: list[dict[str, Any]], metric: str, dose: str) -> dict[str, Any]:
    eligible = [record for record in records if record["dose"] == dose and record.get("endpoint_v_phys") in (0, 1) and finite(record.get(metric))]
    positive = [float(record[metric]) for record in eligible if record["endpoint_v_phys"] == 1]
    negative = [float(record[metric]) for record in eligible if record["endpoint_v_phys"] == 0]
    return {"dose": dose, "metric": metric, "pair_count": len(eligible), "vphys1_count": len(positive), "vphys0_count": len(negative), "mean_given_vphys1": mean(positive), "mean_given_vphys0": mean(negative)}


def seal(root: Path, summary: dict[str, Any]) -> None:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json", "ROOT_SEAL.sha256"}
    entries = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
    sums_sha = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    root_seal = {
        "schema": "STAGE_X_X0_RESULT_ROOT_SEAL_V1",
        "status": summary["status"],
        "summary_sha256": sha256_file(root / "STAGE_X_X0_RESULT.json"),
        "sha256sums_sha256": sums_sha,
        "physical_intervention": False,
        "new_env_steps": 0,
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    (root / "ROOT_SEAL.json").write_text(json.dumps(root_seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "ROOT_SEAL.sha256").write_text(f"{sha256_file(root / 'ROOT_SEAL.json')}  ROOT_SEAL.json\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--worktree", required=True, type=Path)
    parser.add_argument("--availability-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    protocol = load_json(args.protocol)
    availability_root = args.availability_root.resolve()
    availability_summary_path = availability_root / "STAGE_X_X0_MEDIATOR_AVAILABILITY.json"
    availability_summary = load_json(availability_summary_path)
    if availability_summary.get("status") != "AUDIT_COMPLETE" or availability_summary.get("protected_counters_valid") is not True:
        raise ValueError("availability audit is not complete and protected-clean")
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    groups, branch_protected = load_raw_groups(protocol)
    labels, label_protected = load_labels(protocol)
    if any(item.get("counters") != COUNTERS for item in branch_protected + label_protected):
        raise ValueError("protected counter violation in input evidence")
    if not all(all(arm in groups[key] for arm in ARMS) for key in groups):
        raise ValueError("incomplete four-arm group")

    mechanism_rows: list[dict[str, Any]] = []
    for (stage, parent, probe_id, probe_step), arms in sorted(groups.items()):
        mechanism_rows.extend(mechanism_record(stage, parent, probe_id, probe_step, arms, labels))
    if len(mechanism_rows) != 4032:
        raise ValueError(f"unexpected mechanism row count {len(mechanism_rows)}")

    endpoint = {dose: summarize_endpoint(mechanism_rows, dose) for dose in DOSES}
    command = {dose: summarize_numeric(mechanism_rows, dose, "m1_commanded_open_fraction") for dose in DOSES}
    aperture = {dose: summarize_numeric(mechanism_rows, dose, "m2_aperture_excess_auc_vs_control") for dose in DOSES}
    contact = {dose: summarize_binary(mechanism_rows, dose, "m3_any_contact_loss") for dose in DOSES}
    displacement = {dose: summarize_numeric(mechanism_rows, dose, "m4_max_object_displacement") for dose in DOSES}
    paired = {dose: paired_control(mechanism_rows, dose) for dose in DOSES}
    paired_doses = {
        "T5_minus_T3": dose_pairwise(mechanism_rows, "T3", "T5"),
        "T10_minus_T5": dose_pairwise(mechanism_rows, "T5", "T10"),
        "T10_minus_T3": dose_pairwise(mechanism_rows, "T3", "T10"),
    }
    monotone = monotonicity(mechanism_rows)
    heterogeneity = temporal_heterogeneity(mechanism_rows)
    associations = {
        metric: {dose: outcome_association(mechanism_rows, metric, dose) for dose in DOSES}
        for metric in ("m1_commanded_open_fraction", "m2_aperture_excess_auc_vs_control", "m3_any_contact_loss", "m4_max_object_displacement")
    }

    risk = [endpoint[dose]["raw_rate"] for dose in DOSES]
    aperture_mean = [aperture[dose]["raw_mean"] for dose in DOSES]
    risk_monotone = all(value is not None for value in risk) and risk[0] <= risk[1] <= risk[2]
    aperture_monotone = all(value is not None for value in aperture_mean) and aperture_mean[0] <= aperture_mean[1] <= aperture_mean[2]
    combined = availability_summary["combined_availability"]
    m1_available = combined["m1"]["available"]
    m2_available = combined["m2"]["available"]
    m3_available = combined["m3"]["available"]
    m4_available = combined["m4"]["available"]
    if not (m1_available and m2_available and (m3_available or m4_available)):
        status = "STAGE_X_MECHANISM_TELEMETRY_INSUFFICIENT"
    elif risk_monotone and aperture_monotone:
        status = "STAGE_X_PHYSICAL_DUTY_CYCLE_MECHANISM_SUPPORTED"
    else:
        status = "STAGE_X_PHYSICAL_DOSE_RESPONSE_WEAK_OR_NONMONOTONIC"

    summary = {
        "schema": "STAGE_X_X0_RESULT_V1",
        "status": status,
        "x0_authorized": False,
        "x1_clean_no_env_diagnostic_permitted": status != "STAGE_X_MECHANISM_TELEMETRY_INSUFFICIENT",
        "x2_physical_pgd_authorized": False,
        "decision_rule": "A requires exact M1, M2, and at least one exact consequence mediator plus nondecreasing V_phys and aperture dose ordering; otherwise B; missing exact key telemetry is C",
        "source_commit": git_value(args.worktree, "rev-parse", "HEAD"),
        "source_tree": git_value(args.worktree, "rev-parse", "HEAD^{tree}"),
        "source_script_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "availability_summary_sha256": sha256_file(availability_summary_path),
        "availability_root_seal_sha256": sha256_file(availability_root / "ROOT_SEAL.json"),
        "input_population": {"four_arm_group_count": len(groups), "mechanism_row_count": len(mechanism_rows)},
        "endpoint_dose_response": endpoint,
        "command_m1_dose_response": command,
        "aperture_m2_dose_response": aperture,
        "contact_m3_dose_response": contact,
        "object_m4_dose_response": displacement,
        "paired_control_contrasts": paired,
        "paired_dose_contrasts": paired_doses,
        "monotonicity": monotone,
        "temporal_heterogeneity": heterogeneity["macro"],
        "temporal_heterogeneity_parent_rows": len(heterogeneity["rows"]),
        "outcome_association_descriptive": associations,
        "mechanism_chain": {
            "order": "dose -> M2 -> M3 -> M4 -> V_phys",
            "formal_mediation": False,
            "task_failure_taxonomy": "NOT_AVAILABLE" if not combined["task_failure"]["available"] else "AVAILABLE",
            "risk_monotone": risk_monotone,
            "aperture_monotone": aperture_monotone,
            "m1_available": m1_available,
            "m2_available": m2_available,
            "m3_available": m3_available,
            "m4_available": m4_available,
        },
        "privileged_phase_diagnostic": {"status": "NOT_RUN", "reason": "no outcome-independent phase boundary was required for X0"},
        "physical_intervention": False,
        "new_env_steps": 0,
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    (root / "X0_MECHANISM_ROWS.jsonl").write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in mechanism_rows), encoding="utf-8")
    (root / "STAGE_X_X0_RESULT.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "EXECUTION_MANIFEST.json").write_text(json.dumps({
        "schema": "STAGE_X_X0_RESULT_EXECUTION_MANIFEST_V1",
        "protocol": str(args.protocol),
        "availability_root": str(availability_root),
        "source_commit": summary["source_commit"],
        "source_tree": summary["source_tree"],
        "source_script_sha256": summary["source_script_sha256"],
        "read_only": True,
        "physical_intervention": False,
        "new_env_steps": 0,
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal(root, summary)
    print(json.dumps({"status": status, "mechanism_rows": len(mechanism_rows), "output_root": str(root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
