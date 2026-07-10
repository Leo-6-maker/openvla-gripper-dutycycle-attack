#!/usr/bin/env python3
"""CPU-only audit of frozen C2f Teacher-v1 clean step records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Tuple

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
GROUNDED_ROLES = {"primary_attackable", "distractor_or_setup"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(values: List[int], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return float(values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (pos - lo))


def truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def missed_stable_carry_reason(row: Dict[str, Any]) -> str:
    role = str(row.get("teacher_event_role", ""))
    if role == "distractor_or_setup":
        return "V1_TARGET_MATCH_FAILED_OR_DISTRACTOR"
    if role == "unsupported_or_abstain":
        return "V1_NO_GROUNDED_OBJECT"
    if role == "auxiliary_manipulation":
        return "V1_STABLE_CARRY_AUXILIARY_INCONSISTENT"
    return "V1_STABLE_CARRY_ROLE_UNKNOWN"


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_teacher_v1(
    input_root: Path,
    teacher_source: Path | None = None,
    suite_overrides: Mapping[str, Path] | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    aggregates: Dict[Tuple[str, int], Dict[str, Any]] = defaultdict(lambda: {
        "episodes": 0,
        "windows": 0,
        "phases": Counter(),
        "roles": Counter(),
        "first_primary": [],
        "primary_durations": [],
        "stable_carry": 0,
        "grounded_stable_carry": 0,
        "target_match": 0,
        "spatial_absolute_z_fallback_candidates": 0,
        "explicit_grounding_rows": 0,
        "explicit_target_match_rows": 0,
        "explicit_reason_rows": 0,
        "explicit_fallback_rows": 0,
        "clean_success": Counter(),
        "miss_reasons": Counter(),
    })
    read_errors: List[Dict[str, str]] = []
    suite_overrides = dict(suite_overrides or {})
    paths = []
    for path in sorted(input_root.rglob("step_records.jsonl")):
        suite = next((part for part in path.parts if part in SUITES), "")
        if suite not in suite_overrides:
            paths.append(path)
    for suite, root in sorted(suite_overrides.items()):
        for path in sorted(root.rglob("step_records.jsonl")):
            actual = next((part for part in path.parts if part in SUITES), "")
            if actual == suite:
                paths.append(path)
    for path in paths:
        try:
            meta = json.loads(path.with_name("episode_metadata.json").read_text(encoding="utf-8"))
            suite = str(meta.get("suite", ""))
            task = int(meta.get("task_index", meta.get("task_id", -1)))
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception as exc:
            read_errors.append({"path": str(path), "error": str(exc)})
            continue
        agg = aggregates[(suite, task)]
        agg["episodes"] += 1
        agg["windows"] += len(rows)
        clean_success = meta.get("clean_success_observed", meta.get("clean_success"))
        agg["clean_success"]["success" if truthy(clean_success) else "not_success"] += 1
        primary_steps: List[int] = []
        for row in rows:
            phase = str(row.get("teacher_phase", "unknown") or "unknown")
            role = str(row.get("teacher_event_role", "unknown") or "unknown")
            agg["phases"][phase] += 1
            agg["roles"][role] += 1
            primary = truthy(row.get("teacher_primary_attackable")) or role == "primary_attackable"
            if primary:
                primary_steps.append(int(row.get("step", len(primary_steps))))
            if phase == "stable_carry":
                agg["stable_carry"] += 1
                if role in GROUNDED_ROLES:
                    agg["grounded_stable_carry"] += 1
                if primary:
                    agg["target_match"] += 1
                else:
                    agg["miss_reasons"][missed_stable_carry_reason(row)] += 1
                fv = row.get("features_25d") or []
                if suite == "libero_spatial" and len(fv) >= 20:
                    eef_z, relative_z = float(fv[5]), float(fv[19])
                    if eef_z > 0.85 and relative_z < 0.03:
                        agg["spatial_absolute_z_fallback_candidates"] += 1
            agg["explicit_grounding_rows"] += int("teacher_grounded_object" in row)
            agg["explicit_target_match_rows"] += int("teacher_target_match" in row)
            agg["explicit_reason_rows"] += int("teacher_reason_code" in row)
            agg["explicit_fallback_rows"] += int("teacher_used_absolute_z_fallback" in row)
        if primary_steps:
            ordered = sorted(set(primary_steps))
            agg["first_primary"].append(ordered[0])
            run = 1
            for previous, current in zip(ordered, ordered[1:]):
                if current == previous + 1:
                    run += 1
                else:
                    agg["primary_durations"].append(run)
                    run = 1
            agg["primary_durations"].append(run)

    by_task: List[Dict[str, Any]] = []
    reason_rows: List[Dict[str, Any]] = []
    for (suite, task), agg in sorted(aggregates.items()):
        windows = agg["windows"]
        stable = agg["stable_carry"]
        grounded = agg["grounded_stable_carry"]
        row = {
            "suite": suite,
            "task_index": task,
            "episode_count": agg["episodes"],
            "window_count": windows,
            "stable_grasp_count": agg["phases"]["stable_grasp"],
            "stable_carry_count": stable,
            "release_safe_count": agg["phases"]["release_safe"],
            "primary_attackable_count": agg["roles"]["primary_attackable"],
            "unsupported_count": agg["roles"]["unsupported_or_abstain"],
            "stable_carry_rate": stable / windows if windows else 0.0,
            "primary_density": agg["roles"]["primary_attackable"] / windows if windows else 0.0,
            "object_grounding_coverage": grounded / stable if stable else None,
            "target_match_coverage": agg["target_match"] / grounded if grounded else None,
            "episodes_with_primary": len(agg["first_primary"]),
            "first_primary_min": min(agg["first_primary"]) if agg["first_primary"] else None,
            "first_primary_p50": percentile(agg["first_primary"], 0.5),
            "first_primary_p90": percentile(agg["first_primary"], 0.9),
            "first_primary_max": max(agg["first_primary"]) if agg["first_primary"] else None,
            "primary_duration_mean": mean(agg["primary_durations"]) if agg["primary_durations"] else None,
            "spatial_absolute_z_fallback_candidate_count": agg["spatial_absolute_z_fallback_candidates"],
            "explicit_grounding_coverage": agg["explicit_grounding_rows"] / windows if windows else 0.0,
            "explicit_target_match_coverage": agg["explicit_target_match_rows"] / windows if windows else 0.0,
            "explicit_reason_code_coverage": agg["explicit_reason_rows"] / windows if windows else 0.0,
            "explicit_fallback_provenance_coverage": agg["explicit_fallback_rows"] / windows if windows else 0.0,
            "clean_success_episode_count": agg["clean_success"]["success"],
            "clean_not_success_episode_count": agg["clean_success"]["not_success"],
        }
        by_task.append(row)
        for reason, count in sorted(agg["miss_reasons"].items()):
            reason_rows.append({"suite": suite, "task_index": task, "reason_code": reason, "window_count": count})

    suite_counts = Counter()
    for row in by_task:
        suite_counts[row["suite"]] += row["episode_count"]
    source_findings: Dict[str, Any] = {}
    if teacher_source and teacher_source.is_file():
        text = teacher_source.read_text(encoding="utf-8")
        source_findings = {
            "path": str(teacher_source),
            "sha256": sha256_file(teacher_source),
            "absolute_z_fallback_present": "ABSOLUTE_FALLBACK_Z" in text,
            "nearest_body_grounding_present": "_identify_grasped_object" in text and "body_xpos" in text,
            "language_target_match_present": "_object_matches_task_target" in text,
            "release_transition_only_present": 'self._phase = "release_safe"' in text,
        }
    report = {
        "gate": "C2F_TEACHER_V1_LABEL_AUDIT",
        "status": "PASS_C2F_TEACHER_V1_LABEL_AUDIT_WITH_PROVENANCE_LIMITS" if paths and not read_errors and all(suite_counts[s] > 0 for s in SUITES) else "HOLD_C2F_TEACHER_V1_LABEL_AUDIT_INCOMPLETE",
        "input_root": str(input_root),
        "suite_overrides": {suite: str(path) for suite, path in suite_overrides.items()},
        "step_record_file_count": len(paths),
        "read_error_count": len(read_errors),
        "read_errors": read_errors[:100],
        "suite_episode_counts": dict(suite_counts),
        "all_required_suites_present": all(suite_counts[s] > 0 for s in SUITES),
        "source_findings": source_findings,
        "provenance_limits": [
            "Teacher-v1 rows do not record contacted object identity.",
            "Teacher-v1 rows do not record target-match decision provenance or reason code.",
            "Absolute-z fallback usage is a 25D-derived candidate count, not exact branch provenance.",
            "Release-safe is generated by a gripper closed-to-open transition, not target-relative placement evidence.",
        ],
        "boundaries": {
            "CPU_only": True,
            "OpenVLA_loaded": False,
            "LIBERO_runtime": False,
            "counterfactual_replay": False,
            "detector_training": False,
        },
    }
    return report, by_task, reason_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--teacher-source", default="")
    ap.add_argument("--suite-override", action="append", default=[], metavar="SUITE=PATH")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    overrides = {}
    for item in args.suite_override:
        suite, sep, path = item.partition("=")
        if not sep or suite not in SUITES:
            raise SystemExit(f"invalid --suite-override {item!r}; expected SUITE=PATH")
        overrides[suite] = Path(path)
    report, by_task, reasons = audit_teacher_v1(
        Path(args.input_root), Path(args.teacher_source) if args.teacher_source else None, overrides
    )
    (out / "teacher_v1_audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    task_fields = list(by_task[0]) if by_task else ["suite", "task_index"]
    write_csv(out / "teacher_v1_by_suite_task.csv", by_task, task_fields)
    write_csv(out / "teacher_v1_reason_codes.csv", reasons, ["suite", "task_index", "reason_code", "window_count"])
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"].startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
