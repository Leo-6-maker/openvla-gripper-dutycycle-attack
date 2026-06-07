#!/usr/bin/env python3
"""Build unified phase feature table for detector rows."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter


KEY_FIELDS = ["task_key", "state_id", "window_start", "window_end"]
OUT_FIELDS = KEY_FIELDS + [
    "phase_bin_proxy",
    "qpos_phase_class",
    "control_type",
    "phase_is_critical",
    "predicted_phase",
    "phase_confidence",
    "phase_source",
    "phase_missing_reason",
]

CRITICAL = (
    "true_closed",
    "transitional_pre_open",
    "transitional-pre-open",
    "near_closed",
    "pre_lock",
    "pre-lock",
    "contact",
    "grasp",
    "lift",
    "closed_proxy",
    "approach_near_closed_proxy",
    "pre_lock_closed_proxy",
)
NONCRITICAL = (
    "natural_open",
    "stable_post_lock",
    "post_lock",
    "far_too_early",
    "no_contact",
    "after_done",
    "terminal",
    "release",
    "natural_open_or_release_proxy",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector-dataset", default="tables/detector_v26_twostage_dataset.csv")
    ap.add_argument("--labels-v2", default="tables/object_phase_response_labels_v2.csv")
    ap.add_argument("--jobs-state", default="")
    ap.add_argument("--adaptive-candidates", default="tables/object_phase_response_adaptive_candidates.csv")
    ap.add_argument("--clean-control-bank", default="tables/clean_control_negative_bank.csv")
    ap.add_argument("--extra-phase-csv", action="append", default=[])
    ap.add_argument("--output-csv", default="tables/detector_phase_features_v1.csv")
    ap.add_argument("--output-report", default="reports/DETECTOR_PHASE_FEATURE_TABLE_V1.md")
    return ap.parse_args()


def norm(value) -> str:
    return str(value if value is not None else "").strip()


def lower(value) -> str:
    return norm(value).lower()


def read_csv(path: str) -> list[dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [{norm(k).lstrip("\ufeff"): norm(v) for k, v in row.items()} for row in csv.DictReader(f)]


def write_csv(path: str, rows: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return tuple(norm(row.get(f)) for f in KEY_FIELDS)


def row_identity_source(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(args.detector_dataset)
    if rows:
        return rows
    rows = read_csv(args.labels_v2)
    rows.extend(read_csv(args.clean_control_bank))
    return rows


def phase_class(row: dict[str, str]) -> tuple[str, str]:
    explicit = lower(row.get("phase_is_critical"))
    if explicit in {"1", "true", "yes", "critical"}:
        return "true", "explicit_phase_is_critical"
    if explicit in {"0", "false", "no", "noncritical", "non-critical"}:
        return "false", "explicit_phase_is_critical"
    text = " ".join(
        lower(row.get(f))
        for f in ["phase_bin_proxy", "qpos_phase_class", "control_type", "candidate_role", "expected_role", "predicted_phase"]
    )
    if any(tok in text for tok in NONCRITICAL):
        return "false", "rule_noncritical_proxy"
    if any(tok in text for tok in CRITICAL):
        return "true", "rule_critical_proxy"
    return "missing", "no_phase_signal"


def merge_phase(base: dict[str, str], sources: list[tuple[str, dict[str, str]]]) -> dict[str, str]:
    merged = dict(base)
    phase_source_parts = []
    for source_name, row in sources:
        if not row:
            continue
        used = False
        for field in ["phase_bin_proxy", "qpos_phase_class", "control_type", "predicted_phase", "phase_confidence"]:
            if not norm(merged.get(field)) and norm(row.get(field)):
                merged[field] = norm(row.get(field))
                used = True
        if used:
            phase_source_parts.append(source_name)
    phase, reason = phase_class(merged)
    merged["phase_is_critical"] = phase
    merged["phase_source"] = "+".join(phase_source_parts) if phase_source_parts else "row_fields_only"
    merged["phase_missing_reason"] = "" if phase != "missing" else reason
    return {field: norm(merged.get(field)) for field in OUT_FIELDS}


def main() -> int:
    args = parse_args()
    identity_rows = row_identity_source(args)
    candidate_map = {key(r): r for r in read_csv(args.adaptive_candidates)}
    clean_map = {key(r): r for r in read_csv(args.clean_control_bank)}
    labels_map = {key(r): r for r in read_csv(args.labels_v2)}
    jobs_map = {key(r): r for r in read_csv(args.jobs_state)}
    extra_maps = [(os.path.basename(path), {key(r): r for r in read_csv(path)}) for path in args.extra_phase_csv]

    out = []
    seen = set()
    for row in identity_rows:
        k = key(row)
        if not all(k) or k in seen:
            continue
        seen.add(k)
        sources = [
            ("labels_v2", labels_map.get(k, {})),
            ("adaptive_candidates", candidate_map.get(k, {})),
            ("clean_control_bank", clean_map.get(k, {})),
            ("jobs_state", jobs_map.get(k, {})),
        ]
        sources.extend((name, mp.get(k, {})) for name, mp in extra_maps)
        out.append(merge_phase(row, sources))

    write_csv(args.output_csv, out)
    counts = Counter(r["phase_is_critical"] for r in out)
    source_counts = Counter(r["phase_source"] for r in out)
    lines = [
        "# Detector Phase Feature Table V1",
        "",
        f"**Rows**: {len(out)}",
        f"**phase_is_critical=true**: {counts.get('true', 0)}",
        f"**phase_is_critical=false**: {counts.get('false', 0)}",
        f"**phase_is_critical=missing**: {counts.get('missing', 0)}",
        "",
        "Missing phase is kept as `missing`; it is not silently filled as non-critical.",
        "",
        "## Phase Sources",
        "",
    ]
    for source, count in source_counts.most_common():
        lines.append(f"- `{source}`: {count}")
    lines.extend(["", "## Verdict", ""])
    coverage = 0.0 if not out else (len(out) - counts.get("missing", 0)) / float(len(out))
    lines.append(f"- Phase coverage: {coverage:.3f}")
    if coverage >= 0.95:
        lines.append("- Coverage is sufficient for full two-stage diagnostic coverage.")
    else:
        lines.append("- Coverage is below 95%; full two-stage diagnostic is blocked, complete-case can be reported separately.")
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"phase_rows={len(out)} missing={counts.get('missing', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
