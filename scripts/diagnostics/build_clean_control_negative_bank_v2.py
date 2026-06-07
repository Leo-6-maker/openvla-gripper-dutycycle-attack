#!/usr/bin/env python3
"""Build v2 clean-control negative bank with source provenance."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter


KEY_FIELDS = ["task_key", "state_id", "window_start", "window_end"]
OUT_FIELDS = KEY_FIELDS + [
    "label_status",
    "label_vulnerability_ready",
    "label_source",
    "control_type",
    "control_reason",
    "source_type",
    "sample_weight",
    "phase_is_critical",
    "phase_bin_proxy",
    "qpos_phase_class",
    "candidate_id",
]

CONTROL_TOKENS = {
    "natural_open": "natural_open",
    "natural_open_or_release_proxy": "natural_open",
    "stable_post_lock": "stable_post_lock",
    "stable_post_lock_control": "stable_post_lock",
    "post_lock": "post_lock",
    "far_too_early": "far_too_early",
    "no_contact": "no_contact",
    "after_done": "after_done",
    "terminal": "terminal",
}
BAD_TOKENS = ("infra", "manual", "polluted", "ambiguous", "xid", "oom", "localization_fail")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-v2", default="tables/object_phase_response_labels_v2.csv")
    ap.add_argument("--adaptive-candidates", default="tables/object_phase_response_adaptive_candidates.csv")
    ap.add_argument("--phase-features", default="tables/detector_phase_features_v1.csv")
    ap.add_argument("--clean-rollout-controls", action="append", default=[])
    ap.add_argument("--output-csv", default="tables/clean_control_negative_bank_v2.csv")
    ap.add_argument("--output-report", default="reports/CLEAN_CONTROL_NEGATIVE_BANK_V2.md")
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


def bad(row: dict[str, str]) -> bool:
    text = " ".join(lower(v) for v in row.values())
    return any(tok in text for tok in BAD_TOKENS)


def control_type(row: dict[str, str]) -> tuple[str, str]:
    text = " ".join(
        lower(row.get(f))
        for f in ["control_type", "candidate_role", "expected_role", "phase_bin_proxy", "qpos_phase_class", "reason_selected", "source_reason", "control_reason"]
    )
    for token, ctype in CONTROL_TOKENS.items():
        if token in text:
            return ctype, token
    return "", ""


def row_from_source(src: dict[str, str], phase: dict[str, str], source_type: str) -> dict[str, str] | None:
    if bad(src):
        return None
    ctype, reason_token = control_type({**src, **phase})
    if not ctype:
        return None
    phase_value = norm(phase.get("phase_is_critical"))
    if phase_value == "missing":
        return None
    reason = norm(src.get("control_reason") or src.get("reason_selected") or src.get("source_reason") or reason_token)
    if not reason:
        return None
    return {
        "task_key": norm(src.get("task_key")),
        "state_id": norm(src.get("state_id")),
        "window_start": norm(src.get("window_start")),
        "window_end": norm(src.get("window_end")),
        "label_status": "negative",
        "label_vulnerability_ready": "0",
        "label_source": "clean_control_negative",
        "control_type": ctype,
        "control_reason": reason,
        "source_type": source_type,
        "sample_weight": "0.5",
        "phase_is_critical": "false" if phase_value in {"", "false", "0"} else phase_value,
        "phase_bin_proxy": norm(src.get("phase_bin_proxy") or phase.get("phase_bin_proxy")),
        "qpos_phase_class": norm(src.get("qpos_phase_class") or phase.get("qpos_phase_class")),
        "candidate_id": norm(src.get("candidate_id")),
    }


def main() -> int:
    args = parse_args()
    labels = read_csv(args.labels_v2)
    gold_pos = {key(r) for r in labels if lower(r.get("label_status")) == "positive" or norm(r.get("label_vulnerability_ready")) == "1"}
    phase_map = {key(r): r for r in read_csv(args.phase_features)}

    source_rows: list[tuple[dict[str, str], str]] = []
    clean_rows = []
    for path in args.clean_rollout_controls:
        clean_rows.extend(read_csv(path))
    if clean_rows:
        source_rows.extend((r, "clean_rollout_derived") for r in clean_rows)
    else:
        source_rows.extend((r, "candidate_derived") for r in read_csv(args.adaptive_candidates))

    rows = []
    issues = []
    seen = set()
    for src, source_type in source_rows:
        k = key(src)
        if not all(k) or k in seen:
            continue
        row = row_from_source(src, phase_map.get(k, {}), source_type)
        if row is None:
            continue
        if k in gold_pos:
            issues.append("overlap_with_gold_positive:" + "/".join(k))
            continue
        rows.append(row)
        seen.add(k)

    write_csv(args.output_csv, rows)
    source_counts = Counter(r["source_type"] for r in rows)
    control_counts = Counter(r["control_type"] for r in rows)
    status = "HARD_FAIL" if issues else ("OK" if rows else "BLOCKED_NO_CLEAN_CONTROLS")
    lines = [
        "# Clean-Control Negative Bank V2",
        "",
        f"**Status**: {status}",
        f"**Rows**: {len(rows)}",
        f"**Gold-positive overlap issues**: {len(issues)}",
        "",
        "Rows are diagnostic negatives for ablation, not new gold labels.",
        "",
        "## Source Types",
        "",
    ]
    for source, count in source_counts.most_common():
        lines.append(f"- `{source}`: {count}")
    lines.extend(["", "## Control Types", ""])
    for ctype, count in control_counts.most_common():
        lines.append(f"- `{ctype}`: {count}")
    lines.extend(["", "## Issues", ""])
    lines.extend(f"- {i}" for i in issues) if issues else lines.append("- None.")
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"clean_control_v2_rows={len(rows)} issues={len(issues)}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
