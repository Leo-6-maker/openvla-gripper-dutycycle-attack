#!/usr/bin/env python3
"""Audit phase-detector/proxy artifacts for detector two-stage evaluation."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


PHASE_TERMS = [
    "phase_bin_proxy",
    "qpos_phase_class",
    "phase_is_critical",
    "predicted_phase",
    "phase_confidence",
    "ProprioNoStep",
    "proprionostep",
    "TCN",
    "natural_open",
    "stable_post_lock",
    "post_lock",
    "far_too_early",
    "no_contact",
    "after_done",
]

OUT_FIELDS = [
    "path",
    "location",
    "exists",
    "rows",
    "phase_columns",
    "phase_value_hits",
    "phase_coverage_rows",
    "missing_phase_rows",
    "full_two_stage_possible",
    "notes",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-root", default="/data/liuyu/outputs/shared_detector_v25_inputs_20260606")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--output-csv", default="tables/phase_detector_artifact_availability.csv")
    ap.add_argument("--output-report", default="reports/PHASE_DETECTOR_ARTIFACT_AVAILABILITY.md")
    return ap.parse_args()


def norm(value) -> str:
    return str(value if value is not None else "").strip()


def read_rows(path: Path, limit: int | None = None) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists() or path.suffix.lower() != ".csv":
        return [], []
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = [norm(x).lstrip("\ufeff") for x in (reader.fieldnames or [])]
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            rows.append({norm(k).lstrip("\ufeff"): norm(v) for k, v in row.items()})
    return fields, rows


def row_count(path: Path) -> str:
    if not path.exists() or path.suffix.lower() != ".csv":
        return ""
    with path.open(newline="", encoding="utf-8") as f:
        return str(max(0, sum(1 for _ in f) - 1))


def write_csv(path: str, rows: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def candidate_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            low = name.lower()
            if low.endswith(".csv") or low.endswith(".md") or low.endswith(".json"):
                if any(term.lower() in low for term in PHASE_TERMS) or low in {
                    "object_phase_response_labels_v2.csv",
                    "object_phase_response_adaptive_candidates.csv",
                    "jobs_state.csv",
                    "detector_v26_twostage_dataset.csv",
                    "clean_control_negative_bank.csv",
                }:
                    out.append(Path(dirpath) / name)
    return sorted(out)


def audit_file(path: Path, location: str) -> dict[str, str]:
    fields, rows = read_rows(path, limit=None)
    phase_cols = [c for c in fields if any(term.lower() in c.lower() for term in PHASE_TERMS)]
    text_rows = rows[:500]
    text = " ".join(" ".join(norm(v).lower() for v in row.values()) for row in text_rows)
    value_hits = sorted({term for term in PHASE_TERMS if term.lower() in text})
    coverage = ""
    missing = ""
    possible = "false"
    if rows:
        usable_cols = [c for c in ["phase_is_critical", "predicted_phase", "phase_confidence", "phase_bin_proxy", "qpos_phase_class", "control_type"] if c in fields]
        if usable_cols:
            cov = 0
            for row in rows:
                if any(norm(row.get(c)) not in {"", "missing", "unknown"} for c in usable_cols):
                    cov += 1
            coverage = str(cov)
            missing = str(len(rows) - cov)
            if cov == len(rows) and ("phase_is_critical" in fields or "predicted_phase" in fields):
                possible = "true"
    return {
        "path": str(path),
        "location": location,
        "exists": str(path.exists()).lower(),
        "rows": row_count(path),
        "phase_columns": ",".join(phase_cols),
        "phase_value_hits": ",".join(value_hits),
        "phase_coverage_rows": coverage,
        "missing_phase_rows": missing,
        "full_two_stage_possible": possible,
        "notes": "csv_audit" if path.suffix.lower() == ".csv" else "name_or_text_hint",
    }


def main() -> int:
    args = parse_args()
    snapshot = Path(args.snapshot_root)
    repo = Path(args.repo_root)
    paths: list[tuple[Path, str]] = []
    for path in candidate_files(snapshot):
        paths.append((path, "snapshot"))
    for sub in ["tables", "reports", "scripts", "configs"]:
        for path in candidate_files(repo / sub):
            paths.append((path, "repo"))
    seen = set()
    rows = []
    for path, loc in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        rows.append(audit_file(path, loc))
    write_csv(args.output_csv, rows)

    present = [r for r in rows if r["exists"] == "true"]
    full = [r for r in rows if r["full_two_stage_possible"] == "true"]
    phase_files = [r for r in present if r["phase_columns"] or r["phase_value_hits"]]
    lines = [
        "# Phase Detector Artifact Availability",
        "",
        f"**Snapshot root**: `{snapshot}`",
        f"**Repo root**: `{repo}`",
        f"**Files audited**: {len(rows)}",
        f"**Present files**: {len(present)}",
        f"**Files with phase hints**: {len(phase_files)}",
        f"**Full two-stage-ready files**: {len(full)}",
        "",
        "CPU-only discovery. No GPU, VIS, rollout, watcher, or live output was touched.",
        "",
        "## Verdict",
        "",
    ]
    if full:
        lines.append("- At least one file appears to have complete phase fields for direct two-stage evaluation.")
    else:
        lines.append("- Full two-stage evaluation is not established by artifact discovery alone; use the phase feature table coverage audit.")
    lines.extend(["", "## Phase-Hint Files", ""])
    if not phase_files:
        lines.append("- None found.")
    else:
        for row in phase_files:
            lines.append(
                f"- `{row['path']}`: rows={row['rows']}, cols={row['phase_columns'] or '-'}, "
                f"hits={row['phase_value_hits'] or '-'}, coverage={row['phase_coverage_rows'] or '?'}"
            )
    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"audited={len(rows)} phase_files={len(phase_files)} full_ready_files={len(full)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
