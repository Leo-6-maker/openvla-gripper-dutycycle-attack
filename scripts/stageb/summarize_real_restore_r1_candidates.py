#!/usr/bin/env python3
"""Metadata-only summarizer for real LIBERO restore R1 candidate scans.

This tool is intentionally read-only with respect to rollout artifacts: it
only reads small manifests/logs and writes a separate audit directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_candidates(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def recursive_manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows.append(
            {
                "relpath": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def reason_bucket(reason: str) -> str:
    if not reason:
        return "none"
    if "candidate did not produce eligible natural Student emit" in reason:
        return "no_natural_student_emit"
    if "split_mode" in reason:
        return "detector_split_mode_mismatch"
    if "unnorm_key" in reason or "dataset statistics" in reason:
        return "model_unnorm_binding_mismatch"
    if "five remaining steps" in reason:
        return "emit_too_late_for_restore"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output dir exists and is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    candidate_path = run_root / "run" / "candidate_manifest.csv"
    if not candidate_path.is_file():
        raise SystemExit(f"missing candidate manifest: {candidate_path}")
    candidates = read_candidates(candidate_path)
    summary = read_json(run_root / "run" / "single_parent_restore_qualification_summary.json")
    binding = read_json(run_root / "run" / "openvla_model_binding_receipt.json")

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(candidates):
        bucket = reason_bucket(row.get("reason", ""))
        rows.append(
            {
                "candidate_index": idx,
                "protocol_id": row.get("protocol_id", ""),
                "suite": row.get("suite", ""),
                "task_idx": row.get("task_idx", ""),
                "state_id": row.get("state_id", ""),
                "eval_seed": row.get("eval_seed", ""),
                "selection_hash": row.get("selection_hash", ""),
                "status": row.get("status", ""),
                "reason_bucket": bucket,
                "reason": row.get("reason", ""),
            }
        )
    write_csv(
        out / "real_restore_r1_candidate_summary.csv",
        rows,
        [
            "candidate_index",
            "protocol_id",
            "suite",
            "task_idx",
            "state_id",
            "eval_seed",
            "selection_hash",
            "status",
            "reason_bucket",
            "reason",
        ],
    )

    status_counts = Counter(row["status"] for row in rows)
    bucket_counts = Counter(row["reason_bucket"] for row in rows)
    result = {
        "stage": "REAL_LIBERO_RESTORE_R1_CANDIDATE_METADATA_SUMMARY",
        "run_root": str(run_root),
        "candidate_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_bucket_counts": dict(sorted(bucket_counts.items())),
        "selected_count": int(status_counts.get("SELECTED", 0)),
        "ineligible_count": int(status_counts.get("INELIGIBLE", 0)),
        "planned_count": int(status_counts.get("PLANNED", 0)),
        "run_summary": summary,
        "openvla_model_binding": binding,
        "candidate_manifest_sha256": sha256_file(candidate_path),
    }
    write_json(out / "real_restore_r1_candidate_summary.json", result)

    seal_rows = recursive_manifest(run_root)
    write_csv(out / "real_restore_r1_recursive_sha256_manifest.csv", seal_rows, ["relpath", "size_bytes", "sha256"])
    result["recursive_manifest_file_count"] = len(seal_rows)
    result["recursive_manifest_sha256"] = sha256_file(out / "real_restore_r1_recursive_sha256_manifest.csv")
    write_json(out / "real_restore_r1_candidate_summary.json", result)

    report = [
        "# Real Restore R1 Candidate Metadata Summary",
        "",
        f"- run_root: `{run_root}`",
        f"- candidate_count: {len(rows)}",
        f"- status_counts: `{dict(sorted(status_counts.items()))}`",
        f"- reason_bucket_counts: `{dict(sorted(bucket_counts.items()))}`",
        f"- run_result: `{summary.get('result', '')}`",
        "",
        "This is a metadata-only summary. It does not run LIBERO, OpenVLA, VIS, RAND, shuffled, oracle, or attacks.",
    ]
    (out / "REAL_RESTORE_R1_CANDIDATE_SUMMARY.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
