"""Compare sealed V5-A/V5-B FIT evaluation bundles without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from gripper_attack.b3_training_protocol import seal_directory, verify_sealed_directory


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["canonical_parent_key"])] = row
    return rows


def _scheduler_highest(row: dict[str, Any]) -> bool:
    selected = row.get("selected_window_tier")
    best = row.get("causal_best_teacher_tier")
    return selected is not None and best is not None and int(selected) == int(best)


def review(a_root: Path, b_root: Path, output_root: Path) -> dict[str, Any]:
    a_root = a_root.resolve()
    b_root = b_root.resolve()
    output_root = output_root.resolve()
    verify_sealed_directory(a_root)
    verify_sealed_directory(b_root)
    a_summary = _json(a_root / "evaluation_summary.json")
    b_summary = _json(b_root / "evaluation_summary.json")
    a_rows = _jsonl(a_root / "episode_metrics.jsonl")
    b_rows = _jsonl(b_root / "episode_metrics.jsonl")
    if set(a_rows) != set(b_rows):
        raise ValueError("A/B validation identity sets differ")
    rows: list[dict[str, Any]] = []
    for identity in sorted(a_rows):
        a = a_rows[identity]
        b = b_rows[identity]
        row = {
            "canonical_parent_key": identity,
            "category": a.get("category"),
            "causal_anchor_a": bool(a.get("causal_top1_hit")),
            "causal_anchor_b": bool(b.get("causal_top1_hit")),
            "scheduler_highest_a": _scheduler_highest(a),
            "scheduler_highest_b": _scheduler_highest(b),
            "scheduler_tier_ge2_a": a.get("selected_window_tier") is not None and int(a["selected_window_tier"]) >= 2,
            "scheduler_tier_ge2_b": b.get("selected_window_tier") is not None and int(b["selected_window_tier"]) >= 2,
            "emit_a": int(a.get("emit_count", 0)) > 0,
            "emit_b": int(b.get("emit_count", 0)) > 0,
            "release_a": bool(a.get("release_trigger")),
            "release_b": bool(b.get("release_trigger")),
            "regrasp_a": bool(a.get("regrasp_trigger")),
            "regrasp_b": bool(b.get("regrasp_trigger")),
        }
        row["causal_disagreement"] = row["causal_anchor_a"] != row["causal_anchor_b"]
        row["scheduler_disagreement"] = row["scheduler_highest_a"] != row["scheduler_highest_b"]
        row["emit_disagreement"] = row["emit_a"] != row["emit_b"]
        row["release_disagreement"] = row["release_a"] != row["release_b"]
        row["regrasp_disagreement"] = row["regrasp_a"] != row["regrasp_b"]
        rows.append(row)
    if output_root.exists():
        raise FileExistsError(output_root)
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    summary = {
        "schema": "DETECTOR_V5_AB_WORKING_POINT_REVIEW_V1",
        "a_summary": a_summary,
        "b_summary": b_summary,
        "identity_count": len(rows),
        "disagreement_counts": {
            key: sum(bool(row[key]) for row in rows)
            for key in ("causal_disagreement", "scheduler_disagreement", "emit_disagreement", "release_disagreement", "regrasp_disagreement")
        },
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
        "protected_splits_read": [],
    }
    try:
        staging.mkdir(parents=True)
        with (staging / "ab_disagreement.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["canonical_parent_key"])
            writer.writeheader()
            writer.writerows(rows)
        (staging / "review.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        seal_directory(staging)
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-root", type=Path, required=True)
    parser.add_argument("--b-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.a_root, args.b_root, args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
