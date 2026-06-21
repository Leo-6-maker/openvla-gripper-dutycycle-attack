#!/usr/bin/env python3
"""Select provisional two-suite Layer3 smoke parents.

Metadata-only selector. It uses only consumed H2 DEV_CANARY / DIAGNOSTIC_HOLDOUT
rows and the frozen key rule SHA256(review_round_id | canonical_episode_key).
It does not inspect Student emissions, probabilities, attack outcomes, task
success, visual ease, or human judgments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ALLOWED_SUITES = ("libero_spatial", "libero_goal")
ALLOWED_STRATA = ("DEV_CANARY", "DIAGNOSTIC_HOLDOUT")
PROVISIONAL_SENTINEL = "PROVISIONAL_ENGINEERING_ONLY_NOT_FOR_CLAIMS"
SOURCE_FORM = Path("tables/layer1_h2_20260620/h2_diagnostic_review_round_v2_1_form_template.csv")


FORBIDDEN_SELECTION_COLUMNS = {
    "corridor_p",
    "release_p",
    "pred_phase",
    "Student_emit",
    "student_emit",
    "mlp_emit_step",
    "attack_result",
    "VIS_result",
    "RAND_result",
    "SHUFFLED_result",
    "task_success",
    "assistant_review_result",
    "reviewer_id",
    "review_timestamp",
    "object_identity_valid",
    "target_identity_valid",
    "event_exists",
    "close_onset_valid",
    "grasp_established_valid",
    "lift_onset_valid",
    "stable_carry_valid",
    "window_start_valid",
    "anchor_valid",
    "window_end_valid",
    "release_separation_valid",
    "false_positive_carry",
    "abstain_or_fail_closed_correct",
    "reviewer_notes",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def selection_key(row: dict[str, str]) -> str:
    return sha256_text(f"{row.get('review_round_id', '')}|{row.get('episode_key', '')}")


def eligible_candidates(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("review_stratum") not in ALLOWED_STRATA:
            continue
        if row.get("suite") not in ALLOWED_SUITES:
            continue
        if row.get("teacher_status") != "ELIGIBLE_EVENT":
            continue
        out.append(
            {
                "selection_hash": selection_key(row),
                "review_round_id": row.get("review_round_id", ""),
                "review_stratum": row.get("review_stratum", ""),
                "review_id": row.get("review_id", ""),
                "canonical_episode_key": row.get("episode_key", ""),
                "suite": row.get("suite", ""),
                "task_idx": row.get("task_idx", ""),
                "state_id": row.get("state_id", ""),
                "condition": "CLEAN",
                "mechanism_type": row.get("mechanism_type", ""),
                "teacher_status": row.get("teacher_status", ""),
                "event_id": row.get("event_id", ""),
                "teacher_anchor": row.get("proposed_anchor", ""),
                "teacher_window_start": row.get("proposed_window_start", ""),
                "teacher_window_end": row.get("proposed_window_end", ""),
                "raw_video_path": row.get("raw_video_path", ""),
                "teacher_timeline_path": row.get("teacher_only_timeline_path", ""),
                "teacher_overlay_path": row.get("teacher_only_overlay_path", ""),
                "selection_rule": "sha256(review_round_id|canonical_episode_key)_first_two_per_suite",
            }
        )
    return out


def select_parents(rows: list[dict[str, str]], per_suite: int = 2) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = eligible_candidates(rows)
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_suite[row["suite"]].append(row)
    selected: list[dict[str, Any]] = []
    suite_counts: dict[str, Any] = {}
    for suite in ALLOWED_SUITES:
        ordered = sorted(by_suite.get(suite, []), key=lambda r: (r["selection_hash"], r["canonical_episode_key"], r["review_id"]))
        take = ordered[:per_suite]
        for rank, row in enumerate(take):
            row = dict(row)
            row["suite_selection_rank"] = rank
            row["detector_checkpoint_sha256"] = (
                "f0ff9acdc77d1ca000214dae5d2758ba6474d3748248078cf99d1bdc79195da0"
                if suite == "libero_spatial"
                else "d98256ea6c29f5aed4d96b58d0f5a9497358de54de6633aa52fe944828067994"
            )
            row["detector_supervised_source"] = "libero_goal_only" if suite == "libero_spatial" else "libero_spatial_only"
            row["libero10_supervised_rows_used"] = "0"
            selected.append(row)
        suite_counts[suite] = {"candidate_count": len(ordered), "selected_count": len(take)}
    audit = {
        "status": "PASS" if all(suite_counts[s]["selected_count"] == per_suite for s in ALLOWED_SUITES) else "REDUCED_DENOMINATOR",
        "provisional_engineering_only": True,
        "selection_inputs": ["review_round_id", "episode_key", "review_stratum", "suite", "teacher_status"],
        "forbidden_inputs_not_used": sorted(FORBIDDEN_SELECTION_COLUMNS),
        "allowed_suites": list(ALLOWED_SUITES),
        "allowed_strata": list(ALLOWED_STRATA),
        "per_suite_target": per_suite,
        "source_status_counts": Counter(row.get("teacher_status", "") for row in rows),
        "candidate_counts": suite_counts,
        "selected_count": len(selected),
    }
    return selected, audit


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-form", default=str(SOURCE_FORM))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--per-suite", type=int, default=2)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source_form)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / PROVISIONAL_SENTINEL).write_text("Provisional Layer3 two-suite interface smoke. Not final paper evidence.\n", encoding="utf-8")
    rows = read_csv(source)
    selected, audit = select_parents(rows, per_suite=args.per_suite)
    audit["source_form"] = str(source)
    audit["source_form_sha256"] = source_sha256(source)
    write_csv(out / "provisional_layer3_two_suite_parent_manifest.csv", selected)
    write_json(out / "provisional_layer3_two_suite_parent_manifest_audit.json", audit)


if __name__ == "__main__":
    main()

