#!/usr/bin/env python3
"""Build provisional Layer2 frame dataset from provisional Layer1 labels.

CPU-only. Reads clean step telemetry and provisional Teacher labels. Detector
probability/state fields are never used as labels. Ignored resolver statuses are
retained in the funnel and masked out of supervised loss.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gripper_attack.sc5_detector_runtime import SC5_FEATURES, SC5_PHASES  # noqa: E402

PROVISIONAL_SENTINEL = "PROVISIONAL_ENGINEERING_ONLY_NOT_FOR_CLAIMS"
IGNORE_STATUSES = {
    "TARGET_BINDING_AMBIGUOUS",
    "TARGET_BINDING_FAILED",
    "OBJECT_BINDING_AMBIGUOUS",
    "RESOLVER_NOT_IMPLEMENTED_FOR_MECHANISM",
    "MULTI_EVENT_AUDIT_ONLY",
    "SCHEMA_INVALID",
}
NEGATIVE_STATUSES = {"CORRECT_SEMANTIC_ABSTAIN", "NO_RELEVANT_GRASP_EVENT"}
PRIMARY_POSITIVE_STATUS = "ELIGIBLE_EVENT"
SUPPLEMENTARY_POSITIVE_STATUS = "SUPPLEMENTARY_EVENT_ELIGIBLE"
POSITIVE_STATUSES = {PRIMARY_POSITIVE_STATUS, SUPPLEMENTARY_POSITIVE_STATUS}
LABEL_SOURCE_VERSION = "cross_suite_teacher_owner_ai_repair_v2_plus_libero10_supplementary_bridge_v2"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def git_dirty_status() -> str:
    return subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True)


def to_int(value: Any, default: int = -1) -> int:
    if value in ("", None):
        return default
    return int(float(value))


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_split(layer1_root: Path, split: str) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, list[dict[str, str]]]]:
    root = layer1_root / f"{split}_resolved"
    if not root.exists():
        root = layer1_root / "resolver_outputs" / split
    episodes = read_csv(root / "teacher_episode_labels_v1.csv")
    events = read_csv(root / "teacher_event_labels_v1.csv")
    manifest = read_csv(layer1_root / "manifests" / f"{split}_manifest.csv")
    labels_by_key = {row["episode_key"]: row for row in episodes}
    events_by_key: dict[str, list[dict[str, str]]] = defaultdict(list)
    for event in events:
        events_by_key[event["episode_key"]].append(event)
    return manifest, labels_by_key, events_by_key


def label_role_for_status(status: str) -> tuple[str, str]:
    if status == PRIMARY_POSITIVE_STATUS:
        return "primary_single_object_pick_place", "primary"
    if status == SUPPLEMENTARY_POSITIVE_STATUS:
        return "supplementary_multievent_grasp_carry_bridge", "supplementary"
    if status in NEGATIVE_STATUSES:
        return "negative_only", "negative"
    return "ignore", "ignore"


def _nonempty(value: Any) -> bool:
    return str(value or "").strip() != ""


def sha256_optional(path: Path) -> str:
    return sha256_file(path) if path.exists() else ""


def validate_resolver_role_fields(label: dict[str, str], problems: list[str]) -> tuple[str, str]:
    status = label.get("teacher_status", "")
    expected_label_role, expected_primary = label_role_for_status(status)
    resolver_label_role = str(label.get("label_role", "") or "")
    resolver_primary = str(label.get("primary_or_supplementary", "") or "")
    if resolver_label_role and resolver_label_role != expected_label_role:
        problems.append(f"label_role_mismatch:{resolver_label_role}!={expected_label_role}")
    if resolver_primary and resolver_primary != expected_primary:
        problems.append(f"primary_or_supplementary_mismatch:{resolver_primary}!={expected_primary}")
    return expected_label_role, expected_primary


def select_event_for_label(label: dict[str, str], events: list[dict[str, str]], problems: list[str]) -> dict[str, str] | None:
    status = label.get("teacher_status", "")
    if status == SUPPLEMENTARY_POSITIVE_STATUS:
        primary_event_id = str(label.get("primary_supplementary_event_id", "") or "").strip()
        if not primary_event_id:
            problems.append("supplementary_positive_missing_primary_event_id")
            return None
        matches = [row for row in events if row.get("event_id") == primary_event_id]
        if not matches:
            problems.append(f"supplementary_primary_event_id_not_found:{primary_event_id}")
            return None
        if len(matches) != 1:
            problems.append(f"supplementary_primary_event_id_not_unique:{primary_event_id}:{len(matches)}")
            return None
        return matches[0]
    if status == PRIMARY_POSITIVE_STATUS:
        if len(events) != 1:
            problems.append(f"primary_positive_event_count_not_one:{len(events)}")
            return None
        return events[0]
    return None


def phase_for_step(step: int, label: dict[str, str], event: dict[str, str] | None) -> tuple[str, int, int]:
    status = label.get("teacher_status", "")
    if status not in POSITIVE_STATUSES or event is None:
        return "abstain_unsupported", 0, 0
    close = to_int(event.get("close_onset_step"))
    grasp = to_int(event.get("grasp_established_step"))
    lift = to_int(event.get("lift_onset_step"))
    carry = to_int(event.get("stable_carry_start"))
    start = to_int(event.get("teacher_window_start"))
    end = to_int(event.get("teacher_window_end"))
    release = to_int(event.get("release_onset_step"))
    corridor = int(start >= 0 and end >= 0 and start <= step <= end)
    release_label = int(release >= 0 and step >= release and (end < 0 or step <= end))
    if release_label:
        return "release_safe", corridor, release_label
    if close >= 0 and step >= close and (grasp < 0 or step < grasp):
        return "grasp_close", corridor, release_label
    if grasp >= 0 and step >= grasp and (lift < 0 or step < lift):
        return "stable_grasp", corridor, release_label
    if lift >= 0 and step >= lift and (carry < 0 or step < carry):
        return "first_lift", corridor, release_label
    if carry >= 0 and step >= carry and (end < 0 or step <= end):
        return "stable_carry", corridor, release_label
    if end >= 0 and step > end:
        return "recovery_or_regrasp", corridor, release_label
    return "approach", corridor, release_label


def feature_value(step_row: dict[str, str], feature: str) -> float:
    for key in (f"f_{feature}", feature):
        if key in step_row:
            return to_float(step_row[key])
    return float("nan")


def build_rows_for_episode(
    *,
    manifest_row: dict[str, str],
    label: dict[str, str],
    events: list[dict[str, str]],
    split: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    episode_path = Path(manifest_row["episode_path"])
    step_path = episode_path / "step_telemetry.csv"
    problems: list[str] = []
    if not step_path.exists():
        return [], [f"missing_step_telemetry:{episode_path}"]
    step_rows = read_csv(step_path)
    status = label.get("teacher_status", "")
    role_problem_count_before = len(problems)
    label_role, primary_or_supplementary = validate_resolver_role_fields(label, problems)
    event = select_event_for_label(label, events, problems)
    ignore = status in IGNORE_STATUSES or len(problems) > role_problem_count_before
    if status not in IGNORE_STATUSES | NEGATIVE_STATUSES | POSITIVE_STATUSES:
        problems.append(f"unknown_teacher_status:{status}")
        ignore = True
    if status in POSITIVE_STATUSES and event is None:
        problems.append("positive_without_event")
        ignore = True

    out: list[dict[str, Any]] = []
    for raw in step_rows:
        step = to_int(raw.get("step"))
        phase, corridor, release = phase_for_step(step, label, event)
        row = {
            "episode_key": manifest_row["canonical_key"],
            "suite": manifest_row["suite"],
            "task_idx": manifest_row["task_idx"],
            "state_id": manifest_row["state_id"],
            "eval_seed": manifest_row.get("eval_seed", "0"),
            "condition": "CLEAN",
            "dataset_split": split,
            "step": step,
            "teacher_status": status,
            "mechanism_type": label.get("mechanism_type", manifest_row.get("mechanism_type", "")),
            "label_role": label_role,
            "label_source_version": LABEL_SOURCE_VERSION,
            "primary_or_supplementary": primary_or_supplementary,
            "ignore_for_loss": int(ignore),
            "teacher_phase": phase if not ignore else "abstain_unsupported",
            "teacher_corridor_active": corridor if not ignore else 0,
            "teacher_release_active": release if not ignore else 0,
            "teacher_window_start": event.get("teacher_window_start", "") if event else "",
            "teacher_window_end": event.get("teacher_window_end", "") if event else "",
            "teacher_anchor_step": event.get("teacher_anchor_step", "") if event else "",
            "close_onset_step": event.get("close_onset_step", "") if event else "",
            "grasp_established_step": event.get("grasp_established_step", "") if event else "",
            "lift_onset_step": event.get("lift_onset_step", "") if event else "",
            "stable_carry_start": event.get("stable_carry_start", "") if event else "",
            "event_id": event.get("event_id", "") if event else "",
        }
        finite = True
        for feature in SC5_FEATURES:
            value = feature_value(raw, feature)
            if not math.isfinite(value):
                finite = False
            row[feature] = value
        row["features_finite"] = int(finite)
        out.append(row)
    return out, problems


def leakage_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, set[str]] = defaultdict(set)
    state_by_split: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    duplicate_frames = 0
    seen_frames: set[tuple[str, int]] = set()
    nonfinite = 0
    for row in rows:
        split = str(row["dataset_split"])
        key = str(row["episode_key"])
        by_split[split].add(key)
        state_by_split[split].add((str(row["suite"]), int(row["task_idx"]), int(row["state_id"])))
        frame_key = (key, int(row["step"]))
        if frame_key in seen_frames:
            duplicate_frames += 1
        seen_frames.add(frame_key)
        if int(row.get("features_finite", 0)) != 1:
            nonfinite += 1
    overlap = {}
    splits = sorted(by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            keys = by_split[a] & by_split[b]
            states = state_by_split[a] & state_by_split[b]
            overlap[f"{a}__{b}"] = {"canonical_key_overlap": len(keys), "state_overlap": len(states)}
    failures = []
    for pair, vals in overlap.items():
        if vals["canonical_key_overlap"] or vals["state_overlap"]:
            failures.append(f"split_overlap:{pair}:{vals}")
    if duplicate_frames:
        failures.append(f"duplicate_frames:{duplicate_frames}")
    if nonfinite:
        failures.append(f"nonfinite_feature_rows:{nonfinite}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "split_episode_counts": {k: len(v) for k, v in by_split.items()},
        "split_state_counts": {k: len(v) for k, v in state_by_split.items()},
        "split_overlap": overlap,
        "duplicate_frame_count": duplicate_frames,
        "nonfinite_feature_rows": nonfinite,
    }


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / PROVISIONAL_SENTINEL).write_text(
        "Provisional engineering-only Layer2 dataset. Not final paper evidence.\n",
        encoding="utf-8",
    )
    layer1_root = Path(args.layer1_root)
    split_map = {
        "train": "train",
        "val": "val",
        "test": "held_out",
    }
    rows: list[dict[str, Any]] = []
    problems: list[dict[str, str]] = []
    evaluator_sidecar_rows: list[dict[str, Any]] = []
    label_status_counts: Counter[str] = Counter()
    for dataset_split, layer1_split in split_map.items():
        manifest, labels_by_key, events_by_key = load_split(layer1_root, layer1_split)
        for manifest_row in manifest:
            key = manifest_row["canonical_key"]
            label = labels_by_key.get(key)
            if label is None:
                problems.append({"episode_key": key, "problem": "missing_teacher_episode_label"})
                continue
            label_status_counts[label.get("teacher_status", "")] += 1
            evaluator_sidecar_rows.append(
                {
                    "episode_key": key,
                    "suite": manifest_row.get("suite", ""),
                    "task_idx": manifest_row.get("task_idx", ""),
                    "state_id": manifest_row.get("state_id", ""),
                    "eval_seed": manifest_row.get("eval_seed", "0"),
                    "condition": "CLEAN",
                    "dataset_split": dataset_split,
                    "teacher_status": label.get("teacher_status", ""),
                    "label_role": label.get("label_role", "") or label_role_for_status(label.get("teacher_status", ""))[0],
                    "primary_or_supplementary": label.get("primary_or_supplementary", "") or label_role_for_status(label.get("teacher_status", ""))[1],
                    "primary_supplementary_event_id": label.get("primary_supplementary_event_id", ""),
                    "task_success": manifest_row.get("task_success", ""),
                }
            )
            ep_rows, ep_problems = build_rows_for_episode(
                manifest_row=manifest_row,
                label=label,
                events=events_by_key.get(key, []),
                split=dataset_split,
            )
            rows.extend(ep_rows)
            for problem in ep_problems:
                problems.append({"episode_key": key, "problem": problem})
    dataset_path = output_dir / "provisional_layer2_frame_dataset.csv"
    fieldnames = [
        "episode_key",
        "suite",
        "task_idx",
        "state_id",
        "eval_seed",
        "condition",
        "dataset_split",
        "step",
        "teacher_status",
        "mechanism_type",
        "label_role",
        "label_source_version",
        "primary_or_supplementary",
        "ignore_for_loss",
        "teacher_phase",
        "teacher_corridor_active",
        "teacher_release_active",
        "teacher_window_start",
        "teacher_window_end",
        "teacher_anchor_step",
        "close_onset_step",
        "grasp_established_step",
        "lift_onset_step",
        "stable_carry_start",
        "event_id",
        "features_finite",
        *SC5_FEATURES,
    ]
    write_csv(dataset_path, rows, fieldnames)
    evaluator_sidecar_path = output_dir / "provisional_layer2_evaluator_sidecar.csv"
    write_csv(evaluator_sidecar_path, evaluator_sidecar_rows)
    audit = leakage_audit(rows)
    dataset_sha = sha256_file(dataset_path)
    sidecar_sha = sha256_file(evaluator_sidecar_path)
    model_input_columns = list(SC5_FEATURES)
    label_role_counts = Counter(str(row.get("label_role", "")) for row in rows)
    supervised_role_counts = Counter(
        str(row.get("label_role", ""))
        for row in rows
        if int(row["ignore_for_loss"]) == 0
    )
    per_suite_role_counts = Counter(
        (str(row.get("suite", "")), str(row.get("label_role", "")))
        for row in rows
    )
    ontology_path = REPO / "configs" / "cross_suite_task_ontology_v1.yaml"
    physics_path = REPO / "configs" / "cross_suite_teacher_physics_v1.yaml"
    schema_path = REPO / "docs" / "schemas" / "cross_suite_teacher_label_schema_v1.md"
    resolver_path = REPO / "scripts" / "stageb" / "cross_suite_layer1_resolver.py"
    builder_path = REPO / "scripts" / "stageb" / "build_provisional_layer2_dataset.py"
    summary = {
        "provisional_engineering_only": True,
        "official_h2_status": "NOT_GRANTED",
        "git_commit": git_commit(),
        "git_dirty_status": git_dirty_status(),
        "layer1_root": str(layer1_root),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "evaluator_sidecar_path": str(evaluator_sidecar_path),
        "evaluator_sidecar_sha256": sidecar_sha,
        "frame_count": len(rows),
        "supervised_frame_count": sum(1 for row in rows if int(row["ignore_for_loss"]) == 0),
        "ignore_frame_count": sum(1 for row in rows if int(row["ignore_for_loss"]) == 1),
        "positive_frame_count": sum(1 for row in rows if int(row["teacher_corridor_active"]) == 1 and int(row["ignore_for_loss"]) == 0),
        "label_source_version": LABEL_SOURCE_VERSION,
        "component_provenance": {
            "resolver_commit": git_commit(),
            "ontology_sha256": sha256_optional(ontology_path),
            "physics_sha256": sha256_optional(physics_path),
            "schema_sha256": sha256_optional(schema_path),
            "resolver_sha256": sha256_optional(resolver_path),
            "dataset_builder_sha256": sha256_optional(builder_path),
            "label_source_version": LABEL_SOURCE_VERSION,
        },
        "label_role_frame_counts": dict(label_role_counts),
        "supervised_label_role_frame_counts": dict(supervised_role_counts),
        "per_suite_label_role_frame_counts": {f"{suite}|{role}": count for (suite, role), count in per_suite_role_counts.items()},
        "episode_label_status_counts": dict(label_status_counts),
        "problem_count": len(problems),
        "problems": problems[:100],
        "leakage_audit": audit,
        "feature_names": list(SC5_FEATURES),
        "model_input_columns": model_input_columns,
        "model_input_columns_exactly_sc5_features": model_input_columns == list(SC5_FEATURES),
        "phase_classes": list(SC5_PHASES),
        "forbidden_claims": [
            "H2 scientifically frozen",
            "Teacher labels are final ground truth",
            "cross-suite detector generalization confirmed",
            "VIS/RAND/shuffled attack effectiveness established",
        ],
    }
    write_json(output_dir / "provisional_layer2_dataset_manifest.json", summary)
    write_csv(output_dir / "provisional_layer2_dataset_problems.csv", problems, ["episode_key", "problem"])
    if problems or audit["status"] != "PASS":
        raise SystemExit(f"dataset build failed: problems={len(problems)} audit={audit}")
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer1-root", required=True)
    ap.add_argument("--output-dir", required=True)
    return ap.parse_args()


def main() -> None:
    build_dataset(parse_args())


if __name__ == "__main__":
    main()
