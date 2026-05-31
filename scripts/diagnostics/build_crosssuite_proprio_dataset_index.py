#!/usr/bin/env python3
"""Build a CrossSuite-ProprioNoStep-v2 artifact index.

The default output is metadata plus per-step deployable proprio fields.  It
can ingest either the Milestone 2B student CSV or artifact-rich rollout
directories containing ``step_records.jsonl``.  It never trains a model and it
does not read attack/oracle/VIS outcomes as labels.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CSV_FIELDS = [
    "suite",
    "task_id",
    "task_name",
    "state_id",
    "seed",
    "run_id",
    "run_dir",
    "step_idx",
    "n_steps",
    "clean_success",
    "mechanism_type",
    "mechanism_eligible",
    "teacher_label_available",
    "full_eef_position_available",
    "full_eef_velocity_available",
    "gripper_feature_available",
    "action_feature_available",
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_vx",
    "eef_vy",
    "eef_vz",
    "gripper_qpos",
    "gripper_width",
    "gripper_command",
    "action_dx",
    "action_dy",
    "action_dz",
    "action_gripper",
    "available_features",
    "available_labels",
    "missing_fields",
    "label_source",
    "split_candidate",
    "notes",
]
REQUIRED_FEATURES = (
    "gripper_qpos",
    "gripper_width",
    "gripper_command",
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_vx",
    "eef_vy",
    "eef_vz",
    "action_dx",
    "action_dy",
    "action_dz",
    "action_gripper",
)
REQUIRED_LABELS = ("mechanism_eligible", "teacher_label_available")
SUITE_PREFIXES = ("spatial_", "object_", "goal_", "libero10_", "libero_10_")


def write_schema_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def print_schema() -> None:
    for field in CSV_FIELDS:
        print(field)


def _norm_text(value: object) -> str:
    return str("" if value is None else value).strip().lower().replace(" ", "_")


def _norm_task(value: object) -> str:
    text = _norm_text(value)
    for prefix in SUITE_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _state_seed(value: object) -> str:
    text = str("" if value is None else value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _as_float(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    try:
        return float(value)
    except Exception:
        return None


def _as_bool_text(value: object) -> str:
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y"):
        return "true"
    if text in ("0", "false", "no", "n"):
        return "false"
    return ""


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_teacher_windows(paths: Iterable[Path]) -> Dict[Tuple[str, str, str, str], dict]:
    labels: Dict[Tuple[str, str, str, str], dict] = {}
    for path in paths:
        if not path or not path.exists():
            continue
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                suite = _norm_text(row.get("suite"))
                task = _norm_task(row.get("task_id") or row.get("task_name"))
                state = _state_seed(row.get("state_id"))
                seed = _state_seed(row.get("seed"))
                labels[(suite, task, state, seed)] = row
    return labels


def _mechanism_type(task_name: str) -> str:
    text = _norm_text(task_name)
    if any(word in text for word in ("open_the", "close_the", "turn_on", "turn_off")):
        return "articulated_object"
    if text.count("_and_") >= 2 or "both_" in text:
        return "multi_object_transfer"
    if any(word in text for word in ("pick_up", "place_it", "put_the", "put_both")):
        return "pick_place_transfer"
    if any(word in text for word in ("push_the", "move_the", "next_to", "front_of", "left_of", "right_of")):
        return "rearrangement_or_spatial"
    return "unknown_or_low_signal"


def _extract_action(row: dict, idx: int, fallback_keys: Iterable[str]) -> Optional[float]:
    for key in fallback_keys:
        value = row.get(key)
        if isinstance(value, list) and len(value) > idx:
            return _as_float(value[idx])
        value = _as_float(value)
        if value is not None and idx == 0:
            return value
    return None


def _extract_step_features(row: dict, prev_row: Optional[dict]) -> Dict[str, Optional[float]]:
    eef_x = _as_float(row.get("eef_x"))
    eef_y = _as_float(row.get("eef_y"))
    eef_z = _as_float(row.get("eef_z") if row.get("eef_z") not in (None, "") else row.get("eef_z_before"))
    prev_x = _as_float(prev_row.get("eef_x")) if prev_row else None
    prev_y = _as_float(prev_row.get("eef_y")) if prev_row else None
    prev_z = _as_float((prev_row or {}).get("eef_z") if (prev_row or {}).get("eef_z") not in (None, "") else (prev_row or {}).get("eef_z_before")) if prev_row else None
    return {
        "eef_x": eef_x,
        "eef_y": eef_y,
        "eef_z": eef_z,
        "eef_vx": _as_float(row.get("eef_vx")) if row.get("eef_vx") not in (None, "") else (None if eef_x is None or prev_x is None else eef_x - prev_x),
        "eef_vy": _as_float(row.get("eef_vy")) if row.get("eef_vy") not in (None, "") else (None if eef_y is None or prev_y is None else eef_y - prev_y),
        "eef_vz": _as_float(row.get("eef_vz")) if row.get("eef_vz") not in (None, "") else (None if eef_z is None or prev_z is None else eef_z - prev_z),
        "gripper_qpos": _as_float(row.get("gripper_qpos") or row.get("gripper_qpos_sum_before") or row.get("gripper_qpos_abs_sum_before")),
        "gripper_width": _as_float(row.get("gripper_width") or row.get("gripper_qpos_abs_sum_before") or row.get("gripper_qpos_abs_sum_after")),
        "gripper_command": _as_float(row.get("gripper_command") or row.get("clean_gripper_env") or row.get("executed_gripper_env")),
        "action_dx": _as_float(row.get("action_dx")) if row.get("action_dx") not in (None, "") else _extract_action(row, 0, ("env_action", "executed_action", "clean_action", "raw_action")),
        "action_dy": _as_float(row.get("action_dy")) if row.get("action_dy") not in (None, "") else _extract_action(row, 1, ("env_action", "executed_action", "clean_action", "raw_action")),
        "action_dz": _as_float(row.get("action_dz")) if row.get("action_dz") not in (None, "") else _extract_action(row, 2, ("env_action", "executed_action", "clean_action", "raw_action")),
        "action_gripper": _as_float(row.get("action_gripper")) if row.get("action_gripper") not in (None, "") else _extract_action(row, 6, ("env_action", "executed_action", "clean_action", "raw_action")),
    }


def _row_from_features(meta: dict, features: dict, label: Optional[dict], source_note: str) -> dict:
    available = {key for key, value in features.items() if value not in (None, "")}
    missing = set(REQUIRED_FEATURES) - available
    clean_success = _as_bool_text((label or {}).get("clean_success")) or _as_bool_text(meta.get("clean_success"))
    mechanism_eligible = _as_bool_text((label or {}).get("mechanism_eligible"))
    teacher_label_available = "true" if label else "false"
    if not mechanism_eligible and not label:
        mechanism_eligible = "false"
    full_pos = all(features.get(k) is not None for k in ("eef_x", "eef_y", "eef_z"))
    full_vel = all(features.get(k) is not None for k in ("eef_vx", "eef_vy", "eef_vz"))
    gripper_ok = all(features.get(k) is not None for k in ("gripper_qpos", "gripper_width", "gripper_command"))
    action_ok = all(features.get(k) is not None for k in ("action_dx", "action_dy", "action_dz", "action_gripper"))
    split_candidate = "yes" if full_pos and full_vel and gripper_ok and action_ok and label else "no"
    if split_candidate == "no" and features.get("eef_z") is not None and gripper_ok and action_ok and label:
        split_candidate = "partial_eef_z_only"
    labels = ["teacher_window"] if label else []
    out = {
        **meta,
        **{key: "" if value is None else value for key, value in features.items()},
        "clean_success": clean_success,
        "mechanism_type": meta.get("mechanism_type") or _mechanism_type(meta.get("task_name", "")),
        "mechanism_eligible": mechanism_eligible,
        "teacher_label_available": teacher_label_available,
        "full_eef_position_available": str(full_pos).lower(),
        "full_eef_velocity_available": str(full_vel).lower(),
        "gripper_feature_available": str(gripper_ok).lower(),
        "action_feature_available": str(action_ok).lower(),
        "available_features": ";".join(sorted(available)),
        "available_labels": ";".join(labels),
        "missing_fields": ";".join(sorted(missing | (set(REQUIRED_LABELS) if not label else set()))),
        "label_source": (label or {}).get("label_source", ""),
        "split_candidate": split_candidate,
        "notes": source_note,
    }
    return {field: out.get(field, "") for field in CSV_FIELDS}


def _artifact_rows(artifact_roots: Iterable[Path], teacher_labels: Dict[Tuple[str, str, str, str], dict]) -> List[dict]:
    output = []
    for root in artifact_roots:
        for step_path in sorted(root.rglob("step_records.jsonl")):
            rows = list(_read_jsonl(step_path))
            if not rows:
                continue
            prev = None
            for row in rows:
                suite = _norm_text(row.get("suite"))
                task_name = row.get("task_name") or row.get("task_instruction") or row.get("task_id") or step_path.parent.name
                task_id = row.get("task_id") or task_name
                state_id = _state_seed(row.get("state_id"))
                seed = _state_seed(row.get("seed"))
                label = teacher_labels.get((suite, _norm_task(task_id), state_id, seed)) or teacher_labels.get((suite, _norm_task(task_name), state_id, seed))
                meta = {
                    "suite": suite,
                    "task_id": task_id,
                    "task_name": task_name,
                    "state_id": state_id,
                    "seed": seed,
                    "run_id": row.get("run_id") or step_path.parent.name,
                    "run_dir": str(step_path.parent),
                    "step_idx": row.get("step_idx", ""),
                    "clean_success": row.get("success_done") or row.get("success_so_far") or row.get("info_success_if_available"),
                    "mechanism_type": _mechanism_type(str(task_name)),
                }
                features = _extract_step_features(row, prev)
                output.append(_row_from_features(meta, features, label, "artifact_step_records_no_training"))
                prev = row
    return output


def _student_csv_rows(input_csv: Path, teacher_labels: Dict[Tuple[str, str, str, str], dict]) -> List[dict]:
    output = []
    with input_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            suite = _norm_text(row.get("suite"))
            task_id = row.get("task_id") or row.get("task_name")
            state_id = _state_seed(row.get("state_id"))
            seed = _state_seed(row.get("seed"))
            label = teacher_labels.get((suite, _norm_task(task_id), state_id, seed))
            meta = {
                "suite": suite,
                "task_id": task_id,
                "task_name": row.get("task_name", ""),
                "state_id": state_id,
                "seed": seed,
                "run_id": row.get("run_id", ""),
                "run_dir": row.get("run_dir", ""),
                "step_idx": row.get("step_idx", ""),
                "clean_success": row.get("clean_success", ""),
                "mechanism_type": row.get("mechanism_type", ""),
            }
            features = {key: _as_float(row.get(key)) for key in REQUIRED_FEATURES}
            output.append(_row_from_features(meta, features, label, "student_csv_no_training"))
    return output


def _aggregate_episode_rows(rows: List[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, str, str, str, str], dict] = {}
    counts: Dict[Tuple[str, str, str, str, str], int] = {}
    feature_sets: Dict[Tuple[str, str, str, str, str], set] = {}
    label_sets: Dict[Tuple[str, str, str, str, str], set] = {}
    for row in rows:
        key = (
            row.get("suite", ""),
            _norm_task(row.get("task_id")),
            row.get("state_id", ""),
            row.get("seed", ""),
            row.get("run_id") or row.get("run_dir", ""),
        )
        counts[key] = counts.get(key, 0) + 1
        base = grouped.setdefault(key, dict(row))
        for field in CSV_FIELDS:
            if field in ("available_features", "available_labels", "missing_fields", "notes", "n_steps"):
                continue
            if base.get(field) in (None, "") and row.get(field) not in (None, ""):
                base[field] = row.get(field)
        feature_sets.setdefault(key, set()).update(x for x in row.get("available_features", "").split(";") if x)
        label_sets.setdefault(key, set()).update(x for x in row.get("available_labels", "").split(";") if x)
    output = []
    for key, row in grouped.items():
        available = feature_sets.get(key, set())
        labels = label_sets.get(key, set())
        missing = (set(REQUIRED_FEATURES) - available) | (set(REQUIRED_LABELS) if "teacher_window" not in labels else set())
        row["n_steps"] = counts.get(key, 0)
        row["available_features"] = ";".join(sorted(available))
        row["available_labels"] = ";".join(sorted(labels))
        row["missing_fields"] = ";".join(sorted(missing))
        row["full_eef_position_available"] = str({"eef_x", "eef_y", "eef_z"}.issubset(available)).lower()
        row["full_eef_velocity_available"] = str({"eef_vx", "eef_vy", "eef_vz"}.issubset(available)).lower()
        row["gripper_feature_available"] = str({"gripper_qpos", "gripper_width", "gripper_command"}.issubset(available)).lower()
        row["action_feature_available"] = str({"action_dx", "action_dy", "action_dz", "action_gripper"}.issubset(available)).lower()
        has_full = not missing
        has_z = {"eef_z", "gripper_qpos", "gripper_width", "gripper_command", "action_gripper"}.issubset(available) and "teacher_window" in labels
        row["split_candidate"] = "yes" if has_full else ("partial_eef_z_only" if has_z else "no")
        row["notes"] = "episode_aggregate_no_training"
        output.append({field: row.get(field, "") for field in CSV_FIELDS})
    return output


def build_index(
    input_csv: Optional[Path],
    artifact_roots: Iterable[Path],
    teacher_window_csvs: Iterable[Path],
    output_csv: Path,
    row_mode: str = "episode",
) -> None:
    teacher_labels = _load_teacher_windows(teacher_window_csvs)
    rows: List[dict] = []
    if input_csv:
        rows.extend(_student_csv_rows(input_csv, teacher_labels))
    rows.extend(_artifact_rows(artifact_roots, teacher_labels))
    if row_mode == "episode":
        rows = _aggregate_episode_rows(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv")
    parser.add_argument("--artifact_root", action="append", default=[])
    parser.add_argument("--teacher_window_csv", action="append", default=[])
    parser.add_argument("--row_mode", choices=("episode", "per_step"), default="episode")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--dry-run-schema", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-schema", action="store_true")
    args = parser.parse_args()
    output_csv = Path(args.output_csv)
    if args.print_schema:
        print_schema()
    if args.dry_run_schema or args.dry_run or args.print_schema:
        write_schema_csv(output_csv)
        return 0
    input_csv = Path(args.input_csv) if args.input_csv else None
    artifact_roots = [Path(p) for p in args.artifact_root]
    if not input_csv and not artifact_roots:
        raise SystemExit("--input_csv or --artifact_root is required unless --dry-run-schema is set")
    build_index(input_csv, artifact_roots, [Path(p) for p in args.teacher_window_csv], output_csv, row_mode=args.row_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
