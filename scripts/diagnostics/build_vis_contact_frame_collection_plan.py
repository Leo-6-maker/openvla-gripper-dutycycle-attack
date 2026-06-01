#!/usr/bin/env python3
"""Build a clean-only frame collection plan for VIS contact-frame diagnostics.

The planner consumes the read-only selector audit produced by
``select_vis_contact_frames.py`` and emits one row per missing contact/carry
frame candidate. It does not launch LIBERO, OpenVLA, attacks, or training.
Generated commands are clean-only artifact-rich collection commands that save
RGB frames and step_records for the target state.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "plan_id",
    "suite",
    "task_name",
    "task_index",
    "state_id",
    "seed",
    "target_policy_step",
    "request_start_policy_step",
    "request_end_policy_step",
    "source_step_records",
    "source_selector_status",
    "collection_output_root",
    "collection_run_id_prefix",
    "model_path",
    "cuda_visible_devices",
    "render_gpu_device_id",
    "command",
    "boundary",
]


# Verified against ``libero_object`` from openvla_official_libero_20260525.
LIBERO_OBJECT_TASK_INDEX = {
    "pick up the alphabet soup and place it in the basket": 0,
    "pick up the cream cheese and place it in the basket": 1,
    "pick up the salad dressing and place it in the basket": 2,
    "pick up the bbq sauce and place it in the basket": 3,
    "pick up the ketchup and place it in the basket": 4,
    "pick up the tomato sauce and place it in the basket": 5,
    "pick up the butter and place it in the basket": 6,
    "pick up the milk and place it in the basket": 7,
    "pick up the chocolate pudding and place it in the basket": 8,
    "pick up the orange juice and place it in the basket": 9,
}


def norm_task_name(value: str) -> str:
    return str(value or "").replace("_", " ").strip().lower()


def parse_int(value: str, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def build_plan_rows(
    selector_rows: list[dict],
    output_root: str,
    model_path: str,
    cuda_visible_devices: str,
    render_gpu_device_id: str,
    window_radius: int,
    python_executable: str,
) -> list[dict]:
    rows: list[dict] = []
    for row in selector_rows:
        if row.get("selector_status") != "candidate_missing_frame":
            continue
        if row.get("suite") != "libero_object":
            continue
        task_name = row.get("task_name", "")
        task_index = LIBERO_OBJECT_TASK_INDEX.get(norm_task_name(task_name))
        if task_index is None:
            raise ValueError(f"cannot map Object task to task_start index: {task_name}")
        state_id = parse_int(row.get("state_id"), 0)
        target = parse_int(row.get("policy_step_idx") or row.get("step_idx"), 0)
        plan_id = f"{row.get('run_id') or task_index}_contact_s{state_id}"
        run_prefix = f"vis_payload_contact_{task_name.replace(' ', '_')}_clean"
        command_parts = [
            python_executable,
            "scripts/run_official_eval_artifact_rich.py",
            "--model_path", model_path,
            "--task_suite_name", "libero_object",
            "--num_trials_per_task", str(state_id + 1),
            "--task_start", str(task_index),
            "--task_count", "1",
            "--seed", str(parse_int(row.get("seed"), 0)),
            "--output_root", output_root,
            "--render_gpu_device_id", str(render_gpu_device_id),
            "--cuda_visible_devices", cuda_visible_devices,
            "--run_id_prefix", run_prefix,
            "--save_rgb",
            "--save_step_records",
            "--save_privileged_teacher_state",
            "--attack_condition", "clean",
        ]
        rows.append(
            {
                "plan_id": plan_id,
                "suite": row.get("suite", ""),
                "task_name": task_name,
                "task_index": task_index,
                "state_id": state_id,
                "seed": parse_int(row.get("seed"), 0),
                "target_policy_step": target,
                "request_start_policy_step": max(0, target - window_radius),
                "request_end_policy_step": target + window_radius,
                "source_step_records": row.get("step_records", ""),
                "source_selector_status": row.get("selector_status", ""),
                "collection_output_root": output_root,
                "collection_run_id_prefix": run_prefix,
                "model_path": model_path,
                "cuda_visible_devices": cuda_visible_devices,
                "render_gpu_device_id": render_gpu_device_id,
                "command": " ".join(command_parts),
                "boundary": "clean_only_no_attack_no_vis_no_sus30_no_detector_intervention",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector_csv", default="tables/vis_contact_frame_selection_audit.csv")
    parser.add_argument("--output_csv", default="tables/vis_contact_frame_collection_plan.csv")
    parser.add_argument("--collection_output_root", default="/data/liuyu/outputs/vis_payload_contact_frame_collection_20260601")
    parser.add_argument("--model_path", default="/data/aviary/models/openvla/openvla-7b-finetuned-libero-object")
    parser.add_argument("--cuda_visible_devices", default="4,5")
    parser.add_argument("--render_gpu_device_id", default="4")
    parser.add_argument("--window_radius", type=int, default=2)
    parser.add_argument("--python", default="python")
    args = parser.parse_args()

    with Path(args.selector_csv).open(newline="", encoding="utf-8") as f:
        selector_rows = list(csv.DictReader(f))
    rows = build_plan_rows(
        selector_rows=selector_rows,
        output_root=args.collection_output_root,
        model_path=args.model_path,
        cuda_visible_devices=args.cuda_visible_devices,
        render_gpu_device_id=args.render_gpu_device_id,
        window_radius=args.window_radius,
        python_executable=args.python,
    )
    write_csv(Path(args.output_csv), rows)
    print(f"wrote {len(rows)} plan rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
