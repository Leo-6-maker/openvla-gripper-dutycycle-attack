#!/usr/bin/env python3
"""Build a no-rollout plan for collecting clean contact-frame RGB dumps.

This script is intentionally non-executing. It consumes the read-only
``vis_contact_frame_selection_audit.csv`` produced by
``select_vis_contact_frames.py`` and emits a bounded clean-only collection plan
for rows where the selector found a contact/carry candidate but no saved frame.

It does not import LIBERO, load OpenVLA, start MuJoCo, run attacks, or create
rollout output roots. The generated commands are proposals that require
explicit approval before execution.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


OBJECT_TASK_ORDER = [
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    "pick_up_the_salad_dressing_and_place_it_in_the_basket",
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    "pick_up_the_ketchup_and_place_it_in_the_basket",
    "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
    "pick_up_the_butter_and_place_it_in_the_basket",
    "pick_up_the_milk_and_place_it_in_the_basket",
    "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
    "pick_up_the_orange_juice_and_place_it_in_the_basket",
]

CSV_FIELDS = [
    "plan_id",
    "run_id",
    "suite",
    "task_id",
    "task_name",
    "libero_task_index",
    "state_id",
    "seed",
    "source_step_records",
    "selector_score",
    "selector_reason",
    "target_contact_step_idx",
    "target_policy_step_idx",
    "expected_artifact_step_idx",
    "frame_window_start",
    "frame_window_end",
    "requested_policy_frames",
    "expected_frame_paths",
    "existing_frame_available",
    "collection_mode",
    "attack_enabled",
    "vis_enabled",
    "sus30_enabled",
    "detector_enabled",
    "model_path",
    "unnorm_key",
    "max_steps",
    "num_steps_wait",
    "center_crop",
    "postprocess_gripper",
    "official_preprocess",
    "attention_backend",
    "cuda_visible_devices",
    "render_gpu_device_id",
    "physical_render_gpu_note",
    "output_root",
    "planned_run_id_prefix",
    "planned_run_dir",
    "artifact_runner",
    "collection_command",
    "rollouts_if_approved",
    "launch_requires_explicit_approval",
    "collection_status",
    "notes",
]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def normalize_task_name(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace(" ", "_")
    text = text.replace("object_", "")
    return text


def infer_state_id(row: dict) -> int:
    raw = str(row.get("state_id", "")).strip()
    if raw != "":
        return _int(raw, 0)
    match = re.search(r"_s(\d+)(?:\D*$|$)", str(row.get("run_id", "")))
    return int(match.group(1)) if match else 0


def infer_object_task_index(row: dict) -> int:
    raw_task_id = str(row.get("task_id", "")).strip()
    if raw_task_id.isdigit():
        return int(raw_task_id)
    task_name = normalize_task_name(str(row.get("task_name", "")))
    task_id_name = normalize_task_name(raw_task_id)
    for idx, canonical in enumerate(OBJECT_TASK_ORDER):
        if task_name == canonical or task_id_name.endswith(canonical):
            return idx
    raise ValueError(f"cannot infer LIBERO object task index from task_id={raw_task_id!r} task_name={row.get('task_name')!r}")


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def select_missing_contact_rows(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if str(row.get("suite", "")).strip() == "libero_object"
        and str(row.get("selector_status", "")).strip() == "candidate_missing_frame"
        and not _truthy(row.get("frame_available"))
    ]


def planned_runner_run_id(prefix: str, task_name: str, state_id: int) -> str:
    short = normalize_task_name(task_name)
    short = short.replace("pick_up_the_", "").replace("_and_place_it_in_the_basket", "")
    return f"{prefix}_{short}_s{state_id}"


def build_plan_row(
    row: dict,
    *,
    output_root: str,
    model_path: str,
    cuda_visible_devices: str,
    render_gpu_device_id: int,
    frame_context: int,
    run_id_prefix: str,
    num_steps_wait: int,
    max_steps: int,
    attention_backend: str,
) -> dict:
    task_index = infer_object_task_index(row)
    state_id = infer_state_id(row)
    target_policy_step = _int(row.get("step_idx"), 0)
    frame_start = max(0, target_policy_step - frame_context)
    frame_end = target_policy_step + frame_context
    requested = list(range(frame_start, frame_end + 1))
    run_id = planned_runner_run_id(run_id_prefix, str(row.get("task_name", "")), state_id)
    planned_run_dir = f"{output_root}/runs/libero_object/{run_id}"
    expected_paths = [f"{planned_run_dir}/frames/step_{idx:04d}.png" for idx in requested]
    trials_needed = state_id + 1
    command = (
        "python scripts/run_official_eval_artifact_rich.py "
        f"--model_path {model_path} "
        "--task_suite_name libero_object "
        f"--task_start {task_index} --task_count 1 "
        f"--num_trials_per_task {trials_needed} "
        f"--num_steps_wait {num_steps_wait} "
        "--center_crop "
        f"--attn_impl {attention_backend} "
        f"--cuda_visible_devices {cuda_visible_devices} "
        f"--render_gpu_device_id {render_gpu_device_id} "
        f"--output_root {output_root} "
        f"--run_id_prefix {run_id_prefix} "
        "--attack_condition clean "
        "--save_rgb --save_step_records --save_privileged_teacher_state"
    )
    if state_id > 0:
        note = "runner lacks state_ids; command would collect preceding states too unless a state selector is added"
    else:
        note = "state0 exact one-episode clean-only frame dump"
    return {
        "plan_id": f"contact_frame_{task_index}_s{state_id}_step{target_policy_step}",
        "run_id": row.get("run_id", ""),
        "suite": "libero_object",
        "task_id": row.get("task_id", ""),
        "task_name": row.get("task_name", ""),
        "libero_task_index": task_index,
        "state_id": state_id,
        "seed": row.get("seed", "0"),
        "source_step_records": row.get("step_records", ""),
        "selector_score": row.get("score", ""),
        "selector_reason": row.get("reason", ""),
        "target_contact_step_idx": target_policy_step,
        "target_policy_step_idx": target_policy_step,
        "expected_artifact_step_idx": target_policy_step + num_steps_wait,
        "frame_window_start": frame_start,
        "frame_window_end": frame_end,
        "requested_policy_frames": ";".join(str(idx) for idx in requested),
        "expected_frame_paths": ";".join(expected_paths),
        "existing_frame_available": str(_truthy(row.get("frame_available"))).lower(),
        "collection_mode": "clean_only_contact_frame_dump",
        "attack_enabled": "false",
        "vis_enabled": "false",
        "sus30_enabled": "false",
        "detector_enabled": "false",
        "model_path": model_path,
        "unnorm_key": "libero_object",
        "max_steps": max_steps,
        "num_steps_wait": num_steps_wait,
        "center_crop": "true",
        "postprocess_gripper": "true",
        "official_preprocess": "official_pil_lanczos",
        "attention_backend": attention_backend,
        "cuda_visible_devices": cuda_visible_devices,
        "render_gpu_device_id": render_gpu_device_id,
        "physical_render_gpu_note": f"render_gpu_device_id is local inside CUDA_VISIBLE_DEVICES={cuda_visible_devices}; local 0 maps to first listed physical GPU",
        "output_root": output_root,
        "planned_run_id_prefix": run_id_prefix,
        "planned_run_dir": planned_run_dir,
        "artifact_runner": "scripts/run_official_eval_artifact_rich.py",
        "collection_command": command,
        "rollouts_if_approved": trials_needed,
        "launch_requires_explicit_approval": "true",
        "collection_status": "proposed_not_executed",
        "notes": note,
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in CSV_FIELDS} for row in rows)


def write_report(path: Path, rows: list[dict], source_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_rollouts = sum(_int(row.get("rollouts_if_approved"), 0) for row in rows)
    task_lines = "\n".join(
        f"- {row['task_name']} state {row['state_id']}: target policy step {row['target_policy_step_idx']}, frames {row['frame_window_start']}..{row['frame_window_end']}"
        for row in rows
    )
    text = f"""# VIS Contact Frame Collection Proposal

## Status

This is a proposal only. No rollout, attack, training, VIS, sus30, or detector-triggered run was launched by this planner.

Source audit: `{source_csv}`

Planned rows: {len(rows)}
Maximum clean-only rollouts if explicitly approved: {total_rollouts}

## Selected Contact Candidates

{task_lines if task_lines else "- None"}

## Collection Boundary

- Collection mode: clean-only contact frame dump.
- Attack enabled: false.
- VIS enabled: false.
- sus30 enabled: false.
- Detector enabled: false.
- Official config: `libero_object`, `num_steps_wait=10`, `max_steps=280`, center crop, official PIL/Lanczos preprocessing, postprocessed gripper.
- GPU policy in generated commands: `CUDA_VISIBLE_DEVICES=4,5`; local render GPU id 0 maps to physical GPU 4, not physical GPU 0.
- Existing `scripts/run_official_eval_artifact_rich.py` saves RGB as `frames/step_####.png` using policy-step indexing.

## Why This Is Needed

The prior saved VIS diagnostic frames are available but correspond to wait/pre-policy rows. The selector found contact/carry candidates in existing Object artifacts, but those contact candidates do not have RGB frame files. This proposal collects only the missing clean frames needed to rerun one-frame VIS diagnostics at verified contact/carry timesteps.

## Approval Gate

Do not execute the generated commands until explicitly approved. If approved, start with the three planned state-0 clean episodes only and verify that the expected contact-frame PNG files exist before any further VIS diagnostic.
"""
    path.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_csv", default="tables/vis_contact_frame_selection_audit.csv")
    parser.add_argument("--output_csv", default="tables/vis_contact_frame_collection_plan.csv")
    parser.add_argument("--report_md", default="reports/VIS_CONTACT_FRAME_COLLECTION_PROPOSAL.md")
    parser.add_argument("--output_root", default="/data/liuyu/outputs/vis_contact_frame_dump_clean_20260531")
    parser.add_argument("--model_path", default="/data/aviary/models/openvla/openvla-7b-finetuned-libero-object")
    parser.add_argument("--cuda_visible_devices", default="4,5")
    parser.add_argument("--render_gpu_device_id", type=int, default=0)
    parser.add_argument("--frame_context", type=int, default=2)
    parser.add_argument("--run_id_prefix", default="vis_contact_frame_clean")
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=280)
    parser.add_argument("--attention_backend", default="eager")
    parser.add_argument("--print-schema", action="store_true")
    args = parser.parse_args()
    if args.print_schema:
        for field in CSV_FIELDS:
            print(field)
        return 0
    rows = select_missing_contact_rows(read_rows(Path(args.audit_csv)))
    plan = [
        build_plan_row(
            row,
            output_root=args.output_root,
            model_path=args.model_path,
            cuda_visible_devices=args.cuda_visible_devices,
            render_gpu_device_id=args.render_gpu_device_id,
            frame_context=max(0, int(args.frame_context)),
            run_id_prefix=args.run_id_prefix,
            num_steps_wait=int(args.num_steps_wait),
            max_steps=int(args.max_steps),
            attention_backend=args.attention_backend,
        )
        for row in rows
    ]
    write_rows(Path(args.output_csv), plan)
    write_report(Path(args.report_md), plan, Path(args.audit_csv))
    print(f"wrote {len(plan)} proposed clean-only collection rows to {args.output_csv}")
    print(f"wrote proposal report to {args.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
