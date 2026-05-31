#!/usr/bin/env python3
"""One-frame VIS diagnostic loader.

This script does not run a rollout or step a LIBERO environment.  It loads one
saved RGB frame, decodes the clean OpenVLA action, runs TokenPrefixPGD on that
frame, and re-decodes the adversarial action from ``debug["adv_inputs"]``.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.attack_adapter import TokenPrefixPGDAttacker, get_adv_inputs_from_attack_result  # noqa: E402
from gripper_attack.openvla_redecode import redecode_openvla_action_from_adv_inputs  # noqa: E402


OBJECTIVE_MAP = {
    "target_action_ce": "force_gripper_open_token_ce",
    "gripper_open_region_ce": "gripper_open_region_ce",
    "gripper_logit_margin_cw": "gripper_logit_margin_cw",
}
CSV_FIELDS = [
    "frame_path",
    "step_records_path",
    "step_idx",
    "instruction",
    "model_path",
    "unnorm_key",
    "objective",
    "attack_objective",
    "eps",
    "steps",
    "clean_action",
    "adv_action",
    "clean_gripper_token",
    "adv_gripper_token",
    "gripper_token_flipped",
    "clean_gripper_action",
    "adv_gripper_action",
    "gripper_delta",
    "arm_l2",
    "target_ce_before",
    "target_ce_after",
    "open_bin_prob_mass_before",
    "open_bin_prob_mass_after",
    "close_bin_prob_mass_before",
    "close_bin_prob_mass_after",
    "open_close_margin_before",
    "open_close_margin_after",
    "perturbation_linf",
    "perturbation_l2",
    "clean_decode_runtime_sec",
    "adv_decode_runtime_sec",
    "attack_runtime_sec",
    "model_dtype",
    "pixel_values_dtype",
    "cuda_visible_devices",
    "error",
]


@dataclass
class OneFrameContext:
    """Reusable model/frame state for repeated no-rollout VIS diagnostics."""

    runner: Any
    step_row: Dict[str, Any]
    frame_path: Path
    image: np.ndarray
    instruction: str
    model: Any
    processor: Any
    device: Any
    clean_action: Any
    clean_gen: Any
    clean_decode_runtime_sec: float


def parse_budget(value: str) -> float:
    text = str(value).strip()
    if "/" in text:
        num, den = text.split("/", 1)
        return float(num) / float(den)
    return float(text)


def _json(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.astype(float).tolist()
    return json.dumps(value, separators=(",", ":"))


def _load_runner_module():
    path = REPO_ROOT / "scripts" / "v4_run_eval_openvla.py"
    spec = importlib.util.spec_from_file_location("codex_v4_run_eval_openvla", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load runner module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _physical_gpu0_visible() -> bool:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return True
    return any(part.strip() == "0" for part in visible.split(","))


def _read_step_record(path: Optional[str], step_idx: Optional[int]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"step_records path does not exist: {p}")
    rows = []
    with p.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                if step_idx is not None and int(row.get("step_idx", -1)) == int(step_idx):
                    return row
    if step_idx is not None:
        raise ValueError(f"step_idx={step_idx} not found in {p}")
    return rows[0] if rows else {}


def _resolve_frame_path(args, row: Dict[str, Any]) -> Path:
    image_path = getattr(args, "image_path", None) or getattr(args, "frame", None)
    if image_path:
        return Path(image_path)
    for key in ("image_path", "agentview_image_path", "frame_path", "rgb_path"):
        value = row.get(key)
        if value:
            return Path(value)
    if args.step_records and args.step_idx is not None:
        candidate = Path(args.step_records).parent / "frames" / f"step_{int(args.step_idx):04d}.png"
        if candidate.exists():
            return candidate
    raise ValueError("no image_path provided and no frame path found in step record")


def _load_frame(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"image_path does not exist: {path}")
    return np.asarray(Image.open(path).convert("RGB"))


def _gripper_audit(debug: dict, prefix: str) -> dict:
    audit = debug.get(f"{prefix}_logit_audit", {}) or {}
    return {
        "target_ce": debug.get("target_ce_initial" if prefix == "clean" else "target_ce_final", ""),
        "open_mass": audit.get("gripper_open_bin_prob_mass", ""),
        "close_mass": audit.get("gripper_close_bin_prob_mass", ""),
        "margin": (
            ""
            if audit.get("gripper_open_bin_prob_mass", "") == "" or audit.get("gripper_close_bin_prob_mass", "") == ""
            else float(audit.get("gripper_open_bin_prob_mass", 0.0)) - float(audit.get("gripper_close_bin_prob_mass", 0.0))
        ),
    }


def prepare_one_frame_context(args) -> OneFrameContext:
    """Load the model, frame, instruction, and clean decode once.

    Threshold sweeps should reuse this context across objective/epsilon/step
    combinations. This keeps diagnostics no-rollout while avoiding repeated
    OpenVLA model loads for the same frame.
    """

    if _physical_gpu0_visible() and not args.allow_physical_gpu0:
        raise RuntimeError("refusing real diagnostic because physical GPU0 is visible; set CUDA_VISIBLE_DEVICES to nonzero GPUs")
    runner = _load_runner_module()
    step_row = _read_step_record(args.step_records, args.step_idx)
    frame_path = _resolve_frame_path(args, step_row)
    image = _load_frame(frame_path)
    instruction = args.instruction or step_row.get("task_instruction") or step_row.get("task_name") or step_row.get("task_id")
    if not instruction:
        raise ValueError("instruction is required when it is absent from step_records")

    model, processor, device = runner.load_model(args.model_path, model_gpu_device_id=int(args.model_gpu_device_id))
    clean_t0 = time.time()
    clean_action, _, clean_dt, clean_gen = runner.decode_with_scores(
        model,
        processor,
        device,
        image,
        instruction,
        args.unnorm_key,
        int(getattr(args, "k_prefix_logits", 8)),
        libero_official_preprocess=args.libero_official_preprocess,
        libero_preprocess_backend=args.libero_preprocess_backend,
        center_crop=args.center_crop,
        resize_size=int(args.openvla_resize_size),
        drop_attention_mask=True,
    )
    clean_decode_runtime = clean_dt if clean_dt is not None else time.time() - clean_t0
    return OneFrameContext(
        runner=runner,
        step_row=step_row,
        frame_path=frame_path,
        image=image,
        instruction=str(instruction),
        model=model,
        processor=processor,
        device=device,
        clean_action=clean_action,
        clean_gen=clean_gen,
        clean_decode_runtime_sec=float(clean_decode_runtime),
    )


def run_one_frame_attack(context: OneFrameContext, args) -> Dict[str, Any]:
    """Run one TokenPrefixPGD attack and re-decode from debug adv_inputs."""

    target_action = np.asarray(context.clean_action, dtype=np.float32).copy()
    target_action[-1] = float(args.force_open_raw_gripper)
    attack_objective = OBJECTIVE_MAP[args.objective]
    attack_cfg = {
        "attack_optimizer": {
            "method": "token_prefix_pgd",
            "objective": attack_objective,
            "epsilon": float(args.eps),
            "step_size": float(args.step_size),
            "num_steps": int(args.steps),
            "random_start": bool(args.random_start),
            "cw_margin": float(args.cw_margin),
        }
    }
    attacker = TokenPrefixPGDAttacker(
        context.model,
        context.processor,
        attack_cfg,
        seed=int(args.seed),
        preprocess_kwargs={
            "libero_official_preprocess": args.libero_official_preprocess,
            "libero_preprocess_backend": args.libero_preprocess_backend,
            "center_crop": args.center_crop,
            "resize_size": int(args.openvla_resize_size),
            "postprocess_gripper": args.postprocess_gripper,
        },
        device=context.device,
    )
    attack_t0 = time.time()
    attack_result = attacker.attack(
        observation=context.image,
        instruction=context.instruction,
        clean_action=context.clean_action,
        target_action=target_action,
        clean_model_output=context.clean_gen,
        unnorm_key=args.unnorm_key,
    )
    attack_runtime = time.time() - attack_t0
    adv_inputs = get_adv_inputs_from_attack_result(attack_result)
    adv_decoded = redecode_openvla_action_from_adv_inputs(
        model=context.model,
        processor=context.processor,
        adv_inputs=adv_inputs,
        instruction=context.instruction,
        unnorm_key=args.unnorm_key,
    )
    adv_action = np.asarray(adv_decoded.action, dtype=np.float32)
    clean_tokens = context.runner.action_token_ids_from_gen(
        context.clean_gen,
        int(context.model.get_action_dim(args.unnorm_key)),
    )
    adv_tokens = [int(x) for x in adv_decoded.token_ids.tolist()]
    clean_grip_token = clean_tokens[-1] if clean_tokens else ""
    adv_grip_token = adv_tokens[-1] if adv_tokens else ""
    clean_audit = _gripper_audit(attack_result.debug or {}, "clean")
    adv_audit = _gripper_audit(attack_result.debug or {}, "adv")
    clean_action_np = np.asarray(context.clean_action, dtype=np.float32)
    arm_l2 = float(np.linalg.norm(adv_action[:-1] - clean_action_np[:-1]))
    return {
        "frame_path": str(context.frame_path),
        "step_records_path": args.step_records or "",
        "step_idx": args.step_idx if args.step_idx is not None else context.step_row.get("step_idx", ""),
        "instruction": context.instruction,
        "model_path": args.model_path,
        "unnorm_key": args.unnorm_key,
        "objective": args.objective,
        "attack_objective": attack_objective,
        "eps": args.eps,
        "steps": args.steps,
        "clean_action": _json(context.clean_action),
        "adv_action": _json(adv_action),
        "clean_gripper_token": clean_grip_token,
        "adv_gripper_token": adv_grip_token,
        "gripper_token_flipped": str(clean_grip_token != adv_grip_token).lower() if clean_grip_token != "" and adv_grip_token != "" else "",
        "clean_gripper_action": float(clean_action_np[-1]),
        "adv_gripper_action": float(adv_action[-1]),
        "gripper_delta": float(adv_action[-1] - clean_action_np[-1]),
        "arm_l2": arm_l2,
        "target_ce_before": clean_audit["target_ce"],
        "target_ce_after": adv_audit["target_ce"],
        "open_bin_prob_mass_before": clean_audit["open_mass"],
        "open_bin_prob_mass_after": adv_audit["open_mass"],
        "close_bin_prob_mass_before": clean_audit["close_mass"],
        "close_bin_prob_mass_after": adv_audit["close_mass"],
        "open_close_margin_before": clean_audit["margin"],
        "open_close_margin_after": adv_audit["margin"],
        "perturbation_linf": attack_result.observation_perturb_linf,
        "perturbation_l2": attack_result.observation_perturb_l2,
        "clean_decode_runtime_sec": context.clean_decode_runtime_sec,
        "adv_decode_runtime_sec": adv_decoded.runtime_sec,
        "attack_runtime_sec": attack_runtime,
        "model_dtype": adv_decoded.model_dtype,
        "pixel_values_dtype": adv_decoded.pixel_values_dtype,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "error": "",
    }


def run_one_frame(args) -> Dict[str, Any]:
    """Load one diagnostic context and run one TokenPrefixPGD attack."""

    return run_one_frame_attack(prepare_one_frame_context(args), args)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in CSV_FIELDS} for row in rows)


def print_schema() -> None:
    for field in CSV_FIELDS:
        print(field)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_path")
    parser.add_argument("--step_records")
    parser.add_argument("--step_idx", type=int)
    parser.add_argument("--instruction")
    parser.add_argument("--model_path")
    parser.add_argument("--unnorm_key", default="libero_object")
    parser.add_argument("--objective", choices=sorted(OBJECTIVE_MAP), default="target_action_ce")
    parser.add_argument("--eps", type=parse_budget, default=parse_budget("4/255"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--step_size", type=parse_budget, default=parse_budget("1/255"))
    parser.add_argument("--cw_margin", type=float, default=5.0)
    parser.add_argument("--force_open_raw_gripper", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_gpu_device_id", type=int, default=0)
    parser.add_argument("--k_prefix_logits", type=int, default=8)
    parser.add_argument("--openvla_resize_size", type=int, default=224)
    parser.add_argument("--libero_official_preprocess", action="store_true")
    parser.add_argument("--libero_preprocess_backend", choices=["official_pil_lanczos", "tf_jpeg_legacy", "none"], default="official_pil_lanczos")
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--postprocess_gripper", action="store_true")
    parser.add_argument("--random_start", action="store_true")
    parser.add_argument("--allow_physical_gpu0", action="store_true")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-schema", action="store_true")
    args = parser.parse_args()

    if args.print_schema:
        print_schema()
    if args.dry_run or args.print_schema:
        write_rows(Path(args.output_csv), [])
        return 0
    if not args.model_path:
        raise SystemExit("--model_path is required for real one-frame diagnostic")
    if not args.image_path and not args.step_records:
        raise SystemExit("--image_path or --step_records is required for real one-frame diagnostic")
    start = time.time()
    try:
        row = run_one_frame(args)
    except Exception as exc:
        row = {"error": str(exc), "runtime_sec": time.time() - start}
        write_rows(Path(args.output_csv), [row])
        raise
    write_rows(Path(args.output_csv), [row])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
