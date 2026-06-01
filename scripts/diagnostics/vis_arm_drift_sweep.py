#!/usr/bin/env python3
"""VIS arm-drift diagnostic harness.

This is a no-rollout diagnostic. It compares the best gripper-targeted
TokenPrefixPGD configuration against a random same-Linf perturbation on the same
saved frame. It never uses ``action_adv`` and always re-decodes OpenVLA actions
from processor inputs.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gripper_attack.attack_adapter import TokenPrefixPGDAttacker, get_adv_inputs_from_attack_result  # noqa: E402
from gripper_attack.openvla_redecode import redecode_openvla_action_from_adv_inputs  # noqa: E402
from vis_one_frame_loader import OBJECTIVE_MAP, parse_budget, prepare_one_frame_context, run_one_frame_attack  # noqa: E402


LOSS_VARIANTS = (
    "gripper_only",
    "random_same_norm",
)
CSV_FIELDS = [
    "loss_variant",
    "objective",
    "eps",
    "steps",
    "seed",
    "clean_gripper_action",
    "adv_gripper_action",
    "gripper_delta",
    "arm_l2",
    "gripper_to_arm_ratio",
    "clean_gripper_token",
    "adv_gripper_token",
    "token_flip",
    "perturbation_linf",
    "perturbation_l2",
    "target_ce_before",
    "target_ce_after",
    "open_bin_prob_mass_before",
    "open_bin_prob_mass_after",
    "close_bin_prob_mass_before",
    "close_bin_prob_mass_after",
    "random_baseline_gripper_delta",
    "random_baseline_arm_l2",
    "random_baseline_token_flip",
    "runtime_sec",
    "model_dtype",
    "pixel_values_dtype",
    "error",
]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in CSV_FIELDS} for row in rows)


def print_schema() -> None:
    for field in CSV_FIELDS:
        print(field)


def _json_action_to_array(value: Any) -> np.ndarray:
    if isinstance(value, str):
        import json

        return np.asarray(json.loads(value), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _ratio(gripper_delta: float, arm_l2: float) -> float:
    return float(abs(gripper_delta) / max(float(arm_l2), 1e-6))


def _build_clean_inputs(context, args) -> Dict[str, torch.Tensor]:
    target_action = np.asarray(context.clean_action, dtype=np.float32).copy()
    target_action[-1] = float(args.force_open_raw_gripper)
    attacker = TokenPrefixPGDAttacker(
        context.model,
        context.processor,
        {
            "attack_optimizer": {
                "method": "token_prefix_pgd",
                "objective": OBJECTIVE_MAP[args.objective],
                "epsilon": float(args.eps),
                "step_size": float(args.step_size),
                "num_steps": int(args.steps),
            }
        },
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
    target_ids = attacker.action_to_token_ids(target_action, args.unnorm_key)
    clean_ids, _full_ids, _labels, x0 = attacker._build_inputs_and_labels(context.image, context.instruction, target_ids)
    return {"input_ids": clean_ids.detach(), "pixel_values": x0.detach()}


def _random_same_linf_inputs(context, args, perturbation_linf: float) -> Dict[str, torch.Tensor]:
    clean_inputs = _build_clean_inputs(context, args)
    x0 = clean_inputs["pixel_values"]
    gen = torch.Generator(device=x0.device)
    gen.manual_seed(int(args.random_seed))
    rand = torch.empty_like(x0, dtype=torch.float32).uniform_(-1.0, 1.0, generator=gen)
    rand = torch.sign(rand)
    delta = rand * float(perturbation_linf)
    adv_float = x0.float() + delta
    adv_model = adv_float.to(dtype=x0.dtype)
    over = (adv_model.float() - x0.float()).abs() > (float(perturbation_linf) + 1e-7)
    if bool(torch.any(over).detach().cpu()):
        adv_model = torch.where(over, x0, adv_model)
    return {"input_ids": clean_inputs["input_ids"], "pixel_values": adv_model.detach()}


def _decode_row(context, args, adv_inputs: Dict[str, torch.Tensor], *, loss_variant: str, perturbation_linf: float, perturbation_l2: float) -> Dict[str, Any]:
    t0 = time.time()
    decoded = redecode_openvla_action_from_adv_inputs(
        model=context.model,
        processor=context.processor,
        adv_inputs=adv_inputs,
        instruction=context.instruction,
        unnorm_key=args.unnorm_key,
    )
    runtime = time.time() - t0
    clean_action = np.asarray(context.clean_action, dtype=np.float32)
    adv_action = np.asarray(decoded.action, dtype=np.float32)
    clean_tokens = context.runner.action_token_ids_from_gen(
        context.clean_gen,
        int(context.model.get_action_dim(args.unnorm_key)),
    )
    adv_tokens = [int(x) for x in decoded.token_ids.tolist()]
    clean_grip_token = clean_tokens[-1] if clean_tokens else ""
    adv_grip_token = adv_tokens[-1] if adv_tokens else ""
    gripper_delta = float(adv_action[-1] - clean_action[-1])
    arm_l2 = float(np.linalg.norm(adv_action[:-1] - clean_action[:-1]))
    return {
        "loss_variant": loss_variant,
        "objective": args.objective,
        "eps": args.eps,
        "steps": args.steps,
        "seed": args.seed,
        "clean_gripper_action": float(clean_action[-1]),
        "adv_gripper_action": float(adv_action[-1]),
        "gripper_delta": gripper_delta,
        "arm_l2": arm_l2,
        "gripper_to_arm_ratio": _ratio(gripper_delta, arm_l2),
        "clean_gripper_token": clean_grip_token,
        "adv_gripper_token": adv_grip_token,
        "token_flip": str(clean_grip_token != adv_grip_token).lower() if clean_grip_token != "" and adv_grip_token != "" else "",
        "perturbation_linf": perturbation_linf,
        "perturbation_l2": perturbation_l2,
        "runtime_sec": runtime,
        "model_dtype": decoded.model_dtype,
        "pixel_values_dtype": decoded.pixel_values_dtype,
    }


def run_arm_drift(args) -> list[dict]:
    context = prepare_one_frame_context(args)
    rows: list[dict] = []
    t0 = time.time()
    gripper_row = run_one_frame_attack(context, args)
    gripper_delta = float(gripper_row["gripper_delta"])
    arm_l2 = float(gripper_row["arm_l2"])
    targeted = {
        "loss_variant": "gripper_only",
        "objective": gripper_row["objective"],
        "eps": gripper_row["eps"],
        "steps": gripper_row["steps"],
        "seed": args.seed,
        "clean_gripper_action": gripper_row["clean_gripper_action"],
        "adv_gripper_action": gripper_row["adv_gripper_action"],
        "gripper_delta": gripper_delta,
        "arm_l2": arm_l2,
        "gripper_to_arm_ratio": _ratio(gripper_delta, arm_l2),
        "clean_gripper_token": gripper_row["clean_gripper_token"],
        "adv_gripper_token": gripper_row["adv_gripper_token"],
        "token_flip": gripper_row["gripper_token_flipped"],
        "perturbation_linf": gripper_row["perturbation_linf"],
        "perturbation_l2": gripper_row["perturbation_l2"],
        "target_ce_before": gripper_row["target_ce_before"],
        "target_ce_after": gripper_row["target_ce_after"],
        "open_bin_prob_mass_before": gripper_row["open_bin_prob_mass_before"],
        "open_bin_prob_mass_after": gripper_row["open_bin_prob_mass_after"],
        "close_bin_prob_mass_before": gripper_row["close_bin_prob_mass_before"],
        "close_bin_prob_mass_after": gripper_row["close_bin_prob_mass_after"],
        "runtime_sec": float(gripper_row["attack_runtime_sec"]) + float(gripper_row["adv_decode_runtime_sec"]),
        "model_dtype": gripper_row["model_dtype"],
        "pixel_values_dtype": gripper_row["pixel_values_dtype"],
    }
    rows.append(targeted)

    random_inputs = _random_same_linf_inputs(context, args, float(gripper_row["perturbation_linf"]))
    clean_inputs = _build_clean_inputs(context, args)
    diff = (random_inputs["pixel_values"].float() - clean_inputs["pixel_values"].float()).detach()
    random_row = _decode_row(
        context,
        args,
        random_inputs,
        loss_variant="random_same_norm",
        perturbation_linf=float(diff.abs().max().cpu()) if diff.numel() else 0.0,
        perturbation_l2=float(torch.linalg.vector_norm(diff.reshape(-1)).cpu()) if diff.numel() else 0.0,
    )
    random_row["runtime_sec"] = float(random_row["runtime_sec"]) + (time.time() - t0)
    rows.append(random_row)

    targeted["random_baseline_gripper_delta"] = random_row["gripper_delta"]
    targeted["random_baseline_arm_l2"] = random_row["arm_l2"]
    targeted["random_baseline_token_flip"] = random_row["token_flip"]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame")
    parser.add_argument("--image_path")
    parser.add_argument("--step_records")
    parser.add_argument("--step_idx", type=int)
    parser.add_argument("--instruction")
    parser.add_argument("--model_path")
    parser.add_argument("--unnorm_key", default="libero_object")
    parser.add_argument("--objective", choices=sorted(OBJECTIVE_MAP), default="target_action_ce")
    parser.add_argument("--eps", type=parse_budget, default=parse_budget("4/255"))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--step_size", type=parse_budget, default=parse_budget("1/255"))
    parser.add_argument("--cw_margin", type=float, default=5.0)
    parser.add_argument("--force_open_raw_gripper", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--model_gpu_device_id", type=int, default=0)
    parser.add_argument("--k_prefix_logits", type=int, default=8)
    parser.add_argument("--openvla_resize_size", type=int, default=224)
    parser.add_argument("--libero_official_preprocess", action="store_true")
    parser.add_argument("--libero_preprocess_backend", choices=["official_pil_lanczos", "tf_jpeg_legacy", "none"], default="official_pil_lanczos")
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--postprocess_gripper", action="store_true")
    parser.add_argument("--random_start", action="store_true")
    parser.add_argument("--allow_physical_gpu0", action="store_true")
    parser.add_argument("--loss_variant", choices=LOSS_VARIANTS, action="append")
    parser.add_argument("--output_csv")
    parser.add_argument("--dry-run-schema", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-schema", action="store_true")
    args = parser.parse_args()

    if args.print_schema:
        print_schema()
    if args.dry_run_schema or args.dry_run or args.print_schema:
        if args.output_csv:
            write_rows(Path(args.output_csv), [])
        return 0
    if not args.output_csv:
        raise SystemExit("--output_csv is required for real diagnostics")
    if not args.model_path:
        raise SystemExit("--model_path is required for real diagnostics")
    if not args.frame and not args.image_path and not args.step_records:
        raise SystemExit("--frame, --image_path, or --step_records is required for real diagnostics")

    rows: list[dict] = []
    try:
        rows = run_arm_drift(args)
    except Exception as exc:
        rows.append({"error": str(exc)})
        write_rows(Path(args.output_csv), rows)
        raise
    write_rows(Path(args.output_csv), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
