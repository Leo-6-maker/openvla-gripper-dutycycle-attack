#!/usr/bin/env python3
"""VIS token-flip threshold diagnostic harness.

This script is intentionally diagnostic-only. It does not run rollout and it
must re-decode from TokenPrefixPGD ``debug["adv_inputs"]`` rather than using
``action_adv``. The real OpenVLA decode integration is left as an explicit
integration point; when a model/frame loader is unavailable the script fails
loudly instead of fabricating results.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gripper_attack.openvla_redecode import redecode_openvla_action_from_adv_inputs
from vis_one_frame_loader import prepare_one_frame_context, run_one_frame_attack


OBJECTIVES = ("target_action_ce", "gripper_open_region_ce", "gripper_logit_margin_cw")
CSV_FIELDS = [
    "objective",
    "eps",
    "nominal_eps_float",
    "steps",
    "target_ce_before",
    "target_ce_after",
    "open_bin_prob_mass_before",
    "open_bin_prob_mass_after",
    "close_bin_prob_mass_before",
    "close_bin_prob_mass_after",
    "open_close_margin_before",
    "open_close_margin_after",
    "decoded_clean_gripper_token",
    "decoded_adv_gripper_token",
    "decoded_clean_gripper_action",
    "decoded_adv_gripper_action",
    "gripper_token_flipped",
    "arm_action_l2",
    "perturbation_linf",
    "budget_ratio",
    "budget_ok",
    "perturbation_l2",
    "runtime_sec",
    "model_dtype",
    "pixel_values_dtype",
    "error",
]


def parse_budget(value: str) -> float:
    text = str(value).strip()
    if "/" in text:
        num, den = text.split("/", 1)
        return float(num) / float(den)
    return float(text)


def write_schema_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in CSV_FIELDS} for row in rows)


def budget_fields(eps_text: str, perturbation_linf) -> dict:
    nominal = parse_budget(eps_text)
    try:
        linf = float(perturbation_linf)
    except (TypeError, ValueError):
        return {"nominal_eps_float": nominal, "budget_ratio": "", "budget_ok": ""}
    ratio = linf / nominal if nominal > 0 else float("inf")
    return {
        "nominal_eps_float": nominal,
        "budget_ratio": ratio,
        "budget_ok": str(ratio <= 1.01).lower(),
    }


def print_schema() -> None:
    for field in CSV_FIELDS:
        print(field)


def require_real_model_frame_loader() -> None:
    raise RuntimeError(
        "OpenVLA adversarial re-decode helper is implemented, but this diagnostic "
        "still needs a real model/frame/attack-result loader for one-frame smoke. "
        "The real path must call redecode_openvla_action_from_adv_inputs(model, "
        "processor, debug['adv_inputs'], ...); do not use action_adv and do not "
        "fallback to zeros."
    )


def decode_adv_inputs_for_diagnostic(model, processor, adv_inputs, instruction, unnorm_key):
    """Decode prepared adversarial inputs using the shared OpenVLA helper."""

    return redecode_openvla_action_from_adv_inputs(
        model=model,
        processor=processor,
        adv_inputs=adv_inputs,
        instruction=instruction,
        unnorm_key=unnorm_key,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", help="Frame/observation source for a single diagnostic example")
    parser.add_argument("--instruction", help="Task instruction")
    parser.add_argument("--model_path", help="OpenVLA model path")
    parser.add_argument("--unnorm_key", default="libero_object")
    parser.add_argument("--step_records")
    parser.add_argument("--step_idx", type=int)
    parser.add_argument("--step_size", default="1/255")
    parser.add_argument("--model_gpu_device_id", type=int, default=0)
    parser.add_argument("--openvla_resize_size", type=int, default=224)
    parser.add_argument("--libero_official_preprocess", action="store_true")
    parser.add_argument("--libero_preprocess_backend", choices=["official_pil_lanczos", "tf_jpeg_legacy", "none"], default="official_pil_lanczos")
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--postprocess_gripper", action="store_true")
    parser.add_argument("--force_open_raw_gripper", type=float, default=1.0)
    parser.add_argument("--cw_margin", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_physical_gpu0", action="store_true")
    parser.add_argument("--objective", choices=OBJECTIVES, action="append")
    parser.add_argument("--eps", action="append")
    parser.add_argument("--steps", action="append", type=int)
    parser.add_argument("--output_csv")
    parser.add_argument("--dry-run-schema", action="store_true", help="Only write the output CSV header")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --dry-run-schema")
    parser.add_argument("--print-schema", action="store_true", help="Print output fields and exit")
    args = parser.parse_args()

    if args.print_schema:
        print_schema()
    if args.dry_run_schema or args.dry_run or args.print_schema:
        if args.output_csv:
            write_schema_csv(Path(args.output_csv))
        return 0

    if not args.output_csv:
        raise SystemExit("--output_csv is required for real diagnostics")
    output_csv = Path(args.output_csv)

    required = {"--model_path": args.model_path}
    if not args.frame and not args.step_records:
        required["--frame or --step_records"] = None
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"missing required arguments for real diagnostic: {', '.join(missing)}")

    eps_values = args.eps or ["4/255", "8/255", "12/255", "16/255"]
    steps_values = args.steps or [10, 20, 40]
    objectives = args.objective or list(OBJECTIVES)
    context = prepare_one_frame_context(args)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for objective in objectives:
        for eps_text in eps_values:
            for steps in steps_values:
                start = time.time()
                try:
                    loader_args = argparse.Namespace(
                        image_path=args.frame,
                        step_records=args.step_records,
                        step_idx=args.step_idx,
                        instruction=args.instruction,
                        model_path=args.model_path,
                        unnorm_key=args.unnorm_key,
                        objective=objective,
                        eps=parse_budget(eps_text),
                        steps=int(steps),
                        step_size=parse_budget(args.step_size),
                        cw_margin=args.cw_margin,
                        force_open_raw_gripper=args.force_open_raw_gripper,
                        seed=args.seed,
                        model_gpu_device_id=args.model_gpu_device_id,
                        k_prefix_logits=8,
                        openvla_resize_size=args.openvla_resize_size,
                        libero_official_preprocess=args.libero_official_preprocess,
                        libero_preprocess_backend=args.libero_preprocess_backend,
                        center_crop=args.center_crop,
                        postprocess_gripper=args.postprocess_gripper,
                        random_start=False,
                        allow_physical_gpu0=args.allow_physical_gpu0,
                        output_csv=str(output_csv),
                    )
                    result = run_one_frame_attack(context, loader_args)
                    row = {
                        "objective": objective,
                        "eps": eps_text,
                        "steps": steps,
                        "target_ce_before": result.get("target_ce_before", ""),
                        "target_ce_after": result.get("target_ce_after", ""),
                        "open_bin_prob_mass_before": result.get("open_bin_prob_mass_before", ""),
                        "open_bin_prob_mass_after": result.get("open_bin_prob_mass_after", ""),
                        "close_bin_prob_mass_before": result.get("close_bin_prob_mass_before", ""),
                        "close_bin_prob_mass_after": result.get("close_bin_prob_mass_after", ""),
                        "open_close_margin_before": result.get("open_close_margin_before", ""),
                        "open_close_margin_after": result.get("open_close_margin_after", ""),
                        "decoded_clean_gripper_token": result.get("clean_gripper_token", ""),
                        "decoded_adv_gripper_token": result.get("adv_gripper_token", ""),
                        "decoded_clean_gripper_action": result.get("clean_gripper_action", ""),
                        "decoded_adv_gripper_action": result.get("adv_gripper_action", ""),
                        "gripper_token_flipped": result.get("gripper_token_flipped", ""),
                        "arm_action_l2": result.get("arm_l2", ""),
                        "perturbation_linf": result.get("perturbation_linf", ""),
                        "perturbation_l2": result.get("perturbation_l2", ""),
                        "runtime_sec": float(result.get("attack_runtime_sec", 0.0) or 0.0) + float(result.get("adv_decode_runtime_sec", 0.0) or 0.0),
                        "model_dtype": result.get("model_dtype", ""),
                        "pixel_values_dtype": result.get("pixel_values_dtype", ""),
                        "error": "",
                    }
                    row.update(budget_fields(eps_text, row["perturbation_linf"]))
                    rows.append(row)
                    write_rows(output_csv, rows)
                except Exception as exc:
                    row = {
                        "objective": objective,
                        "eps": eps_text,
                        "nominal_eps_float": parse_budget(eps_text),
                        "steps": steps,
                        "runtime_sec": f"{time.time() - start:.6f}",
                        "error": str(exc),
                    }
                    rows.append(row)
                    write_rows(output_csv, rows)
                    raise
    write_rows(output_csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
