#!/usr/bin/env python3
"""VIS arm-drift diagnostic harness.

This is a no-rollout diagnostic scaffold. It records the schema needed to
compare gripper effect against arm drift and random same-norm controls. Real
OpenVLA decode integration must be supplied before producing metric rows.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gripper_attack.openvla_redecode import redecode_openvla_action_from_adv_inputs


LOSS_VARIANTS = (
    "gripper_only",
    "gripper_plus_arm_preservation",
    "gripper_plus_full_action_l2",
    "random_same_norm",
)
CSV_FIELDS = [
    "loss_variant",
    "lambda_arm",
    "random_same_norm",
    "eps",
    "steps",
    "gripper_delta",
    "arm_l2",
    "gripper_to_arm_ratio",
    "token_flip",
    "perturbation_linf",
    "perturbation_l2",
    "random_baseline_gripper_delta",
    "random_baseline_arm_l2",
    "runtime_sec",
    "error",
]


def write_schema_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def print_schema() -> None:
    for field in CSV_FIELDS:
        print(field)


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
    parser.add_argument("--frame")
    parser.add_argument("--instruction")
    parser.add_argument("--model_path")
    parser.add_argument("--unnorm_key", default="libero_object")
    parser.add_argument("--lambda_arm", action="append", type=float)
    parser.add_argument("--loss_variant", choices=LOSS_VARIANTS, action="append")
    parser.add_argument("--random_same_norm", action="store_true")
    parser.add_argument("--output_csv")
    parser.add_argument("--dry-run-schema", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-schema", action="store_true")
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
    _ = args.lambda_arm or [0.1, 0.3, 1.0, 3.0]

    required = {"--frame": args.frame, "--instruction": args.instruction, "--model_path": args.model_path}
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"missing required arguments for real diagnostic: {', '.join(missing)}")
    raise RuntimeError(
        "OpenVLA adversarial re-decode helper is implemented, but this arm-drift "
        "harness still needs a real model/frame/attack-result loader. The real path "
        "must call redecode_openvla_action_from_adv_inputs(model, processor, "
        "debug['adv_inputs'], ...); this harness must not fake decoded actions or "
        "use action_adv."
    )


if __name__ == "__main__":
    raise SystemExit(main())
