#!/usr/bin/env python3
"""VIS token-flip threshold diagnostic harness.

This script is intentionally diagnostic-only. It does not run rollout and it
must re-decode from TokenPrefixPGD ``debug["adv_inputs"]`` rather than using
``action_adv``. The real OpenVLA decode integration is left as an explicit
integration point; when it is unavailable the script fails loudly instead of
fabricating results.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


OBJECTIVES = ("target_action_ce", "gripper_open_region_ce", "gripper_logit_margin_cw")
CSV_FIELDS = [
    "objective",
    "eps",
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


def print_schema() -> None:
    for field in CSV_FIELDS:
        print(field)


def require_real_decode_path() -> None:
    raise RuntimeError(
        "OpenVLA adversarial re-decode integration is not wired in this harness. "
        "Provide a real decoder that consumes debug['adv_inputs']; do not use action_adv "
        "and do not fallback to zeros."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", help="Frame/observation source for a single diagnostic example")
    parser.add_argument("--instruction", help="Task instruction")
    parser.add_argument("--model_path", help="OpenVLA model path")
    parser.add_argument("--unnorm_key", default="libero_object")
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

    required = {"--frame": args.frame, "--instruction": args.instruction, "--model_path": args.model_path}
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"missing required arguments for real diagnostic: {', '.join(missing)}")

    eps_values = args.eps or ["4/255", "8/255", "12/255", "16/255"]
    steps_values = args.steps or [10, 20, 40]
    _ = [parse_budget(v) for v in eps_values]
    _ = steps_values
    _ = args.objective or list(OBJECTIVES)
    start = time.time()
    try:
        require_real_decode_path()
    except Exception as exc:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerow({
                "runtime_sec": f"{time.time() - start:.6f}",
                "error": str(exc),
            })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
