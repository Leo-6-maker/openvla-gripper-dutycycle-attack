#!/usr/bin/env python3
"""Create the 4-fold x 3-seed x 2-variant viability plan only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_training_protocol import VIABILITY_SEEDS, load_fit_fold_bundle, sha256_file


def build_matrix_plan(fold_root: Path) -> dict:
    manifest = load_fit_fold_bundle(fold_root)
    runs = [
        {"run_id": f"fold{fold_id}_{variant}_seed{seed}", "fold_id": fold_id, "variant": variant, "seed": seed, "train_count": 600, "validation_count": 200, "formal_training_authorized": False, "formal_attack_authorized": False}
        for fold_id in range(4)
        for variant in ("B3_25D", "B3_25D9D")
        for seed in VIABILITY_SEEDS
    ]
    return {"schema": "B3_OFFICIAL_V3_FIT_VIABILITY_MATRIX_V1", "fold_manifest_sha256": sha256_file(fold_root / "SHA256SUMS"), "run_count": len(runs), "runs": runs, "status": "PREPARATION_ONLY", "formal_training_authorized": False, "formal_attack_authorized": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite: {args.output}")
    plan = build_matrix_plan(args.fold_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS_PREPARATION_ONLY", "run_count": plan["run_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
