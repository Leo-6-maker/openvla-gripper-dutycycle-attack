"""CPU-only checkpoint-local semantic audit for the frozen X1R target token."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.execution_target import classify_execution_token
from scripts.stage_x.audit_stage_x1r_t1_native_token_authority import load_native_suite


MODEL_PATHS = {
    "libero_10": "/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10",
    "libero_goal": "/mnt/sdc/dty_user/openvla_attack/models/libero-goal",
    "libero_object": "/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object",
    "libero_spatial": "/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    errors = []
    for suite, model_path in MODEL_PATHS.items():
        info = load_native_suite(suite, Path(model_path))
        kwargs = {
            "vocab_eff": int(info["vocab_eff"]),
            "n_bins": len(info["bin_centers"]),
            "bin_centers": info["bin_centers"],
            "action_stats": info["stats"],
        }
        target = classify_execution_token(31745, **kwargs)
        endpoint = classify_execution_token(31744, **kwargs)
        row = {
            "suite": suite,
            "model_path": model_path,
            "target_token_id": 31745,
            "target_execution_class": target.execution_class,
            "target_decoded_raw_gripper": target.decoded_raw_gripper,
            "target_executed_env_gripper": target.executed_env_gripper,
            "endpoint_compatibility_token_id": 31744,
            "endpoint_compatibility_execution_class": endpoint.execution_class,
            "native_authority_open_token_id": int(info["tokenizer_vocab_size"] - np.digitize(1.0, info["bins"])),
            "tokenizer_vocab_size": int(info["tokenizer_vocab_size"]),
            "n_action_bins": int(info["native"].n_bins),
        }
        rows.append(row)
        if target.execution_class != "NATIVE_OPEN":
            errors.append(f"{suite}:31745:{target.execution_class}")
        if endpoint.execution_class == "NATIVE_OPEN":
            errors.append(f"{suite}:31744 unexpectedly NATIVE_OPEN")

    report = {
        "schema": "STAGE_X_X1R_TARGET_TOKEN_CPU_SEMANTICS_V1",
        "status": "PASS_31745_NATIVE_OPEN" if not errors else "HOLD_TARGET_TOKEN_SEMANTICS",
        "official_environment": "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800",
        "rows": rows,
        "errors": errors,
        "model_inference": False,
        "env_step": False,
        "pgd_calls": 0,
        "physical_interventions": 0,
        "vphys_reads": 0,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "errors": errors}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
