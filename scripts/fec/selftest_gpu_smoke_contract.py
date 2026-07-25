#!/usr/bin/env python3
"""CPU-only contract tests for the hardened FEC smoke runner."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_runner(repo_root: Path):
    path = repo_root / "scripts/fec/run_gpu_smoke.py"
    spec = importlib.util.spec_from_file_location("fec_smoke_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mock_result(module, cfg, transform: str, seed: int):
    debug = {
        "strict_route": True,
        "allow_fallback": False,
        "fallback_used": False,
        "resolved_adapter_class": "TokenPrefixPGDAttacker",
        "resolved_objective": module.TARGET_OBJECTIVE,
        "target_action_present": True,
        "target_token_id": module.TARGET_TOKEN_ID,
        "target_execution_class": module.TARGET_EXECUTION_CLASS,
        "gradient_transform": transform,
        "gradient_transform_seed": seed,
        "temporal_init": "none",
        "temporal_prev_delta_used": False,
        "pixel_budget_adv_inputs_linf": 0.03,
        "adv_inputs": {"input_ids": np.zeros((1, 3)), "pixel_values": np.zeros((1, 3, 2, 2))},
        "delta_final_sha256": "d" * 64,
        "processor_input_sha256": "p" * 64,
        "num_backwards": 5,
    }
    return SimpleNamespace(debug=debug, x_adv=None, action_adv=None,
                           num_attack_steps=cfg["attack_optimizer"]["num_steps"],
                           observation_perturb_linf=0.03)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    module = load_runner(args.repo_root)
    cfg = module.load_yaml(args.repo_root / "configs/fec_attack_v3.yaml")
    module.validate_base_config(cfg)

    true_cfg = module.effective_config(cfg, "TRUE_T10", rand_direction_seed=123456)
    rand_cfg = module.effective_config(cfg, "RAND_T10", rand_direction_seed=123456)
    random_cfg = module.effective_config(cfg, "RANDOM_TIME_T10", rand_direction_seed=123456)
    assert true_cfg["attack_optimizer"]["objective"] == module.TARGET_OBJECTIVE
    assert true_cfg["attack_optimizer"]["gradient_transform"] == "none"
    assert rand_cfg["attack_optimizer"]["objective"] == module.TARGET_OBJECTIVE
    assert rand_cfg["attack_optimizer"]["gradient_transform"] == "rademacher"
    assert random_cfg["attack_optimizer"]["gradient_transform"] == "none"

    raw_open = np.asarray([0, 0, 0, 0, 0, 0, 0.9961], dtype=np.float32)
    raw_close = np.asarray([0, 0, 0, 0, 0, 0, 0.0039], dtype=np.float32)
    assert module.normalize_and_invert_gripper(raw_open)[-1] == -1.0
    assert module.normalize_and_invert_gripper(raw_close)[-1] == 1.0

    true_result = mock_result(module, true_cfg, "none", 123456)
    rand_result = mock_result(module, rand_cfg, "rademacher", 123456)
    module.validate_attack_result(true_result, arm="TRUE_T10", config=true_cfg)
    module.validate_attack_result(rand_result, arm="RAND_T10", config=rand_cfg)

    print("FEC smoke CPU contract self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
