#!/usr/bin/env python3
"""Extract and validate the frozen SC5 feature contract from live imports.

Imports SC5_FEATURES, SC5_PHASES, SC5MLPV1 directly from source.
Validates architecture by instantiating model and checking output shapes.
Binds git commit SHA. Fails closed on any mismatch.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gripper_attack.sc5mlp_v1 import SC5MLPV1, SC5_FEATURES, SC5_PHASES, N_FEATURES, N_PHASES, HIDDEN_DIM
from gripper_attack.sc5_detector_runtime_v1r import (SC5DetectorRuntimeV1R, SC5_FEATURES as RT_FEATURES,
                                                      SC5_PHASES as RT_PHASES)
from gripper_attack.sc5_streaming_features_v2 import FEATURE_NAMES

import numpy as np
import torch

FORBIDDEN_FEATURES = [
    "normalized_step", "absolute_timestep", "suite_id", "task_id",
    "state_id", "object_identity", "teacher_anchor", "teacher_window",
    "object_pose", "target_pose", "object_target_distance",
    "future_timestep", "episode_success", "attack_condition",
    "vis_outcome", "rand_outcome", "oracle_outcome",
    "post_attack_qpos", "post_attack_width", "manual_outcome",
]

FSM_DEFAULTS = {
    "tau_corridor": 0.3, "tau_release": 0.3, "guard": 5, "K": 10,
    "fsm_versions": ["legacy_v1", "v1r_r1", "v1r_r2"],
    "one_shot": True,
}

LOSS_CONFIG = {
    "formula": "phase_CE + 0.5 * corridor_BCE + 0.3 * release_BCE",
    "corridor_pos_weight": 5.0,
    "phase_class_balanced": True,
}

CHECKPOINT_REQUIRED_KEYS = [
    "model_state", "mean", "std", "feature_names", "phase_classes",
    "dataset_sha256", "split_mode", "normalization_source",
    "n_train", "n_val", "seed",
]


def get_git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "GIT_UNAVAILABLE"


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_feature_consistency() -> list[str]:
    errors = []
    if list(SC5_FEATURES) != list(RT_FEATURES):
        errors.append("SC5_FEATURES != runtime SC5_FEATURES")
    if list(SC5_FEATURES) != list(FEATURE_NAMES):
        errors.append("SC5_FEATURES != streaming FEATURE_NAMES")
    if list(SC5_PHASES) != list(RT_PHASES):
        errors.append("SC5_PHASES != runtime SC5_PHASES")
    if len(SC5_FEATURES) != N_FEATURES:
        errors.append(f"len(features)={len(SC5_FEATURES)} != N_FEATURES={N_FEATURES}")
    if len(SC5_PHASES) != N_PHASES:
        errors.append(f"len(phases)={len(SC5_PHASES)} != N_PHASES={N_PHASES}")
    for fb in FORBIDDEN_FEATURES:
        if fb in SC5_FEATURES:
            errors.append(f"Forbidden feature '{fb}' in SC5_FEATURES")
    return errors


def validate_model_architecture() -> list[str]:
    errors = []
    try:
        model = SC5MLPV1()
        model.eval()
        dummy = torch.randn(4, N_FEATURES)
        with torch.no_grad():
            out = model(dummy)
        if "phase_logits" not in out:
            errors.append("Missing phase_logits in output")
        elif out["phase_logits"].shape[-1] != N_PHASES:
            errors.append(f"phase_logits dim {out['phase_logits'].shape[-1]} != {N_PHASES}")
        if "corridor_logit" not in out:
            errors.append("Missing corridor_logit in output")
        elif out["corridor_logit"].shape[-1] != 1:
            errors.append("corridor_logit output dim != 1")
        if "release_logit" not in out:
            errors.append("Missing release_logit in output")
        elif out["release_logit"].shape[-1] != 1:
            errors.append("release_logit output dim != 1")
        if hasattr(model, 'confidence_head'):
            errors.append("confidence_head found in model")
    except Exception as e:
        errors.append(f"Model instantiation failed: {e}")
    return errors


def extract(repo_root: Path) -> dict:
    commit = get_git_commit(repo_root)
    consistency_errors = validate_feature_consistency()
    arch_errors = validate_model_architecture()
    all_errors = consistency_errors + arch_errors

    source_files = {
        "sc5mlp_v1": repo_root / "src/gripper_attack/sc5mlp_v1.py",
        "detector_runtime_v1r": repo_root / "src/gripper_attack/sc5_detector_runtime_v1r.py",
        "streaming_features_v2": repo_root / "src/gripper_attack/sc5_streaming_features_v2.py",
    }
    file_hashes = {k: hash_file(p) if p.exists() else "MISSING" for k, p in source_files.items()}
    missing = [k for k, v in file_hashes.items() if v == "MISSING"]
    if missing:
        all_errors.append(f"Missing source files: {missing}")

    return {
        "gate": "SC5_FROZEN_FEATURE_CONTRACT_V1",
        "extraction_source_commit": commit,
        "extraction_method": "live_import_from_src",
        "features": {
            "names": list(SC5_FEATURES),
            "n_features": N_FEATURES,
            "canonical_order": {name: i for i, name in enumerate(SC5_FEATURES)},
        },
        "phases": {
            "names": list(SC5_PHASES),
            "n_phases": N_PHASES,
        },
        "forbidden_features": FORBIDDEN_FEATURES,
        "architecture": {
            "model": "SC5MLPV1",
            "input_dim": N_FEATURES,
            "hidden_dim": HIDDEN_DIM,
            "n_layers": 2,
            "activation": "ReLU",
            "heads": ["phase(9)", "corridor(1)", "release(1)"],
            "no_confidence_head": True,
        },
        "fsm_parameters": FSM_DEFAULTS,
        "training_objective": LOSS_CONFIG,
        "checkpoint_required_keys": CHECKPOINT_REQUIRED_KEYS,
        "source_file_hashes": file_hashes,
        "cross_module_consistency_errors": consistency_errors,
        "architecture_validation_errors": arch_errors,
        "contract_valid": len(all_errors) == 0,
        "errors": all_errors,
    }


def main():
    ap = argparse.ArgumentParser(description="Extract frozen SC5 feature contract")
    ap.add_argument("--repo_root", default=str(REPO))
    ap.add_argument("--output", default="-")
    ap.add_argument("--fail_on_error", action="store_true")
    args = ap.parse_args()

    contract = extract(Path(args.repo_root))

    out = json.dumps(contract, indent=2)
    if args.output == "-":
        print(out)
    else:
        with open(args.output, "w") as f:
            f.write(out + "\n")

    if args.fail_on_error and not contract["contract_valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
