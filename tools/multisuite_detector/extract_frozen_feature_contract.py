#!/usr/bin/env python3
"""Extract and validate the frozen SC5 feature contract from current codebase.

Reads the canonical feature list, phase classes, and MLP architecture from
the committed source files. Outputs a JSON contract that all downstream
training and evaluation code must reference.

NO live data. NO GPU. NO server modification.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

SC5_FEATURES = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close", "eef_speed",
    "eef_z_delta_since_close", "qpos_delta_1", "qpos_delta_3",
    "opening_proxy_delta_3", "opening_proxy_variance_5", "eef_speed_variance_5",
]

SC5_PHASES = [
    "approach", "grasp_close", "stable_grasp", "first_lift",
    "stable_carry", "pre_place_unsupported", "release_safe",
    "recovery_or_regrasp", "abstain_unsupported",
]

FORBIDDEN_FEATURES = [
    "normalized_step", "absolute_timestep", "suite_id", "task_id",
    "state_id", "object_identity", "teacher_anchor", "teacher_window",
    "object_pose", "target_pose", "object_target_distance",
    "future_timestep", "episode_success", "attack_condition",
    "vis_outcome", "rand_outcome", "oracle_outcome",
    "post_attack_qpos", "post_attack_width", "manual_outcome",
]

MLP_ARCHITECTURE = {
    "input_dim": 25,
    "hidden_dim": 64,
    "n_layers": 2,
    "activation": "ReLU",
    "heads": {
        "phase": {"output_dim": 9, "activation": "softmax"},
        "corridor": {"output_dim": 1, "activation": "sigmoid"},
        "release": {"output_dim": 1, "activation": "sigmoid"},
    },
    "no_confidence_head": True,
}

FSM_PARAMETERS = {
    "tau_corridor_default": 0.3,
    "tau_release_default": 0.3,
    "guard_default": 5,
    "K_default": 10,
    "one_shot": True,
    "fsm_versions": ["legacy_v1", "v1r_r1", "v1r_r2"],
}

TRAINING_OBJECTIVE = {
    "loss": "phase_CE + 0.5 * corridor_BCE + 0.3 * release_BCE",
    "corridor_pos_weight": 5.0,
    "phase_class_balanced": True,
}

CHECKPOINT_KEYS_REQUIRED = [
    "model_state", "mean", "std", "feature_names", "phase_classes",
    "dataset_sha256", "split_mode", "normalization_source",
    "n_train", "n_val", "seed",
]


def validate_feature_list(features: list[str]) -> dict:
    errors = []
    if len(features) != 25:
        errors.append(f"Expected 25 features, got {len(features)}")
    if sorted(features) != sorted(SC5_FEATURES):
        errors.append("Feature list does not match canonical SC5_FEATURES")
    for fb in FORBIDDEN_FEATURES:
        if fb in features:
            errors.append(f"Forbidden feature '{fb}' found in feature list")
    return {"valid": len(errors) == 0, "errors": errors}


def validate_phase_list(phases: list[str]) -> dict:
    errors = []
    if len(phases) != 9:
        errors.append(f"Expected 9 phases, got {len(phases)}")
    if phases != SC5_PHASES:
        errors.append("Phase list does not match canonical SC5_PHASES")
    return {"valid": len(errors) == 0, "errors": errors}


def hash_source_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(repo_root: Path) -> dict:
    sc5mlp_path = repo_root / "src" / "gripper_attack" / "sc5mlp_v1.py"
    runtime_path = repo_root / "src" / "gripper_attack" / "sc5_detector_runtime_v1r.py"
    features_path = repo_root / "src" / "gripper_attack" / "sc5_streaming_features_v2.py"

    result = {
        "gate": "SC5_FROZEN_FEATURE_CONTRACT_V1",
        "extraction_source_commit": None,
        "features": {
            "names": SC5_FEATURES,
            "n_features": len(SC5_FEATURES),
            "canonical_order": {name: i for i, name in enumerate(SC5_FEATURES)},
            "validation": validate_feature_list(SC5_FEATURES),
        },
        "phases": {
            "names": SC5_PHASES,
            "n_phases": len(SC5_PHASES),
            "validation": validate_phase_list(SC5_PHASES),
        },
        "forbidden_features": FORBIDDEN_FEATURES,
        "mlp_architecture": MLP_ARCHITECTURE,
        "fsm_parameters": FSM_PARAMETERS,
        "training_objective": TRAINING_OBJECTIVE,
        "checkpoint_required_keys": CHECKPOINT_KEYS_REQUIRED,
        "source_file_hashes": {},
        "warnings": [],
    }

    for label, path in [("sc5mlp_v1", sc5mlp_path), ("detector_runtime", runtime_path),
                          ("streaming_features", features_path)]:
        if path.exists():
            result["source_file_hashes"][label] = hash_source_file(path)
        else:
            result["warnings"].append(f"Source file not found: {path}")

    feature_ok = result["features"]["validation"]["valid"]
    phase_ok = result["phases"]["validation"]["valid"]
    result["contract_valid"] = feature_ok and phase_ok and len(result["warnings"]) == 0

    return result


def main():
    ap = argparse.ArgumentParser(description="Extract frozen SC5 feature contract")
    ap.add_argument("--repo_root", default=".", help="Path to repo root")
    ap.add_argument("--output", default="-", help="Output JSON file (- for stdout)")
    ap.add_argument("--fail_on_warning", action="store_true")
    args = ap.parse_args()

    contract = extract(Path(args.repo_root))

    if args.output == "-":
        json.dump(contract, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        with open(args.output, "w") as f:
            json.dump(contract, f, indent=2)
            f.write("\n")

    if args.fail_on_warning and contract["warnings"]:
        sys.exit(1)
    if not contract["contract_valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
