"""Canonical task identity mapping for experiment tasks.

Separates runner_task_id (passed to --task_id) from semantic_task_name
(used in run_ids, reports, and Table1 cross-reference joins).

Usage:
    from src.utils.task_identity import TASK_IDENTITY, RUNNER_TASK_ID, RUN_ID_TASK_KEY

    runner_cmd = ["--task_id", TASK_IDENTITY["runner_task_id"]]
    run_id = f"{TASK_IDENTITY['semantic_task_name']}_s{state}_r{repeat}_{condition}"
"""

# --- Black-bowl-to-plate spatial task (the primary bowl-on-plate experiment) ---
BOWL_ON_PLATE_SPATIAL = {
    "runner_task_id": "libero_spatial_black_bowl",
    "semantic_task_name": "goal_put_the_bowl_on_the_plate",
    "suite": "libero_spatial",
    "suite_family": "LIBERO-Spatial",
    "libero_official_task_name": (
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
    ),
    "short_label": "black_bowl_to_plate",
    "claim_task_label": "LIBERO Spatial black-bowl-to-plate",
    "is_black_bowl_related": True,
    "is_non_black_bowl_claim": False,
    "target_object": "akita_black_bowl_1",
    "target_receptacle": "plate_1",
    "config_file": "configs/v4_tasks_libero.yaml",
    "full4_config_file": "configs/v4_tasks_libero_full4_20260518.yaml",
    "table1_task_key": "goal_put_the_bowl_on_the_plate",
    "mechanism_type": "pick_place",
    "grasp_type": "gripper_release",
}

# Convenience aliases for the primary experiment task
TASK_IDENTITY = BOWL_ON_PLATE_SPATIAL

# Shorthand for runner --task_id argument
RUNNER_TASK_ID = TASK_IDENTITY["runner_task_id"]

# Shorthand for run_id construction and Table1 manifest join
RUN_ID_TASK_KEY = TASK_IDENTITY["semantic_task_name"]

# Shorthand for Table1 full4 manifest matching
TABLE1_TASK_KEY = TASK_IDENTITY["table1_task_key"]


# --- Condition configs have moved to src/utils/condition_protocols.py ---
# The MATCHED_CONDITIONS previously defined here did NOT match Codex r0/r1/r2 protocol.
# See LEGACY_CODEX_STATE5_MATCHED_CONDITIONS in condition_protocols.py for the correct config.
# Import for backward compatibility (deprecated, do not use for new experiments):
from src.utils.condition_protocols import (
    CLEAN_DETECT_PROTOCOL,
    LEGACY_CODEX_STATE5_PROTOCOL,
    LEGACY_CODEX_STATE5_MATCHED_CONDITIONS,
    COMMAND_OPEN_ORACLE_PROTOCOL,
    DIAGNOSTIC_GRIPPER_MARGIN_PROTOCOL,
    DIAGNOSTIC_OPEN_REGION_CE_PROTOCOL,
)

# DEPRECATED: old MATCHED_CONDITIONS had wrong rho/objective/eps values.
# Kept only for reference — do NOT use for confirmatory experiments.
DEPRECATED_DEEPSEEK_DRIFT_MATCHED_CONDITIONS = [
    {
        "condition_name": "random_same_autowindow",
        "attack_objective": "random_noise",
        "temporal_init": "prev_delta",
        "force_open_raw_gripper": None,
        "rho": 0.0,
        "cw_margin": None,
        "epsilon": 0.10,
        "step_size": 0.020,
        "attack_steps": 0,
        "is_attack": False,
        "is_control": True,
        "deprecated": True,
        "deprecation_reason": "rho=0 disables attack; Codex used rho=1.0 with actual perturbation (linf=0.10)",
    },
    {
        "condition_name": "VIS_current_same_autowindow",
        "attack_objective": "vis_current",
        "temporal_init": "prev_delta",
        "force_open_raw_gripper": None,
        "rho": 0.0,
        "cw_margin": None,
        "epsilon": 0.10,
        "step_size": 0.020,
        "attack_steps": 0,
        "is_attack": False,
        "is_control": True,
        "deprecated": True,
        "deprecation_reason": "rho=0 disables attack; Codex used rho=1.0 with vis_current objective (linf=2.12)",
    },
    {
        "condition_name": "best_VIS_gripper_targeted_stronger_same_autowindow",
        "attack_objective": "gripper_logit_margin_cw",
        "temporal_init": "prev_delta",
        "force_open_raw_gripper": None,
        "rho": 1.0,
        "cw_margin": 5.0,
        "epsilon": 0.10,
        "step_size": 0.020,
        "attack_steps": 20,
        "is_attack": True,
        "is_control": False,
        "deprecated": True,
        "deprecation_reason": "Codex used force_gripper_open_token_ce (not margin), eps=0.25 (not 0.10), asteps=60 (not 20), fo=1.0",
    },
    {
        "condition_name": "command_open_0.75_same_autowindow",
        "attack_objective": "oracle_env_gripper_open",
        "temporal_init": "none",
        "force_open_raw_gripper": 0.75,
        "rho": 0.0,
        "cw_margin": None,
        "epsilon": 0.10,
        "step_size": 0.020,
        "attack_steps": 0,
        "is_attack": False,
        "is_control": False,
        "env_extra": {
            "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE": "-1.0",
            "V4_ORACLE_GRIPPER_PATTERN": "continuous_open",
        },
        "deprecated": True,
        "deprecation_reason": "rho=0 DISABLES oracle override; rho must be >0. Codex used rho=1.0, eps=0.0, asteps=1.",
    },
]

# DEPRECATED: alias kept only for backward compat with broken tests
MATCHED_CONDITIONS = DEPRECATED_DEEPSEEK_DRIFT_MATCHED_CONDITIONS
TRIAGE_MATCHED_CONDITIONS = DEPRECATED_DEEPSEEK_DRIFT_MATCHED_CONDITIONS
OPTIONAL_CONDITION = DIAGNOSTIC_OPEN_REGION_CE_PROTOCOL


def make_run_id(semantic_task_name, state_id, repeat_id, condition_name):
    """Construct a canonical run_id from semantic task name, state, repeat, and condition."""
    return f"{semantic_task_name}_s{state_id}_r{repeat_id}_{condition_name}"


def make_clean_detect_run_id(semantic_task_name, state_id, repeat_id):
    """Construct run_id for a clean_detect run."""
    return make_run_id(semantic_task_name, state_id, repeat_id, "clean_detect")
