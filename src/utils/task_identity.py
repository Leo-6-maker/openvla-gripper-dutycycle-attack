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


# --- Condition definitions (dict-based, avoids tuple-length bugs) ---
MATCHED_CONDITIONS = [
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
    },
]

# Optional condition: VIS_gripper_open_region_ce_same_autowindow
OPTIONAL_CONDITION = {
    "condition_name": "VIS_gripper_open_region_ce_same_autowindow",
    "attack_objective": "gripper_open_region_ce",
    "temporal_init": "prev_delta",
    "force_open_raw_gripper": None,
    "rho": 1.0,
    "cw_margin": None,
    "epsilon": 0.10,
    "step_size": 0.020,
    "attack_steps": 20,
    "is_attack": True,
    "is_control": False,
}

# Queue B conditions (triage: fewer conditions)
TRIAGE_MATCHED_CONDITIONS = [
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
        "env_extra": {
            "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE": "-1.0",
            "V4_ORACLE_GRIPPER_PATTERN": "continuous_open",
        },
    },
    {
        "condition_name": "command_open_1.00_same_autowindow",
        "attack_objective": "oracle_env_gripper_open",
        "temporal_init": "none",
        "force_open_raw_gripper": 1.0,
        "rho": 0.0,
        "cw_margin": None,
        "epsilon": 0.10,
        "step_size": 0.020,
        "attack_steps": 0,
        "env_extra": {
            "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE": "-1.0",
            "V4_ORACLE_GRIPPER_PATTERN": "continuous_open",
        },
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
    },
]


def make_run_id(semantic_task_name, state_id, repeat_id, condition_name):
    """Construct a canonical run_id from semantic task name, state, repeat, and condition."""
    return f"{semantic_task_name}_s{state_id}_r{repeat_id}_{condition_name}"


def make_clean_detect_run_id(semantic_task_name, state_id, repeat_id):
    """Construct run_id for a clean_detect run."""
    return make_run_id(semantic_task_name, state_id, repeat_id, "clean_detect")
