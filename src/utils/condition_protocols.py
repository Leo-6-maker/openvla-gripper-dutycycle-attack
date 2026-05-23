"""
Condition protocol definitions — canonical per-condition attack/control configs.

IMPORTANT: These define what each condition MEANS, not defaults for a specific experiment.
Experiment drivers should select from these protocols and may override parameters.

Protocols are separated from task identity mapping (task_identity.py) so that
condition definitions can evolve independently of task naming conventions.

Sources:
- LEGACY_CODEX_STATE5_PROTOCOL: verified against Codex r0/r1/r2 raw step_records
  (linf, attack_active steps, objective args extracted from manifests/step_records)
- Subagent A runner semantics audit: confirmed rho=0 disables attack_active,
  command_open oracle override gated on bd.attack_active
"""

# ============================================================
# Clean baseline protocol
# ============================================================
CLEAN_DETECT_PROTOCOL = {
    "condition_name": "clean_detect",
    "protocol_name": "clean_detect",
    "protocol_version": "1.0",
    "purpose": "clean_baseline_and_autowindow",
    "attack_enabled": False,
    "attack_objective": "",
    "attack_objective_raw_arg": "",  # empty string passed to suppress config default
    "effective_attack_objective_expected": "",  # no attack, no objective fallback
    "omit_attack_objective_cli_arg": False,  # explicitly pass empty string
    "temporal_init": "",
    "rho": 0.0,
    "epsilon": 0.0,
    "step_size": 0.0,
    "attack_steps": 0,
    "cw_margin": None,
    "force_open_raw_gripper": None,
    "is_attack": False,
    "is_control": False,
    "is_oracle": False,
    "notes": "Must pass attack_objective='' to runner to suppress config default fallback.",
}


# ============================================================
# Legacy Codex state5 protocol (r0/r1/r2)
# Verified against raw step_records audit 2026-05-23:
#   random: 10 active steps, linf=0.10, rho=1.0
#   VIS_current: 10 active steps, linf=2.12, rho=1.0
#   targeted: obj=force_gripper_open_token_ce, eps=0.25, ss=0.050, asteps=60, fo=1.0
#   command_open: obj=oracle_env_gripper_open, rho=1.0, eps=0.0, asteps=1, fo=0.75
#
# IMPORTANT: random and VIS_current in this protocol are ACTUAL attacks
# (not no-attack controls). They used rho=1.0 and had measurable linf perturbation.
# The --attack_objective arg was NOT passed; effective objective fell back to
# config default (targeted_directional_ce for random, vis_current for VIS_current via env).
# ============================================================
LEGACY_CODEX_STATE5_PROTOCOL = {
    "protocol_name": "legacy_codex_state5",
    "protocol_version": "1.0",
    "source": "Codex r0/r1/r2 raw step_records audit 2026-05-23",
    "notes": (
        "Random and VIS_current are actual attacks in this protocol (rho=1.0, linf>0). "
        "They are NOT no-attack controls. The --attack_objective arg was omitted; "
        "effective objective fell back to config/env defaults."
    ),
}

# Random same-window condition (Codex legacy: actual attack with config-default objective)
CODEX_LEGACY_RANDOM_SAME_WINDOW = {
    "condition_name": "random_same_autowindow",
    "attack_objective": None,  # Codex did NOT pass --attack_objective; falls back to config
    "attack_objective_raw_arg": None,  # what Codex passed on CLI (nothing)
    "omit_attack_objective_cli_arg": True,  # driver MUST omit --attack_objective flag entirely
    "effective_attack_objective_expected": "targeted_directional_ce",  # config default fallback
    "requires_execution_audit": True,  # must verify effective objective in raw step_records
    "temporal_init": "prev_delta",
    "force_open_raw_gripper": None,
    "rho": 1.0,
    "cw_margin": None,
    "epsilon": 0.10,
    "step_size": 0.020,
    "attack_steps": 20,
    "is_attack": True,
    "is_control": False,
    "is_oracle": False,
    "notes": (
        "Codex: 10 active steps, linf=0.10, SR=1.0. Not a clean-repeat control. "
        "attack_objective=None + omit_attack_objective_cli_arg=True means "
        "--attack_objective was NOT passed on CLI; effective objective resolved "
        "to 'targeted_directional_ce' via config default. "
        "Driver MUST NOT pass 'None' or empty string as --attack_objective value."
    ),
}

# VIS_current same-window condition (Codex legacy: actual attack with vis_current objective)
CODEX_LEGACY_VIS_CURRENT_SAME_WINDOW = {
    "condition_name": "VIS_current_same_autowindow",
    "attack_objective": "vis_current",
    "attack_objective_raw_arg": "vis_current",  # what Codex passed on CLI or via env
    "effective_attack_objective_expected": "vis_current",  # confirmed in step_records audit
    "requires_execution_audit": True,  # must verify effective objective in raw step_records
    "temporal_init": "prev_delta",
    "force_open_raw_gripper": None,
    "rho": 1.0,
    "cw_margin": None,
    "epsilon": 0.10,
    "step_size": 0.020,
    "attack_steps": 20,
    "is_attack": True,
    "is_control": False,
    "is_oracle": False,
    "notes": (
        "Codex: 10 active steps, linf=2.12, SR=1.0. "
        "attack_objective='vis_current' — this is an actual attack, not a control. "
        "Do NOT misinterpret as a no-attack baseline."
    ),
}

# Targeted force-gripper-open token CE (Codex legacy: primary attack)
CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN = {
    "condition_name": "best_VIS_gripper_targeted_stronger_same_autowindow",
    "attack_objective": "force_gripper_open_token_ce",
    "attack_objective_raw_arg": "force_gripper_open_token_ce",  # what Codex passed on CLI
    "effective_attack_objective_expected": "force_gripper_open_token_ce",  # direct match
    "omit_attack_objective_cli_arg": False,  # this objective IS explicitly passed
    "temporal_init": "prev_delta",
    "force_open_raw_gripper": 1.0,
    "rho": 1.0,
    "cw_margin": None,
    "epsilon": 0.25,
    "step_size": 0.050,
    "attack_steps": 60,
    "is_attack": True,
    "is_control": False,
    "is_oracle": False,
    "notes": (
        "Codex: 10 active steps, linf=2.12, SR=0.0. "
        "Uses force_gripper_open_token_ce (token-level CE targeting gripper-open tokens). "
        "NOT gripper_logit_margin_cw. eps=0.25 is 2.5x larger than random/VIS_current."
    ),
}


# ============================================================
# Command-open oracle protocol
# WARNING: rho must be >0 for oracle_env_gripper_open to function.
# Subagent A confirmed: oracle_override_active gated on bd.attack_active.
# rho=0 forces attack_active=False perpetually → override never fires.
# ============================================================
COMMAND_OPEN_ORACLE_PROTOCOL = {
    "condition_name": "command_open_0.75_same_autowindow",
    "protocol_name": "command_open_oracle",
    "protocol_version": "1.0",
    "purpose": "upper_bound_command_layer_gripper_open",
    "attack_objective": "oracle_env_gripper_open",
    "attack_objective_raw_arg": "oracle_env_gripper_open",  # what Codex passed on CLI
    "effective_attack_objective_expected": "oracle_env_gripper_open",  # direct match
    "omit_attack_objective_cli_arg": False,  # this objective IS explicitly passed
    "temporal_init": "prev_delta",
    "force_open_raw_gripper": 0.75,
    "rho": 1.0,  # MUST be >0 for oracle override to activate
    "cw_margin": None,
    "epsilon": 0.0,
    "step_size": 0.0,
    "attack_steps": 1,  # minimum: 1 step to trigger attack_active
    "is_attack": False,
    "is_control": False,
    "is_oracle": True,
    "env_extra": {
        "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE": "-1.0",
        "V4_ORACLE_GRIPPER_PATTERN": "continuous_open",
    },
    "warnings": [
        "rho=0 DISABLES oracle override (attack_active never True)",
        "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE sign depends on LIBERO suite convention",
        "For LIBERO-Spatial: -1.0 = open. For LIBERO-Object: sign may differ.",
    ],
}


# ============================================================
# Diagnostic protocols (NOT confirmatory — for ablation/diagnosis only)
# ============================================================
DIAGNOSTIC_GRIPPER_MARGIN_PROTOCOL = {
    "condition_name": "diagnostic_gripper_logit_margin_cw",
    "protocol_name": "diagnostic_gripper_margin",
    "protocol_version": "1.0",
    "purpose": "diagnostic_ablation_gripper_logit_margin",
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
    "is_oracle": False,
    "notes": (
        "Uses Carlini-Wagner margin loss (logit gap), NOT token-CE. "
        "This is a DIFFERENT mechanism from force_gripper_open_token_ce. "
        "Not the legacy Codex main targeted attack."
    ),
}

DIAGNOSTIC_OPEN_REGION_CE_PROTOCOL = {
    "condition_name": "VIS_gripper_open_region_ce_same_autowindow",
    "protocol_name": "diagnostic_open_region_ce",
    "protocol_version": "1.0",
    "purpose": "diagnostic_ablation_open_region_ce",
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
    "is_oracle": False,
}


# ============================================================
# Convenience: legacy Codex state5 matched conditions list
# (in the order they should be run: random, VIS_current, targeted, command_open)
# ============================================================
LEGACY_CODEX_STATE5_MATCHED_CONDITIONS = [
    CODEX_LEGACY_RANDOM_SAME_WINDOW,
    CODEX_LEGACY_VIS_CURRENT_SAME_WINDOW,
    CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN,
    COMMAND_OPEN_ORACLE_PROTOCOL,
]

# Optional diagnostic to run if stable
LEGACY_CODEX_STATE5_OPTIONAL = [
    DIAGNOSTIC_OPEN_REGION_CE_PROTOCOL,
]
