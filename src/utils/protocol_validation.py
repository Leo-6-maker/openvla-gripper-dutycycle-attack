"""Protocol validation — fail-fast guards for common protocol drift patterns.

These validators catch silent protocol regression at config-load time.
Import and call in experiment drivers before launching any rollout.
"""


def validate_command_open_protocol(protocol):
    """command_open must have rho > 0 and objective oracle_env_gripper_open.

    rho=0 disables OnlineBudgetController → attack_active never True →
    oracle_override_active never fires.  This is the #1 protocol drift bug.
    """
    assert protocol["rho"] > 0, \
        f"command_open rho={protocol['rho']} disables oracle override (must be >0)"
    assert protocol["attack_objective"] == "oracle_env_gripper_open", \
        f"command_open objective must be oracle_env_gripper_open, got {protocol['attack_objective']}"


def validate_same_seed_protocol(matched_seed, clean_seed):
    """Matched condition seed MUST equal clean seed for same-window validity.

    If seeds differ, the trajectory window (start/end steps) is NOT the same
    and the matched comparison is invalid.
    """
    assert matched_seed == clean_seed, \
        f"Seed drift: matched_seed={matched_seed} != clean_seed={clean_seed}. " \
        "Same-window comparison requires identical seeds."


def validate_window_source(rollout_window_source, expected_hint="fresh_clean_detect_autowindow"):
    """Rollout window input must be a fresh clean_detect autowindow.

    table1_prior_window is a static lookup, not a fresh trajectory window.
    Using table1_prior_window as rollout input breaks the autowindow protocol.
    """
    source_str = str(rollout_window_source).lower()
    assert "table1_prior" not in source_str, \
        f"Rollout window must come from fresh clean_detect autowindow, " \
        f"not table1_prior_window. Got: {rollout_window_source}"
    assert "clean_detect" in source_str or "autowindow" in source_str \
        or expected_hint in source_str, \
        f"Window source should be fresh autowindow. Got: {rollout_window_source}"


def validate_codex_targeted_protocol(protocol):
    """Codex targeted protocol exact parameter check.

    Verified against Codex r0/r1/r2 raw step_records audit 2026-05-23.
    These are NOT configurable — any deviation is a protocol bug.
    """
    assert protocol["attack_objective"] == "force_gripper_open_token_ce", \
        f"Expected force_gripper_open_token_ce, got {protocol['attack_objective']}"
    assert abs(protocol["epsilon"] - 0.25) < 1e-9, \
        f"Expected epsilon=0.25, got {protocol['epsilon']}"
    assert abs(protocol["step_size"] - 0.050) < 1e-9, \
        f"Expected step_size=0.050, got {protocol['step_size']}"
    assert protocol["attack_steps"] == 60, \
        f"Expected attack_steps=60, got {protocol['attack_steps']}"
    assert protocol.get("force_open_raw_gripper") == 1.0, \
        f"Expected force_open_raw_gripper=1.0, got {protocol.get('force_open_raw_gripper')}"


def validate_condition_config_schema(protocol):
    """Every condition must define all required top-level fields."""
    required = [
        "condition_name", "attack_objective", "rho", "epsilon",
        "step_size", "attack_steps", "is_attack", "is_control",
    ]
    missing = [k for k in required if k not in protocol]
    assert not missing, f"Missing required fields in {protocol.get('condition_name', '?')}: {missing}"
