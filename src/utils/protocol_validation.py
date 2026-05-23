"""Protocol validation — fail-fast guards for common protocol drift patterns.

These validators catch silent protocol regression at config-load time.
Import and call in experiment drivers before launching any rollout.

IMPORTANT: All validators use explicit ``if / raise``, NOT ``assert``.
``assert`` is stripped by ``python -O``, which would silently disable
protocol guards during long-running GPU experiments.
"""


class ProtocolValidationError(ValueError):
    """Raised when a condition protocol fails validation.

    This is a hard failure — the protocol is invalid and the experiment
    MUST NOT proceed.  Fix the protocol config before re-launching.
    """


def validate_command_open_protocol(protocol):
    """command_open must have rho > 0 and objective oracle_env_gripper_open.

    rho=0 disables OnlineBudgetController → attack_active never True →
    oracle_override_active never fires.  This is the #1 protocol drift bug.
    """
    if protocol["rho"] <= 0:
        raise ProtocolValidationError(
            f"command_open rho={protocol['rho']} disables oracle override (must be >0)"
        )
    if protocol["attack_objective"] != "oracle_env_gripper_open":
        raise ProtocolValidationError(
            f"command_open objective must be oracle_env_gripper_open, "
            f"got {protocol['attack_objective']}"
        )
    if "env_extra" not in protocol:
        raise ProtocolValidationError(
            "command_open must have env_extra with V4_ORACLE_FORCE_GRIPPER_ENV_VALUE"
        )
    if "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE" not in protocol.get("env_extra", {}):
        raise ProtocolValidationError(
            "command_open env_extra missing V4_ORACLE_FORCE_GRIPPER_ENV_VALUE"
        )
    if protocol.get("attack_steps", 0) < 1:
        raise ProtocolValidationError(
            f"command_open attack_steps={protocol.get('attack_steps')} must be >= 1 "
            "to trigger attack_active"
        )


def validate_same_seed_protocol(matched_seed, clean_seed):
    """Matched condition seed MUST equal clean seed for same-window validity.

    If seeds differ, the trajectory window (start/end steps) is NOT the same
    and the matched comparison is invalid.
    """
    if matched_seed != clean_seed:
        raise ProtocolValidationError(
            f"Seed drift: matched_seed={matched_seed} != clean_seed={clean_seed}. "
            "Same-window comparison requires identical seeds."
        )


def validate_window_source(rollout_window_source, expected_hint="fresh_clean_detect_autowindow"):
    """Rollout window input must be a fresh clean_detect autowindow.

    table1_prior_window is a static lookup, not a fresh trajectory window.
    Using table1_prior_window as rollout input breaks the autowindow protocol.
    """
    source_str = str(rollout_window_source).lower()
    if "table1_prior" in source_str:
        raise ProtocolValidationError(
            f"Rollout window must come from fresh clean_detect autowindow, "
            f"not table1_prior_window. Got: {rollout_window_source}"
        )
    if not ("clean_detect" in source_str or "autowindow" in source_str
            or expected_hint in source_str):
        raise ProtocolValidationError(
            f"Window source should be fresh autowindow. Got: {rollout_window_source}"
        )


def validate_codex_targeted_protocol(protocol):
    """Codex targeted protocol exact parameter check.

    Verified against Codex r0/r1/r2 raw step_records audit 2026-05-23.
    These are NOT configurable — any deviation is a protocol bug.
    """
    if protocol["attack_objective"] != "force_gripper_open_token_ce":
        raise ProtocolValidationError(
            f"Expected force_gripper_open_token_ce, got {protocol['attack_objective']}"
        )
    if abs(protocol["epsilon"] - 0.25) >= 1e-9:
        raise ProtocolValidationError(
            f"Expected epsilon=0.25, got {protocol['epsilon']}"
        )
    if abs(protocol["step_size"] - 0.050) >= 1e-9:
        raise ProtocolValidationError(
            f"Expected step_size=0.050, got {protocol['step_size']}"
        )
    if protocol["attack_steps"] != 60:
        raise ProtocolValidationError(
            f"Expected attack_steps=60, got {protocol['attack_steps']}"
        )
    if protocol.get("force_open_raw_gripper") != 1.0:
        raise ProtocolValidationError(
            f"Expected force_open_raw_gripper=1.0, "
            f"got {protocol.get('force_open_raw_gripper')}"
        )


def validate_condition_config_schema(protocol):
    """Every condition must define all required top-level fields.

    Extended schema checks for fields that have historically been missing
    or wrong in protocol definitions.
    """
    name = protocol.get("condition_name", "?")

    required = [
        "condition_name", "attack_objective", "rho", "epsilon",
        "step_size", "attack_steps", "is_attack", "is_control",
    ]
    missing = [k for k in required if k not in protocol]
    if missing:
        raise ProtocolValidationError(
            f"Missing required fields in {name}: {missing}"
        )

    recommended = [
        "force_open_raw_gripper",
        "is_oracle",
        "attack_objective_raw_arg",
        "effective_attack_objective_expected",
    ]
    rec_missing = [k for k in recommended if k not in protocol]
    if rec_missing:
        raise ProtocolValidationError(
            f"Missing recommended fields in {name}: {rec_missing}. "
            "These fields document protocol provenance and prevent drift."
        )

    # command_open-specific checks
    if "oracle_env_gripper_open" in str(protocol.get("attack_objective", "")):
        if "env_extra" not in protocol:
            raise ProtocolValidationError(
                f"{name}: command_open protocol missing env_extra"
            )
        if "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE" not in protocol.get("env_extra", {}):
            raise ProtocolValidationError(
                f"{name}: command_open protocol missing V4_ORACLE_FORCE_GRIPPER_ENV_VALUE"
            )
        if protocol.get("rho", 0) <= 0:
            raise ProtocolValidationError(
                f"{name}: command_open protocol rho must be > 0, got {protocol.get('rho')}"
            )
        if protocol.get("attack_steps", 0) < 1:
            raise ProtocolValidationError(
                f"{name}: command_open protocol attack_steps must be >= 1"
            )

    # Codex targeted-specific checks
    if "force_gripper_open_token_ce" in str(protocol.get("attack_objective", "")):
        if protocol.get("force_open_raw_gripper") != 1.0:
            raise ProtocolValidationError(
                f"{name}: Codex targeted must have force_open_raw_gripper=1.0"
            )
        if abs(protocol.get("epsilon", 0) - 0.25) >= 1e-9:
            raise ProtocolValidationError(
                f"{name}: Codex targeted must have epsilon=0.25"
            )
        if abs(protocol.get("step_size", 0) - 0.050) >= 1e-9:
            raise ProtocolValidationError(
                f"{name}: Codex targeted must have step_size=0.050"
            )
        if protocol.get("attack_steps", 0) != 60:
            raise ProtocolValidationError(
                f"{name}: Codex targeted must have attack_steps=60"
            )
