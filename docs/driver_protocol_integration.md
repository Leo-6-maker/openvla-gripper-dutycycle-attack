# Driver Protocol Integration Guide

**Baseline**: `fix/protocol-schema-and-condition-config-20260523`
**Last updated**: 2026-05-23

## Overview

This document specifies how experiment drivers MUST integrate with the
protocol definitions in `src/utils/condition_protocols.py` and the validators
in `src/utils/protocol_validation.py`.

## Required Integration Points

### 1. Protocol Import

```python
from src.utils.condition_protocols import (
    CLEAN_DETECT_PROTOCOL,
    LEGACY_CODEX_STATE5_MATCHED_CONDITIONS,
    COMMAND_OPEN_ORACLE_PROTOCOL,
    CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN,
)
```

### 2. Protocol Validation at Config-Load Time

Before launching any rollout, call the validators:

```python
from src.utils.protocol_validation import (
    validate_command_open_protocol,
    validate_codex_targeted_protocol,
    validate_condition_config_schema,
    validate_same_seed_protocol,
)

# Validate each condition
for cond in LEGACY_CODEX_STATE5_MATCHED_CONDITIONS:
    validate_condition_config_schema(cond)
    if cond["condition_name"].startswith("command_open"):
        validate_command_open_protocol(cond)
    if cond.get("attack_objective") == "force_gripper_open_token_ce":
        validate_codex_targeted_protocol(cond)

# Validate seed protocol
validate_same_seed_protocol(matched_seed, clean_seed)
```

### 3. CLI Flag Construction

CRITICAL: `attack_objective=None` DOES NOT mean pass `--attack_objective None`.
It means OMIT the `--attack_objective` flag entirely.

```python
def build_attack_objective_arg(protocol):
    """Build the --attack_objective CLI argument for a condition protocol."""
    if protocol.get("omit_attack_objective_cli_arg"):
        # DO NOT pass --attack_objective at all
        # The runner will fall back to its config default
        return []
    obj = protocol.get("attack_objective")
    if obj is None:
        # Safety: None with omit=False is ambiguous — reject
        raise ValueError(
            f"attack_objective=None but omit_attack_objective_cli_arg=False "
            f"for {protocol['condition_name']}. This is ambiguous."
        )
    if obj == "":
        # Clean detect: pass empty string to suppress config default
        return ["--attack_objective", ""]
    return ["--attack_objective", obj]
```

### 4. Command-Open Oracle Requirements

The command_open oracle protocol requires:

```
rho > 0              # required for attack_active
attack_steps >= 1    # required to trigger attack_active  
epsilon = 0.0        # no visual perturbation
attack_objective = oracle_env_gripper_open
env_extra = {
    "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE": "-1.0",  # sign depends on suite
    "V4_ORACLE_GRIPPER_PATTERN": "continuous_open",
}
```

The sign of `V4_ORACLE_FORCE_GRIPPER_ENV_VALUE` depends on the LIBERO suite:
- LIBERO-Spatial: -1.0 = open
- LIBERO-Object: sign may differ (needs per-suite calibration)

### 5. Window Source Enforcement

Rollout window MUST come from fresh clean_detect autowindow.
table1_prior_window is FORBIDDEN as rollout input.

```python
from src.utils.protocol_validation import validate_window_source

# Before using a window for rollout:
validate_window_source(window_provenance_string)
```

### 6. Seed Protocol

Matched condition seed MUST equal clean seed:

```python
# CORRECT:
clean_seed = repeat_id
matched_seed = repeat_id  # same as clean

# WRONG (DeepSeek drift):
clean_seed = repeat_id
matched_seed = repeat_id + 100  # SEED DRIFT
```

### 7. Deprecated Condition Access

NEVER import MATCHED_CONDITIONS or TRIAGE_MATCHED_CONDITIONS:

```python
# WRONG — will raise RuntimeError:
from src.utils.task_identity import MATCHED_CONDITIONS  

# CORRECT:
from src.utils.condition_protocols import LEGACY_CODEX_STATE5_MATCHED_CONDITIONS
```

## TODO: Unified Command Builder

A unified command builder utility (`src/utils/command_config.py`) is planned
but not yet implemented. When added, it should:

1. Accept a protocol dict and a task identity dict
2. Return a complete CLI argument list
3. Handle all the edge cases documented above
4. Call the relevant validators before returning

Until then, individual driver scripts are responsible for implementing
the integration points described in this document.
