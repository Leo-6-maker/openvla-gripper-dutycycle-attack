# Attack Mechanisms

**Baseline**: `fix/protocol-schema-and-condition-config-20260523`
**Last updated**: 2026-05-23

## Mechanism Taxonomy

### 1. Clean Baseline / Autowindow

- **Protocol**: `CLEAN_DETECT_PROTOCOL`
- **attack_enabled**: false
- **rho**: 0.0
- **Purpose**: Clean trajectory collection for autowindow detection.
  No attack perturbation. Used to compute trajectory windows for matched conditions
  via `detect_window()` (success_done-based autowindow).
- **attack_objective**: "" (empty string — must be passed explicitly to runner
  to suppress config default fallback)

### 2. Legacy Codex Random Same-Window

- **Protocol**: `CODEX_LEGACY_RANDOM_SAME_WINDOW`
- **attack_objective**: None (CLI flag OMITTED — falls back to `targeted_directional_ce`
  via config default)
- **rho**: 1.0
- **is_attack**: True (NOT a no-attack control)
- **epsilon**: 0.10, step_size: 0.020, attack_steps: 20
- **Active steps**: 10 (with OnlineBudgetController)
- **linf**: ~2.12 in pixel space
- **Purpose**: Legacy Codex random-noise condition. Verifies that the attack
  budget is correctly gated and random direction does not cause spurious
  task failure.

### 3. Legacy Codex VIS-Current Same-Window

- **Protocol**: `CODEX_LEGACY_VIS_CURRENT_SAME_WINDOW`
- **attack_objective**: `vis_current`
- **rho**: 1.0
- **is_attack**: True (NOT a no-attack control)
- **epsilon**: 0.10, step_size: 0.020, attack_steps: 20
- **Active steps**: 10
- **linf**: ~2.12
- **Purpose**: Legacy Codex visual-current-direction attack. Uses the
  vis_current objective to perturb in a semantically meaningful direction.
  Same budget as random, different direction.

### 4. Main Targeted Attack: Force Gripper Open Token CE

- **Protocol**: `CODEX_LEGACY_TARGETED_FORCE_GRIPPER_OPEN`
- **attack_objective**: `force_gripper_open_token_ce`
- **rho**: 1.0
- **epsilon**: 0.25 (2.5x larger than random/VIS_current)
- **step_size**: 0.050
- **attack_steps**: 60
- **force_open_raw_gripper**: 1.0
- **Active steps**: 10
- **linf**: ~2.12
- **Purpose**: The primary attack mechanism. Uses token-level cross-entropy loss
  targeting gripper-open token positions in the VLA output sequence. Forces
  the model to predict gripper-open actions during the attack window.
- **THIS IS NOT**: `gripper_logit_margin_cw`. The CW margin loss operates on
  logit gaps; token-CE operates on output token probabilities. These are
  fundamentally different mechanisms.

### 5. Diagnostic: Gripper Logit Margin CW

- **Protocol**: `DIAGNOSTIC_GRIPPER_MARGIN_PROTOCOL`
- **attack_objective**: `gripper_logit_margin_cw`
- **rho**: 1.0
- **cw_margin**: 5.0
- **epsilon**: 0.10, step_size: 0.020, attack_steps: 20
- **Purpose**: Diagnostic ablation only. NOT the legacy Codex main attack.
  Uses Carlini-Wagner margin loss on gripper logit gap.
- **Status**: Diagnostic, not confirmatory.

### 6. Diagnostic: Open Region CE

- **Protocol**: `DIAGNOSTIC_OPEN_REGION_CE_PROTOCOL`
- **attack_objective**: `gripper_open_region_ce`
- **rho**: 1.0
- **epsilon**: 0.10, step_size: 0.020, attack_steps: 20
- **Purpose**: Diagnostic ablation. Targets gripper-open action region via CE loss.
- **Status**: Diagnostic, not confirmatory.

### 7. Oracle: Command-Open Upper Bound

- **Protocol**: `COMMAND_OPEN_ORACLE_PROTOCOL`
- **attack_objective**: `oracle_env_gripper_open`
- **rho**: 1.0 (CRITICAL: rho=0 DISABLES oracle override)
- **epsilon**: 0.0 (no visual perturbation)
- **attack_steps**: 1 (minimum to trigger attack_active)
- **force_open_raw_gripper**: 0.75
- **env_extra**: `V4_ORACLE_FORCE_GRIPPER_ENV_VALUE=-1.0` (LIBERO-Spatial sign convention)
- **Purpose**: Upper-bound oracle that forces gripper-open at the command/env layer
  (not via visual perturbation). Demonstrates that forcing the gripper open
  during the attack window makes the task succeed.
- **Gate**: `oracle_override_active` is gated on `bd.attack_active`.
  rho=0 → attack_active=False → oracle never fires.

## Protocol Invariants

1. command_open rho MUST be >0 (enforced by `validate_command_open_protocol`)
2. Codex targeted MUST use `force_gripper_open_token_ce`, NOT `gripper_logit_margin_cw`
3. `gripper_logit_margin_cw` is DIAGNOSTIC only
4. Random and VIS_current ARE actual attacks (is_attack=True), not no-attack controls
5. attack_objective=None means OMIT the CLI flag entirely (not pass "None" string)
6. Clean detect MUST have attack_enabled=False, attack_objective="", rho=0.0
7. Matched seed MUST equal clean seed (enforced by `validate_same_seed_protocol`)
8. Rollout window source MUST be fresh clean_detect autowindow, NOT table1_prior_window
