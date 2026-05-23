# Autowindow Protocols

**Baseline**: `fix/table1-generic-autowindow-baseline-20260524`
**Last updated**: 2026-05-24

## Table1 Generic Autowindow (Canonical)

The Table1 generic autowindow detector is a **phase-cue-based** detector that consumes
clean step_records only. It is NOT a simple formula — it identifies multiple phase cues
and selects the window based on a priority hierarchy.

### Detector Identity

- **detector_name**: `table1_generic_autowindow_phase_cue`
- **detector_script**: `scripts/detect_contact_window_from_clean.py`
- **detector_config**: `configs/generic_autowindow_detector.yaml`
- **window_source**: `fresh_clean_generic_autowindow`

### How It Works

1. **Reads clean step_records.jsonl** — no attack data, no rollout intervention
2. **Detects phase cues**:
   - `grasp_step` — gripper closes on object
   - `lift_step` — object lifts from surface
   - `carry_start_step` — object begins transport
   - `near_target_step` — object approaches target (quantile-based)
   - `eef_descent_step` — end-effector begins descent toward target
   - `release_intent_step` — gripper begins to open
   - `done_step` — success detection
3. **Selects preplace anchor** from priority order:
   1. `release_intent` (gripper opening near target)
   2. `eef_descent` (end-effector descending)
   3. `near_target_supported` (near target with descent/slowdown support)
   4. `near_target_late` (near target, late enough)
   5. `late_carry_fallback` (lift + offset ratio of done-lift interval)
4. **Computes window**: `[selected_preplace_step - window_len, selected_preplace_step - 1]`
5. **Records detector_config_hash** — SHA256 of the YAML config file

### Detector Modes

| Mode | Description | Eligible for command-open? |
|------|-------------|---------------------------|
| `release_intent` | Window anchored on gripper release intent | YES (if confidence >= medium) |
| `preplace_cue` | Window anchored on preplace cue | YES (if confidence >= medium) |
| `near_target_late` | Window anchored on late near-target | YES (if confidence >= medium) |
| `late_carry_fallback` | Fallback using lift-to-done ratio | Conditional (diagnostic only, unless confidence high) |
| `eef_descent` | Window anchored on EEF descent | YES (if confidence >= medium) |
| `failed_no_signal` | No reliable phase cue detected | NO |

### Key Invariants

1. **Fresh window required for each repeat/candidate.**
   Table1 prior windows are provenance only — they must NOT be used as rollout input.
2. **Clean step_records only.** The detector must not use attack outcomes or matched-run data.
3. **Detector config hash required.** Every output row must record the SHA256 of the config YAML.
4. **Confidence gating.** Low-confidence windows are diagnostic only — not command-open eligible.

## Deprecated: Standard Done-Based Formula

```text
detector_name = standard_done_minus_11_2
window = [done_step - 11, done_step - 2]
```

This formula is **DEPRECATED** for matched rollouts. Reason:

- On the 2026-05-23 true non-BB relay, it produced post-release or gripper-open-throughout
  windows for ALL 13 tested candidates across 3 suites.
- Table1 used the generic_autowindow phase-cue detector, NOT this formula.
- The done_step anchors on success detection, which occurs after gripper release for
  most non-black-bowl tasks.

### Migration

All future matched rollouts MUST use the generic autowindow detector.
If the detector is not available, run it on the clean trajectory first —
do NOT fall back to the standard formula.

## Detector-vNext

When detector behaviour is improved:
1. Freeze the new detector version name BEFORE any matched rollout
2. Record the new detector config hash
3. Validate on clean data only (no attack outcome tuning)
4. Update `TABLE1_GENERIC_AUTOWINDOW_PROTOCOL` if the anchor order changes
