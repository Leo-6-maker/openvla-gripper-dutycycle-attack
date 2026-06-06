# Command Susceptible Label Protocol v2

**Date**: 2026-06-07
**Status**: Draft — replaces implicit v1 label policy from active probe era

## 1. Objective

Define a reproducible protocol for labeling windows as `command_susceptible`
(targeted VIS PGD induces sustained decoded gripper OPEN) from VIS attack traces,
independent of any probe or detector.

## 2. Label Definitions

### 2.1 command_susceptible_positive

A window is labeled **command_susceptible_positive** when ALL of these hold:

| Criterion | Threshold | Measurement |
|---|---|---|
| VIS targeted open_count | >= 6 | Count of attack frames with decoded env_gripper = +1.0 (OPEN) |
| VIS targeted longest_open_streak | >= 3 | Longest consecutive OPEN streak in attack frames |
| Random contrast | random_open_count < targeted_open_count - 3 | Matched random Linf (same eps) must NOT reproduce the OPEN |
| Provenance | clean | No infra error, no manual conflict, no trace contamination |
| Attack validity | PGD converged | loss decreased, no OOM, correct objective used |

### 2.2 command_susceptible_negative

A window is labeled **command_susceptible_negative** when at least ONE of:

| Criterion | Threshold |
|---|---|
| VIS targeted open_count | < 3 AND longest_open_streak < 2 (valid attack, no effect) |
| Random contrast failed | random_open_count >= targeted_open_count - 3 (no selectivity) |
| Provenance | clean (same requirements as positive) |

### 2.3 Excluded from train/eval

| Exclusion reason | Tag |
|---|---|
| VIS trace polluted (object already moved, camera blocked) | `polluted` |
| 1R failed / pending confirmation | `pending_1r` |
| Infra error (OOM, GPU fault, env crash) | `infra_error` |
| Manual merge conflict unresolved | `manual_conflict` |
| Random baseline missing or invalid | `no_random_control` |
| 3R result contradicts 1R | `unconfirmed_3r_conflict` |

## 3. VIS Attack Configuration (Label Generation)

For generating command_susceptible labels, use the **standard VIS attack config**:

| Parameter | Value |
|---|---|
| PGD steps | 20 (medium budget) or 40 (full budget) |
| Restarts | 1 (medium) or 3 (full) |
| eps_raw_pixels | 6 |
| Objective | `prefix_locked_gripper_open_margin` |
| Arm preserve weight | 0.5 |
| Gripper margin | 5.0 |
| Attack frames | full window [ws, we] |
| Env step | YES (rollout mode) |
| Matched random | same eps, random Linf direction, same frames |

## 4. Open Convention

**Decoded env_gripper = +1.0 means OPEN**, -1.0 means CLOSE.

Pipeline: `model.generate → action_tokens → decode_to_continuous → normalize_gripper_action(binarize=True) → invert_gripper_action → env_gripper`

This is the OFFICIAL LIBERO convention. All labels use this convention.

## 5. physical_bridge_label (Downstream)

A window has physical_bridge when it is command_susceptible AND:

| Criterion | Threshold |
|---|---|
| qpos_opening_delta | >= 0.02 (measurable physical opening) |
| qpos_label | `strong` or `weak` (not `none`) |
| Provenance | same clean requirements |

physical_bridge is a **harder downstream label**. Detector primary target is command_susceptible.

## 6. Label Provenance Tiers

| Tier | Criteria | Max use |
|---|---|---|
| **Gold** | 3R-confirmed, random contrast passed, clean provenance, PGD20+ | Train + Eval |
| **Silver** | 1R-positive with random contrast, clean provenance, PGD10+ | Train only (no eval) |
| **Bronze** | 1R-positive without random contrast, or PGD5+ micro attack | Exploratory only |
| **Reject** | Any exclusion reason in section 2.3 | Excluded from all |

## 7. Existing Label Mapping

The existing `object_phase_response_labels_v2.csv` uses:

| Field | Mapping to v2 |
|---|---|
| `vis_open_count` | → command_susceptible open_count |
| `label_physical_response` | → physical_bridge (1.0=positive, 0.5=weak, 0=negative) |
| `taxonomy` | → mechanism classification (informational, NOT used as feature) |
| `label_status` | → positive/negative/ignore (approximate — need random contrast check) |
| `source_batch` | → VIS config provenance (batch1=PGD20, batch3=PGD40x3) |

For v2, we need to add:
- `random_open_count` (from matched random trace)
- `vis_longest_open_streak` (computed from VIS trace)
- `random_longest_open_streak` (computed from random trace)
- `label_tier` (gold/silver/bronze/reject)
- `exclusion_reason` (if rejected)

## 8. Label Generation Workflow

```
1. Select window [ws, we] from clean rollout
2. Run clean rollout to get observation sequence
3. Run VIS targeted PGD attack (PGD20, env.step) on window
4. Run matched random Linf attack (same eps) on window
5. Extract per-frame decoded env_gripper from both traces
6. Compute open_count, longest_open_streak for both
7. Apply criteria from section 2
8. Assign label tier from section 6
9. Record provenance in label CSV
```

## 9. Version History

| Version | Date | Changes |
|---|---|---|
| v1 (implicit) | 2026-06-05 | Used in active probe era; no random contrast, inconsistent provenance |
| v2 | 2026-06-07 | Formal protocol with random contrast, provenance tiers, exclusion rules |
