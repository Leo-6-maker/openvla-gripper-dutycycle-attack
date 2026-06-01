# VIS Gripper Action Semantics Audit

**Date**: 2026-06-01 | **Phase**: 1 — Direction Semantics

## Method

Loaded OpenVLA model and inspected:
1. `norm_stats["libero_object"]["action"]` for q01/q99 unnormalize params
2. `model.bin_centers` for the 255-bin discretization
3. Token → bin → normalized → unnormalized mapping

## Key Configuration

| Parameter | Value |
|-----------|-------|
| vocab_size (real) | 32000 |
| bin_centers count | 255 |
| bin_centers range | [-0.996094, 0.996094] |
| q01 (all 7 dims) | [-1, -1, -1, -1, -1, -1, -1] |
| q99 (all 7 dims) | [1, 1, 1, 1, 1, 1, 1] |
| mask (gripper) | True |
| action_dim | 7 |

## Unnormalize Formula

```
action[i] = 0.5 * (norm + 1.0) * (q99[i] - q01[i]) + q01[i]
          = 0.5 * (norm + 1.0) * 2 + (-1)
          = norm + 1.0 - 1.0
          = norm
```

**The unnormalize is identity for all 7 action dimensions.** Decoded action values equal bin centers directly.

## Action Value Semantics

### 0.0 — NEUTRAL / NO-OP

- Bin: ~127 (middle of 255 bins)
- norm/bin_center: approximately 0.0
- Physical meaning: gripper stays at current position, no opening or closing command
- In LIBERO/robosuite convention: neutral no-op

### 0.996078 — FULLY OPEN

- Bin: 254 (top of 255 bins, near BC[254]=0.996094)
- norm/bin_center: approximately 0.996078
- Physical meaning: strong OPEN command, gripper fully opens
- In LIBERO/robosuite convention: raw_action > 0 → open

## Token → Action Direction

| Property | Value |
|----------|-------|
| Bin 0 action | -0.996094 (MOST CLOSE) |
| Bin 127 action | ~0.0 (NEUTRAL) |
| Bin 254 action | 0.996094 (MOST OPEN) |
| Direction | higher bin → higher action value → MORE OPEN |

## Open Region (gripper_open_region_ce)

| Property | Value |
|----------|-------|
| open_bin_threshold | int(0.75 * 255) = 191 |
| Open bins | [191, 254] inclusive (64 bins) |
| Min open action | ~0.503906 |
| Max open action | ~0.996094 |

## Postprocess Pipeline

```
raw_action[-1] (decoded token value, range [-1, 1])
  → normalize_gripper_action(binarize=True) → {0, 1} discrete
  → invert_gripper_action (for MuJoCo convention)
```

The decoded value 0.0 → binarize → 0 (close, because 0 is not > 0.5 threshold).
The decoded value 0.996078 → binarize → 1 (open).

## Direction Semantics Verdict

| From | To | Physical Meaning |
|------|----|-----------------|
| 0.0 | 0.996078 | NEUTRAL/NO-OP → FULLY OPEN |

**The change 0.0 → 0.996078 represents the gripper transitioning from neutral to fully open.**

This is the **CORRECT direction** for the sustained open-gripper attack payload.

## Gate S1

**PASS** — Direction semantics confirmed correct. 0.0 → 0.996078 means the model goes from "not opening" to "fully opening" the gripper. The `gripper_open_region_ce` objective targets the correct token region.
