# Stage-B Attack Open Semantics — Inversion Audit

**Date**: 2026-06-07
**Scope**: Full semantic chain from model tokens → env action → physical gripper
**Finding**: **VIS PGD objective targets physical CLOSE, not OPEN**

## 1. The normalize → invert → env.step pipeline

```
raw_gripper (model decoded, range [0,1])
    ↓ normalize_gripper_action(binarize=True):  2*raw - 1 → sign → {+1, -1}
    ↓ invert_gripper_action:  × (-1)
    ↓ env.step(env_action_full)
```

### Numerical trace

| raw_gripper | normalize | after binarize | invert | env_action_6 | Physical |
|-------------|-----------|----------------|--------|-------------|----------|
| ≈ 0.0 | -1.0 | -1.0 | +1.0 | **+1.0** | **CLOSE** |
| ≈ 0.5 | 0.0 | +1.0 (0→+1 rule) | -1.0 | -1.0 | OPEN |
| ≈ 1.0 | +1.0 | +1.0 | -1.0 | **-1.0** | **OPEN** |

**Oracle smoke confirmed**: `env_action_6 = -1.0` → gripper OPENS, qpos abs_sum INCREASES.

## 2. Current code semantics (WRONG)

### gripper_semantics.py (line 82-88)
```python
def env_gripper_is_open(env_gripper):
    return float(env_gripper) > 0.0   # WRONG: env=+1 → CLOSE physically
```

The docstring (lines 17-18) claims:
```
raw_action ≈ 0.0  →  env_action = +1  →  OPEN    # WRONG
raw_action ≈ 0.996 →  env_action = -1  →  CLOSE   # WRONG
```

### attack_adapter.py — get_gripper_region_by_decoded_action() (line 325)
```python
is_open_by_env = (env_val > 0)   # WRONG: classifies CLOSE tokens as OPEN
```

**Result**: `open_token_ids` contain tokens where `env_val = +1` → physical CLOSE.

### run_stageb_vis_labeling.py (line 285)
```python
open_count = sum(1 for g in decoded_grips if g > 0)  # WRONG
```

Runner summary `decoded_open_count` counts CLOSE as OPEN.

## 3. Correct code (hotfix postprocess ONLY)

### run_patched_rerun_postprocess_hotfix.py
```python
def is_open(v):
    return float(v) < -0.5   # CORRECT: env=-1 → OPEN
```

**Only the hotfix postprocess uses the correct convention.** The hotfix bypasses runner summary and reads from trace CSVs.

## 4. Impact on 44-row rerun

### The mismatch:

| Component | OPEN test | Token IDs in objective | Physical effect |
|-----------|-----------|----------------------|-----------------|
| VIS PGD objective | `env > 0` | tokens where env=+1 | **Optimizes for CLOSE** |
| Hotfix label | `env < -0.5` | N/A | Measures physical OPEN |

### This explains the 3/44 cmd_susceptible result:

- VIS PGD20 pushes model toward **physical CLOSE** (env=+1 tokens)
- Random Linf has no directional bias → randomly produces some OPEN
- In many windows random produces more OPEN than VIS targeting CLOSE
- `cmd_susceptible = 3/44` means only 3 windows where VIS (targeting CLOSE) accidentally produced more OPEN than random
- `random_confounded = 35/44` means in 35 windows, random perturbation produced ≥6 OPEN actions while VIS targeting CLOSE mostly failed to open

### cream_cheese/orange_juice VIS-immune explained:

These tasks have models that firmly produce CLOSE tokens by default. VIS PGD targeting CLOSE pushes them further toward CLOSE (0 OPEN). Random randomly hits OPEN bins. → VIS 0-1 OPEN, RAND 10-11 OPEN.

## 5. The fix

### What needs to change:

```python
# gripper_semantics.py
def env_gripper_is_open(env_gripper):
    return float(env_gripper) < -0.5  # env=-1 → OPEN

# attack_adapter.py — get_gripper_region_by_decoded_action()
is_open_by_env = (env_val < -0.5)   # env=-1 → OPEN
# This flips open_token_ids ↔ close_token_ids

# run_stageb_vis_labeling.py
open_count = sum(1 for g in decoded_grips if g < -0.5)  # env=-1 → OPEN
```

### The flip:

| | Before fix | After fix |
|---|---|---|
| open_token_ids | {tokens where env=+1} | {tokens where env=-1} |
| Physical meaning of open | CLOSE | OPEN ✓ |
| VIS PGD direction | Push to CLOSE | Push to OPEN ✓ |

## 6. Verification checklist

- [ ] P0-A: Env-only oracle truth table (formal artifact)
- [ ] Fix gripper_semantics.py: env_gripper_is_open → `env < -0.5`
- [ ] Fix attack_adapter.py: is_open_by_env → `env_val < -0.5`
- [ ] Fix run_stageb_vis_labeling.py: L285 → `g < -0.5`
- [ ] Fix tests/test_open_convention.py
- [ ] Fix stageb/postprocess_patched_traces_v1.py
- [ ] Fix stageb/build_pair_labels_v1.py
- [ ] Self-check assertions in gripper_semantics.py must pass
- [ ] P0-C: 3-row corrected VIS smoke (butter s6, ketchup s8, 1 control)

## 7. Verdict

**VIS_PGD_INVERTED = TRUE**

The 44-row rerun result (`cmd_susceptible = 3/44`) is NOT evidence that VIS lacks selectivity. It is evidence that **VIS PGD targeting the wrong physical direction** produces expectedly poor results.

The corrected objective should dramatically change the cmd_susceptible rate.
