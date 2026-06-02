# Sustained Proxy Code Review — Local Audit

**Commit**: `f07c98a0ba4f7aa8b646331946a86eab984a330c`
**Branch**: `exp/sustained-proxy-burst-control-20260530`
**Parent**: `c62214f` (blob-equiv to remote freeze `0870443`)
**Date**: 2026-05-30

## Summary

**code_review_status = PASS**

1 file changed, 9 insertions, 2 deletions. All P0 invariants verified.

## P0 Invariant Checklist

| # | Invariant | Status | Evidence |
|---|-----------|--------|----------|
| 1 | success_official = done_any | ✅ PASS | `if done: success = True` unchanged (line 533) |
| 2 | Clean never attacks | ✅ PASS | `attack_condition != "clean"` guard unchanged (lines 453, 455) |
| 3 | Clean preserves action | ✅ PASS | attack_action returns action unchanged for clean |
| 4 | Oracle unchanged | ✅ PASS | `a[-1] = 1.0` for oracle_open unchanged |
| 5 | Inversion unchanged | ✅ PASS | `np.clip(-action[-1] + noise)` unchanged |
| 6 | sustained_command_open_proxy ≠ VIS | ✅ PASS | New condition name, no VIS in path |
| 7 | attack_burst_steps independent | ✅ PASS | Default 0 = backward compat; >0 overrides trigger_duration |
| 8 | attack_remaining init | ✅ PASS | `burst_steps = args.attack_burst_steps if >0 else det_out["trigger_duration"]` |
| 9 | run_id uniqueness | ✅ PASS | Unchanged |
| 10 | Manifest logging | ✅ PASS | attack_burst_steps + attack_hold_mode logged |
| 11 | Step records | ✅ PASS | attack_applied, attacked_env_action, original_env_action all logged |
| 12 | No VIS_targeted in proxy path | ✅ PASS | VIS_targeted only in backward-compat inversion path |
| 13 | No large outputs staged | ✅ PASS | 1 source file only |
| 14 | Tests pass | ✅ PASS | py_compile OK, unittest 6/6 OK |

## Code Snippets

### attack_action (line 124-132)
```python
def attack_action(action, condition, rng):
    if condition == "clean": return action
    a = action.copy()
    if condition == "oracle_open": a[-1] = 1.0
    elif condition == "random_control": a[-1] = 1.0 if rng.random() > 0.5 else -1.0
    elif condition == "sustained_command_open_proxy": a[-1] = 1.0  # NEW
    elif condition in ("VIS_targeted", "gripper_inversion_proxy"):  # UNCHANGED
        a[-1] = float(np.clip(-action[-1] + rng.normal(0, 0.05), -1.0, 1.0))
    return a
```

### CLI args (lines 159-167)
```python
ap.add_argument("--attack_burst_steps", type=int, default=0, ...)  # NEW
ap.add_argument("--attack_hold_mode", default="fixed", choices=["fixed"], ...)  # NEW
ap.add_argument("--attack_condition", ..., choices=[..."sustained_command_open_proxy"])  # MODIFIED
```

### Attack burst logic (line 456)
```python
burst_steps = args.attack_burst_steps if hasattr(args, "attack_burst_steps") and args.attack_burst_steps > 0 else det_out["trigger_duration"]
attack_remaining = burst_steps
```

### Manifest logging (lines 601-602)
```python
"attack_burst_steps": args.attack_burst_steps if hasattr(args, "attack_burst_steps") else 0,
"attack_hold_mode": args.attack_hold_mode if hasattr(args, "attack_hold_mode") else "none",
```

## Files Changed

| File | Changes |
|------|---------|
| scripts/run_official_eval_artifact_rich.py | +9 -2 |

## Decision

**PASS** — All P0 invariants satisfied. No blockers. Continue Full10 sus30 run.
