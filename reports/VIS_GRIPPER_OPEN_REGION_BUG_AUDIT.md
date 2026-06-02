# VIS Gripper OPEN Region Bug Audit

**Date**: 2026-06-02  
**Severity**: CRITICAL P0  
**Status**: FIXED in `attack_adapter.py`

## Bug Description

`action_bins_for_env_sign(dim=-1, sign="negative", postprocess_gripper=True)` returned tokens that decode to **CLOSE** actions, not OPEN. The sign-string semantics are inverted under the `postprocess_gripper → normalize → invert` pipeline.

## Root Cause

The postprocess chain:

```
raw = 0.5*(center+1)*(high-low)+low     # center=-1→raw≈0, center=+1→raw≈1
env = 2*raw-1                            # raw=0→env=-1, raw=1→env=+1
env = sign(env); env[env==0]=1          # -1→-1, +1→+1
env = -1*env                             # -1→+1(OPEN), +1→-1(CLOSE)
```

After processing: `env=+1` = OPEN, `env=-1` = CLOSE.

Then `sign="negative"` selects `env < -0.5`, which matches CLOSE tokens (env=-1).

## Impact

| Token | disc | Decoded Action | Semantic | Old Region | Corrected Region |
|-------|------|---------------|----------|------------|-----------------|
| 31744 | 254 | 1.000 | CLOSE | CLOSE | CLOSE |
| 31745 | 254 | 1.000 | CLOSE | **OPEN (wrong!)** | CLOSE |
| 31808 | 191 | 0.748 | CLOSE | **OPEN (wrong!)** | CLOSE |
| 31872 | 127 | ~0.500 | boundary | CLOSE | boundary |
| 31873 | 126 | ~0.496 | **OPEN** | CLOSE | **OPEN** |

All 64 tokens in the old loss "OPEN" region [31745–31808] decode to CLOSE-side actions. The true decoded OPEN region starts at token ~31873.

## What This Invalidates

1. **prefix_locked objective results**: 1-token shift from 31744→31745 moves within CLOSE saturation (both decode to same action). `open_after=1.0` measures wrong region. Result: NOT VALID as gripper-open evidence.

2. **margin scan OPEN regions**: `gripper_margin_to_open` computed against wrong region. Raw logits still valid.

3. **open_region_prob_mass in old gate-lite/gripper-only gate CSVs**: measures probability mass on wrong tokens. Invalid as OPEN metric.

## What Survives

1. **gripper_open_region_ce true OPEN flips**: The objective pushed tokens 128 bins past the wrong region, landing on actual OPEN-decoding tokens. These are real successes — the attack accidentally overcame the wrong region definition.

2. **Token-level flip detection**: `clean_token != adv_token` is still valid (it's region-agnostic). But token flip alone doesn't guarantee decoded OPEN.

3. **armL2, NAD_Z, NAD_DoF1_3, linf_raw**: These are objective-agnostic and remain valid.

## Fix

Added `get_gripper_region_by_decoded_action()` which directly decodes every possible token through the production pipeline and classifies by actual decoded action.

Replaced all `action_bins_for_env_sign("negative"/"positive")` calls for gripper dim with the corrected function.

## Corrected Region

- OPEN tokens: decoded action < 0.5 (true OPEN side)
- CLOSE tokens: decoded action >= 0.5
- Boundary tokens: adjacent OPEN/CLOSE tokens at transition

## Pre-Fix Outputs (Marked Invalid)

| File | Reason |
|------|--------|
| gate-lite partial CSV | open_after measures wrong region |
| gripper-only gate partial CSV | open_after measures wrong region |
| P1 smoke CSV | prefix objective results invalid |
| P3-small partial CSV | region definition invalid |
| margin scan CSV | gripper_margin_to_open uses wrong region |
