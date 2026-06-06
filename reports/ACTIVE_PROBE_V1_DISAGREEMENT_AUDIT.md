# Active Probe V1 — Disagreement Audit

**Date**: 2026-06-07
**Disagreements**: 11 from full31

## Config Comparison

| Aspect | Probe v1 | VIS Label |
|---|---|---|
| PGD steps | 3 | 20 (batch1) / 40 (batch3) |
| Restarts | 1 | 1 (batch1) / 3 (batch3) |
| Effective budget | 3 | 20-120 |
| eps_raw | 6 | 6 |
| Objective | prefix_locked_gripper_open_margin | same |
| env.step | NO | YES |
| Open convention | decoded +1=OPEN | same |
| Frames sampled | ~10/window | 18/window (full attack) |

## By Disagreement Type

- **POSITIVE_LABEL_LOW_PROBE**: 5
- **HIGH_PROBE_NEGATIVE_LABEL**: 4
- **CEILING_POSITIVE**: 2

## Diagnosis Summary

- probe_surrogate_mismatch (x11): no-env decode vs env.step rollout
- probe_surrogate_mismatch (x10): PGD3 probe vs PGD120 VIS (40x budget gap)
- probe_surrogate_mismatch (x5): PGD3 too weak
- VIS PGD120 succeeded with 18/18 open (x5): VIS PGD120 succeeded with 18/18 open
- probe_surrogate_mismatch (x5): PGD budget gap 40x — PGD3 severely underpowered vs PGD120
- label_noise_suspected (x2): VIS trace labeled polluted, probe suggests real susceptibility
- label_noise_suspected (x2): VIS label says no action bridge, but probe induces OPEN
- probe_surrogate_mismatch (x1): PGD3 probe vs PGD20 VIS (7x budget gap)
- ceiling_artifact (x1): clean model already OPEN in 10/10 frames
- ceiling_artifact (x1): clean model already OPEN in 9/10 frames

## Per-Disagreement Detail

### salad_dressing_s5_w28_45

- **Type**: HIGH_PROBE_NEGATIVE_LABEL | label=negative taxonomy=no_action_bridge batch=batch3b
- **Probe**: clean=0/10 (rate=0.0), targeted=7/10 (rate=0.7), random=1/10 (rate=0.1)
- **Probe deltas**: t-c=7, t-r=0, streak=5 (clean streak=0)
- **VIS**: open=0/18, qpos=0.0, phys=0.0, PGD budget=40
- **Diagnosis**: probe_surrogate_mismatch: PGD3 probe vs PGD120 VIS (40x budget gap); probe_surrogate_mismatch: no-env decode vs env.step rollout; label_noise_suspected: VIS label says no action bridge, but probe induces OPEN

### butter_s3_w29_46

- **Type**: HIGH_PROBE_NEGATIVE_LABEL | label=ignore taxonomy=polluted batch=batch3
- **Probe**: clean=3/10 (rate=0.3), targeted=9/10 (rate=0.9), random=4/10 (rate=0.4)
- **Probe deltas**: t-c=6, t-r=0, streak=9 (clean streak=1)
- **VIS**: open=0/18, qpos=0.0, phys=0.0, PGD budget=40
- **Diagnosis**: probe_surrogate_mismatch: PGD3 probe vs PGD120 VIS (40x budget gap); probe_surrogate_mismatch: no-env decode vs env.step rollout; label_noise_suspected: VIS trace labeled polluted, probe suggests real susceptibility

### alphabet_soup_s6_w40_57

- **Type**: HIGH_PROBE_NEGATIVE_LABEL | label=negative taxonomy=no_action_bridge batch=batch3b
- **Probe**: clean=1/10 (rate=0.1), targeted=7/10 (rate=0.7), random=3/10 (rate=0.3)
- **Probe deltas**: t-c=6, t-r=0, streak=4 (clean streak=1)
- **VIS**: open=0/18, qpos=0.0, phys=0.0, PGD budget=40
- **Diagnosis**: probe_surrogate_mismatch: PGD3 probe vs PGD120 VIS (40x budget gap); probe_surrogate_mismatch: no-env decode vs env.step rollout; label_noise_suspected: VIS label says no action bridge, but probe induces OPEN

### salad_dressing_s0_w31_48

- **Type**: HIGH_PROBE_NEGATIVE_LABEL | label=ignore taxonomy=polluted batch=batch3
- **Probe**: clean=3/10 (rate=0.3), targeted=7/10 (rate=0.7), random=3/10 (rate=0.3)
- **Probe deltas**: t-c=4, t-r=0, streak=5 (clean streak=1)
- **VIS**: open=0/18, qpos=0.0, phys=0.0, PGD budget=40
- **Diagnosis**: probe_surrogate_mismatch: PGD3 probe vs PGD120 VIS (40x budget gap); probe_surrogate_mismatch: no-env decode vs env.step rollout; label_noise_suspected: VIS trace labeled polluted, probe suggests real susceptibility

### cream_cheese_s4_w28_45

- **Type**: POSITIVE_LABEL_LOW_PROBE | label=positive taxonomy=claim_usable batch=batch3
- **Probe**: clean=3/10 (rate=0.3), targeted=3/10 (rate=0.3), random=3/10 (rate=0.3)
- **Probe deltas**: t-c=0, t-r=0, streak=2 (clean streak=3)
- **VIS**: open=18/18, qpos=0.038149, phys=1.0, PGD budget=40
- **Diagnosis**: probe_surrogate_mismatch: PGD3 probe vs PGD120 VIS (40x budget gap); probe_surrogate_mismatch: no-env decode vs env.step rollout; probe_surrogate_mismatch: PGD3 too weak; VIS PGD120 succeeded with 18/18 open; probe_surrogate_mismatch: PGD budget gap 40x — PGD3 severely underpowered vs PGD120

### milk_s5_w25_42

- **Type**: POSITIVE_LABEL_LOW_PROBE | label=negative taxonomy=action_positive_physical_positive_task_negative batch=batch3
- **Probe**: clean=2/10 (rate=0.2), targeted=2/10 (rate=0.2), random=1/10 (rate=0.1)
- **Probe deltas**: t-c=0, t-r=0, streak=1 (clean streak=1)
- **VIS**: open=18/18, qpos=0.03799, phys=1.0, PGD budget=40
- **Diagnosis**: probe_surrogate_mismatch: PGD3 probe vs PGD120 VIS (40x budget gap); probe_surrogate_mismatch: no-env decode vs env.step rollout; probe_surrogate_mismatch: PGD3 too weak; VIS PGD120 succeeded with 18/18 open; probe_surrogate_mismatch: PGD budget gap 40x — PGD3 severely underpowered vs PGD120

### alphabet_soup_s4_w4_21

- **Type**: POSITIVE_LABEL_LOW_PROBE | label=positive taxonomy=claim_usable batch=batch3
- **Probe**: clean=6/10 (rate=0.6), targeted=5/10 (rate=0.5), random=5/10 (rate=0.5)
- **Probe deltas**: t-c=-1, t-r=0, streak=3 (clean streak=5)
- **VIS**: open=18/18, qpos=0.032241, phys=1.0, PGD budget=40
- **Diagnosis**: probe_surrogate_mismatch: PGD3 probe vs PGD120 VIS (40x budget gap); probe_surrogate_mismatch: no-env decode vs env.step rollout; probe_surrogate_mismatch: PGD3 too weak; VIS PGD120 succeeded with 18/18 open; probe_surrogate_mismatch: PGD budget gap 40x — PGD3 severely underpowered vs PGD120

### ketchup_s1_w21_38

- **Type**: POSITIVE_LABEL_LOW_PROBE | label=positive taxonomy=claim_usable batch=batch3
- **Probe**: clean=5/10 (rate=0.5), targeted=4/10 (rate=0.4), random=6/10 (rate=0.6)
- **Probe deltas**: t-c=-1, t-r=0, streak=2 (clean streak=3)
- **VIS**: open=18/18, qpos=0.037871, phys=1.0, PGD budget=40
- **Diagnosis**: probe_surrogate_mismatch: PGD3 probe vs PGD120 VIS (40x budget gap); probe_surrogate_mismatch: no-env decode vs env.step rollout; probe_surrogate_mismatch: PGD3 too weak; VIS PGD120 succeeded with 18/18 open; probe_surrogate_mismatch: PGD budget gap 40x — PGD3 severely underpowered vs PGD120

### ketchup_s0_w16_33

- **Type**: CEILING_POSITIVE | label=positive taxonomy=action_positive_physical_strong_task_positive batch=batch1
- **Probe**: clean=10/10 (rate=1.0), targeted=8/10 (rate=0.8), random=9/10 (rate=0.9)
- **Probe deltas**: t-c=-2, t-r=0, streak=5 (clean streak=10)
- **VIS**: open=18/18, qpos=0.038042, phys=1.0, PGD budget=20
- **Diagnosis**: probe_surrogate_mismatch: PGD3 probe vs PGD20 VIS (7x budget gap); probe_surrogate_mismatch: no-env decode vs env.step rollout; ceiling_artifact: clean model already OPEN in 10/10 frames

### milk_s4_w19_36

- **Type**: POSITIVE_LABEL_LOW_PROBE | label=positive taxonomy=claim_usable batch=batch3
- **Probe**: clean=4/10 (rate=0.4), targeted=2/10 (rate=0.2), random=4/10 (rate=0.4)
- **Probe deltas**: t-c=-2, t-r=0, streak=1 (clean streak=4)
- **VIS**: open=18/18, qpos=0.03789, phys=1.0, PGD budget=40
- **Diagnosis**: probe_surrogate_mismatch: PGD3 probe vs PGD120 VIS (40x budget gap); probe_surrogate_mismatch: no-env decode vs env.step rollout; probe_surrogate_mismatch: PGD3 too weak; VIS PGD120 succeeded with 18/18 open; probe_surrogate_mismatch: PGD budget gap 40x — PGD3 severely underpowered vs PGD120

### ketchup_s5_w9_26

- **Type**: CEILING_POSITIVE | label=negative taxonomy=action_positive_physical_positive_task_negative batch=batch3
- **Probe**: clean=9/10 (rate=0.9), targeted=6/10 (rate=0.6), random=9/10 (rate=0.9)
- **Probe deltas**: t-c=-3, t-r=0, streak=3 (clean streak=6)
- **VIS**: open=18/18, qpos=0.037403, phys=1.0, PGD budget=40
- **Diagnosis**: probe_surrogate_mismatch: PGD3 probe vs PGD120 VIS (40x budget gap); probe_surrogate_mismatch: no-env decode vs env.step rollout; ceiling_artifact: clean model already OPEN in 9/10 frames

## Root Cause Assessment

### Primary: PGD Budget Gap (PGD3 vs PGD20-120)

The probe uses PGD3 (3 gradient steps, no restarts) while VIS labels come from
PGD20 (batch1) or PGD40x3 (batch3, effective budget 120). The 7-40x budget gap means:

- PGD3 may fail to find perturbations that PGD20/40 finds (false negatives in probe)
- PGD3 may find perturbations that PGD20/40 avoids (different local minima behavior)
- No-env decode vs env.step further amplifies the discrepancy

### Secondary: Label Noise / Contamination

Several "negative" or "polluted" windows show strong probe signal. The probe is not
necessarily wrong — the VIS trace may have been contaminated or the PGD budget/restart
may have missed a valid attack direction.

### Tertiary: Ceiling Effect

2 windows have clean model already commanding OPEN. Delta-to-clean is invalid.

## Recommendation

1. **Do NOT use PGD3 no-env as a surrogate for PGD20+ env.step VIS attack.**
   The budget gap is too large for reliable proxy.
2. **If a cheap probe is needed, run PGD10 no-env on the 11 disagreement rows**
   to check if higher PGD budget closes the gap.
3. **If PGD20 no-env still disagrees, the no-env probe is fundamentally not a
   reliable surrogate for rollout VIS** — the env.step / temporal dynamics matter.
4. **Audit the 4 HIGH_PROBE_NEGATIVE_LABEL windows** with fresh VIS to rule out
   trace contamination vs genuine probe false positive.
