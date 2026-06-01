# VIS Repeatability Results — Correct Decode Path

**Date**: 2026-06-01 | **Branch**: `exp/vis-payload-upgrade-validation-20260601`

## Configuration

- **Model**: OpenVLA 7B finetuned LIBERO-Object
- **Environment**: `openvla_official_libero_20260525`
- **Objective**: `gripper_open_region_ce`
- **Budget**: eps=4/255, steps=20, step_size=1/255
- **Decode**: prompt() wrapper + action prefix token 29871 (corrected)

## Phase B — 10-Seed Deterministic Repeat

| Frame | Clean Grip | Targeted (10 seeds) | Random (5 seeds) | Arm L2 range | Verdict |
|-------|-----------|---------------------|-------------------|-------------|---------|
| ketchup_0098 | 0.0000 | **10/10 open** | 0/5 flip | 0.839 | STRONG POSITIVE |
| tomato_0134 | 0.0000 | **0/10** | 0/5 flip | 0.178 | STRONG NEGATIVE |
| ketchup_0050 | 0.9961 | 10/10 close* | 0/5 flip | 0.512 | Wrong direction* |

*Note: ketchup_0050 already has clean grip=0.9961 (fully open), so flip to close is expected but not useful for attack.

**Gate B: PASS** — ketchup contact 10/10 reproducible, random baseline all negative, tomato negative remains stable.

## Phase C — Multi-Frame Expansion

### Ketchup (robust control)

| Frame | Clean Grip | Open/Total | Arm L2 | Notes |
|-------|-----------|-----------|--------|-------|
| 0096 | ~0.0 | 3/3 | 0.900 | Adjacent also positive |
| 0097 | ~0.0 | 0/3 | — | Frame-specific boundary |
| 0098 | 0.0 | 10/10 | 0.839 | Strongest positive |
| 0099 | ~0.0 | 0/3 | 0.827 | Very next frame fails! |
| 0100 | ~0.0 | 1/3 | 0.900 | Partial |
| 0105 | ~0.0 | 0/3 | — | Negative |
| 0110 | ~0.0 | 3/3 | 0.998 | Positive again |
| 0120 | ~0.0 | 0/3 | 0.824 | Border region |
| 0130 | ~0.0 | 0/3 | — | Negative |

### Tomato Sauce (HIGH-SENSITIVE)

| Frame | Clean Grip | Open/Total | Arm L2 | Notes |
|-------|-----------|-----------|--------|-------|
| 0126 | ~0.0 | 0/3 | 0.131 | No effect |
| **0130** | ~0.0 | **3/3** | **0.131** | **Very low arm L2!** |
| 0134 | 0.0 | 0/10 | 0.178 | Strong negative |
| **0138** | ~0.0 | **3/3** | **0.155-0.191** | **Very low arm L2!** |
| 0142 | ~0.0 | 3/3 | 0.784-1.183 | Higher arm drift |
| 0150 | ~0.0 | 3/3 | 0.922 | Higher arm drift |
| 0160 | ~0.0 | 0/3 | — | Negative |
| 0170 | ~0.0 | 0/3 | 0.861-0.909 | No grip change despite arm L2 |
| 0180 | ~0.0 | 0/2 | 0.515 | Crashed (GPU Xid 31) |

### Cream Cheese (HIGH-SENSITIVE) — GPU4,5 scan

| Frame | Clean Grip | Open/Total | Arm L2 | Notes |
|-------|-----------|-----------|--------|-------|
| 0065 | 0.0 | 3/3 | 0.489 | Positive |
| **0070** | 0.0 | **3/3** | **0.110-0.114** | **LOWEST ARM L2!** |
| 0075 | 0.0 | 0/3 | 0.598-1.074 | Arm L2 high, NO grip change — selectivity! |
| 0080 | 0.0 | 3/3 | 0.597 | Positive |
| 0085 | 0.0 | 3/3 | 0.844 | Positive |

**Gate C: PASS** — Multiple frames across 3 tasks show effect. Both high-sensitive tasks (tomato, cream_cheese) have positive frames. Cream_cheese step_0070 shows the effect with minimal arm drift (0.11).

## Key Insight: Frame-Specificity

The effect is highly frame-specific:
- ketchup step_0098 works (10/10) but step_0099 (1 step later) fails (0/3)
- tomato step_0134 fails (0/10) but step_0138 (4 steps later) works (3/3)
- tomato step_0130 works (3/3) but step_0126 (4 steps earlier) fails (0/3)

This specificity suggests the PGD succeeds only at specific visual states where the open/close token boundary is near a decision threshold.

## Random Baseline

All random same-Linf (4/255) baselines on Phase B frames show **0/5 token flips** and **0/5 meaningful grip changes**. The effect is NOT reproducible by random perturbation.
