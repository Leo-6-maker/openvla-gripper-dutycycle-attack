# VIS EPS Scale Audit

**Date**: 2026-06-02  
**Branch**: `exp/vis-attack-strength-upgrade-20260602`

## Problem

The TokenPrefixPGD attacker applies Linf perturbation directly in **processor-normalized pixel space** (output of `PrismaticImageProcessor`), not in raw RGB `[0,255]` space. The old code used `EPS = 4.0 / 255.0 ≈ 0.0157` as a processor-space budget without accounting for the normalization transform.

### Processor Normalization

OpenVLA's `PrismaticImageProcessor` applies two sequential normalizations:
1. **DINOv2**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
2. **SigLIP**: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

The final normalization (SigLIP) determines the processor-space budget. With `std=0.5`:

```
delta_processor = delta_raw / (255 * std) = delta_raw / 127.5
```

### Impact of Old EPS=4/255 in Processor Space

| Parameter | Old Value | Effective Raw Equivalent |
|-----------|-----------|-------------------------|
| eps_processor | 0.0157 (4/255) | — |
| Effective raw px | — | **~2.0 px** (0.0157 × 127.5) |
| Standard budget | — | 8 px (typical adversarial) |

The old attack was running at **1/4 of standard adversarial strength**.

## Fix

### Explicit raw-pixel semantics

New default: `--eps_raw_pixels 8` (8 raw pixel values in [0,255]).

```python
eps_processor_c = (eps_raw_pixels / 255.0) / image_std_c
EPS = min(eps_processor_c)  # conservative: most-constrained channel
```

### Run-time processor inspection

Read `processor.image_processor.stds[-1]` at model load time. Do not hardcode `std=0.5`.

Fallback: if `image_std` is unavailable, assume `[0.5, 0.5, 0.5]` (SigLIP default).

### Legacy compatibility

`--eps_processor_direct FLOAT` bypasses raw-pixel conversion for exact reproduction of old behavior.

## Conversion Table

See [tables/vis_eps_scale_conversion.csv](../tables/vis_eps_scale_conversion.csv).

Quick reference (SigLIP std=0.5):

| eps_raw_pixels | eps_processor | Strength vs. old (2 px) |
|---------------|---------------|-------------------------|
| 4 | 0.0314 | 2× |
| 8 | 0.0627 | 4× (default) |
| 12 | 0.0941 | 6× |
| 16 | 0.1255 | 8× |

## Logging

Every run now logs:
```
EPS source: eps_raw_pixels (or eps_processor_direct)
Image mean: [0.5, 0.5, 0.5]
Image std:  [0.5, 0.5, 0.5]
eps_raw_pixels: 8
eps_processor_per_channel: [0.062745, 0.062745, 0.062745]
eps_processor (min across channels): 0.062745
effective_raw_eps_recovered: [8.0, 8.0, 8.0]
```

## Multi-Objective Loss Weights

When using `force_open_z_down_token_ce`, per-dimension loss weights are applied:

| Dimension | Default Weight | CLI Flag |
|-----------|---------------|----------|
| gripper (dim -1) | 1.0 | `--gripper_weight` |
| z (dim 2) | 0.5 | `--z_down_weight` |

The weighted CE loss is computed as:
```
total_loss = (w_grip * CE_grip + w_z * CE_z) / 2
```

## Claim Boundary

- This fix corrects a calibration bug: the old attack had ~2 px effective budget.
- The new default (8 px) matches standard adversarial ML practice.
- `force_open_z_down_token_ce` with Z weight > 0 is a **hybrid positive-control**, not pure gripper duty-cycle.
- Do NOT claim this as a selective or efficient attack without further evidence.
