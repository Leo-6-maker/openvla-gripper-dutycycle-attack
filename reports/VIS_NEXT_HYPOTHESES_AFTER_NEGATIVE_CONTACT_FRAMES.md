# VIS Next Hypotheses After Verified Contact-Frame Negative Result

**Date**: 2026-06-01 | **Status**: Future work — do not execute without approval

## H1: Larger ε / Threat-Model Relaxation
- Test ε > 8/255 (e.g., 16/255, 32/255)
- Only meaningful if threat model allows visible perturbation
- Current result is specifically ε≤8/255 — larger budgets untested

## H2: Better Objective
- Direct decoded-token margin (CW-style) instead of CE
- Multi-token open-region objective targeting range of "open" bins
- Temporal accumulation over multiple frames
- Target the action decoding boundary directly

## H3: Alternative Perturbation
- Patch/localized perturbation near gripper/object region
- Image-space trigger / overlay
- Object-centric perturbation

## H4: Different Model Interface
- Attack pre-decode logits instead of token-level CE
- Inspect OpenVLA greedy decoding boundary
- Compare greedy vs sampling/beam decode under perturbation

## Current Status

All VIS experiments BLOCKED. No rollout, no micro, no detector-triggered.

## Valid Negative Result

"Small-budget token-prefix PGD did not produce decoded gripper steering on verified contact/carry frames under ε≤8/255."
