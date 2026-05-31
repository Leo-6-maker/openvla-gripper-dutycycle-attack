# Next-Stage VIS Options After Verified Contact-Frame Negative Result

**Date**: 2026-05-31 | **Status**: Future work only — do not execute without approval

## A. Larger ε / Threat-Model Relaxation

Determine whether VIS can flip gripper token at ε > 8/255 (e.g., 16/255, 32/255). Only meaningful if the threat model allows larger or visible perturbation.

## B. Alternative Objectives

- Margin-based objective over decoded action tokens
- Multi-token open-region CE targeting a range of "open" bins
- Temporal accumulation: attack over multiple frames rather than single frame
- Patch-style perturbation: restrict perturbation to image region near gripper/object

## C. Model-Interface Direction

- Attack pre-token logits directly rather than token-level CE
- Inspect OpenVLA decoding boundary to understand token-flip threshold
- Compare greedy decode vs sampling/beam decode behavior under perturbation

## D. VIS as Documented Negative Result

The current result can be used to state:

> "Small-budget token-prefix PGD did not produce decoded gripper steering on verified contact/carry frames under ε≤8/255."

## E. Cross-Suite

Continue with CrossSuite-ProprioNoStep-v2 only after:
- Clean teacher labels and full feature rows are ready
- Relative EEF features implemented
- Task-holdout split designed
- Do NOT run cross-suite sus30

## Current Status

All experiments BLOCKED. No rollout, no training, no VIS micro.
