# VIS Token-Flip Threshold Diagnostic Plan

## Purpose

Determine whether TokenPrefixPGD can flip the decoded OpenVLA gripper token without running rollout.

This is a diagnostic harness, not a success claim.

## Script

`scripts/diagnostics/vis_token_flip_threshold.py`

The script defines a reproducible output schema and requires real OpenVLA re-decode integration before producing metric rows. It must consume `debug["adv_inputs"]`; it must not use `action_adv` and must not fallback to zeros.

## Objectives

- `target_action_ce`
- `gripper_open_region_ce`
- `gripper_logit_margin_cw`

## Budgets

- epsilon: `4/255`, `8/255`, `12/255`, `16/255`
- steps: `10`, `20`, `40`

## Required Metrics

- target CE before/after
- open/close bin probability mass before/after
- open-close margin before/after
- decoded clean and adversarial gripper token
- decoded clean and adversarial gripper action
- gripper token flip flag
- arm action L2
- perturbation Linf/L2
- dtype information
- error message if failed

## Rollout Gate

Do not recommend VIS rollout unless:

- decoded gripper token flips at epsilon <= 8/255, or decoded gripper action changes meaningfully with controlled arm drift
- random same-norm baseline does not match the targeted effect

If this gate fails, VIS rollout remains blocked.
