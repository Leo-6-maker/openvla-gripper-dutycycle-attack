# Official V3 Detector V5 A/B Working-Point Review

Date: 2026-07-19 CST
Code HEAD used for replay: `6424a96e502855dda34c8415a833982188f30554`
Current branch HEAD after Physics decoder and Physics Teacher fixes: `ce74eb4cf6d55430e14297c45903ba6277e61f92`
Scope: FIT Fold-0 validation only; no training, protected-split read, or attack.

## Execution and sealing

The existing matched-smoke checkpoints were replayed with the latest evaluator
in a clean detached server worktree. Each output was written to a new
non-overwrite root. The previous checkpoint and prediction roots were not
modified.

| Candidate | Replay root `SHA256SUMS` SHA | Prediction audit |
|---|---|---|
| V5-A proprio | `b16c05b28099410912ee4aa9672ce09ff99e0e7c92a7cb324cb227a77f4fbff1` | PASS |
| V5-B proprio + policy intent | `0cb49dac148a1ee5a37dd9dbc293a7c4078d43649aca9c41ea763ecbf671a1c6` | PASS |

The A/B comparison root is:
`OFFICIAL_V3_DETECTOR_V5_AB_WORKING_POINT_REVIEW_6424a96_20260718`

Its `SHA256SUMS` SHA is
`791c8edad3958cb402d1682cf7135ae66917482c58fa80021a402699cac3c3a0`.

Both audits report `formal_training_authorized=false` and
`formal_attack_authorized=false`.

## Fixed working-point result

The evaluator uses the frozen grid `0.05..0.95` and selects the maximum
threshold with critical-window recall at least `0.95`. Neither A nor B had a
threshold satisfying that rule, so both returned `working_point_status=HOLD`
and `selected_threshold=null`.

This is a real HOLD, not a missing print field. The current smoke checkpoints
therefore have no valid selected working point under the current rule.

## Causal and online diagnostics

Both candidates use the same 200 validation identities and the same 126 true
mixed episodes. The causal-anchor argmax diagnostic is identical:

`112 / 126 = 0.8888888889`

This is not scheduler selection accuracy and is not a formal viability result.

| Metric | V5-A | V5-B |
|---|---:|---:|
| raw online emits | 172 | 175 |
| outside-rankable scheduler events | 5 | 5 |
| release triggers | 1 | 1 |
| regrasp triggers | 9 | 7 |
| pure-negative episodes | 3 | 3 |
| pure-negative abstention | 0/3 | 1/3 |
| one-shot compliance | true | true |
| protected split reads | 0 | 0 |

The scheduler-selected fields remain null because the working point is HOLD;
the raw scheduler replay is retained only as a diagnostic. The current
implementation correctly separates causal-anchor argmax from selected
working-point metrics, but it does not authorize deployment or attack.

## A/B disagreement

The sealed comparison reports:

- causal disagreement: `12`
- emit disagreement: `21`
- scheduler disagreement: `33`
- release disagreement: `2`
- regrasp disagreement: `4`

Policy intent was consumed only by V5-B. The runtime causality contract remains
`HOLD_RUNTIME_INTEGRATION_MISSING`: the observed attack runner is a fixed-window
CLI with no same-step clean probe and no previous-step policy-intent path.
This review therefore does not claim runtime policy-intent causality.

## Decision

`V5_R1_A_WORKING_POINT = HOLD`

`V5_R1_B_WORKING_POINT = HOLD`

`V5_R1_PREDICTION_SEALS = PASS`

`V5_R1_FORMAL_TRAINING = NOT_AUTHORIZED`

`V5_R1_ATTACK = NOT_STARTED`

The Physics task decoder and Physics Teacher V2 are now independently audited
for all 800 FIT identities at `ce74eb4`; two non-grasp task roles are explicit
and non-rankable. C2F trajectory binding remains open. No GPU process was
started in this replay.
