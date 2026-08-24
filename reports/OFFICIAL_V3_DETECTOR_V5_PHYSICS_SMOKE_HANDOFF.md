# Official V3 Detector V5 Physics Teacher matched smoke handoff

Date: 2026-07-19 CST  
Branch: `codex/official-v3-detector-v5-20260718`  
HEAD: `7fe496330d50df87ad9da7442352a643bd176d41`  
PR: `#87`, Draft  
Base: `archive/official-v3-b3-25d-execution-5e27d7c`

## Scope

This is a FIT-only development smoke using the independently sealed Physics
Teacher V2 proxy. It uses Fold-0, the fixed 80-identity stratified train
subset, all 200 Fold-0 validation identities, seed `20260717`, three epochs,
FP32, and no validation-based early stopping. It does not read FIT-DEV, CAL,
CHECK, states 30--49 semantics, or attack results.

The Physics Teacher remains a clean-only criticality proxy. It is not a
counterfactual attack label, formal training authorization, or attack
authorization.

## Input bindings

- Physics Teacher root:
  `OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V2_ce74eb4_20260719`
- Physics Teacher independent audit root:
  `OFFICIAL_V3_DETECTOR_V5_TEACHER_PHYSICS_V2_AUDIT_ce74eb4_20260719`
- FIT registry root: `OFFICIAL_V3_CAMPAIGN_REGISTRY_V1_d31187f`
- FIT S1 root: `OFFICIAL_V3_S1_FIT_V1_5e27d7c`
- Fold root: `OFFICIAL_V3_FIT_FOLDS_V1_d31187f`
- Stratified subset:
  `OFFICIAL_V3_DETECTOR_V5_STRATIFIED_SMOKE_SUBSET_R2_179d1e2_20260718`
- Policy-intent root used only by `V5_B_PHYSICS`:
  `OFFICIAL_V3_V5_POLICY_INTENT_BINDING_V1_20260718_02`

The Physics Teacher root has 800 identities, 176,336 steps, 5,143 candidate
windows, and an independent audit status of `PASS`. Both model outputs bind
the Teacher root SHA and audit SHA in their manifests.

## Execution allocation

| Candidate | GPU | PID | Training root SHA256SUMS | Checkpoint SHA256 |
|---|---:|---:|---|---|
| `V5_A_PHYSICS` | 2 | 1970494 | `5d076c40d44c943d607a1a7135f3a192b63079b25386111ef33a4817b1fd20e4` | `9eeff7d51747b67f148d32309c49e6b454707e193747754872a0e380491959d6` |
| `V5_B_PHYSICS` | 6 | 1970495 | `1eded8bf24f7f7793b1cdb77fe8c16dda45c70c8ec18a09b45d4b736099a5e0a` | `0a6fe776313f77a437e97199a749ad0243dc7eaba987318cd39abd0fa82332ae` |

GPU2 was idle at launch. An unrelated VLLM process appeared on GPU6 after
`V5_B_PHYSICS` started; it was not stopped or modified. The V5 process used
approximately 540 MiB on that card and no resource failure occurred.

## Development smoke results

Both checkpoint bundles passed the independent V5 checkpoint auditor. Both
were written to new non-overwrite roots and retain checkpoint, identity,
Teacher-root, and clean-only authorization bindings.

| Metric | `V5_A_PHYSICS` | `V5_B_PHYSICS` |
|---|---:|---:|
| true-mixed validation episodes | 94 | 94 |
| causal-anchor highest-tier hits | 53/94 = 0.5638297872 | 56/94 = 0.5957446809 |
| pure-negative episodes | 96 | 96 |
| pure-negative abstention | 95/96 = 0.9895833333 | 96/96 = 1.0 |
| training loss, epoch 1 | 0.8517791220 | 0.8140267815 |
| training loss, epoch 3 | 0.4768278170 | 0.4532133671 |

The causal-anchor value is an argmax diagnostic, not online scheduler
selection accuracy.

## Causal online replay

Each checkpoint was replayed step by step on the same 200 validation
identities. Prediction roots passed the independent prediction auditor:

- A online root SHA256SUMS:
  `b4ed09fdf781b1f71fd4fa5842e3ad15420c9874abf23f85e9468882e896603d`
- B online root SHA256SUMS:
  `9dddd8719915e2faf0645cd891944b466e4580b4088f7946afeae0d82ebe74bb`
- one-shot compliance: `true` for both;
- outside-rankable emits: `0` for both;
- protected split reads: `0`;
- formal training/attack authorization: `false`.

The fixed threshold grid `0.05..0.95` and rule
`maximum_threshold_with_critical_window_recall_gte_0.95` found no eligible
working point for either candidate. Therefore both working points are
`HOLD`. At the default replay threshold, A produced 2 diagnostic emits and B
produced 0; these are not selected working-point results.

## Gate decision

The Physics matched smoke does not pass the scientific gate. Causal-anchor
top-1 is below the fixed `0.90` smoke target and no threshold meets the fixed
recall requirement. The good pure-negative abstention does not offset the
mixed-window selection failure.

```text
PHYSICS_TEACHER_ROOT              = PASS
PHYSICS_TEACHER_INDEPENDENT_AUDIT = PASS
CHECKPOINT_SEALS                  = PASS (A/B)
PREDICTION_SEALS                  = PASS (A/B)
V5_PHYSICS_MATCHED_SMOKE          = HOLD
V5_A_FULL_FOLD0                   = NOT RUN
V5_B_FULL_FOLD0                   = NOT RUN
V5_C/V5_D                         = HOLD (C2F exact binding unavailable)
FIT_DEV                           = NOT READ
CAL                               = NOT READ
CHECK                             = NOT READ
ATTACK                            = NOT STARTED
```

No additional candidate, full Fold-0 run, multi-seed run, or protected-split
access was started after this HOLD. All smoke and replay roots, logs,
manifests, and negative results remain preserved on the server.

## Mutation declaration

- CLEAN/S1/old Teacher/old checkpoint/prediction roots modified: `0`;
- protected split semantic reads: `0`;
- attack/Direct-open/canary/CS200 started: `0`;
- other-user processes stopped: `0`;
- GitHub PR remained Draft; no merge or Ready transition.
