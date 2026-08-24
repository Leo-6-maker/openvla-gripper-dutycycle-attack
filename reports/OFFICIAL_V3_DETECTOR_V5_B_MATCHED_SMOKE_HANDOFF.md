# Official V3 Detector V5-B matched FIT-only smoke handoff

Date: 2026-07-18  
Branch: `codex/official-v3-detector-v5-20260718`  
PR: #87 (Draft)  
Execution environment: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`  
Execution source before this patch: `179d1e22d1a6214a226aa192cf6d6801f164cdbe`

## Scope and boundary

This handoff records a matched FIT-only development smoke for the policy-intent
source binder. It is not a formal training run, a model-selection result, an
attack-value measurement, or an attack authorization.

Only FIT states 0--19 were read. FIT-DEV, CAL, CHECK, final-parent/CS200
semantics, rollout results, Direct-open, canary, and attack roots were not read.
The CLEAN and S1 roots were read-only. No source artifact, old checkpoint,
old prediction root, or old seal was modified.

## Source closure

The source binder and the V5-B loader were exercised against the sealed policy
root:

```text
policy root:
/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_V5_POLICY_INTENT_BINDING_V1_20260718_02
policy SHA256SUMS:
d0a534da50df1f0e341c06d649cd8f52b89707d50b88ff56e02bb2b234451123
policy manifest SHA256:
b85b16447235c658fbb927f4a6be02d968bb25d6d3ebb1d080d624a01516adef
policy feature-order SHA256:
9e2bcc3278f6e8514a73d0ba56a55470f18a094eba7602cbe3dacebacd5504f4
source-artifact-index SHA256:
19f7d5de804b2ce3abacd87b19c5bbb712a599bff922e041e4d5e05df717dc86
records:
800 identities / 176,336 steps
```

The privileged source auditor now fails closed on required-field coverage,
identity and step closure, finite quaternion/EEF/qpos vectors, contact-pair
schema, contact-count parity, generation parity, and policy/action-token
alignment. The official environment targeted test set passed 22 tests, and
the updated scripts passed `py_compile` and `git diff --check`.

An older smoke subset was correctly rejected because it contained Fold-0
validation identities in the training list. It was not repaired or reused.
The sealed replacement subset is:

```text
OFFICIAL_V3_DETECTOR_V5_STRATIFIED_SMOKE_SUBSET_R2_179d1e2_20260718
identity SHA256:
34d0e384dd6d6aed58b33d553d3f7f0b59a5cc62aba488d4f487a51ae962dbcb
80 identities = 4 suites x 10 tasks x 2 identities
```

## Matched smoke design

Both candidates used exactly the same Fold-0 validation list (200 identities)
and the same fixed 80-identity stratified training subset, seed `20260717`,
three epochs, FP32, and A800 execution. The only intended difference was
policy-intent consumption:

| Candidate | Input | Policy root consumed | Checkpoint SHA256 |
| --- | --- | ---: | --- |
| `V5_A_PROPRIO` | Official 25D Student stream | no | `f142a5ba09ae2945966fb76199f18ae1764c62a1883198bd6995d14236298d17` |
| `V5_B_PROPRIO_POLICY_INTENT` | Official 25D + sealed 9D policy-intent stream | yes | `0feec139dfde75603ab24191f93ccbf4396eca43b180f480b28afef8433a99a9` |

Both checkpoint roots passed `SHA256SUMS`, sidecar, and independent checkpoint
auditor checks. Both manifests include the checkpoint SHA, train/validation
identity SHA, registry/S1/Teacher/fold bindings, and `formal_training_authorized
= false`, `formal_attack_authorized = false`,
`eligible_for_model_selection = false`.

## Smoke results

The Teacher geometry is the strict true-mixed denominator: 126 episodes with
both a known candidate window of tier >=2 and a known candidate window of tier
<=1. There are only three pure-negative episodes in the 200-episode validation
set, so their abstention rate is a small-denominator diagnostic.

| Metric | V5-A | V5-B |
| --- | ---: | ---: |
| true-mixed top-1 | 112/126 = 0.8889 | 112/126 = 0.8889 |
| pure-negative abstention | 0/3 = 0.0000 | 1/3 = 0.3333 |
| one-shot compliance | 1.0 | 1.0 |
| total online emits | 172/200 | 175/200 |
| outside-rankable emits | 5 | 5 |
| release-trigger episodes | 1 | 1 |
| regrasp-trigger episodes | 9 | 7 |

These results do not show a policy-intent scientific gain in this small smoke.
They do show that the 9D root can be consumed with a sealed, train-only
normalization and checkpoint binding without changing the 25D control path.
The result remains engineering/development evidence only.

## Causal online replay

The final replay used the newly sealed checkpoint pair and the causal evaluator.
It runs the ranker step by step, gates the scheduler on candidate-close and
Student-valid state, applies the configured dwell/persistence/veto logic, and
allows at most one emission per episode. Retrospective full-window aggregation
is not used by the primary replay.

Final prediction roots:

```text
V5-A:
/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_DETECTOR_V5_A_PROPRIO_MATCHED_ONLINE_EVAL_F0_S20260717_REVIEW2_20260718
SHA256SUMS SHA256:
d9309551fb23c8640034eff9ebd2038c6ee8a84574a26a175fb11077c71c9f22

V5-B:
/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/OFFICIAL_V3_DETECTOR_V5_B_POLICY_INTENT_MATCHED_ONLINE_EVAL_F0_S20260717_REVIEW2_20260718
SHA256SUMS SHA256:
dc770dc750f6bb4c0ec04ec92a9a41bad5ef124688d1eda2e4ae47f9cd0f3143
```

Both prediction roots contain prediction records, scheduler records, episode
metrics, threshold sweep, manifest, and recursive checksum closure. Both passed
the independent prediction-bundle auditor. The online summaries bind the
validation identity SHA and, for V5-B, the policy-root SHA.

## Interpretation and stopping point

`V5_B_SOURCE_BINDING = PASS` for the matched development path. The smoke does
not pass a scientific Gate: the two candidates have identical strict mixed
top-1, and the pure-negative denominator is only 3 episodes. No full Fold-0
run is authorized by this handoff, and no candidate is eligible for model
selection.

```text
PR87_SOURCE_BINDING                 = PASS
PRIVILEGED_AUDITOR                   = PASS
V5_B_MATCHED_LOADER                 = PASS
V5_B_MATCHED_CHECKPOINT_SEAL        = PASS
V5_B_MATCHED_CAUSAL_REPLAY          = PASS
V5_B_SCIENTIFIC_SMOKE               = HOLD
V5_A_FULL_FOLD0                     = NOT RUN
V5_B_FULL_FOLD0                     = NOT RUN
V5_C / V5_D                         = HOLD
FIT-DEV / CAL / CHECK               = NOT READ
ATTACK                              = NOT STARTED
SOURCE_ARTIFACT_MUTATION            = 0
PROTECTED_SPLITS_READ               = 0
```

All smoke roots, failed invocation evidence, logs, manifests, and seals remain
on the server. Raw checkpoints and prediction records are not committed to
GitHub.
