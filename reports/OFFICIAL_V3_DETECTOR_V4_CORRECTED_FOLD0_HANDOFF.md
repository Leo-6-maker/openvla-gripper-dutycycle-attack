# Official V3 Detector V4 corrected development handoff

Status snapshot: 2026-07-18, server evidence roots under
`/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops`.

This report records the first authorized corrected V4 screening execution. It
does not promote a checkpoint to model-selection status and does not authorize
FIT-DEV, CAL, CHECK, or attack execution.

## Source and environment

- Corrected branch: `codex/official-v3-detector-v4-corrected-fold0-20260718`
- Current code HEAD: `7dca398bf71d0d42c83835e1d2cc6f9b0cb01a75`
- Draft PR: [#86](https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack/pull/86)
- PR base: `archive/official-v3-b3-25d-execution-5e27d7c`
- Official execution archive: `5e27d7c4b1a188bc6a78555f94d2571222587805`
- Server environment: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`
- Python 3.10.20, PyTorch 2.2.0+cu121, Transformers 4.40.1
- Detector dtype: FP32
- Training device: `cuda:0` with `CUDA_VISIBLE_DEVICES=6` for the completed
  Fold-0 runs; the physical GPU6 had unrelated resident work and sufficient
  free memory. No unrelated process was changed.

The official SC5 feature order is name-bound and sealed with:

`3d1101d26567a41bf688587a70c5100a3629ad62f12f9568947ee178a8a63366`

The corrected implementation does not use PR #85's positional View B/C
indices. The stale shared-GRU bottleneck claim in PR #85 is not used as a
scientific conclusion.

## Sealed input roots

- Campaign registry: `OFFICIAL_V3_CAMPAIGN_REGISTRY_V1_d31187f`
  - formal FIT selected: 800
  - registry CSV SHA256:
    `09f71b3a9b8250c80735382ba5deab6dbcadfa21b645e4a981eefb114b236af5`
- FIT S1 root: `OFFICIAL_V3_S1_FIT_V1_5e27d7c`
  - root `SHA256SUMS` SHA256:
    `15c97212fde19682a9e3042d6d051c51606b0989881d471cb8eb80f22354b0cf`
  - 800 identities; source root unchanged
- V4 Teacher derivative:
  `OFFICIAL_V3_DETECTOR_V4_TEACHER_V212_caf995e_20260718`
  - root `SHA256SUMS` SHA256:
    `9fb9b90274a5729a25029f5ada0a4b9e3506efbabefbc86efefc3f2cf7738dab`
  - XOR failures: 0; formal training/attack flags: false
- FIT fold root:
  `OFFICIAL_V3_FIT_FOLDS_V1_d31187f`
  - root `SHA256SUMS` SHA256:
    `efeb24ce17c2de0eaf83aaf54099d8043b456dd6acb5c871cfd8e0daae1ef946`

No FIT-DEV, CAL, CHECK, CS200 rollout, or attack result was read.

## Fold-0 execution matrix

All four candidates used fold 0, seed `20260717`, 600 train / 200 validation,
30 epochs, batch 8, AdamW, learning rate `1e-3`, weight decay `1e-5`, gradient
clip 5.0, and FP32. Every checkpoint is a sealed
`FIT_FOLD_TRAINED_CANDIDATE` with `eligible_for_model_selection=false` and
`formal_attack_authorized=false`.

| Candidate | View / loss | checkpoint SHA256 | checkpoint root seal | prediction root seal | status |
|---|---|---|---|---|---|
| C0 | A / quality BCE | `8bf47c639f7a763a629e02bcf8046f3b83b88ea296507885ee5195143bc208d2` | `6d71cbba51426033d23ae5d28a769309bc83351230efe4889fcc0b65dcf0c312` | `14f6a118b290c82a5a58d629b38602732be4884aa61f9b99a05cd624876892c5` | sealed |
| C1 | B / quality BCE | `6efdfd7d3f92291180971cf4e3b43cd1f0f921bec0d3e9e3f7750c0e2c6f2e37` | `763a14461119b4a8b0f696fb1aad3d2424a8206905cff087261c27d7ac151573` | `6a824dd18d0a401a34923739a401a38bb0f388549852a35232e972a4c8760b07` | sealed |
| C2 | B / quality BCE + window ranking | `f87a15e64894ca14b6d6f1aefbc363aac1341429cc3228f0890ce62d0cf2d3d1` | `230e110c80488d823d1b0a311dbfe391471b1c88322a1b9b1594718d0aaf3f78` | `42f6cac493b324134717c7f6e11fe998389298ba15fdb5175760988a24b03bad` | sealed |
| C3 | C / ranking + release auxiliary | `76735844c0d7f23dc17d5d0ca292428add3bb64f009dcab1cbe7d6a636457dee` | `20a098c1616aab36bfce0fb79bb0988539b060753a1949ed48622d2dfaf4c6df` | `38add275d10106a1f9ee8e4bfa390ee5b9daa13ae862e4ba47a3a7dc60c639c7` | sealed |

The prediction bundles each contain 200 validation identities and pass their
own `SHA256SUMS` closure. The baseline comparison below is computed from the
same sealed prediction records' `candidate_close` field; a separate baseline
bundle has not been created, so these values are diagnostic rather than a
new independent model-selection report.

## Fold-0 matched-recall results

The reported working point is the first threshold in the fixed 0.05 grid that
meets the pre-registered valid-event recall target. Exact numerators and
denominators are retained in the prediction summaries.

| Candidate / threshold | valid event hit | later-event hit | invalid-window any emit | mixed invalid any emit | pure-negative any emit | release overlap |
|---|---:|---:|---:|---:|---:|---:|
| close-only diagnostic | 242/242 (1.0000) | 68/68 (1.0000) | 272/272 (1.0000) | 126/126 (1.0000) | 3/3 (1.0000) | 676/676 (1.0000) |
| C0 / 0.50 | 230/242 (0.9504) | 59/68 (0.8676) | 108/272 (0.3971) | 74/126 (0.5873) | 1/3 (0.3333) | 29/676 (0.0429) |
| C1 / 0.50 | 233/242 (0.9628) | 61/68 (0.8971) | 117/272 (0.4301) | 75/126 (0.5952) | 1/3 (0.3333) | 28/676 (0.0414) |
| C2 / 0.60 | 232/242 (0.9587) | 60/68 (0.8824) | 105/272 (0.3860) | 74/126 (0.5873) | 1/3 (0.3333) | 25/676 (0.0370) |
| C3 / 0.65 | 231/242 (0.9545) | 58/68 (0.8529) | 97/272 (0.3566) | 69/126 (0.5476) | 1/3 (0.3333) | 22/676 (0.0325) |

Fold-0 scientific gate outcome:

- `C0` is the corrected control, not a selection result.
- `C1` increases valid recall but does not improve the matched-recall
  invalid-window metric over C0.
- `C2` and `C3` both improve invalid-window and release overlap while retaining
  at least 0.95 valid-event hit and at least 0.80 later-event hit.
- C2 and C3 are therefore the only candidates authorized for the limited
  4-fold x 1-seed screening. This is a screening authorization, not model
  selection or attack authorization.

## Current 4-fold screening state

Preparation for folds 1–3 is sealed:

- six fold-specific normalization roots for C2/C3 were built from the relevant
  600 train identities only;
- six machine-built authorization roots passed input, runner, and checksum
  validation;
- Fold-1 C2 training completed with checkpoint SHA
  `4918bf592fa1153d3c9e74b832a831b610307f78772f1cb19686345985217ae0`;
- Fold-1 C2 checkpoint root seal:
  `6ef4a58f7f4377d32fa586b277e09086170bb8192f9ba4db48670e32f4af303a`;
- Fold-1 C2 final loss: `0.2573144905765851`;
- Fold-1 C2 prediction is currently running or awaiting final seal in the
  non-overwrite root `OFFICIAL_V3_DETECTOR_V4_PRED_F1_C2_S20260717_7dca398_20260718`.

Current screening status is therefore:

`FOLD0 = COMPLETE; FOLD1/C2 = CHECKPOINT SEALED, PREDICTION PENDING; FOLDS2-3 = NOT STARTED.`

No 4-fold aggregate decision exists yet. It must require at least 3/4 folds
with the pre-registered matched-recall Pareto direction, no catastrophic
release fold, later-event macro at least 0.80, and no severe suite/task
collapse.

## What remains blocked

The following actions are intentionally not performed:

- three-seed screening;
- full-FIT refit;
- FIT-DEV access or model selection;
- CAL or CHECK;
- Direct-open validation;
- 48-cell canary;
- CS200 attack rollout or main attack table;
- merge or Ready-for-review transition.

All historical negative results and all candidate bundles are retained. No
CLEAN/S1 artifact, old manifest, old checksum, or attack result was modified.

## Review request

Please review PR #86 as a Draft against the immutable Official V3 archive.
The important review boundary is that Fold-0 C2/C3 only passed a pre-registered
screening gate and remain candidates; no scientific claim about cross-fold
generalization or attack benefit is made until the limited 4-fold screen is
sealed.

## State matrix

```text
V4_CORRECTED_FEATURE_BINDING       = PASS
V4_TEACHER_XOR_AND_PHASE_CONTRACT  = PASS
V4_FORMAL_AUTHORIZATION            = PASS FOR COMPLETED RUNS
V4_CHECKPOINT_SEAL                 = PASS C0-C3 FOLD0; PASS F1 C2
V4_PREDICTION_SEAL                 = PASS C0-C3 FOLD0; F1 C2 PENDING
CORRECTED_FOLD0                    = PASS SCREENING GATE
4FOLD_X1SEED_SCREENING             = IN PROGRESS (C2/C3 ONLY)
4FOLD_AGGREGATE                    = NOT YET AVAILABLE
3SEED_MATRIX                       = HOLD
FIT_DEV                            = NOT READ
CAL                                = NOT READ
CHECK                              = NOT READ
ATTACK                             = NOT STARTED
SOURCE_ARTIFACT_MUTATION           = 0
RESAMPLING                         = NOT PERFORMED
```
