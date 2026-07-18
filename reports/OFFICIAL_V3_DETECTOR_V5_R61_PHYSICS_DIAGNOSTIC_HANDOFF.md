# Official V3 Detector V5 Physics R6.1 Handoff

Date: 2026-07-19  
Code HEAD: `52cfcb81776fa6f4a3ad00b049d1d572e6ccd884`  
Branch: `codex/official-v3-detector-v5-20260718`  
PR: #87, Draft

## Scope

CPU-only diagnosis of the sealed Physics smoke. No retraining, protected-split read, attack execution, or CLEAN/S1 mutation was performed.

## Input and output seals

| Evidence | Root SHA256SUMS SHA |
|---|---|
| Physics Teacher V2.1 | `18f3520351e1291e462656fb1236baa5bc1b5136848a10174e0a4010cc3d38da` |
| Physics Teacher V2.1 independent audit | `87e92c30e1c8bbb90fbd63e8ecf4cf402861c2cc01c770efa8afb845a74a6acc` |
| R6.1 V2.1 geometry/replay | `c37efeb00bc25f157b9f61de933e0f06fc60cd261ba7db06577c7d675776bb57` |
| R6.1 V2.1 A replay | `e1459e6284d54adc133588af32bcec39808e086dcc8ba5598e8e98d0271391b9` |
| R6.1 V2.1 B replay | `038faa4c41a723dda6cc412c46713d4498748fbb5e34aea336bd0e54cdc3cbbe` |

## Window-contract diagnosis

The old V2 loader did not implement the declared Physics segment contract:

| Census | V2 root / old loader |
|---|---:|
| Raw candidate segments | 5,143 |
| Loader rankable windows | 10,963 |
| Raw segments split by loader | 1,607 |
| Raw segments with tier/phase transition | 1,607 |
| Raw segments with internal tier mixture | 322 |
| TRUE_MIXED episodes | 375 |
| TRUE_MIXED attributed to same raw segment | 198 |
| TRUE_MIXED with multiple raw segments | 356 |

The V2.1 loader preserves one complete candidate segment. Its sealed FIT census is:

| Census | V2.1 |
|---|---:|
| Raw candidate segments | 5,143 |
| Loader rankable windows | 5,069 |
| Raw segments split by loader | 0 |
| Raw segments with tier/phase transition | 1,205 |
| Raw segments with internal tier mixture | 294 |
| TRUE_MIXED episodes | 349 |
| TRUE_MIXED from same raw segment | 0 |
| TRUE_MIXED with multiple raw segments | 349 |
| Window-length/max-tier Pearson correlation | 0.01851 |

V2.1 training/evaluation windows therefore represent independent candidate-close segments; tier transitions within one segment are not promoted to separate candidate windows.

V2.1 step counts:

- identities: 800;
- steps: 176,336;
- known steps: 170,107;
- tier 0/1/2/3 steps: 103,869 / 21,358 / 44,125 / 755;
- explicit non-grasp task roles: 2, with no decoder hold.

## Veto ablation on existing sealed A/B predictions

This is replay only; it does not claim a new trained model. The four modes were:

`U`, `U+R`, `U+G`, and `U+R+G`.

For both A and B, all four modes were identical at threshold 0.05:

| Existing model | Critical recall | Mixed correct selection | Selected tier≥2 precision | Pure-negative abstention | Emits |
|---|---:|---:|---:|---:|---:|
| A, V2.1 windows | 6.59% | 6.90% | 54.55% | 95.96% | 12 |
| B, V2.1 windows | 2.20% | 2.30% | 33.33% | 96.97% | 6 |

No threshold in either replay reached the 0.95 critical-window recall working point. Since disabling either or both vetoes did not change the replay, the current failure is not explained by the fixed release/regrasp veto thresholds. It remains consistent with utility/sampling/window-target mismatch.

## Physics Teacher V2.1 changes

- relative pose stability uses the protocol-declared median reducer;
- zero-displacement co-motion is not treated as positive evidence;
- support removal is evaluated relative to the current candidate segment and pre-segment support;
- unknown target progress is removed from the positive utility contribution and remaining positive weights are renormalized;
- V2.1 records component validity, tier onset, and teacher-only causal eligibility;
- loader preserves complete candidate segments.

The old V2 root remains unchanged and is retained as a historical execution reference.

## Gate

```text
R6.1_CPU_DIAGNOSTIC              = PASS
PHYSICS_TEACHER_V2.1_AUDIT       = PASS
WINDOW_CONTRACT_V2.1             = PASS
VETO_CAUSE_ISOLATION              = PASS
V5_PHYSICS_SCIENTIFIC_SMOKE       = HOLD
FULL_FOLD0                        = NOT RUN
FIT_DEV / CAL / CHECK             = NOT READ
ATTACK                            = NOT STARTED
SOURCE_ARTIFACT_MUTATION          = 0
PROTECTED_SPLIT_READS             = 0
```

The next scientifically meaningful step is a category-balanced CPU diagnostic or a new V2.1 development smoke. No GPU run is authorized by this handoff until that diagnostic is reviewed; this result does not establish a proprio/policy observability limit.
