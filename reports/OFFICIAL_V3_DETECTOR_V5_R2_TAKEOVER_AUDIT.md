# Official V3 Detector V5 R2 takeover audit

## Scope

This R2 line replaces V5's earlier bounded smoke diagnostic.  It reads only
FIT states 0--19 and writes new derived evidence.  FIT-DEV, CAL, CHECK,
final-parent data, attack results, and all attack execution remain unread.

## GitHub and execution state

- PR: #87, Draft, base `archive/official-v3-b3-25d-execution-5e27d7c`.
- Branch: `codex/official-v3-detector-v5-20260718`.
- Current source HEAD: `5707dbac11c84f6d679be055363d493763b48000`.
- The official A800 environment was used for the read-only server census and
  FIT loader audit.  No GPU was started for R0--R2.
- The previous V5-A 32-episode smoke remains preserved and is not reused as a
  strict mixed-episode result.

## R0 findings

The prior V5 implementation had seven R2 blockers:

1. `155/182` included positive-only episodes and was not a true mixed
   denominator.
2. Full-window means were retrospective and unsuitable for an early causal
   decision.
3. `UNKNOWN`, `none:*`, non-candidate, and invalid steps were not fully
   excluded from the candidate set.
4. Support and uncertainty heads had no supervision but were exposed by the
   model/scheduler.
5. The loss used max positive/max negative and mean pure-negative pressure,
   not window-balanced ordinal, all-tier pairwise, listwise, and max-negative
   abstention terms.
6. The scheduler had unit tests but no real per-step replay bundle.
7. The first smoke used `train_keys[:32]`, not a fixed stratified subset.

The R2 code fixes these at shared protocol/data/loss boundaries.  The active
heads are utility, release, and regrasp.  Support and uncertainty are disabled
until valid supervision or an ensemble/MC uncertainty source exists.

## R1 window geometry result

Server evidence root:

`OFFICIAL_V3_DETECTOR_V5_WINDOW_GEOMETRY_R2_0d755176_20260718`

The audit consumed the sealed 2000-row registry only after mechanically
filtering states 0--19, the sealed FIT S1 root, and the sealed V5 Teacher
utility V3 root.  The official environment loader independently loaded all
800 episodes and 2,357 windows.

| quantity | value |
|---|---:|
| FIT identities | 800 |
| rankable candidate windows | 2,357 |
| tier-1 / tier-2 / tier-3 windows | 1,252 / 143 / 962 |
| tier-1 / tier-2 / tier-3 steps | 14,500 / 794 / 53,598 |
| true mixed episodes | 507 |
| positive-only episodes | 224 |
| pure-negative episodes | 9 |
| no-candidate episodes | 60 |
| tier3+tier2 episodes | 222 |
| non-contiguous same-ID count | 0 |
| overlap count | 0 |
| student-invalid candidate overlap | 0 |
| UNKNOWN rows | 100,772 |
| `none:*` rows | 100,772 |

The strict primary denominator for top-1 is therefore 507 true mixed
episodes.  The 100,772 UNKNOWN/`none:*` rows are retained as source rows but
never enter ranking or scheduler candidate windows.

The proxy utility has Pearson correlation 0.5513 with window length and
-0.2883 with `time_since_close`; this is a warning that the T10 proxy may be
partly a duration rule, not evidence of attack vulnerability.

## R3--R7 execution result

The fixed stratified subset was rebuilt from the Fold-0 train identity list
after the first builder correctly failed closed on a validation-identity
leak.  The final subset contains 80 identities, exactly two per suite/task,
with identity SHA
`34d0e384dd6d6aed58b33d553d3f7f0b59a5cc62aba488d4f487a51ae962dbcb`.

The sealed development smoke used the official A800 environment, physical
GPU4, FP32, seed `20260717`, three epochs, and 80 train / 200 validation
episodes.  The final checkpoint SHA is
`ba680e67a9c971097ec200c47bc80e91afc1f762e28bc49112b8f7da222c9e7a`.
The duplicate retry checkpoint has the same byte SHA; both roots remain
preserved.

The strict validation result is:

| metric | V5-A R2 |
|---|---:|
| true mixed episodes | 126 |
| causal top-1 hit | 112/126 = 0.8889 |
| pure-negative episodes | 3 |
| pure-negative abstention at 0.5 | 1/3 = 0.3333 (diagnostic) |
| causal online pure-negative abstention | 0/3 = 0.0000 |
| one-shot compliance | 1.0000 |
| total online emits | 172/200 |

The same candidate geometry gives these simple baseline results on all 507
true-mixed FIT episodes.  B4--B7 require a sealed V4 score stream at the
causal anchor; that source was not silently reconstructed, so those four
baselines remain unavailable rather than being treated as zero-performance
methods.

| baseline | true-mixed top-1 |
|---|---:|
| random | 0.2465 |
| earliest | 0.7988 |
| latest | 0.1677 |
| longest window | 0.9290 |

Thus V5-A causal top-1 (0.8889) is below the longest-window baseline
(0.9290), and pure-negative abstention fails the R2 smoke gate.  This is a
development HOLD, not a vulnerability result.  No full 600-episode Fold-0
run is authorized by the frozen gate.

The final smoke checkpoint and causal prediction root were independently
replayed through the V5 checkpoint/prediction auditors in the official A800
environment.  Both returned `PASS`; the checkpoint SHA matched
`ba680e67a9c971097ec200c47bc80e91afc1f762e28bc49112b8f7da222c9e7a`, the
prediction bundle reported 172 emitted episodes, and both formal-training and
formal-attack authorization flags remained false.

Sealed roots:

- geometry: `OFFICIAL_V3_DETECTOR_V5_WINDOW_GEOMETRY_R2_0d755176_20260718`;
- baselines: `OFFICIAL_V3_DETECTOR_V5_BASELINES_R2_0d755176_20260718`;
- stratified train subset: `OFFICIAL_V3_DETECTOR_V5_STRATIFIED_SMOKE_F0_TRAIN_R2_0d755176_20260718`;
- final smoke: `OFFICIAL_V3_DETECTOR_V5_R2_STRATIFIED_SMOKE_F0_S20260717_0d755176_retry3_20260718`;
- causal replay: `OFFICIAL_V3_DETECTOR_V5_R2_STRATIFIED_ONLINE_EVAL_F0_S20260717_0d755176_20260718`.

## R2 state

```text
V5_R2_INFRASTRUCTURE        = PASS
V5_R2_WINDOW_GEOMETRY       = PASS
V5_R2_DATASET_BOUNDARY      = PASS 800/800
V5_R2_CAUSAL_SCORE_API       = PASS / CAUSAL REPLAY SEALED
V5_R2_LOSS_V2                = IMPLEMENTED / CPU TESTED
V5_R2_BASELINE_CODE         = PARTIAL PASS / B0-B3 SEALED; B4-B7 SOURCE ABSENT
V5_R2_ONLINE_EVALUATOR      = PASS / 200-EPISODE REPLAY SEALED
V5_R2_STRATIFIED_SMOKE      = HOLD; top1 below longest baseline, abstention fails
V5_A_FULL_FOLD0             = NOT RUN
V5_B/C/D                    = HOLD; source roots absent
FIT_DEV                     = NOT READ
CAL                         = NOT READ
CHECK                       = NOT READ
ATTACK                      = NOT STARTED
SOURCE_ARTIFACT_MUTATION    = 0
```

The R2 smoke gate is closed as HOLD.  No full Fold-0 run, extra candidate,
additional seed, or protected split read is permitted on this result.  The
shortest defensible next step is protocol review and recovery/audit of a
policy-intent or causal-visual source; those roots are currently absent, so
V5-B/C/D remain HOLD.
