# S20F Tomato Clip-Mediated Persistence Reproduction

Date: 2026-06-14

Scope: tomato_sauce, state 0, fixed window 70-80, oldbase S20D V4 fixed-window runner, PGD20 VIS, CLI flag `eps_raw_pixels=6`, single-draw matched-seed RAND.

Budget note: the CLI flag name is legacy/misleading. The effective perturbation
budget is `6/255` in the processor-produced `pixel_values` space, not a direct
raw RGB-pixel budget.

## Frozen Status

```text
OLD_S20F_BEHAVIORAL_REPRODUCTION: PASS
VIS_GT_RAND_CLIP_OPEN_DUTY: 3/3 directionally positive
VIS_GT_CLEAN_OPEN_COUNT: seed80 positive, seed81 equal, seed82 negative
RAND_INDUCED_OPEN_TO_CLOSE: 3/3 present
PAIRED_TASK_EFFECT: 2/3 positive
DIRECT_C2O_INJECTION: 0/3 supported
DIRECT_VIS_CLASS_FLIP: 0/3 supported
NATIVE_OPEN_CONTROL: not supported
PHYSICAL_CRITICAL_CLOSURE_DISRUPTION: not yet fully adjudicated
EXACT_NUMERICAL_REPRODUCTION: PARTIAL
CROSS_RUNNER_TRANSFER: not established
CRITICAL_CLOSE_TIMING_CAUSALITY: not fully established
SIMPLE_POST_WINDOW_OPEN_PERSISTENCE_CAUSALITY: contradicted by current traces
FIRST_CLOSE_DELAY_DOSE_RESPONSE_HYPOTHESIS: plausible, not causal
OPTIMIZATION_SPECIFICITY_VS_EQUAL_COMPUTE_RANDOM: not established
RAND_INIT_MATCHED: not established
```

Allowed claim:

```text
On the fixed tomato_sauce_s0 window 70-80, optimized VIS perturbations
produced greater clip-mediated OPEN persistence than single-draw
matched-seed random perturbations in 3/3 perturbation seeds, with paired
task failure in 2/3 seeds.
```

This is a VIS-versus-RAND statement. It must not be read as VIS increasing OPEN
count relative to the clean baseline in all seeds: seed80 is clean-relative
positive, seed81 is equal to clean, and seed82 is clean-relative negative.

Forbidden claims:

```text
native OPEN control
direct C2O injection
detector-selected success
task-wide or LIBERO-wide vulnerability
equal-compute superiority over random search
matched-initialization random superiority
cross-runner stability
```

## Provenance

Original oldbase runner:

```text
/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s20d_v4_fixed_window_l3_runner.py
sha256: 125abeff9d3b525cfaf7f3a76d5882b2ae5c7df2c262fa8633cf432900f587b5
```

Instrumented runner:

```text
/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/scripts/stageb/run_s20f_tomato_instrumented_fixed_window_runner_oldbase.py
sha256: f1c30d23ff92c2be146b15dd532f9adcb4e0cfb78700180e357415ebef72c96b
archived snapshot: scripts/audit/run_s20f_tomato_instrumented_fixed_window_runner_oldbase.py
```

The archived Python file is intentionally byte-preserved. The audit warning is
kept in `scripts/audit/run_s20f_tomato_instrumented_fixed_window_runner_oldbase.README.md`
so the file SHA remains identical to the executed runner.

Model:

```text
/data/aviary/models/openvla/openvla-7b-finetuned-libero-object
files: 22
bytes: 15085126820
fingerprint: 8d327aa66bd8d5d46521e201a4dca77b394d07fc59a763c227c7db9905e76f15
```

Output roots:

```text
seed80: /data/liuyu/outputs/s20f_tomato_instrumented_oldbase_20260614_pair13_seed80_r3
seed81: /data/liuyu/outputs/s20f_tomato_instrumented_oldbase_20260614_pair26_seed81
seed82: /data/liuyu/outputs/s20f_tomato_instrumented_oldbase_20260614_pair45_seed82
```

GPU mapping:

```text
seed80: physical GPU 1,3 render=1
seed81: physical GPU 2,6 render=2
seed82: physical GPU 4,5 render=4
```

Video frame fingerprints:

```text
seed80 VIS frames: n=280 bytes=24296999 hash=419616909b70d4f21fb0d622199fc583a77ba01f93a1848633af3d06ff8f9b72
seed81 VIS frames: n=280 bytes=24440919 hash=ddbeba12ef1553b2794368c165ca587f6e9f54e3e42af54fcccf94479ca931ed
seed82 VIS frames: n=148 bytes=12875645 hash=24cef43e14830a197c7919630f7ac6a8466a67b7892b3471011564feb56fff62
```

Full artifact manifest:

```text
tables/s20f_tomato_artifact_manifest_20260614.csv
```

The manifest includes the original oldbase non-instrumented RAND80 control run:

```text
/data/liuyu/outputs/s20f_tomato_original_oldbase_control_20260614_pair13_rand80
summary sha256: 48ee1232d0865c3e1a31e4f997c2197de726e6060f4cc276b21b3576e59d0152
trace sha256: db8999f6b588edfde41fb700b72a6ec212a3363356237fe916cb89e13c2db40f
video frames: n=204 bytes=17752225 hash=9e6964807b6806524eb1f803c2c329301892c7d66540c0cab36becd059fdf884
```

## Results

| Seed | Condition | Outcome | OPEN / 10 | Clip-open | Native-close | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 0 | clean | success | 9 | 9 | 1 | baseline |
| 80 | RAND | success | 5 | 5 | 5 | single-draw matched seed |
| 80 | VIS | timeout | 10 | 10 | 0 | paired task effect |
| 81 | RAND | success | 5 | 5 | 5 | one tie-ambiguous score row |
| 81 | VIS | timeout | 9 | 9 | 1 | paired task effect |
| 82 | RAND | success | 6 | 6 | 4 | recovers |
| 82 | VIS | success | 7 | 7 | 3 | weak command persistence only |

Directional clip-open persistence gains:

```text
seed80: +5
seed81: +4
seed82: +1
```

Clean-relative and RAND-induced decomposition:

```text
tables/s20f_tomato_effect_decomposition_20260614.csv

seed80: VIS-RAND +5, VIS-clean +1, RAND-clean -4, RAND OPEN->CLOSE 5
seed81: VIS-RAND +4, VIS-clean  0, RAND-clean -4, RAND OPEN->CLOSE 5
seed82: VIS-RAND +1, VIS-clean -2, RAND-clean -3, RAND OPEN->CLOSE 4
```

Thus the current mechanism evidence is not "VIS actively produced extra OPEN
relative to clean in 3/3 seeds." The cleaner statement is that VIS preserved
higher clip-mediated OPEN duty than a single-draw RAND condition that itself
often perturbed clean-shadow OPEN into executed CLOSE.

Paired task effect:

```text
seed80: RAND success, VIS timeout
seed81: RAND success, VIS timeout
seed82: RAND success, VIS success
```

## Historical vs Instrumented Counts

| Seed | Historical RAND / VIS | Instrumented RAND / VIS | Interpretation |
| --- | ---: | ---: | --- |
| 80 | 3 / 10 | 5 / 10 | qualitative task structure repeats, exact count drifts |
| 81 | 5 / 9 | 5 / 9 | exact count repeats |
| 82 | 8 / 10 | 6 / 7 | direction remains positive, exact count drifts |

This supports qualitative paired reproduction, not bitwise trace replay.

## Mechanism Readout

All window OPEN executions in the positive mechanism are token `31744`, classified as `CLIP_MEDIATED_OPEN`. The contrasting CLOSE token is `31872`, classified as `NATIVE_CLOSE`.

No seed supports direct C2O injection:

```text
missed_close_c2o = 0 for clean/RAND/VIS traces
VIS direct class flip count = 0 for seeds 80/81/82
```

Seed81 RAND step 74 is marked `TIE_AMBIGUOUS`:

```text
condition: random_linf
step: 74
emitted token: 31744 CLIP_MEDIATED_OPEN
score argmax: 31872
top1/top2 gap: 0.0
```

This tie can change the RAND count by at most one in either direction; the VIS-RAND direction for seed81 remains positive.

## Critical-Close Timing

The main per-step audit is in:

```text
tables/s20f_tomato_critical_close_timeline_steps68_85_20260614.csv
```

Summary table:

```text
tables/s20f_tomato_clip_persistence_reproduction_summary_20260614.csv
```

Derived event summary:

```text
tables/s20f_tomato_critical_close_event_summary_20260614.csv
```

The event table is a proxy-only adjudication aid. In particular,
`first_qpos_closed_step_abs_le_0p005` is already step 0 for all rows under the
current qpos proxy, `first_object_motion_change_proxy_step` is 0 for all rows,
and `first_EEF_deceleration_proxy_step` is 5 for all rows. These are marked
`NON_DISCRIMINATIVE_PROXY_NOT_CONTACT_EVIDENCE` and must not be used as contact
evidence. The next phase needs a better contact/width-aligned closure
definition.

Key observations:

- Clean baseline has a legal clip-open-heavy window: 9x `31744` and 1x `31872`.
- Seed80 VIS suppresses all window CLOSE events relative to RAND: RAND has five `31872` steps, VIS has zero.
- Seed81 VIS has one `31872` at step 79 but still times out, while clean also has 9/10 OPEN and succeeds. OPEN count alone is therefore not causal enough.
- Seed82 VIS enters `31872` at steps 77-79 and recovers to success, consistent with weak persistence rather than task-level failure.
- Post-window OPEN rate does not explain the paired task effect: seed80 and
  seed81 RAND have higher post-window OPEN rates than VIS while still
  succeeding. The simple "more post-window OPEN causes failure" explanation is
  contradicted by these traces.
- First CLOSE timing is the most useful current hypothesis: clean first CLOSE is
  step 78; VIS first CLOSE is delayed to step 81 in seed80 and step 79 in
  seed81, both task-positive, while seed82 VIS first CLOSE is step 77 and the
  episode succeeds. This is a dose-response hypothesis only, not a causal
  conclusion.
- First stable 3-step CLOSE is non-discriminative for the current task effect:
  clean is step 85, while VIS seeds 80/81/82 are 84/79/77.

The next causal question is not simply "how many OPENs?", but whether the critical CLOSE occurred at the right physical/contact phase.

## Instrumentation Diff Audit

Diff classification table:

```text
tables/s20f_tomato_instrumentation_diff_audit_20260614.csv
```

Important limitation: random generation changes `output_scores=False` to `output_scores=True` so final processed scores can be audited. This is not a strict no-op, although a control run with the original oldbase runner on pair13 seed80 matched the instrumented RAND80 result at `success/open=5`. Therefore:

```text
instrumentation no-op: PARTIAL
EXECUTION_EQUIVALENCE_OBSERVED_ON_TESTED_WINDOW
NOT PROVEN BITWISE EQUIVALENT
observed random token behavior change from instrumentation: not observed in pair13 seed80 control
claim impact: exact numerical reproduction remains PARTIAL
```

## Runner Drift

The merged 6e0a1aa archive runner is not equivalent to the oldbase S20F runner. A trial with the merged archive produced a different RAND80 behavior. This is why all frozen Tomato claims here are scoped to:

```text
oldbase official/V4 fixed-window runner
```

Cross-runner transfer remains unestablished.

## Recommended Next Steps

1. Do not implement native v3.1 from this evidence.
2. Do not train a new detector yet.
3. Keep GPU idle until this follow-up artifact revision is reviewed.
4. Run fixed-frame decomposition next, not rescue: clean-frame replay versus VIS-trajectory clean shadow.
5. Prioritize step 78 as the critical clean decision-boundary frame.
6. Add equal-compute random controls before making an optimization superiority claim.
7. Only after fixed-frame evidence supports a directional score effect, identify critical CLOSE/contact timing and run a minimal oracle critical-closure rescue canary.
8. Treat tomato_sauce_s0_w70-80 as a development parent, not held-out evidence.

## M3 Pre-Registered Frame Set

M3-A should use frozen clean-frame model inference/PGD only, with no LIBERO
rollout:

```text
step74: early RAND CLOSE / tie-adjacent control frame
step78: clean first CLOSE and critical near-boundary frame
step79: post-boundary clean OPEN frame
```

Run each frame under:

```text
CLEAN
RAND_INIT_MATCHED
RAND_BEST_OF_20
VIS_PGD20
```

Required readouts:

```text
delta0 hash
VIS final delta hash
official emitted token
31744 score
31872 score
31744-31872 margin
global top token/class
exec-OPEN basin margin
first six arm tokens
continuous arm L2
Linf
strict invariant
tie-aware invariant
```

M3-B should use fixed VIS/RAND trajectory-state images at seed80 VIS step78/79,
seed81 VIS step78/79, and seed81 RAND step74. The goal is not to show direct
class flips, which are currently 0/3, but to separate:

```text
DIRECT_MARGIN_EFFECT
STATE_MEDIATED_PHASE_SHIFT
MIXED
TIE_SENSITIVE
```

The oracle critical-closure rescue canary is explicitly deferred until M3 shows
that VIS has a directional score effect at the fixed frame level.
