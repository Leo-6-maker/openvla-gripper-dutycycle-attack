# S20F Tomato Legacy Visual Perturbation Reproduction

Date: 2026-06-14

Scope: tomato_sauce, state 0, fixed window 70-80, oldbase S20D V4 fixed-window
runner, CLI flag `eps_raw_pixels=6`.

Important correction: the historical condition label `vis_pgd` did not establish
that PGD or the `prefix_locked_gripper_open_margin` objective executed. The
archived runner and runtime route audit show that the condition should now be
treated as a legacy raw-RGB Rademacher sign-noise condition unless a later audit
proves otherwise.

Budget note: the CLI flag name is legacy/misleading. The effective budget in
the legacy condition A path is the raw-image fallback adapter's epsilon path,
while the legacy condition B path perturbs processor-produced `pixel_values`.
These are not the same perturbation space.

## Frozen Status

```text
OLD_S20F_BEHAVIORAL_REPRODUCTION: PASS
PROVENANCE_ARCHIVE: PASS
ATTACK_ROUTING_AUDIT: FAILED - legacy vis_pgd label did not establish PGD execution
LEGACY_CONDITION_A_GT_CONDITION_B_CLIP_OPEN_DUTY: 3/3 directionally positive
PAIRED_TASK_EFFECT: 2/3 positive
CONDITION_A_EXECUTION: CONSISTENT_WITH RAW_RGB_RADEMACHER FALLBACK
CONDITION_A_HISTORICAL_RUNTIME: NOT BYTE-PROVEN
CONDITION_B_EXECUTION: PROCESSOR_SPACE_UNIFORM_NOISE
PGD20_EXECUTION: NOT ESTABLISHED
PREFIX_LOCKED_OBJECTIVE_EXECUTION: NOT ESTABLISHED
STATIC_ROUTE_EVIDENCE: STRONGLY_INDICATES_FALLBACK_NON_PGD_EXECUTION
FAIR_SAME_SPACE_RANDOM_CONTROL: NOT ESTABLISHED
DIRECT_C2O_INJECTION: 0/3 supported
DIRECT_VIS_CLASS_FLIP: 0/3 supported
NATIVE_OPEN_CONTROL: not supported
CLIP_MEDIATED_OPEN: observed, token 31744
PHYSICAL_CRITICAL_CLOSURE_DISRUPTION: not yet fully adjudicated
HISTORICAL_RESULT_ROLE: behavioral discovery artifact, not attack-specificity evidence
SIMPLE_POST_WINDOW_OPEN_PERSISTENCE_CAUSALITY: contradicted by current traces
FIRST_CLOSE_DELAY_DOSE_RESPONSE_HYPOTHESIS: plausible, not causal
M3_CURRENT_PREREGISTRATION: CANCELLED
GPU: keep idle
```

Allowed claim:

```text
On the fixed tomato_sauce_s0 window 70-80, the legacy raw-RGB Rademacher
perturbation condition produced greater clip-mediated OPEN duty than the
legacy processor-space uniform-noise condition in 3/3 seeds, with paired task
failure in 2/3 seeds. These conditions differ in perturbation space and
distribution, so the result is not evidence of gradient-optimization
specificity.
```

Forbidden claims:

```text
optimized VIS perturbation success
VIS-PGD20 success
gradient-specific effect
prefix_locked_gripper_open_margin succeeded
VIS superiority over matched-initialization random
VIS superiority over equal-compute random search
same-space random-control fairness
native OPEN control
direct C2O injection
detector-selected success
task-wide or LIBERO-wide vulnerability
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

Runtime dependency manifest:

```text
tables/s20f_tomato_runtime_dependency_manifest_20260614.csv
```

Read-only server audit:

```text
repo: /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607
HEAD: a8e14ba78cdd057de4c92d49d9aa2f0dfc8359bc
dirty status: tracked_modified_count=7, untracked_count=329
attack_adapter current worktree blob: 491c983a75a113335e9777302d1253d4167c3d8b
expected GitHub 7f8a0e4 blob from review: 78e985953da0019237907bd6000424eb0b8de7d7
```

Because the runtime repo is dirty, the exact committed-source provenance is not
fully clean. Therefore this report does not byte-prove that historical condition
A executed the fallback adapter. The static route evidence is still strong:
the archived runner omits `method=token_prefix_pgd`, the current runtime
defaults missing `method` to `visual_linf_noise_adapter`, and `target_action is
None` causes non-untargeted token-prefix objectives to fall back to
`ExistingDenseAttackAdapter`.

Model:

```text
/data/aviary/models/openvla/openvla-7b-finetuned-libero-object
files: 22
bytes: 15085126820
fingerprint: 8d327aa66bd8d5d46521e201a4dca77b394d07fc59a763c227c7db9905e76f15
```

Full artifact manifest:

```text
tables/s20f_tomato_artifact_manifest_20260614.csv
```

The manifest includes the original oldbase non-instrumented condition-B seed80
control run:

```text
/data/liuyu/outputs/s20f_tomato_original_oldbase_control_20260614_pair13_rand80
summary sha256: 48ee1232d0865c3e1a31e4f997c2197de726e6060f4cc276b21b3576e59d0152
trace sha256: db8999f6b588edfde41fb700b72a6ec212a3363356237fe916cb89e13c2db40f
video frames: n=204 bytes=17752225 hash=9e6964807b6806524eb1f803c2c329301892c7d66540c0cab36becd059fdf884
```

## Route Audit

The archived runner constructs:

```text
epsilon
step_size
num_steps
random_start
objective = prefix_locked_gripper_open_margin
arm_preserve_weight
gripper_margin
```

It does not set `method=token_prefix_pgd`. In the audited runtime,
`OpenVLAVisualAttacker` uses:

```text
method = cfg.get("method", "visual_linf_noise_adapter")
```

Therefore the static route evidence strongly indicates that the historical
`vis_pgd` condition routes to the dense fallback adapter rather than
`TokenPrefixPGDAttacker`; because the runtime was dirty, this is not a
byte-proven reconstruction of the historical runtime.

The runner also calls:

```text
attacker.attack(img_uint8, instruction, clean_action, None, gen_out)
```

For non-untargeted objectives, the audited `TokenPrefixPGDAttacker` falls back
when `target_action is None`. Even if `method=token_prefix_pgd` had been set,
this call signature would still not establish PGD execution.

Finally, the runner only consumes adversarial decode when:

```text
attack_result.x_adv is not None
```

True token-prefix PGD returns adversarial processor inputs in
`debug["adv_inputs"]` and leaves `x_adv=None` by design. The old runner was
therefore not wired to consume a true token-prefix PGD result.

## Results

Historical labels are retained for trace compatibility only:

```text
VIS label -> legacy condition A: raw-RGB Rademacher sign noise
RAND label -> legacy condition B: processor-space uniform noise
```

| Seed | Condition | Outcome | OPEN / 10 | Clip-open | Native-close | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 0 | clean | success | 9 | 9 | 1 | baseline |
| 80 | condition B | success | 5 | 5 | 5 | processor-space uniform |
| 80 | condition A | timeout | 10 | 10 | 0 | paired task effect |
| 81 | condition B | success | 5 | 5 | 5 | one tie-ambiguous score row |
| 81 | condition A | timeout | 9 | 9 | 1 | paired task effect |
| 82 | condition B | success | 6 | 6 | 4 | recovers |
| 82 | condition A | success | 7 | 7 | 3 | weak command persistence only |

Directional condition A minus condition B clip-open gains:

```text
seed80: +5
seed81: +4
seed82: +1
```

Clean-relative and condition-B-induced decomposition:

```text
tables/s20f_tomato_effect_decomposition_20260614.csv

seed80: A-B +5, A-clean +1, B-clean -4, B OPEN->CLOSE 5
seed81: A-B +4, A-clean  0, B-clean -4, B OPEN->CLOSE 5
seed82: A-B +1, A-clean -2, B-clean -3, B OPEN->CLOSE 4
```

Thus the current mechanism evidence is not "condition A actively produced extra
OPEN relative to clean in 3/3 seeds." The cleaner statement is that condition A
preserved higher clip-mediated OPEN duty than condition B, while condition B
often perturbed clean-shadow OPEN into executed CLOSE.

## Mechanism Readout

All window OPEN executions in this observation are token `31744`, classified as
`CLIP_MEDIATED_OPEN`. The contrasting CLOSE token is `31872`, classified as
`NATIVE_CLOSE`.

No seed supports direct C2O injection:

```text
missed_close_c2o = 0 for clean/condition-B/condition-A traces
direct class flip count = 0 for seeds 80/81/82
```

Seed81 condition-B step 74 is marked `TIE_AMBIGUOUS`:

```text
condition: random_linf
step: 74
emitted token: 31744 CLIP_MEDIATED_OPEN
score argmax: 31872
top1/top2 gap: 0.0
```

This tie can change the condition-B count by at most one in either direction;
the A-B direction for seed81 remains positive.

## Critical-Close Timing

The main per-step audit is in:

```text
tables/s20f_tomato_critical_close_timeline_steps68_85_20260614.csv
```

Derived event summary:

```text
tables/s20f_tomato_critical_close_event_summary_20260614.csv
```

The event table is a proxy-only adjudication aid. `first_qpos_closed_step_abs_le_0p005`
is already step 0 for all rows, `first_object_motion_change_proxy_step` is 0
for all rows, and `first_EEF_deceleration_proxy_step` is 5 for all rows. These
are marked `NON_DISCRIMINATIVE_PROXY_NOT_CONTACT_EVIDENCE` and must not be used
as contact evidence.

Key observations:

- Clean baseline has a legal clip-open-heavy window: 9x `31744` and 1x `31872`.
- Condition A exceeds condition B on clip-mediated OPEN duty in all three seeds.
- Condition A does not exceed clean in all seeds: seed80 is +1, seed81 is 0,
  seed82 is -2.
- Post-window OPEN rate does not explain the paired task effect: condition B in
  seeds 80/81 has higher post-window OPEN rate than condition A while still
  succeeding.
- First CLOSE timing remains the most useful hypothesis: clean first CLOSE is
  step 78; condition A first CLOSE is delayed to step 81 in seed80 and step 79
  in seed81, both task-positive, while seed82 condition A first CLOSE is step
  77 and the episode succeeds. This is a dose-response hypothesis only.
- First stable 3-step CLOSE is non-discriminative for the current task effect:
  clean is step 85, while condition A seeds 80/81/82 are 84/79/77.

## Instrumentation Diff Audit

Diff classification table:

```text
tables/s20f_tomato_instrumentation_diff_audit_20260614.csv
```

Important limitation: random generation changes `output_scores=False` to
`output_scores=True` so final processed scores can be audited. This is not a
strict no-op, although a control run with the original oldbase runner on pair13
seed80 matched the instrumented condition-B seed80 result at `success/open=5`.
Therefore:

```text
instrumentation no-op: PARTIAL
EXECUTION_EQUIVALENCE_OBSERVED_ON_TESTED_WINDOW
NOT PROVEN BITWISE EQUIVALENT
observed random token behavior change from instrumentation: not observed in pair13 seed80 control
claim impact: exact numerical reproduction remains PARTIAL
```

## M3 Cancellation And Rewrite

The previous M3 preregistration assumed the historical condition A was PGD and
therefore planned to analyze `VIS final delta`, `RAND_INIT_MATCHED`, and a
gradient directional score effect. That premise is invalid.

```text
M3 AS PREVIOUSLY WRITTEN: CANCELLED
```

Clean step74 is also not tie-adjacent:

```text
clean step74: 31744 score 59.0, 31872 score 26.5, gap 32.5
clean step78: 31744 score 41.0, 31872 score 41.25, gap -0.25
seed81 condition-B trajectory step74: true tie-sensitive frame
```

Corrected next GPU milestone, after route assertions are implemented and
reviewed:

```text
M3_CORRECTED_TRUE_PGD_FIXED_FRAME_FEASIBILITY
```

This is a new method feasibility test, not a replay of historical S20F.

M3-0 route assertions must hard-fail unless:

```text
method = token_prefix_pgd
adapter class = TokenPrefixPGDAttacker
target_action = clean_action, or prefix objective explicitly obtains arm target from clean generation
attack_result.debug["adv_inputs"] is used for official generation
attack_method starts with token_prefix_pgd
directional_loss_available == True
num_backwards == 20
num_loss_forwards >= 21
fallback_reason absent
adv_inputs present
x_adv is None by design
Linf <= 6/255
```

M3-1 should run only clean step78 as the first fixed-frame canary, with no
LIBERO rollout:

```text
CLEAN
LEGACY_RADEMACHER
LEGACY_PROCESSOR_UNIFORM
TRUE_PGD_DELTA0
TRUE_PGD_FINAL
RAND20_SAME_PROCESSOR_SPACE
```

Required readouts:

```text
surrogate objective_initial/objective_final
open_region_logsumexp
best_close_score
surrogate margin
gradient norm
exact 7 generated tokens
official emitted gripper token
official processed-score argmax
31744 score
31872 score
global top token/class
official 31744-31872 margin
prefix match
tensor hashes
```

M3-1 gate:

```text
TRUE_TOKEN_PREFIX_PGD_ROUTE
NO_FALLBACK
20 BACKWARDS
EXACT 7 TOKENS
SCORE INVARIANT PASS
TENSOR HASHES PRESENT
LINF PASS
PGD_FINAL surrogate margin > PGD_DELTA0 surrogate margin
PGD_FINAL official 31744-31872 margin > PGD_DELTA0 official margin
```

If surrogate improves but official generation does not improve:

```text
SURROGATE_TO_OFFICIAL_TRANSFER_FAIL
```

Stop before rollout.

Only after step78 passes should M3-2 expand to:

```text
clean step79: near-boundary OPEN frame
clean step74: strong OPEN ceiling/control frame
seed81 condition-B trajectory step74: true tie-sensitive frame
```

Trajectory-state experiments must be marked `REOPTIMIZED_FIXED_STATE`; historical
perturbation tensors were not saved, so they cannot be exact replay.

## Next Order

1. Keep GPU idle until this route-audit revision is reviewed.
2. Implement M3-0 true-PGD route assertions and CPU tests.
3. Run the step78 fixed-frame true-PGD canary only after M3-0 passes review.
4. Expand to step79, clean step74, and seed81 condition-B step74 only if step78 passes.
5. Consider full-window true-PGD rollout only after fixed-frame surrogate-to-official transfer is established.
6. Defer oracle critical-closure rescue until true-PGD reproduces a relevant task or command effect.
