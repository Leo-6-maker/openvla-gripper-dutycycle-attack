# Codex Layer3 Feasibility Audit - 2026-06-13

## Executive verdict

`execspec-v2` is correct about the corrected RC1a OPEN/CLOSE token region and
the gripper logit row, but it does not optimize the same conditional
distribution that controls the actually executed seventh action token whenever
the perturbed image changes the first six autoregressively generated action
tokens.

Verdict: a scientifically defensible Layer3-v3 route exists under the frozen
official execution protocol. The selected primary route is
`autoregressive_prefix_gripper_open_execspec_v3`, implemented as a generated
arm-prefix-conditioned gripper objective with stop-gradient discrete prefix
refresh.

This is not evidence of a successful VIS-specific robot effect. It is an
implementation-ready route and CPU-tested prototype. Future GPU rollouts must
still beat matched random perturbations under pre-registered gates.

## Current Layer3 status

- Active base branch inspected: `origin/exp/vis-prefix-margin-repair-20260603`.
- Selected base SHA: `c30064a37b656d834c631601153e5d059b914362`.
- Reviewed handoff SHA in the task: `d813031c4b68721b315091251b656b5507ba3c42`.
- `d813031...` is an ancestor of the current remote experiment branch.
- New remote commits after `d813031...`:
  - `7f60014...`: adds B3/qpos observation fields.
  - `10fedf9...`: adds the 4-parent mini launcher.
  - `c30064a...`: adds the V7 autonomous experiment controller.
- These newer commits affect telemetry and launch support, not the official
  OpenVLA decode semantics. The v3 branch is based on `c30064a...` for
  compatibility.

Current frozen parameters remain:

- trigger: first pre-success shadow-clean CLOSE onset;
- epsilon: `6/255`;
- PGD steps: `20`;
- step size: `1.5 * epsilon / 20`;
- margin: `0.5`;
- event horizon: `H=5`;
- attacked clean-CLOSE budget: `B=3`;
- current objective: `prefix_locked_gripper_top1_open_vs_close_execspec_v2`.

The frozen butter result remains `NO_VIS_SPECIFICITY`: matched RAND and VIS
were equivalent at command duty cycle, physical response, and task outcome,
with RAND OPEN commands associated with official decoder clipping.

## Exact execspec-v2 computation graph

1. The runner reads `obs["agentview_image"]`.
2. `decode_with_scores(...)` preprocesses the clean image through the corrected
   OpenVLA image path: prompt text, processor image, dropped `attention_mask`,
   optional EOS-style `29871`, and greedy `model.generate`.
3. `model.generate(..., max_new_tokens=action_dim)` produces seven action
   tokens autoregressively.
4. The clean action tokens are extracted from `gen.sequences[0, -action_dim:]`.
5. `decode_with_scores` computes discrete indices as
   `vocab_size - token_ids - 1`.
6. The official decoder applies `np.clip(..., 0, n_bins - 1)`.
7. Bin centers are unnormalized through `model.get_action_stats`.
8. The raw seven-dimensional action is postprocessed by
   `normalize_gripper_action(binarize=True)` and `invert_gripper_action`.
9. Raw gripper `> 0.5` is physical/env OPEN; raw `< 0.5` is CLOSE; raw `== 0.5`
   is boundary and must be reported separately.
10. For VIS, the runner calls `OpenVLAVisualAttacker.attack(...)` with
    `target_action = clean_action`.
11. `TokenPrefixPGDAttacker.action_to_token_ids(clean_action, unnorm_key)`
    converts the clean action into seven target token ids.
12. `_build_inputs_and_labels` builds prompt `input_ids`, appends the seven
    target action tokens, and creates labels only at the action-token positions.
13. Prefix-locked v2 masks the seventh gripper label to `-100` and preserves
    labels for arm dims 0 through 5.
14. `_loss(... objective=prefix_locked_gripper_top1_open_vs_close_execspec_v2)`
    runs a teacher-forced forward over `prompt + target_action_tokens`.
15. `action_token_logit_row_index(6, 7)` returns `-2`; this is the row that
    predicts the appended seventh action token. It is the correct row for the
    teacher-forced sequence and does not use prompt or EOS rows.
16. The gripper term is
    `relu(max_close_logit - max_open_logit + margin)` using corrected decoded
    raw-action token regions.
17. The arm preservation term is CE over exactly the first six action dims;
    the gripper dim is excluded defensively.
18. The total selected loss is minimized by signed gradient descent on
    processor-space `pixel_values`.
19. The FP32 master perturbation is projected to `x_orig +/- epsilon`.
20. The projected image is cast to model dtype; any cast-induced budget
    overshoot is reset to the original model-dtype value.
21. `AttackResult` stores `action_adv=None`, `x_adv=None`, and
    `debug["adv_inputs"] = {"input_ids": clean prompt ids, "pixel_values":
    adversarial model input}`.
22. The runner extracts `adv_inputs` and performs a fresh greedy
    autoregressive generation from `prompt + adversarial image`.
23. The generated seven tokens are decoded through the same official
    token-to-index, `np.clip`, unnormalization, and gripper postprocessing path.
24. The final `env.step` action comes from this actual autoregressive decode.

## Teacher-forced/generated-prefix verdict

The current v2 loss optimizes:

`p(gripper token | perturbed image, clean/target teacher-forced first-six action-token prefix)`

The executed path uses:

`p(gripper token | perturbed image, first-six action tokens generated from that perturbed image)`

These are equal only if the first six generated tokens match the clean/target
prefix or if the model's gripper distribution is invariant to the prefix. The
current implementation does not enforce either condition. Deterministic CPU
test `test_teacher_forced_and_generated_prefix_conditioning_are_distinct`
constructs a mock model where teacher-forced context favors OPEN while the
generated-prefix context favors CLOSE, proving the practical distinction.

## Demonstrated limitations

1. Teacher-forced/AR mismatch:
   - v2 computes gripper loss on `full_input_ids = prompt + target tokens`.
   - actual execution regenerates all seven action tokens from `adv_inputs`.
   - CPU tests show these contexts can produce opposite gripper margins.

2. Decoder clipping collapse:
   - official decoding clips out-of-native-range token-derived indices.
   - matched RAND and VIS can both map to the same clipped OPEN bin.
   - if both conditions achieve the same OPEN duty cycle through the same clip
     boundary, this cannot establish VIS specificity.

3. Arm-prefix drift:
   - v2 has an arm CE term over dims 0..5, but it does not directly measure or
     condition the gripper loss on the first six generated tokens.
   - first-six token drift can change the gripper context even when the
     teacher-forced arm CE improves.

4. Temporal limitation:
   - `TokenPrefixPGDAttacker` has optional `temporal_init` and smoothing state.
   - the frozen S20D runner config does not set these options.
   - there is no explicit B3 duty-cycle objective, no multi-frame loss, and no
     anti-cancellation term across the first three attacked opportunities.

5. Numerical sensitivity:
   - optimization uses an FP32 master image and casts to model dtype for
     forward passes.
   - BF16/fp16 cast overshoot is corrected in budget telemetry.
   - BF16 may matter near greedy argmax boundaries, but no current source
     evidence supports attributing the butter collapse solely to BF16.

## Decoder-clipping sensitivity

Relevant runner telemetry already records:

- generated token id;
- discrete index before clip;
- discrete index after clip;
- clipped boolean;
- decoded raw gripper;
- executed environment gripper;
- native OPEN and clip-mediated OPEN counts.

Answer: the current objective cannot establish VIS specificity if optimized
VIS and matched RAND frequently saturate the same official clipping boundary
and produce equivalent command duty cycle, physical response, and task outcome.
The official clipping behavior must remain unchanged, but equal clipping
behavior must be labeled a collapse, not a win.

## Arm-prefix preservation audit

The v2 arm CE indexing is correct for the teacher-forced sequence:

- action dim is seven;
- dims 0 through 5 are arm dims;
- dim 6 is gripper;
- `action_token_logit_row_index(0, 7) == -8`;
- `action_token_logit_row_index(5, 7) == -3`;
- `action_token_logit_row_index(6, 7) == -2`;
- the gripper label is masked in prefix-locked mode;
- `_active_label_rows` sees only the arm rows after masking;
- the gripper row is excluded defensively in the arm CE loop.

The limitation is not an off-by-one bug. The limitation is that the arm CE is
computed under the teacher-forced target prefix and does not guarantee the
first six autoregressively generated tokens remain identical under the
perturbed image.

## Temporal audit

The current frozen runner attacks up to the first three clean-CLOSE
opportunities inside the event horizon. It treats each eligible frame as a
separate `attacker.attack(...)` call, with random start enabled and no explicit
temporal duty-cycle loss. The attacker class can carry previous perturbations
when `temporal_init` is configured, but the frozen S20D setup does not use that
as a contribution.

Therefore there is no explicit mechanism for:

- persistent OPEN direction across B3;
- multi-frame cumulative OPEN objective;
- prevention of frame-to-frame sign cancellation;
- causal temporal smoothing in the frozen config.

## Ranked route table

See `tables/codex_layer3_route_ranking_20260613.csv`.

Summary:

1. Generated-prefix-conditioned gripper objective: primary, implemented.
2. Native-OPEN-focused margin: fallback candidate only.
3. Stronger arm-prefix preservation: useful component, not standalone primary.
4. Causal temporal duty-cycle objective: useful later, not the smallest answer
   to the demonstrated v2 mismatch.
5. Decoder-boundary sensitivity objective: rejected as scientifically fragile.

## Selected primary route

Name: `autoregressive_prefix_gripper_open_execspec_v3`.

Algorithm:

1. Build the same clean prompt inputs and clean/target action token labels as
   v2.
2. Preserve clean arm token labels for dims 0..5.
3. For each refresh point during PGD, greedily generate the first six action
   tokens from the current perturbed image.
4. Treat those six token ids as stop-gradient discrete context.
5. Run a differentiable forward on `prompt + generated_first_six_tokens`.
6. Read row `-1`, the row predicting the next token, as the seventh gripper
   token distribution under that generated prefix.
7. Minimize `relu(max_native_CLOSE_logit - max_native_OPEN_logit + margin)`.
8. Add the existing clean-arm CE preservation term.
9. Update only processor-space image pixels and project within epsilon.
10. Return only adversarial processor inputs; downstream execution still
    performs fresh full autoregressive generation and official decoding.

Gradient validity:

- gradients flow through the model forward from image pixels to the gripper
  logits conditioned on fixed token ids;
- gradients do not flow through greedy token selection;
- the discrete generated prefix is explicitly stop-gradient context.

## Selected fallback route

Fallback candidate: native-OPEN-focused margin under the same generated-prefix
conditioning. It would be selected only if v3 transfer is dominated by
clip-mediated OPEN collapse. It remains a future route; it is not implemented
as a separate objective in this PR.

## Rejected alternatives

- Stronger arm preservation alone: it may reduce drift but does not optimize
  the gripper token under the actual generated prefix.
- Temporal duty-cycle objective alone: it addresses B3 persistence but not the
  demonstrated teacher-forced/AR mismatch.
- Decoder-boundary sensitivity objective: it risks overfitting the official
  clipping implementation and has weak claim language unless it beats matched
  RAND without clipping collapse.
- Changing official decoding, masking token ranges, or replacing actions:
  rejected because these violate official parity.

## Implementation summary

Changed code:

- `src/gripper_attack/attack_adapter.py`
  - adds `autoregressive_prefix_gripper_open_execspec_v3`;
  - adds generated-prefix refresh helpers;
  - adds generated-prefix margin telemetry;
  - preserves execspec-v2 unchanged.
- `scripts/stageb/run_s20d_v6_online_trigger_l3_runner.py`
  - adds `--attack_objective` with default v2;
  - default behavior is unchanged;
  - v3 can be selected explicitly for future runs.
- `scripts/stageb/audit_layer3_objective_transfer.py`
  - offline artifact-only CSV/JSON diagnostic;
  - no model load, no rollout, no GPU.
- `tests/stageb/test_layer3_autoregressive_prefix_v3.py`
  - deterministic CPU mock-model tests.

Key debug fields added for v3:

- `objective_name`;
- `method_version`;
- `prefix_refresh_strategy`;
- `prefix_refresh_interval`;
- `prefix_refresh_count`;
- `clean_arm_prefix_token_ids`;
- `generated_arm_prefix_token_ids`;
- `arm_token_match_rate`;
- `teacher_forced_gripper_margin_initial/final`;
- `generated_prefix_gripper_margin_initial/final`;
- `selected_loss_initial/final`;
- `arm_preservation_loss_initial/final`;
- `num_loss_forwards`;
- `num_generation_forwards`;
- `num_backwards`;
- `pixel_budget_master_linf`;
- `pixel_budget_adv_inputs_linf`.

## CPU test results

Commands run:

```bash
python -m py_compile src/gripper_attack/attack_adapter.py scripts/stageb/run_s20d_v6_online_trigger_l3_runner.py scripts/stageb/audit_layer3_objective_transfer.py tests/stageb/test_layer3_autoregressive_prefix_v3.py
pytest -q tests/stageb/test_layer3_autoregressive_prefix_v3.py
pytest -q tests/stageb/test_layer3_autoregressive_prefix_v3.py tests/stageb/test_openvla_libero_exec_spec.py tests/stageb/test_attack_open_token_region.py tests/v4/test_prefix_locked_loss_contains_gripper.py
```

Observed results:

- `py_compile`: pass.
- focused v3 tests: `5 passed`.
- focused regression set: `26 passed`.

Coverage:

- action-position indexing;
- teacher-forced versus generated-prefix distinction;
- gradient direction and epsilon projection;
- OPEN/CLOSE/boundary token regions;
- official clip parity;
- no direct action replacement;
- arm-preservation indexing;
- prefix refresh count and stop-gradient prefix behavior.

## Future minimal experiment

See `reports/CODEX_LAYER3_MINIMAL_EXPERIMENT_SPEC_20260613.md`.

## Claim boundaries

Allowed after this PR:

- v2 optimizes a teacher-forced clean/target-prefix gripper distribution.
- actual execution uses full autoregressive regeneration from adversarial
  inputs.
- v3 is a CPU-tested prototype that conditions the gripper loss on refreshed
  generated arm-prefix tokens while preserving official execution parity.

Not allowed:

- claiming v3 demonstrates VIS specificity;
- claiming command-only improvement implies physical response;
- claiming physical response implies task failure;
- treating equal RAND/VIS clip-mediated OPEN as VIS specificity.

## Remaining risks

- Generated-prefix refresh adds runtime cost and may be noisy near greedy
  argmax boundaries.
- The stop-gradient prefix is a principled surrogate but not a gradient through
  discrete generation.
- Arm CE may still be too weak to preserve executed arm behavior.
- If RAND and v3 both hit the same clip boundary, the result remains
  non-specific.
- Temporal B3 duty cycle is still evaluated by future rollouts, not optimized
  directly in this prototype.

## Final recommendation

Run the future 3-parent paired mini only after review. Advance a parent only if
v3 beats matched random on `open_count_B3` under the pre-registered gates and
does not collapse to the same clip-mediated OPEN pattern as RAND.

If the future mini fails those gates, the required conclusion is:

`NO_VIS_SPECIFIC_LAYER3_V3_EFFECT_UNDER_THIS_ROUTE`
