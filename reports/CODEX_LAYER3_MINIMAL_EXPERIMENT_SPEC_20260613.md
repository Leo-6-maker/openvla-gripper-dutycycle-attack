# Codex Layer3 Minimal Experiment Spec - 2026-06-13

This is a future GPU experiment specification. It was not executed by Codex.

## Objective

Test whether `autoregressive_prefix_gripper_open_execspec_v3` produces a
matched advantage over random perturbations and over execspec-v2 under the
unchanged official OpenVLA inference and LIBERO execution pipeline.

## Development parents

Use no more than three development parents:

1. `cream_cheese_s2`
2. `bbq_sauce_s0`
3. `chocolate_pudding_s2`

`butter` may be retained only as a known negative/control reference. It must
not be the sole development parent.

## Conditions

For each parent:

- clean observer: `2` runs only if equivalent clean traces are unavailable;
- matched random perturbation: `3` paired seeds;
- execspec-v2 optimized perturbation: `3` paired seeds;
- proposed v3 optimized perturbation: `3` paired seeds.

Use identical paired seeds across random, v2, and v3.

## Frozen settings

Do not change:

- trigger: first pre-success shadow-clean CLOSE onset;
- epsilon: `6/255`;
- PGD steps: `20`;
- step size: `1.5 * epsilon / 20`;
- gripper margin: `0.5`;
- event horizon: `H=5`;
- attacked clean-CLOSE budget: `B=3`;
- official prompt construction;
- official image preprocessing;
- greedy autoregressive generation;
- official token-to-index decode;
- official `np.clip` behavior;
- raw-action unnormalization;
- LIBERO gripper postprocessing;
- environment stepping;
- official success semantics.

## Candidate commands

Default v2 remains unchanged:

```bash
python scripts/stageb/run_s20d_v6_online_trigger_l3_runner.py \
  --task <task> \
  --state_id <state_id> \
  --condition online_vis_pgd \
  --attack_seed <seed> \
  --pgd_steps 20 \
  --eps_raw_pixels 6 \
  --attack_objective prefix_locked_gripper_top1_open_vs_close_execspec_v2 \
  --event_horizon 5 \
  --max_perturb_frames 3 \
  --output_dir <output_dir>
```

Proposed v3:

```bash
python scripts/stageb/run_s20d_v6_online_trigger_l3_runner.py \
  --task <task> \
  --state_id <state_id> \
  --condition online_vis_pgd \
  --attack_seed <seed> \
  --pgd_steps 20 \
  --eps_raw_pixels 6 \
  --attack_objective autoregressive_prefix_gripper_open_execspec_v3 \
  --event_horizon 5 \
  --max_perturb_frames 3 \
  --output_dir <output_dir>
```

Matched random:

```bash
python scripts/stageb/run_s20d_v6_online_trigger_l3_runner.py \
  --task <task> \
  --state_id <state_id> \
  --condition online_random_linf \
  --attack_seed <seed> \
  --pgd_steps 20 \
  --eps_raw_pixels 6 \
  --event_horizon 5 \
  --max_perturb_frames 3 \
  --output_dir <output_dir>
```

## Primary command metric

`open_count_B3`

Definition: number of official executed OPEN commands among the first three
attacked shadow-clean CLOSE opportunities.

## Secondary command metrics

- `open_duty_B3`;
- maximum OPEN streak;
- first OPEN latency;
- native-token OPEN count;
- clip-mediated OPEN count.

## Physical metrics

Use the best available fixed-window physical readouts:

- `qpos_abs_peak_delta`;
- directly available gripper-width peak change, if present;
- fixed-window gripper response after trigger.

Physical response must remain a separate claim from command duty cycle.

## Task metric

Official environment success. Task success/failure must remain separate from
command and physical claims.

## Offline transfer audit

After artifacts are available, run:

```bash
python scripts/stageb/audit_layer3_objective_transfer.py <artifact_dir> \
  --output-dir <audit_output_dir>
```

Review:

- surrogate loss movement;
- teacher-forced gripper margin;
- generated-prefix gripper margin;
- arm-token match rate;
- final gripper token;
- discrete index before/after official clip;
- native-token OPEN versus clip-mediated OPEN category.

## Advancement gate versus random

A parent advances only when v3 shows matched advantage over random:

- at least two of three paired seeds satisfy
  `open_count_B3(v3) > open_count_B3(random)`;
- and
  `sum(open_count_B3(v3)) - sum(open_count_B3(random)) >= 2`.

## Improvement gate versus v2

V3 must be at least non-regressive against v2:

- at least two of three paired seeds satisfy
  `open_count_B3(v3) >= open_count_B3(v2)`;
- at least one paired seed has strict improvement.

## Clipping-collapse rule

If v3 and random produce the same OPEN duty cycle through the same official
clipping boundary, the parent does not demonstrate VIS specificity.

This rule applies even if both produce many OPEN commands.

## Arm behavior criterion

Do not directly clamp or replace generated arm actions.

Pre-register this comparative criterion:

- v3 must not increase decoded arm-action L2 drift relative to v2 by more than
  25 percent on the paired attacked opportunities, unless it also meets the
  command gate and a physical or task gate.

Report:

- generated arm-token match rate;
- per-dimension token changes;
- decoded arm-action L2;
- per-dimension arm-action differences.

## Strong candidate condition

A strong Layer3 candidate requires command superiority plus at least one of:

- stronger physical gripper response than matched random;
- lower official task success than matched random.

A command-only outcome must remain labeled command-only.

## Falsification condition

For this route, the exact falsification condition is:

`NO_VIS_SPECIFIC_LAYER3_V3_EFFECT_UNDER_THIS_ROUTE`

This conclusion is required for a parent if v3 does not beat matched random by
the command gate, or if apparent gains are entirely explained by the same
clip-mediated OPEN pattern seen in random.

## Allowed claim if the experiment passes

Allowed:

`Under the frozen trigger, visual budget, official OpenVLA generation, official
token clipping, and LIBERO postprocessing, the generated-prefix-conditioned v3
objective produced a matched command-duty-cycle advantage over random on the
tested development parents.`

Add physical or task claims only if those separate gates pass.

## Required claim if the experiment fails

Required:

`The generated-prefix-conditioned v3 objective did not demonstrate
VIS-specific Layer3 behavior under the tested frozen protocol.`
