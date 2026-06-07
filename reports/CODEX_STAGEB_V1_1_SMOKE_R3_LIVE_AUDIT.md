# Codex Stage-B v1.1 Smoke R3 Live Audit

Date: 2026-06-07

Server output root: `/data/liuyu/outputs/stageb_v1_1_smoke_r3_20260607`

Mode: CPU-only read-only server audit. No GPU job, VIS rerun, rollout worker, watcher, or server live output mutation was started by Codex.

## Verdict

`TRACE_SCHEMA_PASS_WINDOW_SELECTION_FAIL_LABEL_BUILDER_PATCH_REQUIRED`

The actual smoke traces validate the v1.1 runner schema and corrected gripper semantics, but the selected butter/ketchup windows were unreachable under the official prompt + official image preprocessing trajectory. These no-window traces must not become negative labels.

Codex patched `scripts/stageb/build_pair_labels_v1_1.py` so any paired row with `n_window_steps <= 0` or `n_attack_steps <= 0` hard-fails instead of silently becoming a negative label.

## Server State Observed

- Server repo worktree: `/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605`
- Server git HEAD observed: `ca3a97e73965e2582e28066a93892e3dd9c24617`
- Server worktree is dirty/manual-uploaded, not a clean checkout of `e878bae`.
- Actual runner trace metadata records `git_commit=ca3a97e`, `git_dirty=1`.
- The smoke command lines used explicit shared pair IDs: `p01`, `p02`, `p03`.

## Completed Smoke Outputs

Six traces and six summaries were present after the smoke completed:

- `butter s6 [182,192]`: VIS + random, pair `p01`
- `ketchup s8 [154,164]`: VIS + random, pair `p02`
- `cream_cheese s7 [91,101]`: VIS + random, pair `p03`

Validator result:

```text
Valid: 6  Failed: 0
```

Warnings:

- butter VIS/random: `NO_IN_WINDOW_STEPS`
- ketchup VIS/random: `NO_IN_WINDOW_STEPS`
- cream_cheese VIS/random: no validator warning

## Trace Schema Audit

All six traces had:

- required v1.1 trace columns present
- `trace_version = corrected_stageb_v1_1`
- `qpos_source = obs_robot0_gripper_qpos`
- `prompt_style = official_in_out`
- `image_preprocess_style = official_rot180_only`
- corrected open convention via `env_action_6 < -0.5`
- shared pair IDs within matched tasks

## Window Reachability

| Pair | Task | Window | VIS n_window_steps | Random n_window_steps | Status |
|---|---|---:|---:|---:|---|
| p01 | butter s6 | 182-192 | 0 | 0 | unreachable |
| p02 | ketchup s8 | 154-164 | 0 | 0 | unreachable |
| p03 | cream_cheese s7 | 91-101 | 11 | 11 | reachable |

The official prompt + official image preprocessing changed the rollout timing enough that the old selected windows are not generally reachable. Future smoke windows should be selected from actual v1.1 clean/official traces, not from old open-convention or legacy-preprocessing traces.

## Postprocess / Label Builder Dry Audit

Codex ran postprocess and label building into `/tmp`, not the live output root.

Observed qpos rows:

- butter VIS/random: `n_window_steps=0`, `n_attack_steps=0`
- ketchup VIS/random: `n_window_steps=0`, `n_attack_steps=0`
- cream_cheese VIS/random: `n_window_steps=11`, `n_attack_steps=11`

Before the Codex patch, `build_pair_labels_v1_1.py` produced three label rows, including the two unreachable pairs as all-zero negatives. That is scientifically unsafe.

After the Codex patch, unreachable/no-intervention pairs hard-fail with:

```text
REJECT: VIS <trace> has n_window_steps=0; unreachable/no-intervention windows cannot become labels
```

Synthetic validation confirmed:

- unreachable pair: rc=1
- old trace: rc=1
- duplicate condition row: rc=1
- unpaired row: rc=1
- valid pair: rc=0

## Scientific Interpretation

This smoke is useful for code/schema validation only.

It does not provide evidence that Stage-B VIS works or fails, because two of three pairs never reached their intervention windows. The one reachable cream_cheese pair is negative for VIS-specific physical response in this smoke:

- VIS open count: 0
- random open count: 4
- VIS shifted qpos delta: negative
- random shifted qpos delta: positive

This should not be generalized beyond the smoke.

## Next Action

1. Sync the latest Codex patch before running label builder on smoke outputs.
2. Treat current butter/ketchup smoke rows as reachability failures, not negative labels.
3. Choose earlier windows from actual v1.1 official clean traces, or generate a v1.1 reachability precheck before the next smoke.
4. Re-run a small smoke only after window reachability is repaired.
