# C2F Canary V1 Audit - 2026-07-10

AUDIT STATUS: HOLD

Scope: audit-only review of `/mnt/sdc/dty_user/openvla_attack_evidence/c2f/table1_candidate_gpu17_20260709_235106` per `reports/C2F_CODEX_HANDOFF_20260710.md` from commit `8f0e9f37a53cedb037e41d19b7cc8e085913dd95`.

No new GPU experiment was launched. No rollout, attack rerun, threshold tuning, or model loading was performed.

## Repository / Provenance

- Server checkout inspected: `/mnt/sdc/dty_user/openvla_attack_codex_tools_pr50_f3e6b0`
- Checkout HEAD at inspection time: `199af7ba983dc8613242244a1ea9070502a17e4c`
- GitHub branch `plan/codex-gated-experiment-v1` / PR ref visible from this checkout: `7c1b4988b989578115686f69a1cc8248dd58cedb`
- Requested handoff commit `8f0e9f37a53cedb037e41d19b7cc8e085913dd95` was not present in the server checkout object database, but was readable from GitHub raw.
- Worktree was dirty before this report was written; pre-existing modified/untracked files were not changed.

Root SHA256 files found:

```text
checkpoint.sha256: 3dfdf229533eeca6d197f5042f7dce11d35e21c38f1287367f7ea6f8b8f42bc1  /mnt/sdc/dty_user/openvla_attack_evidence/c2f/siglip_full_final/train_D/c2f_rgb_lang_temporal_detector_v0.pt
parent_csv.sha256: fb2262854f25c9113d955d2f8bfb7107c29d6b4e99eee45419a86d03dc943ee8  /mnt/sdc/dty_user/openvla_attack_evidence/condition_matrix/d7_table1_manifest/d7_table1_preregistered_parent_keys.csv
runtime.sha256: 816d352ce57c2d7c36f4e5f0d1d4133a9485af7b90eeff98a60ba392fc722043  src/gripper_attack/c2f_siglip_detector_runtime.py
worker.sha256: 5c62f9ed36f49081ea8a6e7cd04f86a1a2a91daf44f9b635d5f134601f235ea8  scripts/stageb/run_c2f_canary_worker.py
```

## Episode Completeness

Raw recomputation from `output/**/episode_metadata.json` and `output/**/step_records.jsonl`:

```text
metadata files = 144
step_records files = 144
suites = libero_10:36, libero_goal:36, libero_object:36, libero_spatial:36
conditions = CLEAN:48, TRUE_T10:48, RAND_T10:48
unique parents = 48
parent-condition duplicates = 0
missing parent-condition keys = 0
```

Completeness is PASS.

## Runtime Validity

Scan criteria included metadata/step JSON parse, empty step records, `error`, `traceback`, `EGL`, `OOM`, `CUDA out of memory`, `exception`, `step=-1`, and blank-RGB markers.

```text
runtime_valid episodes = 144
runtime-error episodes = 0
step length min/max = 68 / 300
```

Runtime validity is PASS for the artifacts scanned. This does not fix the mixed-commit/protocol issue below.

## Mixed Commit / Protocol Risk

Per-episode `git_commit` distribution:

```text
ace18762 = 62
172b78d = 42
f3c9fc0 = 6
1c181f8 = 30
1616f52 = 4
missing git_commit = 0
```

By suite / condition / git_commit:

```text
libero_10 CLEAN ace18762: 12
libero_10 TRUE_T10 172b78d: 5
libero_10 TRUE_T10 ace18762: 7
libero_10 RAND_T10 172b78d: 5
libero_10 RAND_T10 ace18762: 7

libero_goal CLEAN f3c9fc0: 6
libero_goal CLEAN 1c181f8: 6
libero_goal TRUE_T10 1c181f8: 12
libero_goal RAND_T10 1c181f8: 12

libero_object CLEAN ace18762: 12
libero_object TRUE_T10 172b78d: 6
libero_object TRUE_T10 ace18762: 6
libero_object RAND_T10 172b78d: 6
libero_object RAND_T10 ace18762: 6

libero_spatial CLEAN ace18762: 12
libero_spatial TRUE_T10 172b78d: 10
libero_spatial TRUE_T10 1616f52: 2
libero_spatial RAND_T10 172b78d: 10
libero_spatial RAND_T10 1616f52: 2
```

HOLD: the 144 episodes are not one homogeneous final-protocol experiment. They mix at least five worker commits, including commits in the known bug-fix window. This blocks formal pooled claims and blocks new replication until the final protocol/worker commit is frozen.

## Model Path / Unnorm Key Status

Metadata fields present:

```text
detector_checkpoint = /mnt/sdc/dty_user/openvla_attack_evidence/c2f/siglip_full_final/train_D/c2f_rgb_lang_temporal_detector_v0.pt
checkpoint_sha256 = 3dfdf229533eeca6d197f5042f7dce11d35e21c38f1287367f7ea6f8b8f42bc1
```

Metadata fields absent in all 144 episodes:

```text
policy model path
processor path
unnorm_key / norm_stats key
```

HOLD: Goal remains invalid for final interpretation until the authentic Goal model path, model shards, processor path, norm_stats keys, and chosen unnorm_key are audited and recorded. The current metadata is insufficient.

## Recomputed Outcomes From Raw Artifacts

Valid denominator only; all 144 scanned episodes were runtime-valid.

```text
libero_object CLEAN: n=12 success=10 emit=6 full_delivery=0 trunc_delivery=0
libero_object TRUE_T10: n=12 success=6 emit=6 full_delivery=4 trunc_delivery=2
libero_object RAND_T10: n=12 success=10 emit=6 full_delivery=6 trunc_delivery=0

libero_spatial CLEAN: n=12 success=11 emit=12
libero_spatial TRUE_T10: n=12 success=11 emit=12 full_delivery=12
libero_spatial RAND_T10: n=12 success=10 emit=12 full_delivery=12

libero_10 CLEAN: n=12 success=5 emit=5
libero_10 TRUE_T10: n=12 success=4 emit=5 full_delivery=5
libero_10 RAND_T10: n=12 success=4 emit=5 full_delivery=5

libero_goal CLEAN: n=12 success=0 emit=5
libero_goal TRUE_T10: n=12 success=0 emit=4 full_delivery=4
libero_goal RAND_T10: n=12 success=0 emit=4 full_delivery=4
```

Condition naming caveat: current TRUE/RAND are command/action-space interventions, better named `TRUE_CMDOPEN_T10_C2f` and `RAND_ACTION_NOISE_T10_C2f`. They are not D7 image-space PGD evidence.

## TRUE/RAND Pre-trigger Parity

Compared TRUE/RAND pairs for matching emitted-parent, first emit step, and pre-trigger score traces.

```text
libero_object: pairs=12 same_emit_parent=12 same_first_emit=12 same_pretrace=12
libero_10: pairs=12 same_emit_parent=12 same_first_emit=12 same_pretrace=10
libero_goal: pairs=12 same_emit_parent=12 same_first_emit=9 same_pretrace=1
libero_spatial: pairs=12 same_emit_parent=12 same_first_emit=11 same_pretrace=5
```

Object pairing is strong. Goal/Spatial pre-trigger parity is weak and must not be used for formal paired claims without instrumentation and deterministic rerun.

## RAND Reproducibility

No deterministic per-job RNG seed was found in episode metadata. HOLD for replication readiness until stable per-job seeds are derived and recorded.

## Delivery Audit

Object delivery count shows a real TRUE/RAND delivery difference:

```text
Object TRUE_T10: mean delivery_count=4.33, full_delivery=4/12, truncated=2/12
Object RAND_T10: mean delivery_count=5.00, full_delivery=6/12, truncated=0/12
```

Object TRUE truncated-delivery parents:

```text
libero_object/task_03/state_000: TRUE delivery=6, RAND delivery=10, success TRUE/RAND=True/True
libero_object/task_06/state_030: TRUE delivery=6, RAND delivery=10, success TRUE/RAND=True/True
```

The Object harmful discordants all had full TRUE delivery_count=10.

## Object Paired Analysis

Exact 12-parent table:

```text
parent                                      CLEAN TRUE RAND T_first R_first T_delivery R_delivery
libero_object/task_00/state_005             T     F    T    78      78      10         10
libero_object/task_01/state_015             T     T    T    -       -       0          0
libero_object/task_01/state_021             T     T    T    -       -       0          0
libero_object/task_01/state_028             T     T    T    -       -       0          0
libero_object/task_01/state_043             F     F    F    -       -       0          0
libero_object/task_02/state_013             T     F    T    80      80      10         10
libero_object/task_03/state_000             T     T    T    96      96      6          10
libero_object/task_03/state_025             T     F    T    146     146     10         10
libero_object/task_05/state_035             T     T    T    -       -       0          0
libero_object/task_06/state_030             T     T    T    119     119     6          10
libero_object/task_07/state_003             T     F    T    84      84      10         10
libero_object/task_08/state_016             F     F    F    -       -       0          0
```

McNemar discordants:

```text
TRUE fail / RAND success b = 4
RAND fail / TRUE success c = 0
exact two-sided p = 0.125
exact one-sided directional p = 0.0625
```

Interpretation: Object signal is strong preliminary paired evidence, but not statistically significant at conventional two-sided p<0.05 on n=12. Replication is required.

## Spatial Interpretation

Spatial emits on 12/12 parents in all conditions, including CLEAN. TRUE and RAND both full-deliver 12/12, but success is essentially unchanged:

```text
CLEAN 11/12
TRUE_T10 11/12
RAND_T10 10/12
```

HOLD for expansion. This is either broad primary-phase coverage, label/gate over-emission, or payload-insensitive suite behavior. Do not spend large GPU budget until first-emit/phase samples are manually audited.

## L10 Audit Snapshot

Online sampled L10 tasks:

```text
task_00 parents=2 emit_count=0 total_success=0/6
task_01 parents=1 emit_count=0 total_success=3/3
task_02 parents=3 emit_count=9 total_success=4/9
task_05 parents=1 emit_count=3 total_success=3/3
task_06 parents=2 emit_count=0 total_success=3/6
task_07 parents=2 emit_count=0 total_success=0/6
task_09 parents=1 emit_count=3 total_success=0/3
```

HOLD for L10 claims. The sampled set has low/uneven base success and no TRUE/RAND separation. The claim about zero primary labels for tasks 00/01/06/07 still needs direct training-dataset verification.

## GO / HOLD Decisions

```text
Goal real-model rerun: HOLD
Reason: authentic Goal model path/shards/processor/norm_stats/unnorm_key are not recorded in canary metadata and must be audited first. Goal canary v1 remains invalid substitute-model evidence.

Object replication: HOLD
Reason: Object paired table is verified and promising, but the 144-episode run mixes commits, RAND seed is not recorded, and final command-space protocol naming/worker freeze are not complete.

Spatial expansion: HOLD
Reason: 100% emit with no harmful TRUE/RAND separation needs label/phase over-emission audit before spending GPU budget.

D7-parity experiment: HOLD
Reason: current TRUE/RAND are command/action-space interventions, not D7 image-space PGD. Track B needs a separate D7 helper parity patch and smoke test after P0 fixes.
```

## Files / Commits To Change Before Experiments

Minimal required fixes before any new jobs:

1. Freeze one final worker commit and write `protocol_id` in metadata.
2. Rename current conditions to `TRUE_CMDOPEN_T10_C2f` and `RAND_ACTION_NOISE_T10_C2f` for Track A.
3. Record `runtime_valid`, `error_type`, `error_message`, `model_path`, `processor_path`, `unnorm_key`, `rng_seed`, `first_emit_step`, `delivery_steps`, and termination reason.
4. Replace global `np.random.randn()` with deterministic per-job local RNG seeded from parent/condition/base seed.
5. Add Goal authentic-model integrity audit before any Goal rerun.
6. Keep D7 image-space PGD parity as a separate Track B protocol and output root.

## Experiments To Launch After Approval

None now. Audit result is HOLD. Next step is code/metadata hardening plus Goal model integrity audit, not GPU episode expansion.