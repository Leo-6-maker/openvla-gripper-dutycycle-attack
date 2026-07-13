# C2G R9Q Experiment Progress

Initial snapshot: `2026-07-13T09:38:57+08:00`
Latest live update: `2026-07-13T09:52:58+08:00`

This is a provenance report, not a new experiment result. It records the
current detector gates, immutable artifacts, live attack campaign state, exact
launch entry points, and explicit non-results.

## Repository

- Repository: `Leo-6-maker/openvla-gripper-dutycycle-attack`
- Branch: `codex/c2g-r9q-final-detector-attack-20260713`
- Executed head: `8ddbd03503c66e7efca9fcc84ad7d49974af33e0`
- Worktree: clean at the snapshot
- D7 Table1: frozen; no D7 artifact was modified

## Detector pipeline

| Gate | Status | Evidence |
|---|---|---|
| B2 overfit smoke | PASS | `c2g_r9q_b2_overfit_smoke_5828dc90_20260713_v1/overfit_smoke_report.json`; 8 fixed FIT episodes |
| B2 training | PASS | seeds `42,123,456`; 30 epochs each; FIT `960`; CAL `116` |
| CAL selection | PASS | `PASS_C2G_R9Q_CALIBRATION_SELECTED_2OF3` |
| CHECK | PASS | one-shot consumption `1`; 92 rows; no CAL reads |
| Streaming replay | PASS | 24 episodes; prefix recompute causal equivalence |
| FSM validation | PASS | 24 episodes; burst length `10`; multi-trigger count `0` |
| Detector bundle | PASS | `PASS_C2G_R9Q_DETECTOR_VALIDATION` |
| Attack canary | HOLD | `32/32` runtime-valid cells; audit failed closed with 4 audit failures |
| Attack panel | NOT_RUN | queued by the supervisor after canary audit |
| Full Table1 | NOT_RUN | queued by the supervisor after panel audit |

The current detector is therefore validated for the offline detector gates and
authorized for the R9Q attack preview path. It is not yet a completed main-table
result.

## Detector definition

- Input: `25D clean proprio/action + 9D clean policy intent`
- Model: causal GRU, hidden size `128`
- Heads: `window_start`, `burst_feasible`, `critical_window`, `release_safe`,
  `contact_grasp`, `grounding_confidence`
- Visual input: disabled
- Language conditioning: disabled; language remains provenance only
- Training: AdamW, `lr=1e-3`, `weight_decay=1e-5`, `batch_size=8`, `30` epochs,
  gradient clip `5`
- Runtime gate: `tau_critical=0.7`, `tau_release=0.4`, `tau_ground=0.3`
- Persistence: `2-of-3`
- Burst: exactly `10` steps
- Selected checkpoint: seed `456`, epoch `11`

### Training evidence

Training root:
`/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_b2_models_5828dc90_20260713_v1`

| Seed | Training report checkpoint SHA256 | Epochs | FIT | CAL |
|---:|---|---:|---:|---:|
| 42 | `9d80fa82f5cc9f9c11aee8b37024082c055ab0d452ba3590ee58ed8b325fce8d` | 30 | 960 | 116 |
| 123 | `1d218190c5d319445e7a0bac183d91eae0eadb25b167b51e5f09f6c255adae40` | 30 | 960 | 116 |
| 456 | `a93f0eb08fb6f356116f1b4df1c2c87da01c725727f20f0262914e9241283ec3` | 30 | 960 | 116 |

Reported final training losses were `0.378566`, `0.375062`, and `0.368634`
for seeds `42`, `123`, and `456`. No training NaN, Inf, or OOM was reported.

## Selected detector and offline gates

Immutable bundle:
`/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_final_detector_bundle_5576d46_20260713_v1`

- `checkpoint.pt`: `336a7723096a8939efd82a77d21c5c6e0b76e8f1ed369d237961b47834548807`
- `detector_config.json`: `196966c80d1b31cc6d04f47982968b8ec73c503c78a415c4ba779e1d7634d93e`
- `normalization.json`: `1b31424aa1dfd56834345934da56749331b36640447e169041d0d99f9fad66f9`
- `training_manifest.json`: `312cd6da72f66101dd955c57692f158ecf799a74d33f0d040a03cf2ec4fab2b3`
- `class_balance.json`: `78efdfb6134708d20932e039817cd02aa8f3b5767ac170618a1bae859f791672`
- bundle `SHA256SUMS`: `c6e2a6562f73ce143890f49524f497e6f50951891f4d156f3a1d59833491b0ea`

CAL selected `90/95` feasible positive episodes (`94.7368%`), `90/95`
full-T10 containment, one false trigger, and zero release-safe emits in the
116-row CAL audit. CAL had limited release-safe support and partial L10 support.

CHECK was consumed exactly once:

- rows: `92`
- feasible hit: `64/69 = 92.7536%`
- full-T10 containment: `64/69 = 92.7536%`
- negative any-trigger: `3/22 = 13.6364%`
- release-safe emit: `0`
- CAL reads during CHECK: `0`

Streaming/FSM evidence:

- streaming report SHA: `44843da11896a954df935dca407cb4c54e0a2dc31573686619fe7a0fdebe7a1c`
- FSM report SHA: `8ad7bc87eefddaf74ebfc2c56c7a5631179bed147903ad3aeeb4836e344ea7fe`
- 24 replay episodes
- maximum six-head logit error below the recorded `0.002` tolerance
- total triggers `18`; multi-trigger count `0`
- mode: `PREFIX_RECOMPUTE_CAUSAL_EQUIVALENCE`

## Attack plans

The source-only frozen Table1 manifest is:

`/mnt/sdc/dty_user/openvla_attack_evidence/condition_matrix/d7_table1_manifest/d7_table1_queue_manifest.csv`

Its SHA256 is:
`bdb0921db53541ca11bc5d038306bafaf6c400dbd340270987f3a460d807f185`

It resolves to `179` unique parents and `716` matched cells (`4` conditions
per parent): `CLEAN`, `R9Q_DETECTOR_T10`, `RAND_T10`, and
`COMMAND_OPEN_ORACLE`. Parent identities and initial states are not changed.

| Stage | Parent/cell scope | Plan root | Manifest SHA256 |
|---|---:|---|---|
| Canary | 8 parents / 32 cells | `c2g_r9q_attack_canary_8ddbd035_20260713_v1` | `7a921dc37df25b274111745d1c85d8655f10fec1827d6f20f2e00671fadc6442` |
| Panel | 40 parents / 160 cells | `c2g_r9q_attack_panel_8ddbd035_20260713_v1` | `cfe5ea9d90776f6a99b98946f2a3db4047d0d9b59714cf4dc07d76ba1c6139f5` |
| Full | 179 parents / 716 cells | `c2g_r9q_attack_main_plan_8ddbd035_20260713_v1` | `56b32b7fa7e69ead3ac3c85b8662bbd4ba8e4ba4f8a48c91f975a1435e7234bf` |

The current scheduler uses logical workers `g6_object`, `g6_spatial`,
`g6_goal`, `g6_l10`, `g7_object`, `g7_spatial`, `g7_goal`, and `g7_l10`.
It uses two resident workers per GPU in waves and a global serialized model
load lock. The requested GPU2/3/4 expansion was not launched and is not part
of this provenance chain.

## Live canary result

Campaign root:
`/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_attack_campaign_8ddbd035_20260713_v1`

At the latest live update:

- valid episode cells: `32/32`
- invalid cells: `0`
- worker states: `8/8 PASS`
- observed success values: `20 true`, `12 false`
- no attack cell was selected from outcomes; all cells remain manifest-bound

Canary audit:

- status: `HOLD_C2G_R9Q_ATTACK_RUN_AUDIT`
- audit report SHA256: `23509655808797283c368af8b140377cc4c077bba38a59f895413ce4d0adba72`
- audit `SHA256SUMS` SHA256: `122dd29f45b01d91bfad492b052afeb2877983d7719b3ad9ab51d17895474075`
- audit failures: `4`
- `R9Q_ALWAYS_OR_NEVER_TRIGGER`: `1`
- `RAND_BURST_NOT_EXACT_T10`: `3`
- R9Q detector condition: `0/8` triggered
- R9Q detector full-T10 deliveries: `0/8`

The supervisor stopped after the canary audit at `2026-07-13T09:39:55+08:00`.
Panel and full run roots were not launched.

## Detector signal interpretation

There is a clear **offline** detector signal:

- CAL: `90/95` feasible positive episodes and `90/95` full-T10 containment
- CHECK: `64/69` feasible hit and `64/69` full-T10 containment
- Streaming replay: `18` triggers across `24` offline traces

There is currently **no positive online deployment signal** in the canary:

- `R9Q_DETECTOR_T10`: `0/8` triggered
- `CLEAN`: `0/8` triggered, as expected
- `RAND_T10`: random branch triggered on several cells, but this does not
  demonstrate detector timing and three RAND cells failed exact-T10 delivery
- `COMMAND_OPEN_ORACLE`: no detector trigger is expected from this condition

Therefore the detector is scientifically promising at the offline/CAL level,
but the detector-to-runtime path is not yet validated. The current canary is a
deployment integration HOLD, not a main-table result.

## Root-cause analysis of zero online triggers

The canary records distinguish model inference from scheduler admission:

- all eight R9Q cells produced non-empty detector outputs after the W16 warmup;
- 1,395 post-warmup rows were scored;
- the observed maximum `critical_window` probability was approximately `0.9999`;
- the observed maximum `grounding_confidence` probability was approximately
  `0.9996`;
- the scheduler state remained `IDLE` on every recorded R9Q row.

This is not consistent with a missing model load or an all-zero detector head.
The selected `epoch_011.pt` checkpoint was inspected and contains no
`susceptibility` field. In `src/gripper_attack/c2g_clean_window_runtime.py`,
the runtime therefore falls back to `require_clean_close=true`,
`minimum_open_minus_close_log_mass=-8`, and `minimum_entropy=0`. The runtime
passes the resulting `susceptibility_gate` as `valid` into the 2-of-3 FSM.
High detector probabilities cannot trigger if that gate is false.

The offline R9Q loss and CAL/CHECK selection use the detector-head gate
`critical * (1 - release_safe) * grounding`; they do not reproduce the full
online clean-policy susceptibility gate. This is an evaluation-contract gap,
not evidence that the learned critical-window head has no signal.

The worker currently records `detector_ready` and `detector_outputs`, but not
`susceptibility_gate`, `clean_top1_is_close`, open/close log-mass margin,
entropy, or the gate history. Therefore the exact blocked subcondition cannot
be identified from the canary artifacts alone. The next minimal diagnostic is
telemetry and CPU/synthetic gate replay. Threshold changes, gate removal, and
detector retraining are not justified before that diagnostic.

## Previous failures and fixes

1. `bd105d13` campaign: stopped before useful cells because the runtime did not
   accept the final checkpoint schema.
2. `903660d5` campaign: one RAND cell failed because the attacker used native
   open token `31745` instead of the frozen CLIP-mediated-open token `31744`.
3. `8ddbd035` fixed the token binding. The current canary has no recurrence of
   that failure at the snapshot.

Old roots remain untouched and are reference evidence only.

## Exact launch entry points

Committed scripts:

```text
scripts/stageb/build_c2g_r9q_attack_manifest.py
scripts/stageb/run_c2g_r9q_attack_worker.py
scripts/stageb/run_c2g_r9q_attack_scheduler.py
tools/multisuite_detector/audit_c2g_r9q_attack_run.py
scripts/stageb/run_c2g_r9q_attack_campaign_8ddbd035.sh
```

The checked-in campaign wrapper is a reproduction of the currently running
supervisor. It is fail-closed on the exact repository head and clean worktree,
uses a fresh stage root, waits for the 6/7 resource admission gate, runs
canary `32`, panel `160`, then full `716`, and audits each stage before the
next one.

Direct scheduler command for one stage:

```bash
REPO=/mnt/sdc/dty_user/openvla_attack_codex_r9q_final_20260713
PY=/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3.10
HEAD=8ddbd03503c66e7efca9fcc84ad7d49974af33e0
BUNDLE=/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_final_detector_bundle_5576d46_20260713_v1
PLAN=/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_attack_canary_8ddbd035_20260713_v1
RUN=/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_attack_campaign_8ddbd035_20260713_v1/canary_run

"$PY" "$REPO/scripts/stageb/run_c2g_r9q_attack_scheduler.py" \
  --mode run \
  --plan-root "$PLAN" \
  --detector-bundle "$BUNDLE" \
  --output-root "$RUN" \
  --expected-git-commit "$HEAD" \
  --worker-budget-mib 18000 \
  --gpu-reserve-mib 8000 \
  --max-resident-workers-per-gpu 2 \
  --poll-seconds 20
```

The wrapper is the preferred entry point because it performs the stage audit
and refuses to continue on a nonzero scheduler or audit result.

## Current decision

- Detector offline validation: `PASS`
- Attack canary: `HOLD_C2G_R9Q_ATTACK_RUN_AUDIT`
- Panel: `NOT_RUN`
- Full Table1: `NOT_RUN`
- GPU2/3/4 extra workers: `NOT_STARTED`
- New training: `0`
- New detector materialization: `0`
- D7 modifications: `0`
- Final main-table claim: `NOT_AVAILABLE`

The next automatic step remains canary audit review. A full main-table result
must not be reported until the panel and full audit roots are present and their
cell, provenance, and matched-condition checks pass.
