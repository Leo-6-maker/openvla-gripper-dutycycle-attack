# Official V3 Detector V4 corrected takeover audit

Date: 2026-07-18

This is the read-only F0 takeover record for the corrected V4 development line.
It is an audit of the current source and sealed evidence, not a training or
attack result.

## GitHub source state

- Repository: `Leo-6-maker/openvla-gripper-dutycycle-attack`
- Historical PR #85: Draft, HEAD `539137568883c7e1a9799425a3399703e6699f45`.
- PR #85 body is stale and its shared-GRU bottleneck conclusion is not used as
  a current scientific conclusion.
- Official execution archive: commit
  `5e27d7c4b1a188bc6a78555f94d2571222587805`, parent
  `0b053945ec77906c5114f59161936ba0e5ac1edf`, tree
  `1a8634cdb92d4a6ab7ee95ca3ceed4c0328dd9d7`.
- Archive ref: `archive/official-v3-b3-25d-execution-5e27d7c`.
- Corrected development branch: `codex/official-v3-detector-v4-corrected-fold0-20260718`.
- Proposed Draft PR base: the official archive ref above; PR #85 is not rewritten.

## Server state

The read-only audit used SSH key authentication and the official environment:

```text
repo: /mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/trainer_checkout_f98715b
HEAD: 5e27d7c4b1a188bc6a78555f94d2571222587805
worktree: clean
python: 3.10.20
torch: 2.2.0+cu121
transformers: 4.40.1
cuda: available, 8 A800-visible devices
```

At the audit time GPUs 0, 1, 4 and 5 had unrelated resident processes; no
corrected V4 process was running. No GPU job, worker, relay or rollout was
started.

## Sealed evidence inspected

- S1 root: `OFFICIAL_V3_S1_FIT_V1_5e27d7c`, top checksum seal PASS,
  800 identities and 800 episode audits.
- Teacher aggregate: status PASS, identity count 800, invariant violations 0,
  four-suite counts 200 each. Teacher labels are source evidence for the
  training line; they are not student features.
- Corrected Teacher V2.1.1 source root:
  `OFFICIAL_V3_DETECTOR_V4_TEACHER_V211_5e27d7c_20260718`, root seal PASS,
  800 episodes, zero label conflicts.
- Historical V4 candidate-window root: seal PASS, 1,386 candidate windows.
- Historical Fold-0 checkpoint root: seal PASS, checkpoint SHA
  `6db857008e4fcbdf93b7056e75c6434a50036ffde708aeb39cb7fc0c5b686911`.
  It remains a historical `FIT_FOLD_TRAINED_CANDIDATE`, not a model-selection
  or attack-authorized checkpoint.
- Historical B3 viability root: seal PASS, 12 runs, but the recorded result is
  `B3_25D_VIABILITY = HOLD`; it is preserved as a negative result.

No FIT-DEV, CAL, CHECK, CS200 result or attack result was read by this audit.

## Feature-order finding

The official SC5 order from the execution archive is:

```text
0  gripper_command
1  gripper_qpos
2  gripper_opening_proxy
3  eef_x
4  eef_y
5  eef_z
6  eef_vx
7  eef_vy
8  eef_vz
9  action_dx
10 action_dy
11 action_dz
12 action_gripper
13 recent_close_streak
14 recent_open_streak
15 recent_gripper_flip_count
16 close_onset
17 time_since_close
18 eef_speed
19 eef_z_delta_since_close
20 qpos_delta_1
21 qpos_delta_3
22 opening_proxy_delta_3
23 opening_proxy_variance_5
24 eef_speed_variance_5
```

The canonical order SHA is
`3d1101d26567a41bf688587a70c5100a3629ad62f12f9568947ee178a8a63366`.

The PR #85 V4 prototype used positional indices that treated qpos as index 0,
command as index 1, EEF xyz as 4/5/6 and time-since-close as 23. Those are not
the official fields. Its View B/C results are therefore frozen as
`V4_V211_INDEX_BUG_SMOKE`, engineering-only, and not eligible for model
selection.

## Confirmed blockers carried into the corrected line

1. Feature derivation must be name-bound to `SC5_FEATURES`.
2. Invalid steps must not update dynamic history.
3. Quality supervision must use `quality_valid XOR veto_invalid`; unknown and
   pre-support rows are masked, not converted to negative labels.
4. Release supervision must use an explicit release-known mask.
5. Ranking must operate on causal phase/window IDs and retain pure-negative
   windows; no detached ranking term is allowed.
6. The trainer must require a sealed, machine-built authorization and enforce
   fold/candidate/seed coordinates.
7. Normalization must be derived from the 600 training identities only and be
   recomputed before training.
8. Prediction/evaluation must use the exact 200 held-out identities and a
   phase-aware threshold sweep.

## Corrected implementation boundary

The new branch contains the shared V4 contract, name-bound A/B/C feature
construction, V2.1.2 Teacher derivative builder, sealed normalization and
authorization builders, authorization-gated trainer, phase-aware evaluator,
checkpoint sealing and a dedicated CPU contract workflow.

The branch intentionally does not change the official execution checkout or
any sealed evidence root. It also does not grant model-selection, CAL, CHECK or
attack authorization.

## Mutation declaration

```text
GitHub mutations during F0: 0
Server artifact mutations: 0
CLEAN/S1 mutation: 0
GPU training started: NO
Teacher materialization started: NO
FIT-DEV/CAL/CHECK read: NO
Attack started: NO
```

Next gate is local/server CPU contract testing. Only after the corrected
feature, Teacher, authorization, fold, checkpoint and evaluator gates pass will
Fold-0 C0-C3 be considered for execution on a newly sealed output root.
