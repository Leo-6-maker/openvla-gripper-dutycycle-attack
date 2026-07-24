# Progress Freeze — D7 / D8F / C2f Clean2000

**Date:** 2026-07-08  
**Branch:** `plan/codex-gated-experiment-v1-c2e0`  
**Purpose:** Freeze the current scientific and engineering state before Clean2000 post-processing, CLIP materialization, C2f ablations, and any D8/C2f online rollout. Future experiments should reference this document as the baseline handoff.

---

## 0. One-line status

```text
D7_TABLE1_MAIN_RESULT = PASS_AUDITED_WITH_L10_STRATIFIED_CAVEAT
D8F_25D_ONLY_ROUTE = CLOSED
C2F_SMOKE3 = PASS_WITH_LABEL_SIGNAL_INCONCLUSIVE
C2F_CLEAN2000 = RUNNING_SHARDED_COLLECTION
```

The paper mainline should remain anchored on D7. C2f is an upgrade/generalization track and must not overwrite or silently replace the frozen D7 Table1 evidence.

---

## 1. Frozen D7 main result

### Status

D7 Table1 is complete and frozen:

- Rollout: `716/716` completed.
- D7C audit: `PASS` with 0 missing, 0 unpaired, 0 contract violations.
- D7D aggregate and D7E render: complete.
- Paired McNemar statistics: complete.
- Main pooled evidence: Object + Goal + Spatial, `N=129`.

### Main O/G/S pooled result

| Condition | Success/N | SR |
|---|---:|---:|
| CLEAN | 115/129 | 89.1% |
| TRUE_T10 | 74/129 | 57.4% |
| RAND_T10 | 114/129 | 88.4% |
| COMMAND_OPEN_ORACLE | 92/129 | 71.3% |

Primary claim:

```text
On Object/Goal/Spatial pooled, C2e3-triggered TRUE_T10 reduces success from 89.1% to 57.4%, while RAND_T10 remains 88.4%; paired TRUE_T10 vs RAND_T10 McNemar p < 0.001.
```

### D7 claim boundary

Allowed claims:

- Clean-only online detector can localize timing-sensitive gripper duty-cycle windows on the O/G/S pooled benchmark.
- TRUE_T10 is direction-specific relative to RAND_T10 under the frozen D7 protocol.
- L10 exposes a grounding/generalization limitation of 25D-only detectors and should be reported as stratified caveat.

Forbidden / not-yet-supported claims:

- Do not claim universal four-suite breakage.
- Do not claim C2e3 solves L10 grounding.
- Do not claim official LIBERO SR alone proves physical contact failure.
- Do not mix the old Object-only report with D7E as a strict regression; detector, parent set, and protocol changed.

---

## 2. D8F 25D-only route closure

### Status

D8F is closed as the 25D-only detector upgrade route.

Observed pattern:

- 7 variants tested.
- L10 recall remained below the required ceiling for a deployable upgrade.
- FP floor remained approximately 31–39%.
- Selective abstention / suite-balanced variants did not solve L10 primary-vs-distractor grounding.

Conclusion:

```text
D8F_25D_ONLY_ROUTE = CLOSED
```

Interpretation:

- 25D-only proprio/action signals can capture contact/carry timing.
- They are insufficient for L10 multi-object primary-object grounding.
- Further detector work should use RGB + language + 25D + context, i.e. C2f.

---

## 3. C2f current status

### 3.1 Smoke3 v2

Smoke3 v2 established that the collection and materialization path works:

| Gate | Result |
|---|---|
| Collection | 3 episodes, 732 steps |
| RGB parity | 732 step rows = 732 PNGs |
| RGB stream | `agentview_image`, victim-aligned |
| 25D features | length=25, no NaN/Inf in smoke |
| task_language | non-empty via robust resolver |
| Stats materialization | 687 windows, stats backend PASS |
| Label signal | primary/hazard all-zero, expected/inconclusive for L10 smoke |

Status:

```text
C2F_SMOKE3 = PASS_WITH_LABEL_SIGNAL_INCONCLUSIVE
```

Meaning:

- RGB / 25D / language / stats materialization chain is valid.
- Smoke3 does not prove teacher labels are trainable.
- Positive label quality must be assessed on Clean2000 after merge + audit.

### 3.2 Clean2000 collection

Current collection is running as 12 isolated workers:

```text
GPUs: 4,5,6,7
Workers: 3 per GPU, 12 total
Threads: 3 per worker, 36 total CPU threads
Episodes: 2000 total
Suite balance: libero_10=500, libero_object=500, libero_goal=500, libero_spatial=500
```

Important engineering boundary:

- Workers must write isolated shard roots.
- Do not allow multiple collectors to write the same collection root.
- After completion, merge shards into a canonical root before any full materialization/training.

---

## 4. C2f collection fields and student/teacher boundary

Each step row records only student-safe observation fields plus teacher labels:

```text
step
rgb_path
features_25d
task_language
teacher_hazard
teacher_primary_attackable
teacher_release_safe
teacher_event_role
teacher_phase
```

The 25D order is frozen:

```text
gripper_command
gripper_qpos
gripper_opening_proxy
eef_x, eef_y, eef_z
eef_vx, eef_vy, eef_vz
action_dx, action_dy, action_dz, action_gripper
recent_close_streak
recent_open_streak
recent_gripper_flip_count
close_onset
time_since_close
eef_speed
eef_z_delta_since_close
qpos_delta_1
qpos_delta_3
opening_proxy_delta_3
opening_proxy_variance_5
eef_speed_variance_5
```

Student-allowed modalities:

```text
rgb
task_language
features_25d
context_108d
```

Forbidden student inputs:

```text
object_pose
target_pose
object_to_target_distance
manual_failure_label
attack_outcome
matched attacked result
future frames/actions
explicit timing shortcut
```

C2f is a sidecar detector. It should not modify OpenVLA weights and should not be trained from D7 attack outcomes.

---

## 5. Required post-Clean2000 gate order

After all 12 workers finish, run the following gates in order.

### 5.1 Merge shards

```bash
COMMIT=<current_commit>
SHARDED=<clean2000_sharded_root>
MERGED=<clean2000_merged_root>

python scripts/stageb/merge_c2f_sharded_collection.py \
  --sharded-root "$SHARDED" \
  --output-root "$MERGED" \
  --mode hardlink \
  --git-commit "$COMMIT"
```

### 5.2 Hygiene check

```bash
python scripts/stageb/check_c2f_collection_hygiene.py \
  --c2f-root "$MERGED" \
  --expected-episodes 2000
```

Do **not** pass `--allow-primary-all-zero` for Clean2000.

### 5.3 Observation audit

```bash
python scripts/stageb/audit_c2f_observation_collection.py \
  --c2f-root "$MERGED" \
  --expected-episodes 2000 \
  --mode pilot200 \
  --strict-primary-nondegenerate
```

### 5.4 Stats materialization

```bash
python tools/multisuite_detector/materialize_c2f_frozen_embeddings.py \
  --c2f-root "$MERGED" \
  --output-dir <clean2000_embeddings_stats> \
  --window 16 \
  --backend stats \
  --device cpu \
  --git-commit "$COMMIT"
```

### 5.5 CLIP materialization

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> \
python tools/multisuite_detector/materialize_c2f_frozen_embeddings.py \
  --c2f-root "$MERGED" \
  --output-dir <clean2000_embeddings_clip> \
  --window 16 \
  --backend clip \
  --model-name openai/clip-vit-base-patch32 \
  --device cuda \
  --git-commit "$COMMIT"
```

### 5.6 A/B/C/D ablation datasets

```bash
python tools/multisuite_detector/make_c2f_ablation_datasets.py \
  --dataset <clean2000_embeddings_clip>/c2f_w16_clip_dataset.npz \
  --output-dir <clean2000_ablation_datasets> \
  --git-commit "$COMMIT"
```

### 5.7 A/B/C/D training

```bash
for NAME in A_25d_only B_25d_language C_25d_rgb D_full_rgb_language; do
  CUDA_VISIBLE_DEVICES=<idle_gpu> \
  python tools/multisuite_detector/train_c2f_rgb_lang_temporal_detector_v0.py \
    --dataset <clean2000_ablation_datasets>/${NAME}.npz \
    --output-dir <c2f_ablation_runs>/${NAME} \
    --device cuda \
    --epochs 30 \
    --batch-size 256 \
    --git-commit "$COMMIT"
done
```

---

## 6. C2f offline gate

C2f should only be considered a successful detector upgrade if the offline ablation supports all of the following:

```text
D_full_rgb_language > A_25d_only on L10 primary/hazard recall
L10 recall > C2e3 baseline recall threshold
Overall FP <= 30%
O/G/S recall does not collapse
Per-suite label distribution is non-degenerate
Primary/hazard labels are not all-zero or dominated by one artifact class
```

If Clean2000 labels are weak, do not train or do not interpret training as meaningful. Instead freeze as:

```text
C2F_CLEAN2000_COLLECTION = PASS_INPUT_CHAIN_BUT_LABEL_SIGNAL_WEAK
```

---

## 7. C2f online rollout policy

No C2f online D8 rollout is authorized by this freeze unless offline gates pass.

If offline gates pass, the first online experiment should be a small canary, not a D7 replacement:

```text
Parent set:
  D7 L10 parents, plus small O/G/S no-collapse check
Conditions:
  CLEAN
  TRUE_T10_C2e3
  TRUE_T10_C2f
  RAND_T10_C2f
  COMMAND_OPEN_ORACLE_C2f
```

Primary questions:

- Does C2f improve L10 primary-window localization?
- Does TRUE_T10_C2f beat RAND_T10_C2f?
- Does COMMAND_OPEN_ORACLE_C2f become more oracle-sensitive than C2e3?
- Does O/G/S avoid detector collapse?

---

## 8. Paper positioning frozen at this point

Current paper mainline:

```text
Clean-only online detector enables timing-specific gripper duty-cycle attacks on OpenVLA.
```

Core contribution framing:

1. Identify a gripper duty-cycle vulnerability in OpenVLA-style VLA policies.
2. Use a clean-only online detector to localize attackable contact/carry windows.
3. Demonstrate direction/timing specificity with CLEAN / TRUE_T10 / RAND_T10 / COMMAND_OPEN_ORACLE controls.
4. Diagnose 25D-only detector limitations on L10 and introduce C2f as visual-language grounding upgrade.

D7 remains Table1 main result. C2f may become Table4/secondary result if and only if offline ablation and online canary pass.

---

## 9. Final frozen status labels

```text
D7_TABLE1_MAIN_RESULT = PASS_AUDITED_WITH_L10_STRATIFIED_CAVEAT
D8F_25D_ONLY_ROUTE = CLOSED
C2F_SMOKE3 = PASS_WITH_LABEL_SIGNAL_INCONCLUSIVE
C2F_CLEAN2000 = RUNNING_SHARDED_COLLECTION
C2F_CLIP_MATERIALIZATION = PENDING_CLEAN2000_MERGE_AND_AUDIT
C2F_ABLATION_TRAINING = PENDING_CLIP_MATERIALIZATION
D8_C2F_ONLINE_CANARY = NOT_AUTHORIZED_UNTIL_OFFLINE_PASS
```
