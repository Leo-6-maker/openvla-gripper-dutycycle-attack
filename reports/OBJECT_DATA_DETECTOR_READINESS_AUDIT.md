# Object-100 Data Detector Readiness Audit

**Date**: 2026-06-04
**Branch**: `exp/vis-prefix-margin-repair-20260603`
**Audit script**: `scripts/diagnostics/audit_object_data_detector_readiness.py`
**Data source**: `milestone_2e2_object100_privileged_artifact_rich_20260527`

---

## 1. Executive Summary

| Metric | Value |
|--------|-------|
| Total episodes | 104 |
| Total per-step rows | 18875 |
| Clean rollouts (all) | 104 |
| Successful (manifest) | 81 (77.9%) |
| Failed/Incomplete | 19 |
| Unique tasks | 10 |
| Unique scenes | 100 |
| Avg steps/episode | 181.5 |
| Min/Max steps | 104 / 280 |
| Runtime feature completeness | 100.0% |
| Heuristic label validity | 98.1% |

### Verdict

**CURRENT DATA IS SUFFICIENT FOR LEARNED CAUSAL EARLY-GRASP DETECTOR TRAINING.**

Key findings:
1. T_gform varies massively across episodes: range 7-221, mean=84, std=29 (see Section 4)
2. This makes a learned detector NECESSARY -- a fixed rule-based trigger cannot cover this variation
3. The data has full temporal traces with complete runtime features (100% coverage)
4. The flat dataset can be converted to temporal traces for TCN training
5. Per-step phase labels must be built heuristically (done in this audit)
6. The existing teacher labels target release/pre-place (different mechanism); NOT usable for early-grasp

Caveats:
- Only seed=0 data available (no seed robustness)
- chocolate_pudding has duplicate reruns (needs dedup)
- ~2 episodes show oscillating gripper (early heuristic false positives)
- Need to verify late-T_gform episodes are physically correct

---

## 2. Data Inventory

| Item | Count |
|------|-------|
| Unique tasks | 10 |
| Unique scenes (task_state) | 100 |
| States per task | 10 |
| Seeds per scene | 1 (seed=0) |
| Clean episodes | 104 |
| Successful | 81 |
| Failed (no grasp, env termination) | 19 |
| Total per-step rows | 18875 |
| Avg steps per episode | 181.5 |

### Tasks

- pick_up_the_alphabet_soup_and_place_it_in_the_basket
- pick_up_the_bbq_sauce_and_place_it_in_the_basket
- pick_up_the_butter_and_place_it_in_the_basket
- pick_up_the_chocolate_pudding_and_place_it_in_the_basket
- pick_up_the_cream_cheese_and_place_it_in_the_basket
- pick_up_the_ketchup_and_place_it_in_the_basket
- pick_up_the_milk_and_place_it_in_the_basket
- pick_up_the_orange_juice_and_place_it_in_the_basket
- pick_up_the_salad_dressing_and_place_it_in_the_basket
- pick_up_the_tomato_sauce_and_place_it_in_the_basket

### Data Format

The data is stored as a **flat per-step dataset** (`no_timestep_visual_proprio_student_dataset.csv`),
NOT as per-episode trace CSVs. Each row is one timestep with all features.
Episodes can be reconstructed by grouping on `episode_key` and sorting by `step_idx`.

**Data type**: Full temporal traces (not just initial states). Each episode has ~130-280 steps
from start to termination.

Images (frames) exist on disk in `runs/libero_object/<task>_state<id>/frames/`.
18,415 total files, mostly PNG frames.

---

## 3. Feature Coverage

### Runtime features (available at deployment)

| Feature | Status |
|---------|--------|
| gripper_command (raw) | Present (100%) |
| gripper_qpos | Present (100%) |
| gripper_width | Present (100%) |
| eef_x, eef_y, eef_z | Present (100%) |
| eef_vx, eef_vy, eef_vz | Present (100%) |
| action_dx, action_dy, action_dz | Present (100%) |
| action_gripper | Present (100%) |
| recent_close_streak | Present (100%) |
| recent_open_streak | Present (100%) |
| recent_gripper_flip_count | Present (100%) |

### Forbidden features (input leakage audit)

| Feature | In dataset? |
|---------|------------|
| object_pose | No (PASS) |
| target_pose | No (PASS) |
| object_to_target_distance | No (PASS) |
| normalized_step | No (PASS) |

**Runtime feature completeness**: 100.0% of episodes have all required runtime features.
**Input leakage risk**: None detected. No privileged features in the flat dataset.

### Missing for detector training

- `done` flag (can be inferred from last step per episode)
- `reward` (not needed for grasp detection)
- Per-step phase labels (not in dataset; must be built heuristically)

---

## 4. Phase Label Quality

### Label validity distribution

```
  heuristic: 102
  incomplete_no_grasp_formation: 2
```

### T_gform (early grasp formation) distribution

| Statistic | Value |
|-----------|-------|
| n (with T_gform) | 102 |
| min | 7 |
| max | 221 |
| mean | 84.34 |
| median | 83 |
| std | 29.44 |
| % T_gform in {0,1} | 0.0% |
| % T_gform <= 3 | 0.0% |
| % T_gform <= 5 | 0.0% |

### Per-task T_gform stats

```
Task                    n    min   max   mean  median
pick_up_the_alphabet_soup_and_place_it_in_the_basket  10    53   110   74.5    75
pick_up_the_bbq_sauce_and_place_it_in_the_basket   9    64   151  89.11    77
pick_up_the_butter_and_place_it_in_the_basket  10    40    94   73.1    79
pick_up_the_chocolate_pudding_and_place_it_in_the_basket  14    37    96  69.71    65
pick_up_the_cream_cheese_and_place_it_in_the_basket  10    78   128  104.8   104
pick_up_the_ketchup_and_place_it_in_the_basket  10    10   179   85.4    83
pick_up_the_milk_and_place_it_in_the_basket  10     7    98   75.0    87
pick_up_the_orange_juice_and_place_it_in_the_basket   9    49   221  98.44    87
pick_up_the_salad_dressing_and_place_it_in_the_basket  10    57   124   86.9    85
pick_up_the_tomato_sauce_and_place_it_in_the_basket  10    25   146   94.2   101
```

### Interpretation

T_gform shows meaningful variation across episodes.
This is a good candidate for learned detector training, as the grasp timing
varies enough to require temporal context beyond a simple close-onset rule.

**Proceed with learned causal TCN detector training.**

---

## 5. Split Plan

Split by scene (task + state_id), NOT random rows. This prevents trajectory identity leakage.

| Split | Scenes | Episodes | % |
|-------|--------|----------|---|
| train | ~70 | 74 | 70% |
| val | ~15 | 14 | 15% |
| test | ~15 | 16 | 15% |

Has method: MD5 hash of scene_id % 100.

---

## 6. Detector Strategy Recommendation

### Recommendation: LEARNED CAUSAL TCN DETECTOR

**Rationale**: T_gform varies 30x across episodes (7-221), with std=29.
A fixed rule-based trigger (e.g., "attack at step 10") would fail on most episodes.
A learned causal TCN detector is necessary to capture the temporal context that predicts
when grasp formation will occur.

### Recommended pipeline

1. **Convert flat to traces**: Group flat dataset by episode_key, sort by step_idx
2. **Build per-step phase labels**: Use heuristic pipeline from this audit
3. **Train causal TCN**: 13-D runtime input, 3-class output (pre_grasp / grasp_formation / post_grasp)
4. **Online trigger**: T_pred = first step where P(grasp_formation) > threshold for K consecutive steps
5. **Attack window**: [T_pred + Delta, T_pred + Delta + 17] where Delta in {5, 10}

### Why learn instead of rule

A rule-based CLOSE->OPEN transition detector would fire at the correct step,
but ONLY after the OPEN command is generated. This is detection, not anticipation.

A learned TCN may detect PRE-GRASP signatures (EEF deceleration, approach trajectory,
gripper alignment) BEFORE the OPEN command, enabling earlier and more robust triggering.

### When a rule-based baseline is sufficient

If the TCN does not outperform rule-based (fire on CLOSE->OPEN), then
the rule IS the detector. But the wide T_gform variation justifies the attempt.

### Recommended rule-based trigger

```
T_trigger = first step where gripper_command < 0.5 for K=2 consecutive steps
attack_window = [T_trigger + 5, T_trigger + 22]  # or +10 to +27
```

### When to train a learned detector

Only if:
1. T_gform shows meaningful variation (>20% not at 0/1) → current data may not satisfy this
2. Rule-based baseline shows false-positives on non-grasp episodes
3. Cross-task generalization requires temporal context beyond the close-onset signal

### Alternative: Teacher-Student

If privileged simulator state becomes available:
- Teacher: full state + object pose → oracle grasp formation labels
- Student: runtime-only features → predict grasp_formation phase
- Current data doesn't support this (no privileged per-step labels)

---

## 7. Blockers Before Training

1. **Deduplicate chocolate_pudding reruns**: 4 rerun episodes duplicate original states.
   Keep the successful one (or the rerun if original failed).

2. **Verify early-oscillation episodes**: ketchup_s8 and milk_s2 may need manual
   label correction (the heuristic may have detected pre-grasp oscillation as T_gform).

3. **Convert flat to temporal traces**: The flat dataset must be grouped by episode_key
   and sorted by step_idx to create temporal sequences for TCN training.

4. **Add seed diversity**: All data is seed=0. Need at least seed 1 data for
   robustness validation.

5. **Filter failed episodes**: 19 failed episodes (no grasp / env termination)
   should be excluded from training, or used only as negative examples.

6. **Validate heuristic labels on held-out tasks**: Before training, verify that
   the heuristic label quality is consistent across all 10 tasks.

All blockers are addressable. None is a hard block to starting detector training.

## 8. Next Commands (after gate approval)

```bash
# 1. Convert flat dataset to trace CSVs (if needed)
python scripts/diagnostics/convert_object_flat_to_traces.py \
  --dataset-csv <path> --output-dir tables/object_traces/

# 2. Build phase labels
python scripts/diagnostics/build_clean_phase_dataset.py \
  --run-dirs tables/object_traces/ \
  --output-csv tables/object_phase_alignment_clean_rollouts.csv \
  --summary-csv tables/object_phase_event_summary.csv

# 3. Evaluate rule-based baseline
python scripts/diagnostics/evaluate_phase_selector_windows.py \
  --mode oracle_phase --phase-csv tables/object_phase_alignment_clean_rollouts.csv \
  --window-policy Tplus10_to_Tplus27 \
  --output-csv tables/object_rule_based_window_proposals.csv

# 4. (Future) Train learned detector only if T_gform varies enough
python scripts/train_phase_selector_scaffold.py \
  --features tables/object_traces/ --labels tables/object_phase_alignment_clean_rollouts.csv
```

---

## 9. Appendix: Data Provenance

- **Source**: LIBERO Object suite (10 tasks) with MuJoCo 2.3.7 physics
- **Collection**: `milestone_2e2` — privileged artifact-rich Object-100 dataset
- **Model**: OpenVLA-7B fine-tuned on LIBERO-Object
- **Controller**: official preprocessing with center crop and postprocess_gripper
- **Preprocessing**: Image mean=[0.5,0.5,0.5] std=[0.5,0.5,0.5]
- **dtype**: bfloat16, eager attention backend
- **Wait steps**: 10 (step_idx starts at 10, policy step 0 = dataset step 10)
