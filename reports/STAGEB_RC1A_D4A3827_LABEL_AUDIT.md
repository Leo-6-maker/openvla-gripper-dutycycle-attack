# Stage-B RC1a d4a3827 Master Label Audit

**Date**: 2026-06-08
**Anchor**: d4a3827
**Input**: 3 detector feature tables + silver stability CSV
**Output**: `tables/stageb_v1_1_all_pair_labels_aggregated_rc1a_d4a3827.csv`

## Provenance confirmation

All 45 windows share (spot-checked against trace samples):
- `trace_version = corrected_stageb_v1_1`
- `source_snapshot_id = f9840cb1`
- `prompt_style = official_in_out`
- `image_preprocess_style = official_rot180_only`
- `qpos_source = obs_robot0_gripper_qpos`

## Label tier distribution

| Tier | Count | Source |
|------|-------|--------|
| bronze (original) | 22 | Bronze batch, no silver/rescue override |
| silver_cmd | 9 | Silver P1A stable cmd (may include +phys) |
| silver_hard_neg | 2 | Silver P1A confirmed neg (zero VIS, zero RAND) |
| rescue_cmd | 4 | Rescue cmd override (random-confounded bronzes reclassified) |
| rescue_hard_neg | 2 | Rescue confirmed neg |
| rescue_phys | 1 | Rescue phys-only override |
| rescue_rand | 2 | Rescue random-sensitive |
| rescue_unstable | 3 | Rescue parent with unstable repeats |
| **Total** | **45** | |

Note: silver_phys and silver_rand tiers don't appear because:
1. silver phys windows got `silver_cmd` (cmd takes priority over phys in override logic)
2. Some silver_rand windows were bumped to rescue tier

## Task distribution

| Task | Total | cmd_specific | vis_specific_phys | random_sensitive |
|------|-------|-------------|-------------------|-----------------|
| alphabet_soup | 12 | 1 | 0 | 1 |
| butter | 6 | 2 | 2 | 2 |
| tomato_sauce | 6 | 5 | 2 | 0 |
| cream_cheese | 5 | 1 | 0 | 0 |
| bbq_sauce | 4 | 0 | 0 | 0 |
| milk | 4 | 2 | 2 | 1 |
| orange_juice | 4 | 1 | 0 | 0 |
| salad_dressing | 4 | 1 | 0 | 0 |
| **Total** | **45** | **13** | **6** | **4** |

## Gap analysis (for targeted expansion)

### Critical gaps

1. **cmd_specific dominated by tomato_sauce (5/13 = 38%)**. Need non-tomato cmd positives:
   - alphabet_soup: 12 windows, only 1 cmd_specific
   - bbq_sauce: 4 windows, 0 cmd_specific
   - cream_cheese: 5 windows, only 1 cmd_specific
   - orange_juice, salad_dressing: only 1 each

2. **vis_specific_phys underpowered (6 total)**. Only 3 tasks have phys positives (butter, tomato_sauce, milk). Need physical bridge enrichment across more tasks.

3. **random_sensitive only 4 windows**. Need 20-25+ for reliable abstain head training. Currently all concentrated in butter (2), alphabet_soup (1), milk (1).

4. **hard_neg only 4 confirmed** (2 silver_hard_neg + 2 rescue_hard_neg). Need 40+ for robust negative training. Most "negatives" are simply windows without observed effect, not confirmed hard negatives.

5. **Same-task contrast lacking**: No task has balanced (cmd_pos, phys_pos, rand, hard_neg) within-task. All labels are sparse.

6. **alphabet_soup over-sampled**: 12/45 = 27% of windows, but yields only 1 cmd_specific. Low yield rate for labels that matter.

### Data model gaps

1. **cmd_any_raw unavailable**: Current pair labels store `cmd_susceptible` which already excludes random-confounded windows (vis_meets AND NOT rand_meets). There's no column tracking windows where VIS produced ≥6 opens regardless of random confound. Feature table's `target_cmd_any` IS `cmd_specific`.

2. **phys_any_raw unavailable**: Same issue — `vis_specific_physical_response` in pair labels already excludes windows where random also produced physical response.

3. **P1b traceability**: P1b contributed 18 pairs but they're merged into silver feature table. Individual P1b windows can't be identified from the feature table alone.

## Expansion constraints (from user)

- Max 36 parent windows (24-30 recommended for sleep)
- butter ≤ 20% of expansion windows
- Each task ≤ 3-4 parent windows
- random_sensitive → abstain, NOT negative
- Same-task contrast preferred
- Every parent must trace back to clean reachable candidate
- smoke first (6 parents), then sleep expansion

## Recommended expansion targets

Based on gap analysis, priority targets:

| Type | Target N | Priority tasks |
|------|----------|---------------|
| Non-tomato cmd_specific | 8-10 | alphabet_soup, bbq_sauce, cream_cheese, orange_juice |
| Physical bridge enrichment | 6-8 | cream_cheese, orange_juice, salad_dressing, bbq_sauce |
| Hard negatives | 6-8 | All tasks, same-episode adjacency |
| Random_sensitive abstain | 4-6 | Non-butter tasks (alphabet_soup, cream_cheese, tomato_sauce) |
| Sentinel repeats | 2-3 | Known stable positives (NOT butter-only) |
| **Total** | **24-30** | |
