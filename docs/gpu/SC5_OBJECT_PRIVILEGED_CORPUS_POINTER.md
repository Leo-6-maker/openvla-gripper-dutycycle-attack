# SC5 Object Privileged LOTO Corpus — Combined 500/500

**Source Commit**: 0280c8564773a5e6ca0482c740891d8f9eddad84
**Worktree**: /mnt/sdc/dty_user/worktrees/sc5_wave1_0280c85

## External Artifact Roots

- **Wave 1**: `/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/wave1_50_0280c85_20260627T175204Z`
- **Wave 2**: `/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/wave2_remaining_states_0280c85_20260627T183812Z`
- **Combined Audit**: Wave2/reports/COMBINED_CORPUS_AUDIT.json

## Combined Corpus

| Metric | Value |
|--------|-------|
| Total Episodes | 500 (10 tasks × 50 states) |
| Schema Valid | 500/500 |
| Opening Proxy Pass | 500/500 |
| Attack Contamination | 0 |
| qpos Parity Max Error | 0.0 |
| width Parity Max Error | 0.0 |
| Row Parity (telemetry=privileged) | 500/500 |
| Hash Overlap Wave1-Wave2 | 0 |
| Unique Telemetry Hashes | 500/500 |

## Attempt Accounting (Wave 1 Only)

| Class | Count |
|-------|-------|
| Accepted Primary | 50 |
| Pre-Run Infra Launch Failures | 17 |
| Scientific Infra Failures | 0 |
| Wave 2 Infra Failures | 0 |

## Per Task

| Task | Name | States | Fold 0 Role |
|------|------|--------|-------------|
| 0 | alphabet_soup | 50/50 | train |
| 1 | cream_cheese | 50/50 | train |
| 2 | salad_dressing | 50/50 | train |
| 3 | bbq_sauce | 50/50 | train |
| 4 | ketchup | 50/50 | train |
| 5 | tomato_sauce | 50/50 | train |
| 6 | butter | 50/50 | validation |
| 7 | milk | 50/50 | train |
| 8 | chocolate_pudding | 50/50 | held-out test |
| 9 | orange_juice | 50/50 | train |

## Reproduction

```bash
# Bridge command (frozen at 0280c85)
python -u scripts/stageb/run_v2_vis_sc5_mlp_bridge_telemetry_phase8_frozen.py \
  --condition CLEAN --task_idx <0-9> --state_id <0-49> --anchor -1 \
  --seed_id $((TASK*1000+STATE)) --eval_seed $((TASK*1000+STATE)) \
  --output_dir <dir> --render_gpu 0 --suite_name libero_object \
  --unnorm_key libero_object --max_env_steps 400 \
  --mlp_path /mnt/sdc/dty_user/openvla_attack/outputs/sc5_v2_seed42/sc5_mlp_v2.pt \
  --object_site_registry <worktree>/configs/phase8_primary_object_sites.json

# Converter
python scripts/stageb/convert_telemetry_to_privileged_v1.py \
  <dir>/step_telemetry.csv <dir>/privileged_step_records.jsonl
```

## Claims Boundary
- Teacher NOT calibrated on combined corpus
- Student NOT trained
- Held-out NOT evaluated
- Attack rollouts: 0
- Condition: CLEAN only
- All 500 trajectories: real MuJoCo basket_1_default_site target coordinates
