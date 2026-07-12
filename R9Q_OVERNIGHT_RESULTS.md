# R9Q Overnight Training Results

**Date:** 2026-07-13
**Server:** pm-364c0001 (dty_user@10.60.2.56:33571)
**Git HEAD (codex):** f47cb752610800b3cbdd6be8290e4562e88fd447

## Dataset

- **Combined root:** `/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_combined_ogs_l10_f47cb75_20260713_v2/`
- **Total episodes:** 1168 (OGS-1500: 900 + Partial-L10: 268)
- **FIT:** 960 | **CAL:** 116 | **CHECK:** 92
- L10 salvage: 268/300 DETECTOR_TRAIN, 382 total with receipts

## Model Configurations

| Config | Data | Features | Params |
|--------|------|----------|--------|
| A1 | OGS-only | 25D | 510K |
| A2 | OGS-only | 25D+9D | 784K |
| B1 | OGS+Partial-L10 | 25D | 510K |
| B2 | OGS+Partial-L10 | 25D+9D | 784K |

All models: GRU(hidden=128), language-conditioned, no visual, 3 seeds (42/123/456), max 30 epochs, AdamW(lr=1e-3)

## Model Checkpoints

```
/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_overnight_models_f47cb75_20260713_v1/
  A1_seed42/checkpoint.pt     (510K)
  A1_seed123/checkpoint.pt    (510K)
  A1_seed456/checkpoint.pt    (510K)
  A2_seed42/checkpoint.pt     (784K)
  A2_seed123/checkpoint.pt    (784K)
  A2_seed456/checkpoint.pt    (784K)
  B1_seed42/checkpoint.pt     (510K)
  B1_seed123/checkpoint.pt    (510K)
  B1_seed456/checkpoint.pt    (510K)
  B2_seed42/checkpoint.pt     (784K)
  B2_seed123/checkpoint.pt    (784K)
  B2_seed456/checkpoint.pt    (784K)
  calibration_results.json
```

## CAL Threshold Calibration Results

Grid search: tau_critical∈{0.3,0.4,0.5,0.6,0.7} × tau_release∈{0.3,0.4,0.5,0.6} × tau_ground∈{0.3,0.5,0.7} × persistence∈{(1,1),(3,2),(5,3)}
Safety filter: false_trigger≤10%, release_safe_emit≤2%

### Best per Config

| Config | Best Seed | Feasible Hit | Full T10 | False Trigger | Thresholds |
|--------|-----------|-------------|----------|---------------|------------|
| A1 | seed42 | 90.5% | 86/95 | 4.8% | τ_crit=0.7 τ_rel=0.3 τ_grnd=0.3 pw=5 pr=3 |
| A2 | seed456 | 91.6% | 87/95 | 4.8% | τ_crit=0.7 τ_rel=0.3 τ_grnd=0.3 pw=5 pr=3 |
| B1 | seed42 | 90.5% | 86/95 | 4.8% | τ_crit=0.7 τ_rel=0.3 τ_grnd=0.3 pw=5 pr=3 |
| B2 | seed456 | 91.6% | 87/95 | 4.8% | τ_crit=0.7 τ_rel=0.3 τ_grnd=0.3 pw=5 pr=3 |

### Key Findings

1. **9D policy intent helps:** A2(91.6%) > A1(90.5%), +1.1pp feasible-hit
2. **Partial-L10 no gain:** B1≈A1, B2≈A2 — 268 partial L10 episodes insufficient to improve detector
3. **False trigger consistently 4.8%** — well under 10% safety gate
4. **All configs converge to same optimal thresholds** (τ_crit=0.7, τ_rel=0.3, τ_grnd=0.3, 3-of-5 persistence)
5. **OGS-1500 alone is sufficient** for preview-quality detector

### Selected Model

**B2_seed456** — Best overall (91.6% feasible-hit, 87 full-T10 containment)
Path: `/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_r9q_overnight_models_f47cb75_20260713_v1/B2_seed456/checkpoint.pt`

## Artifacts

- Training script: `train_r9q.py`
- Calibration script: `calibrate_all.py`
- Monitor script: `monitor_calibrate.py`
- Combined dataset: `c2g_r9q_combined_ogs_l10_f47cb75_20260713_v2/`
- L10 labels: `c2g_r9q_partial_l10_labels_f47cb75_20260713_v1/`
- L10 materialized: `c2g_r9q_partial_l10_materialized_f47cb75_20260713_v1/`
- L10 salvage: `c2g_r8y_l10_520_salvage_f47cb75_20260713_v1/`
- L10 forensic: `c2g_r8y_l10_520_forensic_f47cb75_20260713_v1/`
