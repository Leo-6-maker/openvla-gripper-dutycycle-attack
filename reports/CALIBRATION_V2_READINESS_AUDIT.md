# Calibration v2 Readiness Audit

**Date**: 2026-06-06
**Candidates**: `tables/vis_1r_vs_3r_calibration_v2_candidates.csv` (10 rows: 5 pos, 5 neg)

---

## Per-Candidate Readiness Check

| # | Task | State | Window | Expected | Reason | Config OK? | GPU? | Ready? |
|---|------|-------|--------|----------|--------|-----------|------|--------|
| 1 | ketchup | 1 | [21,38] | positive | batch3 gold pos | YES | 0,1 | READY |
| 2 | milk | 1 | [8,25] | positive | batch3 gold pos | YES | 0,1 | READY |
| 3 | milk | 4 | [19,36] | positive | batch3 gold pos | YES | 0,1 | READY |
| 4 | butter | 5 | [25,42] | positive | batch3 gold pos | YES | 0,1 | READY |
| 5 | ketchup | 0 | [16,33] | positive | batch1 gold pos | YES | 0,1 | READY |
| 6 | ketchup | 5 | [9,26] | negative | batch3 gold neg | YES | 0,1 | READY |
| 7 | milk | 5 | [25,42] | negative | batch3 gold neg | YES | 0,1 | READY |
| 8 | salad_dressing | 0 | [7,24] | negative | batch3 gold neg | YES | 0,1 | READY |
| 9 | ketchup | 2 | [15,32] | negative | batch4 hard neg | YES | 0,1 | READY |
| 10 | ketchup | 3 | [25,42] | negative | batch4 hard neg | YES | 0,1 | READY |

---

## Config Match Verification

All 10 candidates share IDENTICAL config except `pgd_restarts`:

```
Common: --eps_raw_pixels 6 --pgd_steps 40 --objective prefix_locked_gripper_open_margin --seed 0
1R:     --pgd_restarts 1
3R:     --pgd_restarts 3
```

- Action transform: `normalize_gripper_action(raw, binarize=True)` → `invert_gripper_action()` → `env.step()` (official, verified in Phase E canary)
- MuJoCo qpos: `env.sim.data.qpos` from finger joints (verified, obs qpos always 0.0)

**Verdict**: Config hash is identical, no legacy trace reuse, no batch1/3 mixing (all candidates are from batches 1/3/4 with known provenance).

---

## Output Roots

| Run | Path |
|-----|------|
| 1R outputs | `/data/liuyu/outputs/vis_calibration_matched_v2_1r_20260606/` |
| 3R outputs | `/data/liuyu/outputs/vis_calibration_matched_v2_3r_20260606/` |
| Summary CSV | `tables/vis_1r_vs_3r_calibration_v2_results.csv` |
| Report | `reports/VIS_1R_VS_3R_CALIBRATION_V2.md` |

---

## Legacy Trace Reuse Check

| Check | Status |
|-------|--------|
| Calibration v1 1R traces reused? | NO — separate output roots |
| Calibration v1 3R traces reused? | NO — v1 had config mismatch from batch1/3 |
| Global runs traces reused? | NO — fresh runs |
| Adaptive overnight traces reused? | NO — those are uncalibrated 1R only |

**Verdict**: No legacy trace reuse. All runs are fresh.

---

## GPU Assignment

- Primary: GPU pair 0,1
- Status: Currently finishing adp_0060 (overnight task), then available
- Estimated runtime: ~5h per run type (10 candidates × ~30min), serial
- Total: ~10h for all 1R+3R on one pair, or ~5h if 1R and 3R can run on separate pairs

---

## Launch Command

```bash
nohup /home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python \
  /data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605/scripts/run_calibration_v2_matched.py \
  --gpu_pair 0,1 \
  > /data/liuyu/outputs/vis_calibration_matched_v2_1r_20260606/launch.log 2>&1 &
```

---

## Readiness Verdict: **READY — launch when GPU 0,1 free**
