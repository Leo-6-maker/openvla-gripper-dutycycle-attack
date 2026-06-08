# Stage-B RC1a Freeze Note — d4a3827

**Date**: 2026-06-08
**Branch**: `exp/vis-prefix-margin-repair-20260603`
**Anchor SHA**: `d4a3827` (local = remote)
**Server**: `klfy-SYS-4028GR-TR2`
**User**: `liuyu`
**Python**: conda `openvla_official_libero_20260525` (Python 3.10.13)
**Repo path (server)**: `/data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/`
**Server SHA**: `ca3a97e` (first 3 commits behind local — docs/test/early-fix only, semantics unchanged)

## GPU inventory

| Physical | Status | Pairing |
|----------|--------|---------|
| 0, 1 | Idle, usable | `CUDA_VISIBLE_DEVICES=1,0` (GPU1 primary; GPU0 intermittent faults) |
| 2, 6 | Idle, usable | `CUDA_VISIBLE_DEVICES=2,6` |
| 3, 7 | **BLACKLISTED** | Xid31 MMU fault — NEVER use |
| 4, 5 | Idle, usable | `CUDA_VISIBLE_DEVICES=4,5` |

All 8 GPUs: NVIDIA GeForce RTX 2080 Ti, 11 GiB, 32–41°C, 0 MiB used (2026-06-08 14:05 CST).

## RC1a semantic verification

```
env_gripper_is_open(-1.0): True   ✓
env_gripper_is_open(+1.0): False  ✓
raw_gripper_is_open(0.7): True    ✓
raw_gripper_is_open(0.3): False   ✓
```

All tests: 47 passed (server `PYTHONPATH=src pytest tests/stageb/ -q`).

## Provenance constants

All RC1a v1.1 traces share:
- `trace_version = corrected_stageb_v1_1`
- `source_snapshot_id = f9840cb1`
- `prompt_style = official_in_out`
- `image_preprocess_style = official_rot180_only`
- `qpos_source = obs_robot0_gripper_qpos`
- `open_convention = env_action_6_lt_neg_0p5_means_OPEN`

## Data roots (server)

| Phase | Path | Traces |
|-------|------|--------|
| Clean reachability scan | `/data/liuyu/outputs/stageb_v1_1_clean_reachability_scan_rc1a_20260607/` | 27 |
| Bronze batch (48 windows) | `/data/liuyu/outputs/stageb_v1_1_bronze_batch_rc1a_20260607/` | 96 |
| Silver P1A confirm | `/data/liuyu/outputs/stageb_v1_1_silver_confirm_rc1a_20260608/` | 84 |
| Silver P1b | `/data/liuyu/outputs/stageb_v1_1_silver_p1b_rc1a_20260608/` | 36 |
| Random-confounded rescue | `/data/liuyu/outputs/stageb_v1_1_random_confounded_rescue_rc1a_20260608/` | 42 |
| Detector v0 readouts | `/data/liuyu/outputs/stageb_v1_1_detector_v0_rc1a_20260608/` | — |

## Quarantine policy (reconfirmed)

1. Old 44-row patched rerun — inverted VIS objective
2. Old overnight Stage-B labels — wrong open convention
3. Pre-v1.1 traces — missing metadata
4. Active Probe v0b/v1 — no-env surrogate unreliable
5. ProprioNoStep as detector — detects contact, not VIS vulnerability
6. Pre-fix detector readouts — DIAGNOSTIC_ONLY_PRE_FIX

## Key constraint recap

- Do NOT use VIS outcome, post-attack qpos, or manual labels as detector input features
- Do NOT train final detector yet
- Do NOT use old labels / old windows / pre-v1.1 traces
- Never launch Bronze Expansion without review
- Never exceed 36 expansion windows while asleep
- VIS/RAND must share same seed per pair
- GroupKFold by task_state_seed, not random window split
- random_sensitive → abstain head, NEVER negative
