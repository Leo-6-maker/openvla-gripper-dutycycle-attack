# Stage-B RC1a S9b VIS/RAND Smoke PASS — Phase1-port Physical Bridge POC

**Date**: 2026-06-10
**Branch**: exp/vis-prefix-margin-repair-20260603
**S9b sanity**: 9070561 (ORACLE pos=+0.295)
**S9b smoke**: TBD

## Executive Summary

**Verdict: PASS.** Under the Phase1-port runner where ORACLE physical reachability is restored (baseline ~0.002, ORACLE pos=+0.295), VIS gripper PGD attack transfers from command OPEN to positive qpos opening under matched random control. This is the first matched-control evidence of physical bridge in this project.

## Infrastructure

| Gate | Result |
|------|--------|
| 4/4 summary JSON | PASS |
| 4/4 trace CSV | PASS |
| 4/4 infra_status=ok | PASS |
| 0 FAILs / CUDA / EGL / pgd_error | PASS |
| GPU released | PASS |

## Results

ORACLE reference (S9b sanity): `qpos_pos_area = 0.29489932`

| Cond | Seed | Baseline | pos_area | neg_area | Open | Streak | ORACLE norm |
|------|------|----------|----------|----------|------|--------|-------------|
| **VIS** | 9 | 0.001967 | **+0.1248** | 0.007 | **7/10** | 4 | **0.423** |
| RAND | 9 | 0.001967 | 0.0000 | 0.022 | 0/10 | 0 | 0.000 |
| **VIS** | 10 | 0.001749 | **+0.0783** | 0.006 | **6/10** | 5 | **0.265** |
| RAND | 10 | 0.001235 | 0.0161 | 0.004 | 0/10 | 0 | 0.055 |

RAND seed10 has a small positive qpos drift (norm 0.055) without any decoded OPEN commands. It is far below matched VIS (0.265) and does not reproduce the command/qpos bridge.

## Gates

| Gate | Result |
|------|--------|
| Command: VIS open > RAND open | **PASS** (6-7 vs 0) |
| Physical: VIS pos > 0, VIS pos > RAND pos | **PASS** |
| ORACLE norm: VIS >= 0.2 (seed9=0.42, seed10=0.27) | **PASS** |
| Control: RAND no comparable pos | **PASS** |
| Baseline: Phase1-like ~0.002 | **PASS** |

## Runner Provenance

- Runner: `scripts/stageb/run_s9b_phase1_runner_attack_port.py`
- Env init: `seed → reset → set_init_state` (Phase1, no qvel zero)
- Image: `env.sim.render()` + `rotate(180)` (Phase1)
- Prompt: hand-rolled `In: What action...?\nOut:` (Phase1)
- VIS attack: `TokenPrefixPGDAttacker`, `prefix_locked_gripper_open_margin`, pgd20, eps6
- VIS decode: `generate_action_from_inputs()` — model.generate on adv pixel_values (not prompt ids)
- RAND: seeded torch.Generator, `random_seed_str = attack_seed + job_id`
- Qpos: `env.sim.data.qpos[7]`, sampled before `env.step()`
- Metrics: bidirectional pos/neg/abs area over 40-step post-window

## Artifact Paths (Server)

```
/data/liuyu/outputs/stageb_v1_1_k5c_targeted_expansion_rc1a_ca3a97e/s9b_phase1_runner_visrand_smoke/
  summary_milk_s0_w70_80_phase1port_seed9_vispgd_job950410.json
  summary_milk_s0_w70_80_phase1port_seed9_randomlinf_job950411.json
  summary_milk_s0_w70_80_phase1port_seed10_vispgd_job950412.json
  summary_milk_s0_w70_80_phase1port_seed10_randomlinf_job950413.json
  trace_*.csv (4 files)
```

## Claim Boundary

**Allowed:**
- Phase1-port milk-only physical bridge proof-of-concept PASS
- VIS gripper attack transfers command OPEN to positive qpos opening under matched random control
- VIS reaches 26-42% of ORACLE qpos-opening reference
- RAND controls produce no comparable command or physical response

**Forbidden:**
- Layer-3 physical bridge solved
- VIS attack causes task failure / object drop
- Object-wide physical attack success
- Detector solved
- S6/S7 claims modified
- Full expansion without replication

## Next Step

Milk-only replication with new attack seeds (11, 12) before any expansion. No full queue.
