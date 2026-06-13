# V7 butter_s2 execspec-v2 Confirmation — Preregistration

**Date:** 2026-06-13
**Branch:** exp/vis-prefix-margin-repair-20260603
**Preregistration commit:** (to be filled after commit)
**Runner SHA256:** 7d4997a7596cbaa51e703791eb8c68d3e6b1303b97f4c58e600d0e6b28200f86
**adapter SHA256:** 642e468e67d86993f605dea851ab3bf459689864737e304396958cdf6b74ccad
**semantics SHA256:** 699d13442b5ec03c11c80d1353c513cd57b0e0dc842e943f3e7033e4d506d8c4
**exec_spec SHA256:** 13ea43ffe674dae22c5d1c5a1a2c54c92a27c9fcc79ef13dc057594bbe31bf92

## Mission

Obtain a scientifically valid, pre-registered conclusion for whether the corrected
execspec-v2 online VIS attack replicates the butter_s2 command-level C2O effect
under matched RAND control.

A valid non-replication conclusion is an acceptable completion.

## Frozen Protocol

| Parameter | Value |
|-----------|-------|
| task | butter |
| state_id | 2 |
| epsilon | 6/255 |
| budget tolerance | 5e-5 processor units |
| PGD steps | 20 |
| step_size | 1.5 * epsilon / 20 |
| margin | 0.5 |
| event horizon H | 5 |
| attacked budget B | 3 |
| objective | prefix_locked_gripper_top1_open_vs_close_execspec_v2 |
| trigger | first pre-success clean CLOSE onset |
| model_gpu_device_id | -1 (auto device_map) |

## Primary Metric

**strict C2O:**
- clean_raw < 0.5
- executed_raw > 0.5
- executed_env < -0.5

Boundary-only (raw==0.5) and env-only C2O are NOT counted toward gate.

## Retry Policy

Max one retry per logical key (parent, condition, seed, commit).
Only for: ILLEGAL_TOKEN, EARLY_EOS, GPU_XID, CUDA_OOM, CRASH, MISSING_ARTIFACT, IMPORT_MISMATCH.
Never for: C2O=0, task success/failure, weak effect, unfavorable result.

## GPU Pairs

| Pair | CUDA_VISIBLE_DEVICES | Render GPU |
|------|---------------------|------------|
| GPU10 | 0,1 | 1 |
| GPU26 | 2,6 | 2 |
| GPU45 | 4,5 | 4 |

## Mini-Replication (8 rollouts)

| GPU | Order 1 | Order 2 | Order 3 |
|-----|---------|---------|---------|
| GPU10 | clean rep0 | RAND 401 | VIS 401 |
| GPU26 | VIS 402 | RAND 402 | clean rep1 |
| GPU45 | RAND 403 | VIS 403 | — |

Seeds 401-403. Matched RAND/VIS on same GPU pair.

## Mini Gate

- clean valid 2/2
- RAND resolved valid 3/3
- VIS resolved valid 3/3
- VIS strict-C2O >= 2/3
- RAND strict-C2O <= 1/3
- boundary-only positive = 0

### Branches

| VIS C2O | RAND C2O | Action |
|---------|----------|--------|
| >=2/3 | <=1/3 | STRONG_PASS → formal confirmation |
| 1/3 | <=1/3 | Extend to 6 seeds (404-406) |
| 0/3 | any | Stop: NOT_REPLICATED |
| any | >=2/3 | Stop: RANDOM_SENSITIVE |

## Formal Confirmation (30 rollouts)

Only if mini advancement gate passes.

Seeds 411-422 across GPU10/26/45.
clean×6, RAND×12, VIS×12.
Mini seeds excluded from formal statistics.

### Formal Gate

- VIS valid trigger >= 10/12
- RAND valid trigger >= 10/12
- VIS strict-C2O >= 8/12
- RAND strict-C2O <= 2/12
- risk difference >= 0.50
- fallback=0, budget=0

## Claim Limits

Even if all gates pass:
- Physical bridge: NOT_ESTABLISHED
- Task effect: NOT_ESTABLISHED
- Layer3 solved: No
- LIBERO-wide generalization: No
- Real-robot transfer: No

Maximum label: ONLINE_CMD_REPLICATED_WITHIN_PARENT_EXECSPEC_V2
