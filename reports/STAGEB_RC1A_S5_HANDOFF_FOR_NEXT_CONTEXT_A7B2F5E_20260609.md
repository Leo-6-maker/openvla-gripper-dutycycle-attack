# Stage-B RC1a S5 Handoff — Next Context

**Date**: 2026-06-09
**Commit**: a7b2f5e
**Branch**: exp/vis-prefix-margin-repair-20260603
**Server**: klfy-SYS-4028GR-TR2, GPU 1,0 / 2,6 / 4,5 (3,7 BLACKLISTED)

## What is trusted

1. RC1a gripper semantics (raw>0.5→OPEN, spec module single source of truth)
2. Fixed-env K-repeat protocol (--env_seed fixed, --attack_seed 0..4)
3. S5 stable pool: 24 parents, 8 tasks, 240/240 validator PASS
4. K5 probability labels (pV_cmd, pR_cmd, yield_cmd, risk_rand)
5. Layer-1 rand/abstain detector (AUROC=0.833, selector validated)
6. Runner at 0e3428f (seeded RAND generator, env_seed/attack_seed in summary/trace)

## What is quarantined

1. Old 72-pair single-shot labels (exploratory only, NOT training labels)
2. Old 8/8 unstable finding (reinterpreted as seed-coupling protocol bug)
3. Pre-a20379f runner outputs (no --env_seed/--attack_seed separation)
4. Global visual SigLIP sidecar results (tested on protocol-bugged labels)
5. Selector v0 oracle-yield results (upper bound, not online)

## Current best claim

> Layer-1 abstain detector validated. CleanRand abstain + TaskRank improves window selection over TaskOnly baseline. Full online selector: Layer-1 PASS, Layer-2 WIP.

## Current best strategy

```
Abstain(CleanRand) + TaskRank
→ rand_hit: 0.25→0.12
→ cmd_hit: 0.62→0.75
```

## What NOT to do

1. Do NOT train detector on 72-pair single-shot labels
2. Do NOT claim detector solved or pipeline closed
3. Do NOT restart global visual SigLIP
4. Do NOT run cross-suite before Layer-0 mechanism eligibility
5. Do NOT use GPU 3/7
6. Do NOT use oracle yield for online ranking claims

## Immediate next experiment

**K5c targeted expansion** (16 parents × K=5 = 160 jobs):
- Rand-sensitive: 6 → 10+ across more tasks
- Strict phys: 5 → 8–10
- Same-task contrast for cmd de-biasing

## Key files

```
scripts/run_stageb_vis_labeling.py          # Runner (0e3428f)
scripts/diagnostics/run_s5_stable_detector_readout.py  # Detector v0.1
scripts/diagnostics/run_selector_v0_2_leakage_free.py  # Selector v0.2
scripts/diagnostics/audit_prefix_determinism.py        # Prefix audit
scripts/stageb/generate_k5b_queue.py                   # K5b queue generator
outputs: /data/liuyu/outputs/stageb_v1_1_k5*_rc1a_*
```

## Server access

```bash
ssh vla  # jump: scene@10.60.133.3 → liuyu@10.60.133.4
cd /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607
PYTHONPATH=src /home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python
```
