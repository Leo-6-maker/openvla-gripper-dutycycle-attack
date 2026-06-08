# S5 K5b Stable-Label Expansion — 10-Hour Autonomous Runbook

**Target session**: Next DeepSeek
**Commit**: 0e3428f
**Branch**: exp/vis-prefix-margin-repair-20260603

## Context

K5 completed: 80/80 PASS, 8/8 parents stable. The old "8/8 unstable" was a seed-coupling protocol bug. Fixed-env K-repeat reveals clear stable label structure. Detector path REOPENED on stable labels.

K5 stable labels:
- stable_cmd_specific: 4 (milk[70,80], milk[230,240], tomato[90,100], tomato[150,160])
- stable_negative: 3 (bbq[100,110], salad[120,130], alphabet[65,75])
- stable_rand_sensitive: 1 (tomato[115,125])
- stable_vis_phys: 1 (tomato[90,100])

Goal: Expand stable parent pool from 8 to 20-24. Only train detector if Gate 6 passes.

## Server

```bash
ssh vla  # jump: scene@10.60.133.3 → liuyu@10.60.133.4
```
GPU: 1,0 / 2,6 / 4,5 (3,7 BLACKLISTED)
Conda: openvla_official_libero_20260525
Repo: /data/liuyu/repos/codex_stageb_openvla_alignment_rc1a_20260607/

## Phase Timeline

| Phase | Time | Task | Gate |
|-------|------|------|------|
| 0 | 0:00-0:45 | Seal K5 + provenance audit | 0A, 0B |
| 1 | 0:45-1:15 | Update claim boundary | 1 |
| 2 | 1:15-2:15 | Design K5b queue (12-16 parents, 120-160 jobs) | 2 |
| 3 | 2:15-2:45 | K5b 2-parent smoke | 3 |
| 4 | 2:45-7:00 | K5b full run | 4 |
| 5 | 7:00-8:00 | K5b postprocess + probability labels | 5 |
| 6 | 8:00-8:45 | Combined stable pool | 6 |
| 7 | 8:45-9:30 | Detector sanity (only if Gate 6 passes) | 7 |
| 8 | 9:30-10:00 | Decision + summary | 8 |

## Critical Rules

1. Every job: --env_seed fixed, --attack_seed 0..4
2. Gate fail → STOP, write failure report, do NOT proceed
3. No detector training on 72-pair single-shot labels
4. No GPU 3/7, no cross-suite, no visual sidecar
5. Do not pull new code or modify runner during run

## Final Deliverable

1-page summary with gate results, stable pool stats, detector sanity (or skip reason), next action.
