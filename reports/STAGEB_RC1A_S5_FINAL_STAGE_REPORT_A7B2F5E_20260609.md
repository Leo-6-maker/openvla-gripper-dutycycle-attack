# Stage-B RC1a S5 Final Report: Fixed-Env K-Repeat Stable Labels and Abstain-First Selector v0

**Date**: 2026-06-09
**Commit**: a7b2f5e
**Branch**: exp/vis-prefix-margin-repair-20260603

---

## A. Executive Summary

S5的核心成果不是"完整detector solved"，而是：

1. Fixed-env K-repeat标签协议成立
2. 旧8/8 unstable被追溯为seed-coupling protocol bug
3. K5+K5b得到24-parent stable pool
4. Layer-1 rand-risk/abstain detector在stable labels上初步成立
5. 完整online pipeline当前最强策略是Abstain(CleanRand)+TaskRank
6. Layer-2 cmd ranking仍task-biased/WIP
7. Layer-3 strict physical bridge仍underpowered

---

## B. Commit / Provenance Chain

```
d4a3827  RC1a freeze anchor
14cfabe  72-pair exploratory pool
a20379f  --env_seed / --attack_seed split (P0 fix)
0e3428f  seeded RAND generator + env_seed/attack_seed provenance
de1a7be  S5 stable pool / K5b completion
394b04d  detector sanity v0.1
7fe209e  selector v0 (oracle-yield ranking, upper bound only)
a7b2f5e  leakage-free selector v0.2 ← CURRENT HEAD
```

---

## C. RC1a Gripper Semantics

```
raw_gripper > 0.5  → env_action_6 = -1.0 → physical OPEN
raw_gripper < 0.5  → env_action_6 = +1.0 → physical CLOSE
raw_gripper == 0.5 → boundary / neutral
```

所有S5 K-repeat jobs使用RC1a corrected semantics (`raw_gripper_to_env_gripper(binarize=True)`).

---

## D. Old 8/8 Unstable — Reinterpretation

旧Silver confirmation发现8/8 parents unstable。但prefix audit发现6/8 parents在pre-window clean prefix上不一致。根因是旧runner使用单一`--seed`同时控制env replay和attack perturbation。因此旧8/8 unstable**不能**解释为attack intrinsic stochasticity——它是seed-coupled replay protocol bug。

**旧8/8 unstable结论被撤回，重新解释为protocol bug。**

---

## E. S5 Fixed-Env K-Repeat Protocol

```
env_seed fixed within parent
attack_seed varies over {0,1,2,3,4}
each parent: 5 VIS + 5 RAND
VIS/RAND pair shares pair_id, env_seed, attack_seed
labels are probability labels, not single-shot
```

**Probability definitions:**
```
pV_cmd = #VIS_cmd_success / K
pR_cmd = #RAND_cmd_success / K
yield_cmd = pV_cmd - pR_cmd
risk_rand = max(pR_cmd, pR_phys)
```

**Stability label rules (K=5):**
```
stable_cmd_specific:    pV_cmd≥0.6, pR_cmd≤0.2, yield≥0.4
stable_rand_sensitive:  pR_cmd≥0.4 or pR_phys≥0.4
stable_negative:        pV_cmd≤0.2, pR_cmd≤0.2, pV_phys≤0.2, pR_phys≤0.2
stable_vis_phys:        pV_phys≥0.6, pR_phys≤0.2, yield_phys≥0.4, NOT shared_phys
```

---

## F. K5 + K5b Results

| Round | Jobs | Result | Parents |
|-------|------|--------|---------|
| K5 | 80 | 80/80 PASS, 0 failures | 8 |
| K5b | 160 | 160/160 PASS, 0 failures | 16 |
| **Combined** | **240** | **240/240 PASS** | **24 (22 CMD-stable)** |

**Combined stable pool:**
```
stable_cmd_specific:    11
stable_rand_sensitive:   6
stable_negative:         5
unstable_or_unknown:     2
stable_vis_phys:         5
Task coverage:           8 tasks
```

**Key windows:**
```
tomato [55,65]:   GOLD cmd+phys (pV=1.0,pR=0.0, VIS=[7,7,7,8,7], strict vis_phys)
salad  [70,80]:   clearest rand confound (pV=0.0,pR=1.0, RAND=[10,8,10,10,10])
tomato [115,125]: genuine rand-sensitive (pV=0.8,pR=0.8)
milk   [230,240]: perfectly stable cmd (VIS=[11,11,11,11,11])
bbq    [100,110]: stable negative after seed fix (VIS=[0,0,0,1,1])
```

---

## G. Detector v0.1 (a7b2f5e, stable labels only, 19 parents)

| Head | AUROC | P@3 | Verdict |
|------|-------|-----|---------|
| C: rand/abstain | 0.833 | 1.00 | **PASS** — Clean > TaskOnly, shuffle collapses |
| A: cmd_specific | 0.650 | 0.67 | Task-biased, shuffle improves (spurious) |
| B: strict phys | 0.667 | 0.67 | Underpowered (pos=3 in pool) |

**Rand head is the only claimable detector signal.**

---

## H. Selector v0.2 — Leakage-Free (a7b2f5e)

v0 uses oracle yield for ranking (UPPER BOUND only, not online).
v0.2 uses OOF rand/cmd scores — fully leakage-free.

| Strategy | rand_hit | cmd_hit | mean_yield |
|----------|----------|---------|------------|
| Random | 0.12 | 0.62 | 0.57 |
| TaskOnly (no abstain) | 0.25 | 0.62 | 0.65 |
| **Abstain(CleanRand)+TaskRank** | **0.12** | **0.75** | **0.57** |
| Abstain(CleanRand)+CleanCmd | 0.12 | 0.62 | 0.50 |
| Oracle UB | 0.00 | 1.00 | 1.00 |

**Conclusion**: Abstain filter reduces rand risk (0.25→0.12) and improves cmd selectivity (0.62→0.75). CleanCmd OOF ranking is weak — TaskOnly still stronger for positive ranking. Layer-1 PASS, Layer-2 WIP.

---

## I. Current Pipeline Status

```
Layer 0: mechanism eligibility
  NOT YET IMPLEMENTED. Needed for cross-suite.

Layer 1: rand-risk / abstain detector
  PASS at sanity + selector level. Current strongest contribution.

Layer 2: cmd-specific ranking
  WIP. TaskOnly > CleanCmd. Needs same-task contrast + action-hidden features.

Layer 3: strict physical bridge
  Underpowered (pos=5). Needs more strict phys positives.
```

---

## J. Allowed Claims

1. Fixed-env K-repeat protocol resolves seed-coupled instability
2. K5+K5b produced 24-parent stable pool across 8 tasks
3. Random-sensitive/abstain head is learnable from clean online features (AUROC=0.833)
4. Abstain-first selection improves window selection quality over TaskOnly baseline
5. Current best online strategy is CleanRand abstain + TaskRank
6. Cmd ranking and strict phys prediction remain open problems

---

## K. Forbidden Claims

1. ❌ Detector solved
2. ❌ Full online selector generalized
3. ❌ Cmd vulnerability detector works across tasks
4. ❌ Strict physical detector is solved
5. ❌ Visual route is permanently dead
6. ❌ 72-pair single-shot labels are ground truth
7. ❌ Old 8/8 unstable proves intrinsic attack stochasticity
8. ❌ Selector v0 oracle-yield result is a fully online result

---

## L. Visual Route Status

Global OpenVLA SigLIP sidecar did not help under old single-shot exploratory labels. Stable labels now exist — this is not a final rejection. If visual/action representation is resumed, priority order:

1. Action-token logits / hidden states (closest to VIS mechanism)
2. Gripper-object crop features
3. Temporal phase features
4. Global SigLIP only as low-priority ablation

---

## M. Next Recommended Experiments

### Priority 1: K5c targeted expansion (160 jobs)

Goal: stable pool 24 → ~40 parents.

Targets:
- stable_rand_sensitive: 6 → 10+
- stable_vis_phys: 5 → 8–10
- More same-task contrast for cmd head

### Priority 2: Detector v0.3 after K5c

Re-run leakage-free selector on K5+K5b+K5c stable pool.

### Priority 3: Action-hidden sidecar

Extract clean OpenVLA action-token features to improve Layer-2 cmd ranking.
