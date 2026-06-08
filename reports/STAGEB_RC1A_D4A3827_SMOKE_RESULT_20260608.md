# Stage-B RC1a d4a3827 — 6-Parent Smoke Result

**Date**: 2026-06-08
**Anchor**: d4a3827
**Output**: `/data/liuyu/outputs/stageb_v1_1_targeted_expansion_smoke_rc1a_d4a3827/`

## Gate summary

| Criterion | Result |
|-----------|--------|
| Validator PASS | 12/12 |
| Infra fail | 0 |
| VIS/RAND seed+pair_id match | 6/6 |
| GPU 3/7 usage | 0 |
| RAND command opens (=0 across all) | 6/6 |
| cmd_specific yield | 2/6 (33%) |
| hard_neg confirmation | 1/1 failed — clean heuristic unreliable |
| rand_abstain signal | Confirmed — RAND phys confound pattern observed |
| Pair label postprocess | 6 pairs, 0 reject |

## Per-pair results

| Pair | Category | VIS open/streak | RAND open/streak | VIS qpos | RAND qpos | pair label verdict |
|------|----------|-----------------|-------------------|----------|-----------|-------------------|
| alphabet_soup s1 [50,60] | cmd | 2/1 | 0/0 | 0.0001 | -0.003 | **negative** |
| bbq_sauce s1 [55,65] | cmd | 3/3 | 0/0 | -0.017 | -0.017 | **negative** (shared qpos) |
| **bbq_sauce s2 [100,110]** | hard_neg_candidate | **6/5** | 0/0 | **0.026** | -0.001 | **cmd_sus + vis_spec_phys** |
| cream_cheese s2 [50,60] | phys | 8/7 | 0/0 | 0.038 | -0.030 | **cmd_sus + vis_spec_phys** |
| orange_juice s2 [20,30] | phys | 2/2 | 0/0 | -0.035 | -0.039 | **negative, early term (35 steps)** |
| tomato_sauce s2 [90,100] | rand_abstain | 5/5 | 0/0 | 0.00004 | -0.065 | **negative (cmd) + rand phys confound** |

## Key findings

### 1. Hard negative heuristic FAILED
`bbq_sauce s2 [100,110]` was selected as hard_neg_candidate based on clean features (opens=0, qpos_pre=0.001).
VIS PGD produced 6 opens + 0.026 qpos. RAND produced 0 opens + negligible qpos.
**This window is cmd+phys specific, not hard_neg.**

Conclusion: hard_neg cannot be assigned by clean heuristics alone. All hard_neg candidates must be verified by actual VIS/RAND pairs.

### 2. rand_abstain physical confound confirmed
`tomato_sauce s2 [90,100]`: RAND produced 0 command opens but significant physical response (|qpos|=0.065).
VIS produced 5 opens (below threshold) and negligible qpos.
This is a **physical-level random sensitivity** — distinct from command-level random sensitivity.

Implication: abstain head must cover both command AND physical random confound.
Current label: RAND phys confound, NOT random_command_sensitive.

### 3. Shared physical qpos is common
cream_cheese, orange_juice, bbq_sauce cmd all show both VIS and RAND producing significant |qpos|.
VIS-specific physical response (cream_cheese: 0.038 vs RAND: -0.030) is not cleanly separable
by absolute qpos_delta alone. Direction (sign) matters, and shared base qpos response exists.

### 4. Early termination risk
`orange_juice s2 [20,30]` terminated at step 35 (window starts at 20). Only 15 post-window steps.
The episode completed naturally before the attack window effect could play out.
Windows near episode boundaries must be flagged as `edge_candidate` or excluded.

## Label taxonomy update

Based on smoke findings, the label taxonomy needs these categories:

| Label | Definition |
|-------|-----------|
| `cmd_specific` | VIS open≥6 AND RAND open<6 |
| `vis_specific_phys` | VIS |qpos|≥0.01 AND RAND |qpos|<0.01 AND not early termination |
| `rand_command_sensitive` | RAND open≥6 |
| `rand_phys_confound` | RAND open<6 AND RAND |qpos|≥0.01 |
| `shared_qpos_response` | Both VIS and RAND |qpos|≥0.01 |
| `hard_negative` | VIS open<6 AND RAND open<6 AND no significant phys AND no early term |
| `unstable_or_edge` | Episode ≤50 total steps or window near boundary or early done_inside_window |
| `hard_neg_candidate` | **Pre-experiment label only** — NOT a verified hard_neg. Must be replaced by actual label after VIS/RAND testing. |

## Expansion readiness

- [x] job_id uniqueness: fixed in expansion worker generator (global range 300000+)
- [x] pair uniqueness: validator now checks 1 VIS + 1 RAND per pair_id
- [x] hard_neg renamed to hard_neg_candidate in expansion queue
- [x] orange_juice-like edge windows flagged
- [x] Output directory: new, not overlapping with smoke
- [ ] Queue CSV updated with revised category names
- [ ] Commit and push
