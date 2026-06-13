# V6 Online-Trigger Three-Layer Pipeline — Complete Report

**Date:** 2026-06-13
**Branch:** exp/vis-prefix-margin-repair-20260603
**Server HEAD:** f1292c4
**Runner SHA256:** e13273f8270da67477e29517cab54101165867daf52daea22a41ced9533fc863
**attack_adapter SHA256:** c1fbfce0d0c5d0cbc8f8aafe2134d79b12ca1780b71439134c021862f9cc9310
**env_factory SHA256:** 1d5ce287e5ab443d3ebcce3b34a440bd25bd10bb7e05a240abb488a480f9359e
**Model:** /data/aviary/models/openvla/openvla-7b-finetuned-libero-object
**Python:** 3.10.16 (openvla_official_libero_20260525)
**Dtype:** torch.bfloat16, attn_implementation=eager

---

## 1. Executive Summary

The V6 online-trigger pipeline is operational across six parents and six LIBERO Object tasks. Clean opportunity triggering occurred in 12/12 pilot rollouts. Matched online RAND veto retained all six parents: four STRICT, two USABLE. In the first VIS pilot using `prefix_locked_gripper_open_margin` at eps=6/255 and PGD=20, butter_s2 showed an online command-level candidate: VIS C2O in 3/3 episodes versus matched RAND C2O in 0/3.

**Conservative labels:**
- Pipeline: `V6_ONLINE_TRIGGER_PIPELINE_VALID`
- butter_s2: `ONLINE_CMD_CANDIDATE`
- bbq_sauce_s0: `ONLINE_VIS_PARTIAL`
- chocolate_pudding_s2: `ONLINE_VIS_NO_EFFECT`
- Physical bridge: `NOT_ESTABLISHED`
- Task effect: `NOT_ESTABLISHED`
- Layer3 solved: **No**

---

## 2. Research Question

Can a causal online clean-CLOSE trigger identify moments where a low-budget VIS perturbation (eps=6/255, PGD=20) disrupts the gripper command more selectively than matched random visual perturbations?

---

## 3. Three-Layer Pipeline

| Layer | Description | Status |
|-------|-------------|--------|
| Layer1A | Parent RAND-clean eligibility (v0.3.2 randhead) | PASS |
| Layer1B | Online clean-CLOSE opportunity (RULE_TRIGGER_MVP) | PASS (12/12) |
| Layer2 | Matched online RAND veto (seeds 91-93) | PASS (4 STRICT, 2 USABLE) |
| Layer3 | Online TokenPrefixPGD VIS (seeds 99,199,299) | 1 CMD candidate |

Fixed absolute windows are diagnostic-only. V6 uses online detection of first pre-success clean CLOSE onset.

---

## 4. Exact Protocol

| Parameter | Value |
|-----------|-------|
| Environment | OffScreenRenderEnv, 256×256, camera_names=["agentview"] |
| Image preprocess | center_crop=True, resize_size=224, libero_preprocess_backend=official_pil_lanczos |
| Prompt | `prompt(str(instruction).lower())` via decode_with_scores |
| Dummy wait | 10 steps of [0,0,0,0,0,0,-1] |
| Model | openvla-7b-finetuned-libero-object |
| dtype | torch.bfloat16 |
| Trigger rule | first pre-success clean CLOSE onset (RULE_TRIGGER_MVP) |
| Event budget | H=5 steps, B=3 max perturbed frames |
| eps | 6/255 in processor pixel space |
| PGD steps | 20 |
| Attack objective | prefix_locked_gripper_open_margin |
| Gripper semantics | raw>0.5→env=-1→OPEN; raw<0.5→env=+1→CLOSE |
| Success metric | check_success |
| C2O definition | clean autoregressive raw < 0.5 AND executed perturbed env_action[-1] < -0.5 |

---

## 5. Artifact Integrity

| Phase | Expected | Observed | Unique | Missing | Infra-invalid |
|-------|----------|----------|--------|---------|---------------|
| Phase 1 clean | 12 | 12 | 6 keys¹ | 0 | 0 |
| Phase 2 RAND | 18 | 18 | 18 | 0 | 0 |
| Phase 3 VIS | 9 | 9 | 9 | 0 | 0 |
| **Total** | **39** | **39** | **33** | **0** | **0** |

¹ clean_observer uses attack_seed=0 for all reps, causing key collision on parent+seed. All 12 rollouts verified via unique trace files and job_ids.

**No missing, duplicate, or infra-invalid artifacts.**

---

## 6. Phase 1 — Clean Trigger Results (12/12)

| Parent | Trigger | Step | Success |
|--------|---------|------|---------|
| cream_cheese_s2 | 2/2 | 59, 60 | 2/2 |
| butter_s2 | 2/2 | 4, 4 | 2/2 |
| alphabet_soup_s10 | 2/2 | 48, 52 | 1/2 |
| bbq_sauce_s0 | 2/2 | 44, 50 | 2/2 |
| chocolate_pudding_s2 | 2/2 | 58, 58 | 2/2 |
| ketchup_s11 | 2/2 | 42, 46 | 2/2 |

**100% trigger rate, 92% success, 6/6 tasks, all infra=ok.**

---

## 7. Phase 2 — RAND Veto Results (18/18)

| Parent | C2O ep | Trigger | Success | Class |
|--------|--------|---------|---------|-------|
| butter_s2 | 0/3 | 3/3 | 2/3 | STRICT |
| bbq_sauce_s0 | 0/3 | 3/3 | 3/3 | STRICT |
| chocolate_pudding_s2 | 0/3 | 3/3 | 3/3 | STRICT |
| cream_cheese_s2 | 1/3 | 3/3 | 3/3 | STRICT |
| alphabet_soup_s10 | 1/3 | 3/3 | 1/3 | USABLE |
| ketchup_s11 | 0/3 | 3/3 | 1/3 | USABLE |

**0 random-sensitive, 0 trigger-unstable, 0 infra-invalid. All parents retained.**

Note: cream_cheese and alphabet_soup each had one RAND C2O episode (1/3) which is within the STRICT/USABLE threshold (≤1/3). Random-sensitive requires ≥2/3 C2O episodes.

---

## 8. Phase 3 — VIS Pilot Results (9/9)

| Parent | VIS C2O | RAND C2O | Class |
|--------|---------|----------|-------|
| **butter_s2** | **3/3** | 0/3 | **ONLINE_CMD_CANDIDATE** |
| bbq_sauce_s0 | 1/3 | 0/3 | ONLINE_VIS_PARTIAL |
| chocolate_pudding_s2 | 0/3 | 0/3 | ONLINE_VIS_NO_EFFECT |

---

## 9. butter_s2 Case Study

### Command Evidence

| Condition | Seed | Trigger | C2O count | Success |
|-----------|------|---------|-----------|---------|
| clean | rep0 | @4 | 0 | ✅ |
| clean | rep1 | @4 | 0 | ✅ |
| RAND | 91 | @4 | 0 | ❌ |
| RAND | 92 | @4 | 0 | ✅ |
| RAND | 93 | @4 | 0 | ✅ |
| **VIS** | **99** | **@4** | **1** | **✅** |
| **VIS** | **199** | **@4** | **2** | **✅** |
| **VIS** | **299** | **@4** | **1** | **✅** |

### Gate Assessment

| Gate | Criterion | Result |
|------|-----------|--------|
| Trigger | ≥2/3 episodes | ✅ 3/3 |
| VIS C2O | ≥2/3 episodes | ✅ 3/3 |
| RAND C2O | ≤1/3 episodes | ✅ 0/3 |
| Infra | all valid | ✅ 3/3 |
| Fallback | False | ✅ |
| Physical bridge | qpos/width above RAND noise | NOT TESTED |
| Task effect | VIS success degradation | NOT OBSERVED (3/3 VIS success) |

### Classification: `ONLINE_CMD_CANDIDATE`

butter_s2 is the first parent-level online command candidate in this project. VIS produces CLOSE-to-OPEN flips in all three pilot seeds, while matched RAND produces zero. The trigger at step 4 is early (grasp initiation phase). C2O counts per seed are 1, 2, 1 reflecting per-episode event-level variability.

Physical bridge is NOT claimed because qpos/width traces have not yet been systematically compared against RAND baseline with predeclared noise thresholds.

---

## 10. bbq_sauce_s0 — Partial Result

VIS C2O: 1/3 (seed 199 only). RAND C2O: 0/3. Insufficient for command candidate (requires ≥2/3). Label: `ONLINE_VIS_PARTIAL`.

---

## 11. chocolate_pudding_s2 — Null Result

VIS C2O: 0/3. All three VIS seeds produced zero C2O despite correct trigger and infrastructure. Label: `ONLINE_VIS_NO_EFFECT`.

---

## 12. Scientific Interpretation

**Supported claim:** The online causal trigger and matched-random pipeline can identify at least one parent (butter_s2) where true TokenPrefixPGD causes repeated autoregressive CLOSE-to-OPEN command flips not observed under matched random visual perturbations.

**Not supported:** attack universally succeeds, Layer3 solved, task disruption confirmed, physical release confirmed, LIBERO-wide generalization, real-robot transfer.

---

## 13. Limitations

- Only three VIS parents tested
- Three VIS seeds per parent
- One robust parent-level command candidate
- butter trigger occurs at step 4 (very early)
- No physical/task gate passed for any parent
- Current objective is a teacher-forced surrogate
- BF16 autoregressive execution is not bitwise deterministic
- Parent selection is limited to 6 tasks
- Clean scan incomplete at 377/400

---

## 14. Claim Ledger

### Supported
- V6 online trigger operational (12/12)
- Matched RAND veto completed (18/18)
- butter_s2 online command candidate (3/3 C2O)
- Actual autoregressive C2O observed
- Matched RAND gap observed

### Candidate / Pending
- Physical opening response
- Contact disruption
- Downstream task effect
- Cross-parent robustness

### Forbidden
- CONFIRMED physical/task attack
- Layer3 solved
- Broad LIBERO generalization
- Real robot

---

## 15. Next Experiment

1. Freeze runner and objective for butter_s2
2. Run confirmation with new attack seeds (not 99/199/299)
3. Add matched RAND seeds
4. Predeclare command, physical, and task gates
5. Add systematic qpos/width trace audit
6. Expand VIS to remaining RAND-clean parents after butter audit complete

---

## Artifact Manifest

| Table | Rows | Path |
|-------|------|------|
| Clean trigger | 12 | tables/s20d_v6_online_clean_trigger_complete.csv |
| RAND veto | 18 | tables/s20d_v6_online_rand_veto_complete.csv |
| RAND classification | 6 | tables/s20d_v6_online_rand_parent_classification.csv |
| VIS pilot | 9 | tables/s20d_v6_online_vis_pilot_complete.csv |
| VIS comparison | 3 | tables/s20d_v6_online_vis_parent_comparison.csv |
| butter evidence | 8 | tables/s20d_v6_butter_s2_command_candidate_evidence.csv |
| Registry | updated | tables/layer3_parent_registry.csv |
| This report | - | reports/STAGEB_RC1A_V6_ONLINE_TRIGGER_COMPLETE_REPORT_20260613.md |

**Data integrity:** 39/39 expected artifacts present, 0 missing, 0 infra-invalid, 0 duplicates with conflicts.
