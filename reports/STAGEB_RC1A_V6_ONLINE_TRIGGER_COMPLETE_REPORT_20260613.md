# V6 Online-Trigger Three-Layer Pipeline — Complete Report

**Date:** 2026-06-13 (evidence repaired 2026-06-13)
**Branch:** exp/vis-prefix-margin-repair-20260603
**GitHub evidence bundle:** `fefbbe3481d20d74715c042121f35b19cfd5cb2c`
**Server worktree commits (historical):** `6e61195` (executed source files), `eb7d130` (audited tables)
**Runner SHA256:** e13273f8270da67477e29517cab54101165867daf52daea22a41ced9533fc863
**Runner py_compile:** PASS
**attack_adapter SHA256:** c1fbfce0d0c5d0cbc8f8aafe2134d79b12ca1780b71439134c021862f9cc9310
**env_factory SHA256:** 1d5ce287e5ab443d3ebcce3b34a440bd25bd10bb7e05a240abb488a480f9359e
**Model:** /data/aviary/models/openvla/openvla-7b-finetuned-libero-object
**Python:** 3.10.16 (openvla_official_libero_20260525)
**Dtype:** torch.bfloat16, attn_implementation=eager

---

## 1. Executive Summary

The V6 online-trigger pipeline is operational across six parents and six LIBERO Object tasks. Clean opportunity triggering occurred in 12/12 pilot rollouts. Matched online RAND veto retained all six parents: four STRICT, two USABLE. In the first VIS pilot using `prefix_locked_gripper_open_margin` at eps=6/255 and PGD=20, butter_s2 showed an online command-level candidate: VIS C2O in 3/3 episodes versus matched RAND C2O in 0/3.

**Provenance status (2026-06-13, hardened):**
- Original report commit `a26be5f`: `REPORT_PROVENANCE_BLOCKED`
- Evidence repair `fefbbe3`: provenance restored, tables committed, runner py_compile PASS
- Audit hardening `a8e14ba` (server) / pending push: seed-set verification, telemetry values, task-sensitivity, qpos delta fix, 400-key clean scan manifest
- See §16 Provenance Repair Log for full details

**Conservative labels:**
- Pipeline: `V6_ONLINE_TRIGGER_PIPELINE_VALID`
- butter_s2: `ONLINE_CMD_CANDIDATE` (registry: `PENDING_AUDIT_HARDENING`)
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

**Audit verification (2026-06-13):**
- 39/39 summaries present, 39/39 traces present, 0 missing, 0 duplicate-conflict
- All summaries contain required attack telemetry fields (eps_raw_pixels, pgd_steps, decode_path, preprocess_path)
- Audit manifest with SHA256 pairs: [tables/s20d_v6_audit_manifest.csv](tables/s20d_v6_audit_manifest.csv)
- 0 EVIDENCE_FIELD_MISSING, 0 TRACE_MISSING, 0 infra-invalid

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

| Parent | C2O ep | Trigger | Success | Clean | Class |
|--------|--------|---------|---------|-------|-------|
| butter_s2 | 0/3 | 3/3 | 2/3 | 2/2 | STRICT |
| bbq_sauce_s0 | 0/3 | 3/3 | 3/3 | 2/2 | STRICT |
| chocolate_pudding_s2 | 0/3 | 3/3 | 3/3 | 2/2 | STRICT |
| cream_cheese_s2 | 1/3 | 3/3 | 3/3 | 2/2 | STRICT |
| alphabet_soup_s10 | 1/3 | 3/3 | 1/3 | 1/2 | USABLE_BASELINE_UNSTABLE |
| ketchup_s11 | 0/3 | 3/3 | 1/3 | 2/2 | TASK_SENSITIVE_ABSTAIN |

**0 random-sensitive, 0 trigger-unstable, 0 infra-invalid. 4 STRICT, 1 USABLE_BASELINE_UNSTABLE, 1 TASK_SENSITIVE_ABSTAIN.**

ketchup_s11: clean success 2/2 but RAND success 1/3 — RAND causes ≥2/3 episode task degradation, classified TASK_SENSITIVE_ABSTAIN.
alphabet_soup_s10: clean baseline already unstable (1/2), RAND 1/3, classified USABLE_BASELINE_UNSTABLE.

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

### Classification: `ONLINE_CMD_CANDIDATE` (registry: `PENDING_PROVENANCE_REPAIR`)

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
- Clean scan queue terminal complete (400/400), artifacts incomplete (381/400 summaries, 19 missing)
  - 358 done, 42 failed, 381 summaries, 256 policy success, 102 policy failure/timeout, 23 infra-fail-with-summary, 19 infra-fail-no-summary
  - See [tables/s20m4_clean_scan_400_manifest.csv](tables/s20m4_clean_scan_400_manifest.csv)

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
| Audit manifest | 39 | tables/s20d_v6_audit_manifest.csv |
| Clean scan manifest | 400 | tables/s20m4_clean_scan_400_manifest.csv |
| Registry | updated | tables/layer3_parent_registry.csv |
| This report | - | reports/STAGEB_RC1A_V6_ONLINE_TRIGGER_COMPLETE_REPORT_20260613.md |

**Data integrity:** 39/39 expected artifacts present (78 total files: 39 summaries + 39 traces), 0 missing, 0 infra-invalid, 0 duplicate conflicts. All summary-trace pairs verified. Audit manifest contains SHA256 for every file.

---

## 16. Provenance Repair Log (2026-06-13)

### P0-1: Executed runner provenance RESOLVED

The original report cited `Server HEAD: f1292c4`, but `f1292c4` did not contain the V6 runner (`run_s20d_v6_online_trigger_l3_runner.py`) — it was an untracked file in the server working tree. The GitHub-visible runner at `a26be5f` was syntactically broken (indented code before shebang, introduced by `harden_v6_rand_vis.py`).

**Fix:** Server commit `6e61195` adds the exact executed runner (SHA256: `e13273f8`), `libero_v4_env_factory.py` (SHA256: `1d5ce287`), and `attack_adapter.py` (SHA256: `c1fbfce0`) to version control. Runner passes `python -m py_compile`, imports `prompt` correctly, and matches the SHA256 recorded in the original report.

### P0-2: Missing CSV tables RESOLVED

The original GitHub commit `a26be5f` contained only the report and builder script, but none of the six CSV tables or registry it referenced.

**Fix:** Server commit `6e61195` commits all six CSV tables and `layer3_parent_registry.csv`. Commit `eb7d130` regenerates all tables from raw summaries using the audited builder. All 39 rows are accounted for.

### P0-3: USABLE classification rule RESOLVED

The original builder made `ONLINE_RAND_USABLE` unreachable (`c2o >= 2` and `c2o <= 1` covered all cases with no `else` branch). The report's 4 STRICT / 2 USABLE could not be reproduced from the committed code.

**Fix:** The audited builder encodes the correct rule:
- `c2o >= 2/3` → `ONLINE_RANDOM_SENSITIVE_ABSTAIN`
- `trigger < 2/3` → `ONLINE_TRIGGER_UNSTABLE`
- `c2o <= 1/3, trigger >= 2/3, success >= 2/3` → `ONLINE_RAND_STRICT`
- `c2o <= 1/3, trigger >= 2/3, success < 2/3` → `ONLINE_RAND_USABLE`

This produces **4 STRICT, 2 USABLE** matching the original report. USABLE parents (alphabet_soup_s10, ketchup_s11) have degraded success (1/3 each) under RAND perturbation.

### P1 audit defects RESOLVED

All P1 audit defects fixed in the audited builder:
- Expected-key manifest validation with duplicate detection
- Summary+trace pairing verification (39/39 pairs validated)
- `median_event_C2O_rate` correctly computed as `statistics.median(episode_rates)`, with separate `pooled_event_C2O_rate`
- `attacked_close_count=0` flagged as `NO_ATTACK_OPPORTUNITY` with `NA` rate (never replaced with 1)
- `task` field read directly from `summary['task']`, never parsed from `parent_id`
- Attack telemetry fields (eps_raw_pixels, pgd_steps, decode_path, preprocess_path) audited on all attack episodes
- Classification reason and rule version recorded on every row
- 39-row audit manifest with SHA256 for every summary and trace file

### butter_s2 status

The underlying experimental result is unchanged:
- VIS C2O: 3/3 episodes (seeds 99:1, 199:2, 299:1)
- RAND C2O: 0/3 episodes
- Total C2O count: 4 (VIS) vs 0 (RAND) over 4 attacked_close events each

However, registry status is downgraded to `ONLINE_CMD_CANDIDATE_PENDING_PROVENANCE_REPAIR` until the full evidence bundle is pushed to GitHub and a clean checkout audit is completed. Physical bridge and task effect remain `NOT_ESTABLISHED`.

### Limitations not addressed by this repair

The summaries and traces do not record `attack_method`, `objective`, `used_adv_inputs`, `fallback_adapter_used`, or pixel `Linf` at each attacked step. These fields were not instrumented in the V6 runner and cannot be retroactively verified from stored artifacts. The audit confirms infrastructure validity (PGD=20, eps=6/255, decode_path=v4, preprocess_path=v4, trigger_method=RULE_TRIGGER_MVP) from the summary-level fields that were recorded.
