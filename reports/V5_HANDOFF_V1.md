# V5 Handoff: V4 Detector Status, V5 Upgrade Rationale, Gaps

**Date**: 2026-07-26 | **Branch**: `deepseek/integration-final-detector-20260724` | **Commit**: `e99fa27`

---

## 1. V4 Detector Status

### 1.1 Architecture

| Parameter | Value |
|-----------|-------|
| Model | Dual-branch CausalTCN (RF32 + RF128) |
| Input | 51D (25D spatial + 9D policy + 9D goal + 8D auxiliary) |
| Checkpoint SHA | `685ddadf90ad2ac4...` |
| Provider SHA | `6a7ab61d8dba8cb...` |
| Platt calibration | a=0.519, b=0.813 |
| Threshold τ | 0.855 |
| Persistence D | 6 |
| Candidate close gate | raw_gripper ≤ 0.5 |

### 1.2 Formal Matrix Results (20 parents × 5 arms)

```
20/20 accepted: 19 DONE_VALID, 1 DONE_CLASSIFIED_TC (spatial s0 K10 truncation)

Per-suite emission:
  libero_10:     5/5 emit (K10=10 for all attack arms)
  libero_object: 5/5 emit (K10=10 for all attack arms)
  libero_goal:   0/5 emit (detector never fires on goal tasks)
  libero_spatial:2/5 emit (s0=TC, s4=K10=10; s1/s3 no emit)

TRUE/RAND K10 completion:
  TRUE_T10 K10=10: 11/20 parents
  RAND_T10 K10=10: 13/20 parents
  ORACLE K10=10:   12/20 parents
  Clean success:   19/20 parents
```

### 1.3 Root Cause: Goal Zero Coverage

The V4 K10 labeler uses `candidate_close` (raw_gripper ≤ 0.5) as the **first funnel filter**. In goal tasks, the gripper is predominantly OPEN, so cc=False for ~70% of steps. Steps with cc=False are constructively eliminated before physical assessment. The Student learns: goal features → negative raw_logit ≈ -11.

```text
Root cause chain:
  cc is first filter → cc=False → label=0 (70% of goal steps)
  → cc=True but K10 corridor < 10 → label=0
  → Student learns "goal → negative score"
  → raw_logit ≈ -11, cal_prob ≈ 0.005
  → No scheduler configuration triggers emission
```

### 1.4 V4 VIS Attack: INVALID IMPLEMENTATION (3 bugs)

**Bug 1 — Wrong target**: `target_action = clean_raw_action` (grip=0.0, CLOSE). PGD optimizes to **maintain** CLOSE, not flip to OPEN. This is release-blocking, not premature-open.

**Bug 2 — Invalid token**: `TARGET_TOKEN_ID = 31744` maps to disc=255, which is OUT OF BOUNDS for all 3 tested models (max disc=254). No action bin exists at this token ID.

**Bug 3 — Empty execution class**: `target_execution_class = CLIP_MEDIATED_OPEN` has **0 tokens** in all models. No model output can satisfy this class constraint.

```text
Model token audit (all 3 models: libero_10, libero_spatial, libero_object):
  bins=255, vocab_size=32000
  NATIVE_OPEN:        127 tokens (31745-31871) raw>0.5
  NATIVE_CLOSE:       127 tokens (31873-31999) raw<0.5
  BOUNDARY:           1 token  (31872) raw=0.5000
  CLIP_MEDIATED_OPEN: 0 tokens ← EMPTY CLASS
  Token 31744:        disc=255 OUT OF BOUNDS ← INVALID
  Correct OPEN token: 31745 NATIVE_OPEN
```

**V4 Formal TRUE/RAND verdict**: `LEGACY_V4_VIS_OPEN = INVALID_ATTACK_IMPLEMENTATION`. Old results cannot support any scientific conclusion about V4 window quality, VIS feasibility, or model robustness.

### 1.5 Fixed VIS Canary Result (libero_10 s1, emit=83)

After fixing all 3 bugs:
- `target_action[6] = 1.0` (CANONICAL_RAW_OPEN)
- `TARGET_TOKEN_ID = 31745` (NATIVE_OPEN)
- `target_execution_class = NATIVE_OPEN`

Result: 10 PGD frames executed, `final_env_grip = +1.0` (CLOSE) for all frames. VIS does **not** flip the decoded gripper action at this emission point with ε=0.03, PGD-5.

Logit margin analysis:
```
Clean:
  OPEN logit: 22.0
  Top logit:  37.75 (at token 31744 — INVALID/non-action token)
  OPEN prob:  ~0
  Margin:     -15.75

After PGD-5:
  OPEN logit: 8.06
  Top logit:  25.38 (at token 31872 — BOUNDARY, raw=0.5)
  Margin:     -17.31
  OPEN mass:  0.0
  Loss:       49.16 → 17.63 (good optimization progress)
  Margin:     -49.16 → -17.31 (improving but still negative)
```

This single canary proves: at libero_10 s1 emit=83, the frozen ε=0.03/PGD-5 configuration cannot flip the gripper. It does **not** prove that all V4 windows are unreachable, nor that the V4 window is physically safe.

---

## 2. Why Upgrade to V5 Detector

### 2.1 V4's Three Structural Problems

| Problem | Consequence |
|---------|-------------|
| `candidate_close` funnel in training labels | Goal suite 0% coverage; Detector only fires when gripper is near-closed |
| Label depends on cc (policy action), not physics | Student learns policy-correlated state, not physical criticality |
| No stable-grasp / manipulation / safe-release phase model | Emit timing is release-proximity, not physical-vulnerability |

### 2.2 V5 Fixes

| V4 Problem | V5 Solution |
|------------|-------------|
| cc funnel deletes goal steps | Label Contract V2: `candidate_close` is an auxiliary head, never a gate for physical criticality |
| cc→label dependency | Physics Teacher V22: 5 independent heads with evidence lattice, no cc in physics computation |
| No phase awareness | Phase taxonomy: PREGRASP → STABLE_GRASP → TRANSPORT → RELEASE_APPROACH → SAFE_RELEASE |

### 2.3 V5 Emission Target

V5 should emit in the **intersection** of:

```text
Physical Criticality  ∩  VIS Flip Susceptibility
```

Not just "earlier than V4" — the target is windows where:
1. Object is firmly held (stable grasp established)
2. Object is gripper-dependent (lifted, support removed)
3. Safe release has NOT begun
4. Model is NOT yet deep-committed to CLOSE (VIS can actually flip)

---

## 3. Current Architecture

### 3.1 Label Contract V2

Five independent tri-state heads with evidence lattice:

| Head | Definition | Status |
|------|-----------|--------|
| `physical_criticality` | Gripper-dependent critical phase | Implemented, 15/15 tests |
| `k10_feasible` | K=10 attack window available | Implemented |
| `safe_release` | Placement-confirmed planned release | Implemented (placement-gated) |
| `instability` | Contact loss, slip, unplanned opening | Implemented (target-relative) |
| `gripper_closing_state` | Physical qpos closure measurement | Implemented |

Evidence lattice: any known+positive → value=1; all known+negative → value=0; else unknown (valid_mask=false). Unknown is never converted to negative.

### 3.2 Physics Teacher V22.1

| Factor | Description | Source |
|--------|-------------|--------|
| `grasp_state` | Target-finger contact + sustained dwell | `_contact_flags()` from V5 physics (mature code) |
| `contact_state` | Filtered contact: object_contact + gripper_contact | mujoco_contact_pairs |
| `comotion_state` | Object-EEF co-motion via cosine similarity | object_state qpos slices |
| `lift_state` | Target object Z displacement from initial | BDDL object slices |
| `instability_indicators` | Slip, contact_loss, pose_anomaly, width_increase | Target-relative measurements |
| `terminal_state` | Task success from episode_summary.json | episode_summary |
| `placement_state` | Object proximity to target region | Object slices + target_names |
| `safe_release` | Release event + placement confirmation | Gripper qpos + placement |
| `gripper_closing_state` | Physical gripper closure from qpos velocity | Gripper qpos |
| `gripper_physics` | Raw qpos, width, velocity | Sidecar |

### 3.3 N5 Student Model

```text
Dual CausalTCN (kernel_size=2):
  RF32 branch  → exact 32-step receptive field
  RF128 branch → exact 128-step receptive field
  → shared fusion
  → 5 independent ScalarHeads (one per Label Contract head)

Features: 51D input, timestep padding mask, frozen per-head pos_weight
Status: 18/18 tests PASS (including CUDA A800)
Schema SHA: 468c34b6ae...
```

### 3.4 Pilot V3 Results (12 episodes, state_35)

```text
12/12 identity match, 0 unknown→negative, 0 NaN/Inf
Critical:     36.7% (was 96.7% in V1 with EEF proxy)
Known-neg:    53.9%
Unknown:       9.4% (Goal t00/t07 — drawer/stove not in BDDL qpos)
Gripper close: 420 steps
Safe release:  0 (placement-gated, correct behavior)
Instability:   8 events
cc_in_physics: False (compile audit)
Formal unchanged: True
```

Per-episode breakdown (goal tasks):
```
t00 (open drawer):   0.0% crit, 100% unknown — drawer not in BDDL qpos
t01 (bowl on stove): 61.5% crit, 38.5% known-neg
t06 (cream cheese):  67.7% crit, 32.3% known-neg
t07 (turn on stove): 0.0% crit, 100% unknown — stove knob not in BDDL
t08 (bowl on plate): 55.8% crit, 44.2% known-neg
t09 (wine on rack):  52.0% crit, 48.0% known-neg
```

---

## 4. Current Gaps

### 4.1 Critical — VIS-PGD Attack Implementation

| Gap | Detail |
|-----|--------|
| PGD bugs fixed | `target_action`, `TARGET_TOKEN_ID=31745`, `execution_class=NATIVE_OPEN` — all corrected |
| ORACLE at V4 emit | NOT YET RUN — need ORACLE-only canary to determine physical vulnerability of V4 window |
| PGD-5 vs PGD-20 | Logit margin improved -49→-17 in 5 steps but didn't cross zero. More steps may succeed. |
| ε=0.03 vs ε larger | Model may be outside 0.03 reachable radius at V4 emit |
| Config frozen? | Attack config (`fec_attack_v3.yaml`) has hardcoded 31744; new config `fec_attack_v5_open.yaml` fixes this |

### 4.2 High — Formal Evidence

| Gap | Detail |
|-----|--------|
| V4 Formal canonical ledger | 100-arm ledger exists at `/tmp/FORMAL20_CANONICAL_ARM_LEDGER_V1.jsonl` but not committed |
| V4D-OPEN baseline | No corrected-attack Formal matrix exists yet |
| V5 Detector-only replay | Not yet run on V5-trained Student |
| ORACLE physical vulnerability | Not yet demonstrated at V4 emit with matched snapshot |

### 4.3 High — Articulated Tasks

| Gap | Detail |
|-----|--------|
| Drawer tasks (goal t00, t03) | 100% unknown criticality — no drawer joint qpos in CS200 sidecar |
| Handle contact detection | Cabinet geoms (wooden_cabinet_1_g*) contact gripper fingers — can detect engagement |
| Joint progress | CANNOT measure without joint qpos — would need trajectory replay with articulated sidecar |
| Stove knob (goal t07) | Same issue — no joint/knob state in privileged data |

Current recommendation: N5-v1 supports pick-place tasks with strong Teacher; articulated tasks abstain (unknown). N5-v2 would add articulated Teacher after joint data is available.

### 4.4 Medium — Pending Infrastructure

| Gap | Detail |
|-----|--------|
| 1400 full label production | Not yet run — blocked on B2 canary (32-48 episode) |
| V5 Student training | Not yet run — blocked on Label Seal |
| Calibration + Scheduler | Not yet designed |
| Exact-prefix matched-arm runner | Not yet built for V4D-OPEN or V5 Formal |
| ORACLE matched-snapshot baseline | Need to confirm V4 window physical vulnerability |

### 4.5 Low — Documentation

| Gap | Detail |
|-----|--------|
| V5 Scientific Contract | Written (`configs/V5_SCIENTIFIC_CONTRACT_V1.json`), committed |
| Action Semantics Seal | Written (`configs/V5_ACTION_SEMANTICS_SEAL_V1.json`), committed |
| Formal canonical ledger | Generated, not committed |

---

## 5. Next Steps (Priority Order)

### Immediate (this session)

1. **Run ORACLE-only canary at V4 emit** — determine if V4 window is physically vulnerable
2. **Run PGD-20 canary** — determine if more steps can flip the gripper at V4 emit
3. **Commit logit margin analysis results**

### Short-term (next session)

4. **B1: Teacher V5 upgrade** — add `stable_grasp` and `manipulation_active` heads with phase codes
5. **B2: 32-48 episode production canary** — validate Label V2 across all mechanisms
6. **G6: 2000→1400 cohort join** — verify training identities
7. **B3: 1400 full label production** — 16 CPU shards

### Medium-term

8. **B4-B5: Student training** — Prior, MLP, RF32, RF128, Dual baselines + V5
9. **B6: Calibration + Scheduler** — freeze thresholds on calibration split
10. **B7: Detector-only replay** — compare V4 vs V5 on held-out episodes
11. **A3-A5: V4D-OPEN Formal** — frozen V4 Detector + corrected VIS-OPEN executor

---

## 6. Key Files

| File | Description | Status |
|------|-------------|--------|
| `scripts/fec/label_contract_v2.py` | Label Contract V2 (15/15 tests) | P0 closed |
| `n5/phase2_labels/v22_production_v2.py` | V22 Physics Teacher V2 (19/19 tests) | P0 fixed |
| `n5/phase2_labels/physics_teacher_v22.py` | V22 Schema (9/9 tests) | Updated |
| `n5/phase2_labels/run_pilot_12_v3.py` | Pilot V3 pipeline | 12/12 PASS |
| `n5/phase3_student/n5_student_model.py` | N5 Student model (18/18 tests) | Ready |
| `configs/V5_SCIENTIFIC_CONTRACT_V1.json` | V5 scientific mechanism | Frozen |
| `configs/V5_ACTION_SEMANTICS_SEAL_V1.json` | Action space mapping | Frozen |
| `scripts/fec/reconcile_direct_run.py` | Formal reconciliation | Ready |
| `reports/V4_FORMAL_FINAL_SEALED_RECEIPT_V2.json` | Formal 20/20 seal | Committed |
| `reports/PILOT_V3_RECEIPT.json` | Pilot V3 results | Committed |

---

## 7. Critical Constants

```text
V22 Schema SHA: 48472ce8ec9593851227e8969b6217ecbfd9e8f300e7ce2cf388613bd7def506
V22 Config SHA: 936aab0292bf691b9d0ce9d88ae250945b38a3ba49cbda9a14a763fc85f339b5
V5 Pilot V3 receipt SHA: (on server at pilot_12_v3_output/PILOT_RECEIPT_V3.json)

Action semantics:
  CANONICAL_RAW_OPEN  = 1.0
  CANONICAL_RAW_CLOSE = 0.0
  CANONICAL_ENV_OPEN  = -1.0
  CANONICAL_ENV_CLOSE = +1.0
  CORRECT_OPEN_TOKEN  = 31745 (NATIVE_OPEN, disc=254)

Formal queue: 20/20 accepted (19 DONE_VALID, 1 DONE_CLASSIFIED_TC)
```
