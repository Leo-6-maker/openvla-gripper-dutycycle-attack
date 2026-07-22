# Student Trigger Canary Handoff — 2026-07-22

> **Status:** DOCUMENTATION ONLY — NO training, NO attack rollout, NO code changes.
> **Recipient:** Codex execution agent.
> **Branch:** `deepseek/student-trigger-canary-handoff-20260722`

---

## 1. Executive Decision

**The frozen V2 Stage-1 search space has produced zero safety-eligible Student configurations under the preregistered any-head background-false-emit-rate ≤ 0.10 gate.**

This is NOT an AUPRC failure — models demonstrate clear release discrimination superiority over the LR baseline (mean ΔAUPRC ≈ +0.07). The failure is systematic: the **grasp head** emits on background steps at a mean rate of ~0.128, which exceeds the 0.10 any-head union gate in 359/400 observed runs.

**Key question for the next phase:** Do these grasp-head background emissions actually cause false FSM attack triggers? Or does persistence/state-machine filtering make the any-head union gate overly conservative?

**Recommended path:** Offline FSM replay → Passive online shadow → Small active engineering canary. Do NOT use attack results to back-select Student, threshold, or persistence. Do NOT enter formal Stage-2.

---

## 2. Repository and Commit Snapshot

| Item | Value |
|------|-------|
| **Working branch** | `codex/r10-final-factorized-detector-20260720` |
| **Handoff branch** | `deepseek/student-trigger-canary-handoff-20260722` |
| **HEAD** | `ea10e1d81db845bceac54424bb2669e436f6365c` |
| **Dirty files** | `scripts/detector_v5/build_factorized_oof_eval_auth.py` (modified, pre-existing, NOT related to V2) |
| **Untracked** | Various reports, tmp files, audit artifacts — do NOT delete |
| **Last 3 commits** | `ea10e1d` (LR key fix), `877b06b` (aggregator HOLD semantics), `d5a43af` (global HOLD) |

### Provenance Discrepancies (REPORTED — NOT VERIFIED BY HANDOFF AUTHOR)

| Claim | Source | Confidence |
|-------|--------|------------|
| Server execution commit = `f55e1849...` | User report | REPORTED_BY_USER |
| Archive authorization.source_commit = `f9e42f6f...` | VERIFIED_FROM_ARCHIVE | VERIFIED |
| Aggregator code chain HEAD = `ea10e1d` | VERIFIED_FROM_REPOSITORY | VERIFIED |
| `f55e1849` exists on server | User report | REPORTED — NOT VERIFIED from GitHub |
| `f55e1849` pushed to GitHub | NOT CONFIRMED | BLOCKER for remote disaster recovery |

**These three commits serve different roles:**
- `f55e1849`: Execution snapshot (server-local, not on GitHub)
- `f9e42f6f`: Authorization binding (found in archive authorization.json)
- `ea10e1d`: Aggregator/analysis code (GitHub HEAD)

**Before formal closeout:** verify `source_binding.json` in training artifacts matches authorization receipt and server git HEAD.

---

## 3. Authorization Boundaries

VERIFIED_FROM_ARCHIVE (authorization.json in partial archive):

```json
{
  "v2_inner_cv_authorized": true,
  "stage2_authorized": false,
  "engineering_oof_authorized": false,
  "full_fit_authorized": false,
  "cal_authorized": false,
  "check_authorized": false,
  "attack_authorized": false,
  "allowed_seeds": [42],
  "stage2_seeds_not_yet_authorized": [123, 456],
  "stage1_job_count": 864
}
```

**Currently permitted:**
- Code review and artifact audit
- Offline replay preparation
- New engineering canary protocol and authorization template preparation

**Requires new independent authorization before execution:**
- Passive online shadow
- Active engineering canary (any scale)
- Stage-2 multi-seed confirmation
- Engineering OOF
- CAL/CHECK

---

## 4. Server/Runtime Incident History

REPORTED_BY_USER — summarized from execution logs and user reports:

| Wave | Start | End | Workers | Outcome |
|------|-------|-----|---------|---------|
| 1-5 | ~20:38 | ~21:37 | 80 | Killed by pkill in restart scripts |
| 6 (definitive) | 22:14 | ~08:00+ | 80 | Completed 371 jobs; launcher crashed on TimeoutExpired |
| 7 (resume) | ~08:00 | ~09:09 | 80 | Launcher crashed again |
| 8 (resume) | 09:09 | 10:30 | 80 | Launcher crashed again |
| 9 (rescue V2AB) | 10:30 | ~10:45 | 16 | Stopped by user |
| 10 (rescue 6-GPU) | ~11:07 | running | 6 | 1 worker/GPU, currently active |

**Root causes of repeated launcher crashes:**
1. `pkill` in every restart script killed active workers
2. `timeout=900s` insufficient for V2C windowed-GRU on 400 episodes
3. CPU oversubscription (80 workers × 11 PyTorch threads = 880 threads)
4. `subprocess.TimeoutExpired` not caught in `run_cmd()` (fixed in `460aead`)
5. `timeout=3600s` still insufficient for V2C under GPU oversubscription

**Artifact impact:** 0 sealed artifacts lost from aborted waves. All 416 sealed results are from definitive wave 6 and subsequent postprocessing.

---

## 5. Immutable Artifact Inventory

### Sealed Artifacts on Server

| Artifact | Path (relative to OPS) | Status |
|----------|------------------------|--------|
| Inner-CV splits | `OFFICIAL_V3_FACTORIZED_STUDENT_V2_INNER_CV_SPLITS_V1_20260721` | SEALED |
| Job inventory (orig) | `OFFICIAL_V3_FACTORIZED_STUDENT_V2_JOB_INVENTORY_V1_20260721` | SEALED |
| Job inventory (stage1) | `OFFICIAL_V3_FACTORIZED_STUDENT_V2_STAGE1_JOB_INVENTORY_V1_20260721` | SEALED |
| LR baseline | `OFFICIAL_V3_FACTORIZED_STUDENT_V2_LR_BASELINE_V1_20260721` | SEALED |
| Stage-1 authorization | `OFFICIAL_V3_FACTORIZED_STUDENT_V2_STAGE1_AUTHORIZATION_V1_20260721` | SEALED |
| Stage-1 runs | `OFFICIAL_V3_FACTORIZED_STUDENT_V2_STAGE1_RUNS_V1_20260721` | 416/864 complete |
| Preauth quarantine | `V2_PREAUTH_QUARANTINE_20260721` | 80 artifacts, EXCLUDED |
| Sidecar output | `OFFICIAL_V3_FACTORIZED_STUDENT_V2_RECOMMENDED_EXACT_W32_V1_20260721` | 1/12 splits |
| Sidecar worktree | `/mnt/sdc/dty_user/openvla_attack_v2b_recommended` | Branch `codex/v2b-exact-w32-recommended-canary-20260721` |

### Current Grid State (REPORTED — NOT LIVE-VERIFIED)

| Metric | Count |
|--------|-------|
| Train sealed | 416 |
| Prediction sealed | 416 |
| Evaluation files | 416 |
| Audit PASS | 416 |
| Remaining | 448 |
| Staging directories | ~560 (mostly orphaned from aborted waves) |
| Active processes | ~6 (rescue launcher, 1/GPU) |

### Staging Directory Warning

The ~560 staging directories are predominantly UUID-tagged orphans from killed waves. They are NOT active training jobs. Do NOT:
- Count them as active workers
- Delete them without a formal quarantine manifest
- Use them to estimate completion rate

---

## 6. Partial Archive Integrity

**Archive:** `stage1_partial_400.tar.gz`
**SHA256:** `08ebdddd06530864204abd8e9c2b7c6f2f669a582afcc1c72956bcb05d352074`

VERIFIED_FROM_ARCHIVE:
- 400 eval files present
- 400 audit files present
- 400/400 exact eval-audit pairing
- All audit status = PASS
- All audit issues = []
- All full parity = True
- No duplicate labels
- No unexpected labels
- Included SHA256SUMS verifiable against included files
- LR pooled and per_split metrics hash-match

PENDING (requires server access):
- Complete LR root seal verification (archive is a subset)
- Complete training root seal verification
- authorization_receipt.json in training directories
- source_binding.json source_commit verification against authorization

---

## 7. Stage-1 Coverage

VERIFIED_FROM_ARCHIVE (400 runs):

| Candidate | Complete | Total | Coverage |
|-----------|----------|-------|----------|
| V2A | 187 | 288 | 64.9% |
| V2B | 140 | 288 | 48.6% |
| V2C | 73 | 288 | 25.3% |

| Window | Complete | Total |
|--------|----------|-------|
| W16 | 156 | 288 |
| W32 | 115 | 288 |
| W64 | 129 | 288 |

**Config completion:**
- 2/72 configs have 12/12 split closure
- 67/72 have at least 1 result
- 5/72 have 0 results (all V2C)

**The 5 untested configs (all V2C, 0/12):**
1. V2C_W32_H64_D0.1_WD1e-4
2. V2C_W32_H128_D0.0_WD1e-5
3. V2C_W32_H128_D0.1_WD1e-4
4. V2C_W64_H64_D0.0_WD1e-4
5. V2C_W64_H64_D0.1_WD1e-5

**Runtime censoring is severe and non-random:** V2C is systematically underrepresented due to timeout. W32 and W64 are underrepresented. WD=1e-5 is underrepresented relative to WD=1e-4. Do NOT rank configs by partial means.

---

## 8. Complete-Config Results (12/12 Splits)

VERIFIED_FROM_ARCHIVE:

### Config 1: V2A_W16_H64_D0.0_WD1e-05

| Metric | Value |
|--------|-------|
| Release AUROC mean | 0.6769 |
| Release AUPRC mean | 0.8363 |
| Release short AUPRC mean | 0.9478 |
| Mean first/later gap | 0.1611 |
| Worst background false emit | **0.1941** (FAIL) |
| Worst release overlap emit | 0.0018 (PASS) |
| Unsupported route emit | 0.0 (PASS) |
| ΔAUROC vs LR | +0.0463 |
| ΔAUPRC vs LR | +0.0697 |
| Background safety | 11/12 splits FAIL |

### Config 2: V2C_W64_H128_D0.1_WD1e-4

| Metric | Value |
|--------|-------|
| Release AUROC mean | 0.6484 |
| Release AUPRC mean | 0.8289 |
| Release short AUPRC mean | 0.9460 |
| Mean first/later gap | 0.2643 |
| Worst background false emit | **0.1837** (FAIL) |
| Worst release overlap emit | 0.0156 (PASS) |
| Unsupported route emit | 0.0 (PASS) |
| ΔAUROC vs LR | +0.0177 |
| ΔAUPRC vs LR | +0.0623 |
| Background safety | 12/12 splits FAIL |

**Both complete configs definitively fail the frozen safety gate. Neither is eligible for formal Stage-2 selection.**

---

## 9. Safety-Gate Diagnosis

VERIFIED_FROM_ARCHIVE:

**Frozen gate:**
- `background_false_emit_rate ≤ 0.10`
- `release_overlap_emit_rate ≤ 0.05`
- `unsupported_route_emit_rate = 0`

**400-run results:**

| Gate | Pass | Fail | Max Observed |
|------|------|------|-------------|
| Background false emit | 41 | 359 | 0.2329 |
| Release overlap emit | 400 | 0 | 0.0203 |
| Unsupported route emit | 400 | 0 | 0.0 |

**Background false emit distribution:**
- Mean: 0.1390
- Median: 0.1376
- Min: 0.0696
- Max: 0.2329

**Config-level conclusion:**
- 67/67 configs with ≥1 result have at least one observed background safety failure
- Under worst-split gate: all 67 are DEFINITIVE_SAFETY_FAIL
- 5 untested V2C configs: UNTESTED
- **0/72 configs: SAFETY_PASS**

---

## 10. Threshold Sensitivity

VERIFIED_FROM_ARCHIVE (diagnostic only — NOT authorizing threshold change):

| Background ≤ | Run Pass % | Complete Configs Pass |
|-------------|------------|----------------------|
| 0.10 (frozen) | 10.2% | 0/2 |
| 0.12 | 26.8% | 0/2 |
| 0.15 | 65.2% | 0/2 |
| 0.18 | 91.0% | 0/2 |
| 0.19 | 94.8% | 1/2 |
| 0.20 | 97.3% | 2/2 |

**Interpretation:** Rescuing both complete configs requires approximately doubling the background threshold (0.10 → 0.19-0.20). This is a fundamental gate redesign, not a minor calibration adjustment. Must have deployment risk justification — cannot be done solely to produce a shortlist.

---

## 11. Head-Level Decomposition

VERIFIED_FROM_ARCHIVE (400-run means):

| Background Head | Mean Emit Rate | Pass @ 0.10 |
|----------------|---------------|-------------|
| Grasp | 0.1282 | ~69/400 |
| Manipulation | 0.0000207 | 400/400 |
| Release | 0.0134 | 400/400 |
| Any-head union | 0.1390 | 41/400 |

**The failure is almost entirely from the grasp head.** Release head background emission is well-controlled (mean 0.013, max observed 0.049). The any-head union gate is dominated by grasp.

**Critical deployment question:** Does the grasp head emission cause false FSM attack triggers? If the attack executor only consumes release head (or requires grasp→manipulation→release FSM chain), the any-head gate may be measuring the wrong risk.

**This question must be answered by offline FSM replay before any gate redesign.**

---

## 12. Split/Identity Instability

VERIFIED_FROM_ARCHIVE:

- Splits `o0_i2` and `o3_i2` are the hardest: 0/33 and 0/33 runs pass background ≤ 0.10
- Split `o3_i0` is relatively easier
- Mean background rates vary by ~0.03-0.05 across splits
- Investigate: identity composition, background denominator, label boundaries, episode concentration, confidence intervals

Do NOT substitute worst-split with mean-split without protocol amendment.

---

## 13. Matched Architecture Comparison

VERIFIED_FROM_ARCHIVE:

### V2B vs V2A (~140 matched config/split pairs)

| Metric | Mean Δ (V2B − V2A) | Interpretation |
|--------|-------------------|----------------|
| Release AUROC | +0.0079 | Marginal improvement |
| Release AUPRC | +0.0040 | Negligible |
| Short AUPRC | −0.0018 | Slightly worse |
| Background emit | +0.0043 | Worse (undesirable) |
| Release overlap | +0.0083 | Worse — 140/140 higher |
| First/later gap | +0.0600 | Substantially worse |

### V2C vs V2A (~66 matched)

| Metric | Mean Δ (V2C − V2A) |
|--------|-------------------|
| AUPRC | +0.0092 |
| Background | +0.0278 (worse) |
| Overlap | +0.0097 (worse) |
| Gap | +0.0420 (worse) |

**Conclusion:** Event-balanced V2B provides minimal AUPRC gain with worse safety and stability. V2C shows marginally better AUPRC but substantially worse background, higher compute cost, and systemic timeout failures. Current evidence favors V2A as the engineering reference architecture.

---

## 14. LR Comparison

VERIFIED_FROM_ARCHIVE:

| Candidate | Mean ΔAUROC | Mean ΔAUPRC | % Beats LR |
|-----------|------------|------------|------------|
| V2A | +0.0458 | +0.0714 | 95.7% |
| V2B | +0.0478 | +0.0708 | 95.7% |
| V2C | +0.0524 | +0.0779 | 98.6% |

**Students consistently beat LR on release discrimination.** The failure is NOT that models cannot learn release — it's that the current safety gate (any-head union, fixed 0.5 threshold) is not passed by any configuration.

---

## 15. Sidecar Status and Limitations

VERIFIED_FROM_REPOSITORY (code at `codex/v2b-exact-w32-recommended-canary-20260721`, HEAD `401f79a05`):

| Metric | Value |
|--------|-------|
| Architecture | V2B exact-W32 causal TCN |
| Splits complete | 1/12 |
| Tests | 3/3 PASS |
| Release AUROC | 0.726 |
| Release AUPRC | 0.854 |
| Release short AUPRC | 0.959 |
| Background false emit | 0.142 (FAIL @ 0.10) |
| Audit | PASS |
| formal_selection_eligible | **false** |

**Fixes implemented in sidecar (vs original V2):**
1. Exact receptive field (RF=32, not ~63)
2. Route-specific class weights correctly propagated to loss
3. Jitter invalid prefix excluded from Teacher supervision mask

**Limitations:**
- Single split only — cannot draw config-level conclusions
- Background emit 0.142 > 0.10 — would be safety-eliminated
- Must NOT be promoted to formal Stage-1 result
- Must NOT be mixed into 864-job aggregation
- Engineering diagnostic only

---

## 16. Provenance Discrepancies

| Item | Status | Action |
|------|--------|--------|
| `f55e1849` on GitHub | NOT FOUND | Push from server or create git bundle for disaster recovery |
| `f9e42f6f` vs `f55e1849` | Both reported | Verify relationship — likely f9e42f6f is a later snapshot commit on server |
| LR key format | `o0_i0` (underscore) confirmed in archive | Aggregator fixed in `ea10e1d` |
| Archive LR completeness | Subset | Full LR root seal requires server artifacts |
| Authorization receipt in training dirs | UNVERIFIED | Check before formal closeout |

---

## 17. Aggregator Verification Checklist

VERIFIED_FROM_REPOSITORY (HEAD `ea10e1d`):

- [x] Reads `source_commit` from `source_binding.json`
- [x] Full `verify_sealed_directory()` on train and predict
- [x] 72 configs × 12 splits closure check
- [x] `math.isfinite()` on all metric values
- [x] Missing/NaN/Inf metric → global HOLD
- [x] Safety elimination (worst-split)
- [x] Same-split LR comparison (12 expected keys)
- [x] LR root seal verification
- [x] Per-split `abs(first - later)` gap (not cross-split mean cancellation)
- [x] Lexicographic selection (safety→AUPRC→short→gap→params)
- [x] Zero eligible → `HOLD_NO_ELIGIBLE_CONFIG`
- [x] Tie keeping (all ties survive)
- [x] LR incomplete vs no-eligible status separation
- [x] HOLD clears shortlist to empty, exits non-zero
- [x] LR key format: `o0_i0` (underscore) — FIXED in `ea10e1d`

**One additional verification needed:**
- [ ] Confirm `lr_splits` keys in actual JSON match `o0_i0` format (underscore). If server JSON uses `o0i0` (no underscore), aggregator will falsely report LR incomplete. Check before running.

---

## 18. Why Active Attack Is NOT Yet a Formal Selection Tool

1. Attack rollout uses a SINGLE checkpoint — cannot replace 72-config inner-CV selection
2. Attack success depends on PGD strength, target, and environment — not solely Student quality
3. Observing attack results then selecting Student/threshold creates data leakage
4. Current authorization explicitly sets `attack_authorized = false`
5. Active canary is ENGINEERING DIAGNOSTIC, not formal gate

---

## 19. Offline Replay Protocol

**PROPOSED_NOT_AUTHORIZED — requires new engineering authorization before execution.**

### Student Selection

1. **Primary:** V2A_W16_H64_D0.0_WD1e-05 (12/12 complete, lowest compute cost)
2. **Optional comparison:** Exact-W32 sidecar (1/12, engineering-only)
3. **Optional diagnostic:** A low-background but incomplete V2A (for threshold/Pareto exploration only)

### Required Data from Server

Per selected Student checkpoint (minimum 1 split, ideally 12):
- `heldout_step_predictions.jsonl` (all columns)
- `heldout_event_predictions.jsonl`
- `checkpoint.pt` (for potential re-inference)
- `run_config.json`, `source_binding.json`, `authorization_receipt.json`

### Offline Analysis

For each prediction shard, compute:

1. **Raw head scores:** grasp/manipulation/release probability per step
2. **Raw any-head emit:** `any(g≥τ, m≥τ, r≥τ)` per step
3. **FSM simulation:**
   - State machine: IDLE → GRASPED → MANIPULATING → RELEASED → IDLE
   - Persistence k ∈ {1, 2, 3, 4}
   - Attack triggered on transition to GRASPED or MANIPULATING (TBD per deployment spec)
4. **Per-episode metrics:**
   - False attack starts (background steps where FSM triggers)
   - Vulnerable-window recall (does FSM trigger during known oracle window?)
   - Trigger timing offset (how early/late vs oracle onset)
   - Trigger-window overlap/IoU
5. **Threshold/Persistence Pareto:**
   - Sweep τ ∈ [0.3, 0.9], k ∈ {1,2,3,4}
   - Plot: false-trigger rate vs vulnerable-window recall
   - Select operating point from inner-train/CAL data ONLY

### Fixed Parameters (from existing experiments)

- Attack duration: 8 or 10 steps
- Maximum window: ≤12 steps
- These are physical constraints from Black Bowl and Moka experiments

### Output

- `fsm_replay_metrics.json`
- `threshold_persistence_pareto.csv`
- `selected_operating_point.json` (frozen before any active canary)
- Must set minimum event recall/coverage constraint to prevent degenerate threshold selection

---

## 20. Passive Shadow Protocol

**PROPOSED_NOT_AUTHORIZED — requires new engineering authorization before execution.**

### Prerequisites
- Offline FSM replay complete
- Operating point (τ, k) frozen
- FSM false-trigger rate acceptable in offline analysis

### Execution
1. Load frozen Student checkpoint
2. Run online inference (no perturbation applied)
3. Record: head scores, FSM state, trigger windows
4. Compare: online vs offline predictions (must match)
5. Verify: causal input window, no future leakage
6. Verify: unsupported route never triggers

### Target Tasks
- **Priority:** Black Bowl State5 (oracle window ~78-87) and State7 (oracle window ~75-84)
- **Defer:** Moka (control fragility, second-pot localization complexity)

### Pass/Fail Conditions
- Online/offline prediction consistency: Δ ≤ tolerance
- No unsupported route triggers
- False FSM attack starts per episode: ≤ threshold (TBD in protocol)
- Trigger windows overlap known vulnerable phases
- Non-vulnerable phases do not produce sustained FSM starts

---

## 21. Active 8-Episode Smoke

**PROPOSED_NOT_AUTHORIZED — requires new engineering authorization before execution.**

### Configuration
- 1 frozen Student (from offline replay selection)
- Black Bowl State5 only
- 2 pre-frozen seeds
- 4 conditions × 2 seeds = 8 episodes

### Four Conditions

1. **Clean:** No attack, baseline execution
2. **Oracle fixed-window:** Gripper-targeted attack at known oracle window (e.g., steps 75-84)
3. **Student-triggered:** Gripper-targeted attack triggered by Student FSM
4. **Matched control:** Student-chosen window, but matched control perturbation (NOT gripper-targeted)

**The matched control must use an existing mechanism from the repository (e.g., `random_gripper_clean` or equivalent).** VERIFY the actual CLI from scripts before writing executable commands. Do NOT invent new control mechanisms.

### Fixed Attack Parameters (from existing experiments — VERIFY FROM REPOSITORY)

Reported from prior mechanism experiments:
- PGD epsilon: 0.10
- PGD steps (K): 20
- PGD alpha: 0.020
- Check `--force_open_raw_gripper 1.0` in actual CLI — if not set, gripper attack may reinforce wrong direction

### Required Metrics (per episode)

- Student head scores (grasp/manipulation/release)
- Threshold τ and persistence k
- FSM state trace and transition reasons
- Trigger start/end step
- Oracle-window offset and overlap/IoU
- Actual PGD steps executed
- Executed gripper open rate
- Maximum consecutive open streak
- Gripper qpos trace
- Object slip/drop flag
- Failure phase classification
- Clean/Control/Oracle/Student video
- Manual review label (blind to condition)
- LIBERO check_success (SR) — but do NOT rely on it as sole outcome

---

## 22. Optional 24-Episode Expansion

**PROPOSED_NOT_AUTHORIZED — only after 8-episode smoke passes.**

- State5 + State7
- 3 seeds
- 4 conditions
- = 24 episodes

Same metrics as 8-episode smoke. Moka deferred to future extension.

---

## 23. Frozen Parameter Sources

| Parameter | Source | Status |
|-----------|--------|--------|
| Student architecture, W, H, D, WD | 400-result partial diagnostic | INFERENCE — not formally selected |
| Head threshold τ | Offline FSM replay (inner-train only) | PROPOSED |
| Persistence k | Offline FSM replay | PROPOSED |
| Attack duration | Prior mechanism experiments (Black Bowl) | REPORTED — VERIFY from repo |
| PGD epsilon, K, alpha | Prior mechanism experiments | REPORTED — VERIFY from repo |
| Window length max | Physical constraints from oracle windows | REPORTED |

---

## 24. Required Metrics and Manual Review

**Beyond LIBERO SR:**

LIBERO `check_success` has been shown to miss gripper-open and drop failures. Every episode must record:

- Gripper qpos trace
- Executed open commands
- Object state (position, contact, support)
- Manual review label (blind to condition)
- Failure taxonomy: grasp-failure / premature-release / drop / slip / no-attack-executed / wrong-timing / control-failure

**Manual video audit protocol:**
- Blinded to condition name
- Two independent reviewers (if available)
- Pre-defined failure taxonomy
- Disagreement arbitration
- Key frames saved with video hash

---

## 25. Stop/HOLD Conditions

**Stop active canary immediately if:**
- Online/offline prediction mismatch exceeds tolerance
- Unsupported route triggers in deployment
- False FSM attack starts per episode exceeds pre-registered threshold
- Student window never overlaps oracle window
- Clean or control conditions show systematic failures unrelated to attack
- Any GPU OOM, NaN, or Traceback in attack pipeline

**Do NOT proceed to 24-episode expansion if:**
- 8-episode smoke does not show clean separation between conditions
- Matched control shows similar outcomes to Student-triggered attack
- Manual review cannot confirm attack causality

---

## 26. Codex Execution Checklist

Strict order. Do NOT skip steps.

1. **Checkout and verify:**
   ```bash
   git checkout deepseek/student-trigger-canary-handoff-20260722
   git rev-parse HEAD
   git status --short  # must show only handoff file + pre-existing dirty files
   ```

2. **Verify aggregator at `ea10e1d`:**
   - Read `scripts/detector_v5/aggregate_factorized_v2_stage1.py`
   - Confirm LR key uses `o0_i0` (underscore) format
   - Confirm global HOLD on metric integrity

3. **Check LR key format in actual JSON:**
   ```bash
   python3 -c "import json; d=json.load(open('LR_ROOT/per_split_metrics.json')); print(list(d.keys())[:3])"
   ```
   Expected: `['o0_i0', 'o0_i1', 'o0_i2']`
   If `['o0i0', ...]` (no underscore): aggregator will falsely report LR incomplete. BLOCKER.

4. **Verify authorization/source_binding/execution commit relationship:**
   - Read authorization.json source_commit
   - Sample 5 training dirs: read `source_binding.json` source_commit and `authorization_receipt.json`
   - All must match. Report discrepancies.

5. **Re-run partial diagnostic on 400 archive:**
   - Extract archive
   - Verify SHA256 of archive
   - Run aggregator in partial mode
   - Confirm 2 complete configs, 67 safety-fail, 5 untested

6. **Select ≤2 engineering Students:**
   - Primary: V2A_W16_H64_D0.0_WD1e-05
   - Optional: exact-W32 sidecar (engineering-only label)

7. **Acquire minimal step-prediction subset:**
   - At minimum: 1 split × 1 Student × step_predictions.jsonl
   - Preferred: 12 splits × primary Student

8. **Build offline FSM replay:**
   - Implement state machine with configurable τ and k
   - Output per-episode false-trigger count and oracle-window recall
   - Sweep τ ∈ [0.3, 0.9], k ∈ {1,2,3,4}

9. **Output threshold/persistence Pareto:**
   - Plot and save
   - Identify operating point candidates

10. **Freeze one operating point:**
    - Must use only inner-train/engineering data
    - Must set minimum recall/coverage constraint
    - Document selection rationale

11. **Create new engineering-only authorization:**
    - Independent root
    - `formal_selection_eligible = false`
    - `attack_authorized = false` (this is canary, not formal attack)
    - Bind specific Student checkpoint, τ, k, attack parameters

12. **Passive online shadow:**
    - Black Bowl State5 and State7
    - Verify online/offline consistency
    - Record FSM traces

13. **Check pre-registered pass/fail:**
    - False FSM starts per episode ≤ threshold
    - Oracle window overlap exists
    - No unsupported route triggers

14. **Only if shadow passes: 8-episode active smoke**

15. **Seal results:** Do NOT use for back-selection.

16. **Decision point:** Extend to 24 episodes OR report findings.

17. **Do NOT enter Stage-2** unless new formal protocol explicitly authorizes it.

---

## 27. Exact Commands Verified From Repository

**None in this handoff.** All CLI examples are PROPOSED_NOT_AUTHORIZED. The recipient (Codex) must verify actual script arguments from the repository before constructing any execution command.

Key scripts to verify:
- `scripts/detector_v5/predict_factorized_v2_inner_cv.py`
- `scripts/detector_v5/evaluate_factorized_v2_inner_cv.py`
- `scripts/detector_v5/audit_factorized_v2_inner_cv_predictions.py`
- `scripts/detector_v5/aggregate_factorized_v2_stage1.py`
- `scripts/detector_v5/launch_factorized_v2_inner_cv.py`
- `src/gripper_attack/v5_factorized_student_v2.py`
- Existing attack scripts (verify from repo, not from memory)

---

## 28. Open Questions / Blockers

| # | Question | Severity | Action |
|---|----------|----------|--------|
| 1 | Does grasp head background emit cause false FSM attack starts? | **CRITICAL** | Offline FSM replay required before any gate redesign |
| 2 | What is the relationship between `f55e1849`, `f9e42f6f`, and `ea10e1d`? | HIGH | Verify source_binding.json and authorization receipt |
| 3 | Are LR split keys `o0_i0` or `o0i0` in actual JSON? | HIGH | Check before running aggregator |
| 4 | Can `f55e1849` be pushed to GitHub for disaster recovery? | MEDIUM | Try from independent clone |
| 5 | Do the 5 untested V2C configs also fail background safety? | MEDIUM | Safety-closure rescue running (1 split each, GPU-isolated) |
| 6 | Does the any-head union gate measure the right deployment risk? | MEDIUM | Requires FSM replay + deployment spec review |
| 7 | What is the actual attack CLI for `random_gripper_clean` matched control? | MEDIUM | VERIFY FROM REPOSITORY |

---

## 29. Evidence-to-Claim Table

| Claim | Evidence Level |
|-------|---------------|
| 0/72 configs pass frozen safety gate | VERIFIED_FROM_ARCHIVE (400 runs) |
| Failure is from grasp head, not release | VERIFIED_FROM_ARCHIVE (head decomposition) |
| V2A is preferred engineering reference | VERIFIED_FROM_ARCHIVE (matched comparison) |
| LR baseline integrated correctly | VERIFIED_FROM_ARCHIVE (same-split keys) |
| Aggregator implements frozen lexicographic selection | VERIFIED_FROM_REPOSITORY (`ea10e1d`) |
| 416 train sealed, 400 audit PASS | REPORTED_BY_USER |
| Server execution commit = f55e1849 | REPORTED_BY_USER |
| Grid currently running at 6 workers, 1/GPU | REPORTED_BY_USER |
| Sidecar exact-W32 fixes work | VERIFIED_FROM_REPOSITORY (code + 3 tests) |
| Sidecar 1/12 split metrics | REPORTED_BY_USER |
| FSM replay will show grasp-head false triggers are harmless | INFERENCE — NOT YET TESTED |

---

## 30. Final GO/HOLD Matrix

| Activity | Status | Condition to Proceed |
|----------|--------|---------------------|
| Code review / artifact audit | **GO** | — |
| Offline FSM replay | **GO** (after engineering auth) | New engineering authorization |
| Passive online shadow | **HOLD** | Offline replay shows acceptable FSM false-trigger rate |
| 8-episode active smoke | **HOLD** | Passive shadow passes all pre-registered checks |
| 24-episode expansion | **HOLD** | 8-episode smoke shows clean condition separation |
| Formal Stage-2 | **HOLD** | New formal protocol + complete 864 or protocol amendment |
| Engineering OOF | **HOLD** | — |
| CAL / CHECK | **HOLD** | — |
| Formal attack matrix | **HOLD** | — |
| Modifying background gate threshold | **HOLD** | Must have deployment risk justification + new protocol |
| Deleting orphan staging directories | **HOLD** | Formal quarantine manifest required first |
| Back-selecting Student from attack results | **FORBIDDEN** | — |

---

*Handoff generated 2026-07-22. This document is a snapshot — server state may have changed. All PROPOSED actions require independent authorization before execution.*
