# Stage-B RC1a Session Handoff — 2026-06-10 S9b Ready

**Repository**: `Leo-6-maker/openvla-gripper-dutycycle-attack`  
**Branch**: `exp/vis-prefix-margin-repair-20260603`  
**Latest reviewed GitHub HEAD in this session**: `9ce0a7b` (`merge: S9b VIS fix + smoke fca5c3f`)  
**Purpose**: conservative handoff for a new ChatGPT/Codex window. Do not continue from chat memory alone; re-read the files listed below.

---

## 0. First instruction to the next window

Before proposing experiments or launching anything, read these files from GitHub at the current branch/head:

1. `reports/STAGEB_RC1A_S8_LAYER3_FINAL_HANDOFF_20260610.md`
2. `tables/s9b_phase1_port_sanity_results.csv`
3. `scripts/stageb/run_s9b_phase1_runner_attack_port.py`
4. `scripts/stageb/run_s9b_phase1_runner_visrand_smoke.sh`
5. `tables/s9b_phase1_runner_visrand_smoke_manifest.csv`

The previous assistant made several terminology mistakes. Treat this handoff as the corrected terminology reference.

---

## 1. Correct layer definitions

### Layer 1 — detector v0.3 / CleanRand abstain-first selector

Layer 1 is **not merely the VIS command result**. Layer 1 is the validated **CleanRand abstain-first detector/selector**.

Its role is:

```text
clean-only features / CleanRand score
→ abstain random-sensitive / random-confounded windows
→ keep cleaner VIS-specific candidate windows
```

Allowed wording:

```text
Layer 1 detector v0.3 is validated as an abstain-first random-confound filter for command-level VIS-specific OPEN selection.
```

Do **not** say:

```text
Layer 1 proves physical task failure.
Layer 1 solves the whole detector.
Layer 1 is just an attack result rather than the CleanRand abstain-first selector.
```

### Layer 2 — post-Layer1 second-stage selector / ranker

Layer 2 is the attempted enhancement after Layer 1, originally using action dynamics / action logits / hidden features / HiddenSafe-style ranking. This is **not validated**.

Current status:

```text
Layer 2 top-K/ranker direction failed fresh confirmation.
It should not be used as a main claim.
```

Future direction should be reframed as:

```text
Layer2A: hard-case veto / confound abstain
  - clean drift
  - high qpos baseline
  - natural qpos instability
  - random-trigger risk not fully captured by Layer1

Layer2B: physical-transfer-aware selector
  - only after Layer3 generates reliable physical labels
```

### Layer 3 — physical bridge evaluation

Layer 3 asks whether command-level OPEN transfers into physical gripper qpos/width response:

```text
VIS-induced OPEN command
→ gripper qpos / width opening
→ possible contact-quality failure or task failure
```

Layer 3 is not a detector. It is a physical bridge / causal validation stage.

---

## 2. Current validated / non-validated status

### Validated

1. **Layer 1**: detector v0.3 / CleanRand abstain-first selector is the only validated detector-like pipeline so far.
2. **S8 Phase1 ORACLE**: Phase1 runner shows ORACLE forced-open physical reachability. Milk L=10 is cleanest reference.
3. **S9b Phase1-port sanity**: the Phase1-port runner restores Phase1-like physical dynamics.

### Not validated

1. Layer 2 top-K / HiddenSafe / action-hidden ranker.
2. VIS/RAND physical bridge on the Phase1-port runner. It is prepared but **not yet run** in the reviewed state.
3. Full automatic physical-vulnerable-window discovery.
4. Object-wide or LIBERO-wide physical attack success.
5. “Detector solved.” Never say this.

---

## 3. Critical gripper semantics

Use the corrected RC1a gripper convention:

```text
raw_gripper > 0.5  → env_action_6 = -1.0 → physical OPEN
raw_gripper < 0.5  → env_action_6 = +1.0 → physical CLOSE
raw_gripper == 0.5 → boundary / neutral
```

In trace summaries, executable physical OPEN is:

```text
env_action_6 < -0.5
```

Do not revive old/open-inverted labels or pre-RC1a traces.

---

## 4. Layer-3 experiment chain so far

### S8 Phase1 — ORACLE upper-bound, Phase1 runner

Status: **PASS**.

Known result:

```text
Phase1 ORACLE L=10 on milk:
baseline ≈ 0.0019
ORACLE qpos_pos_area ≈ +0.261
response_delay_pos = 0
```

Meaning:

```text
Physical qpos opening is reachable in the Phase1 runner under forced OPEN.
```

This does **not** prove VIS/RAND physical success.

### S8 Phase2 — VIS/RAND smoke, Phase2 runner

Status: command gate PASS, physical bridge FAIL under Phase2 runner.

Known result:

```text
VIS produced many OPEN commands.
RAND produced few or no OPEN commands in short condition.
But qpos_pos_area was 0 for VIS/RAND in Phase2 runner.
```

Initial interpretation “VIS duty-cycle weakness” was superseded by S8b/S8c.

### S8b — same-runner ORACLE calibration in Phase2 runner

Status: runner/reference mismatch confirmed.

Known result:

```text
Phase2 same-runner baseline ≈ 0.0395
Phase2 same-runner ORACLE_OPEN had 10/10 or 30/31 OPEN
but qpos_pos_area ≈ 0
```

Meaning:

```text
Phase2 physical bridge failure cannot be attributed to VIS objective weakness,
because same-runner ORACLE also cannot reproduce Phase1 positive qpos opening.
```

### S8c — parity isolation: half-open + post_horizon=40

Status: mismatch persists.

Known result:

```text
Post horizon and window convention are not the primary mismatch source.
Phase2 ORACLE still does not reproduce Phase1 positive qpos opening.
```

### S9a — init-order parity A/B

Status: FAIL / negative parity result.

Known result:

```text
Removing qvel[:]=0 + sim.forward() before set_init_state did not change Phase2-style result.
Baseline remained ≈0.0395 and ORACLE qpos_pos_area remained 0.
```

Meaning:

```text
qvel-zero/sim.forward init-order is not the sole mismatch cause.
The mismatch is elsewhere, likely including reset/init-state behavior and/or the pre-window closed-loop trajectory induced by runner input/action path.
```

### S9b — Phase1-runner attack port sanity

Status: **PASS**.

Frozen table:

```text
condition     baseline    qpos_pos_area    qpos_neg_area    open_count    streak
clean         0.001967    0.000000         0.028322         0/10          0
oracle_open   0.001967    0.294899         0.003694         10/10         10
```

Meaning:

```text
The Phase1-port runner restores Phase1-like physical dynamics.
It is now a valid physical bridge testbed for a small VIS/RAND smoke.
```

This still does **not** prove VIS physical bridge.

---

## 5. Current GitHub state at 9ce0a7b

Commit `9ce0a7b` includes:

1. S9b sanity result table:
   - `tables/s9b_phase1_port_sanity_results.csv`
2. VIS P0 fix in:
   - `scripts/stageb/run_s9b_phase1_runner_attack_port.py`
3. VIS/RAND smoke manifest:
   - `tables/s9b_phase1_runner_visrand_smoke_manifest.csv`
4. VIS/RAND smoke launch script:
   - `scripts/stageb/run_s9b_phase1_runner_visrand_smoke.sh`

### VIS P0 fix status

The runner now uses:

```python
generate_action_from_inputs(adv_ids, adv_pv)
```

which calls autoregressive `model.generate(...)` on adversarial pixel values. This replaces the earlier wrong behavior of decoding prompt/input ids as if they were action tokens.

Summary records:

```text
adv_redecode_mode = model_generate_from_adv_inputs
```

---

## 6. Current approved next experiment

The next approved experiment is **only**:

```text
S9b Phase1-runner VIS/RAND physical bridge smoke
milk only
state_id = 0
window = [70, 80) half-open
open_duration = 10
seeds = 9, 10
conditions = vis_pgd, random_linf
4 jobs total
```

Manifest:

```text
950410 seed9  vis_pgd      GPU 1,0
950411 seed9  random_linf  GPU 1,0
950412 seed10 vis_pgd      GPU 4,5
950413 seed10 random_linf  GPU 4,5
```

Launch script:

```bash
scripts/stageb/run_s9b_phase1_runner_visrand_smoke.sh
```

Do not add seeds, tasks, windows, pattern replay, sustained objective, full queue, or Layer2 experiments before reviewing the 4-job smoke.

---

## 7. S9b VIS/RAND smoke gates

### Infra gate

After launch, require:

```text
4/4 summary JSON
4/4 trace CSV
4/4 infra_status = ok
no FAIL_/fatal/traceback/cuda/egl/mujoco/pgd_error/random_error
GPU released
```

### Science gate

ORACLE reference from sanity:

```text
oracle_ref_pos_area = 0.294899
```

Check per matched seed pair:

```text
Command:
VIS decoded_open_count > RAND decoded_open_count
VIS max_open_streak > RAND max_open_streak

Physical:
VIS qpos_pos_area > 0
VIS qpos_pos_area > matched RAND qpos_pos_area
VIS_norm = VIS qpos_pos_area / 0.294899 >= 0.2 or 0.3

Control:
RAND must not reproduce the same positive qpos response.
```

### Interpretation branches

If VIS command and qpos are both stronger than RAND:

```text
Allowed claim:
Phase1-port milk-only physical bridge proof-of-concept PASS.
```

If VIS command OPEN happens but qpos does not respond:

```text
Interpret as current VIS duty-cycle/objective insufficient under a physical-reachable runner.
Only then consider sustained-open / close-interruption penalty objective.
```

If RAND also shows positive qpos:

```text
Treat as random/confounded, not VIS-specific physical success.
```

---

## 8. Forbidden claims / actions

Do not claim:

```text
Detector solved.
Layer 2 validated.
Layer 3 solved.
Object-wide physical attack success.
Full automatic vulnerable-window discovery.
VIS physical bridge restored before S9b smoke results exist.
```

Do not use:

```text
old 44-row table as final labels
old overnight labels
pre-v1.1 traces
Bronze labels as final labels
random_sensitive as negative
old inverted OPEN convention
```

Do not launch without explicit review:

```text
full 24-job queue
extra seeds
extra windows
sustained objective
pattern replay
Layer2 ranker reruns
```

---

## 9. Recommended message for the next ChatGPT window

Paste this to the next assistant:

```text
You are taking over the OpenVLA Gripper Duty-Cycle Attack project.
Repository: Leo-6-maker/openvla-gripper-dutycycle-attack
Branch: exp/vis-prefix-margin-repair-20260603
Latest reviewed HEAD: 9ce0a7b

First read:
reports/STAGEB_RC1A_SESSION_HANDOFF_20260610_S9B_READY.md
reports/STAGEB_RC1A_S8_LAYER3_FINAL_HANDOFF_20260610.md

do not continue from old chat memory.

Important corrected terminology:
Layer1 = detector v0.3 / CleanRand abstain-first random-confound filter, validated.
Layer2 = post-Layer1 second-stage ranker/selector, not validated; future should be veto/physical-aware.
Layer3 = physical bridge command→qpos, not a detector.

Current state:
S9b Phase1-port sanity PASS. VIS P0 re-decode fixed. S9b 4-job VIS/RAND smoke is prepared but not launched.
Only review/launch those 4 jobs if explicitly asked. Do not add seeds/windows/objectives.
```
