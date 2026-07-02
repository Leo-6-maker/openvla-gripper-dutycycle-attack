# OpenVLA Gripper Duty-Cycle Attack

> **Current project status and handoff guide — updated 2026-07-02.**  
> This README supersedes the old fixed-window / single black-bowl description. The repository has progressed to a clean-only privileged-Teacher → causal-Student detector, detector-triggered online visual attack pipeline, frozen LIBERO-Object evidence, and a cross-suite OpenVLA benchmark extension.

## TL;DR

This project studies a narrow inference-time vulnerability of Vision-Language-Action policies:

> **A short, gripper-targeted visual perturbation becomes highly destructive when it is aligned with a contact-critical manipulation phase.**

The current main scientific story is not “generic PGD breaks OpenVLA” and not “our loss is uniquely superior.” It is:

1. **timing specificity** — the same payload is much weaker at random or shifted times;
2. **direction/selectivity** — matched random-direction perturbations do not reproduce the effect;
3. **deployable timing** — the online trigger uses only causal proprioception/action history;
4. **mechanistic sufficiency** — a command-level OPEN oracle reproduces failure on the same emitted episodes.

The next paper-facing milestone is a **BadVLA-style, standardized OpenVLA/LIBERO benchmark presentation**, not further expansion of external attack baseline reproductions.

---

## Read this first: status hierarchy for humans and AI agents

When files disagree, use this order:

1. **Newest dated protocol / acceptance report in `reports/`**
2. **Frozen configs and manifests in `configs/` and `evidence/manifests/`**
3. **This README**
4. `docs/claim_and_evidence.md` and old `scripts/v4_*` documentation
5. `archive/`, legacy branch names, and historical run-directory names

Do **not** infer the current project state from the former README, a single old commit message, or a server checkout that has not been synchronized.

Generated rollout evidence is intentionally not all stored on `main`; large artifacts live in external evidence bundles / server roots with SHA256 manifests. See `docs/INDEX.md`.

---

## Current repository state

### Merged on `main`

- Clean-only SC5 canonical corpus, privileged Teacher, 25D causal Student detector, and detector-triggered Layer 3 bridge.
- Cross-suite clean collector, manifests, deep-integrity audits, and the accepted 300-episode non-Object clean corpus.
- Paper table schemas, claim gates, branch governance, and CPU tests.

Important merge/evidence lineage:

- `75dc110` — merged clean-only SC5 detector and Layer 1–3 mainline.
- `141657f` — merged cross-suite CLEAN300 collector and audit tooling.

### Separate formal-evaluation lineage

The LIBERO-Object formal Table 1 / SOTA-canary execution work was developed on a separate evaluation lineage ending at least at:

- `209b98c` — corrected TMA Random-Time canary accounting and added the fail-closed non-negative trigger guard.

Those adapted TMA/UMA experiments are now **exploratory/archival**, not the main paper direction after the latest advisor meeting.

---

## System architecture

```text
Layer 1 — Offline privileged Teacher
  clean rollout + simulator object/target/contact state
  → phase, corridor, release-safe, abstention labels

Layer 2 — Online causal Student trigger
  25D proprioception/action features only
  → phase/corridor/release predictions
  → IDLE → ARMED → EMITTED one-shot state machine

Layer 3 — Online visual attack executor
  detector emit
  → K=10 gripper-targeted visual perturbation
  → OPEN-duty shaping
  → physical grasp/lift/carry failure
```

### Layer 1: privileged Teacher

Privileged simulator state is used **offline only** to construct clean supervision. The Teacher does not run during deployment. The canonical build is fail-closed on missing fields, attack contamination, ambiguous object/target identity, duplicate trajectories, and split leakage.

### Layer 2: 25D causal Student

The current Student is a lightweight MLP, not the earlier 13D TCN pilot.

Input features:

- 13 direct proprio/action values: gripper command, qpos/opening proxy, EEF pose/velocity, action translation, action gripper;
- 12 causal history features: close/open streaks, flip count, close onset, time since close, EEF speed, post-close lift proxy, qpos/opening deltas and short-window variances.

Model:

- shared `25 → 64 → 64` MLP;
- phase head over 9 phases;
- corridor head;
- release head;
- frozen train-only normalization;
- strict checkpoint feature-order, phase-class, dataset-SHA, and split-mode checks.

Frozen online trigger logic:

```text
IDLE
  → predicted phase == stable_carry and corridor_p > 0.3
ARMED
  → after guard=5, corridor_p > 0.3 and release_p < 0.3
EMITTED
  → one-shot latch
```

The online detector does **not** use RGB, object pose, target pose, normalized episode time, future frames, attack outcomes, or manual trigger anchors.

### Layer 3: frozen attack payload

Current frozen Object payload:

- attack length: `K = 10` frames;
- epsilon: `6/255` in processor pixel space;
- PGD steps: `20`;
- target token: `31744`;
- objective: `autoregressive_prefix_gripper_target_token_logratio_arm_v3`;
- target execution class: `CLIP_MEDIATED_OPEN`;
- arm gate: at least `5/6` arm dimensions preserved;
- strict route, no fallback.

Primary implementation:

- `scripts/stageb/run_v2_vis_sc5_mlp_bridge.py`
- `src/gripper_attack/sc5_detector_runtime.py`
- `src/gripper_attack/sc5_streaming_features_v2.py`
- `src/gripper_attack/attack_adapter.py`
- `src/gripper_attack/route_contract.py`

---

## Detector training and corpus status

Authoritative closeout: `reports/V2_SC5_C16_CLOSEOUT.md`.

Canonical corpus summary:

| Item | Current audited value |
|---|---:|
| Source census | 3,013 episodes |
| Eligible before provenance filters | 688 |
| Attack-contaminated episodes caught | 332 |
| Clean after provenance | 354 |
| Unique after dedup | 314 |
| Included canonical episodes | 142 |
| Canonical rows | 20,438 |
| SC5-valid episodes | 113 |
| Frozen split | 112 train / 28 val / 2 strict held-out episodes |
| Unique tasks represented | 38 |

Known limitation: semantic object/target binding is still only partial for some Tier-B examples; multi-stage Tier-C parents are conservatively routed to abstention rather than treated as regular positives.

Provisional 3-seed replay metrics on the frozen Object corpus:

| Metric | Mean | Gate |
|---|---:|---:|
| Coverage | 0.873 | ≥ 0.80 |
| False-early | 0.025 | ≤ 0.10 |
| Post-release | 0.000 | ≤ 0.05 |
| Median absolute timing error | 2.7 steps | ≤ 8 |
| K10 containment | 0.974 | ≥ 0.85 |
| No-corridor abstention | 0.954 | ≥ 0.90 |

These are **Object-corpus detector results**, not proof of universal cross-suite generalization.

---

## Frozen LIBERO-Object evidence

The current formal Object panel uses 162 preregistered clean-success parents. Lower task success rate (SR) / higher failure rate (FR) means a stronger attack.

| Condition | Evaluation | Success / N | SR | FR | Attack frames | Status |
|---|---|---:|---:|---:|---:|---|
| CLEAN | ITT | 162 / 162 | 1.000 | 0.000 | 0 | FROZEN |
| RAND_T10 | ITT | 162 / 162 | 1.000 | 0.000 | 1,620 | FROZEN |
| RANDOM_TIME_V3 | ITT | 119 / 162 | 0.735 | 0.265 | 1,620 | FROZEN |
| EARLY_SHIFT_T10 | emission-matched | 98 / 141 | 0.695 | 0.305 | 1,410 | FROZEN |
| TRUE_T10 | ITT | 21 / 162 | 0.130 | 0.870 | 1,410 | FROZEN |
| TRUE_T10, emitted only | emission-conditional | 0 / 141 | 0.000 | 1.000 | 1,410 | FROZEN |
| COMMAND_OPEN_ORACLE | emission-matched | 0 / 141 | 0.000 | 1.000 | 1,410 | AUDIT PASS |

Interpretation boundaries:

- `141/162 = 87.0%` of the preregistered panel emitted; all 141 attacked episodes failed.
- The 21 no-emission episodes remain in the ITT denominator and succeeded.
- `RAND_T10` rules out arbitrary matched-budget perturbation as the explanation.
- Random-time and early-shift conditions show strong timing specificity.
- The command-level OPEN oracle shows that sustained OPEN in the same window is **sufficient** to reproduce failure; it does not, by itself, prove complete causal mediation of every visual-attack effect.

### Exploratory adapted-baseline canaries

The following 9-episode results were used to validate routing and scientific contracts. They are **not final full baselines and are no longer the main experiment priority**:

| Exploratory condition | Success / N | SR | Status |
|---|---:|---:|---|
| Adapted TMA-OPEN, Student timing | 1 / 9 | 0.111 | validator pass |
| Adapted TMA-OPEN, V3 random timing | 5 / 9 | 0.556 | validator pass |
| Adapted UMA | 9 / 9 | 1.000 | validator pass |
| Shuffled gradient | 9 / 9 | 1.000 | validator pass |

Do not present these canaries as a completed SOTA comparison.

---

## Cross-suite status

Authoritative reports:

- `reports/CROSS_SUITE_CLEAN300_FINAL_ACCEPTANCE_20260619.md`
- `reports/CROSS_SUITE_GENERALIZATION_PLAN_20260619.md`
- `configs/sc5_cross_suite_protocol_v1.yaml`

Accepted clean corpus outside LIBERO-Object:

| Suite | Planned / valid | Clean success | Clean failure | Raw Object-detector emits* |
|---|---:|---:|---:|---:|
| LIBERO-10 | 100 / 100 | 43 | 57 | 58 |
| LIBERO-Goal | 100 / 100 | 78 | 22 | 3 |
| LIBERO-Spatial | 100 / 100 | 76 | 24 | 3 |
| **Total** | **300 / 300** | **197** | **103** | — |

`*` Raw emit counts are diagnostics only. They are not timing-correctness or generalization evidence.

What CLEAN300 proves:

- planned and discovered denominators reconcile exactly;
- no missing, extra, duplicate-primary, replacement, or schema-invalid episodes;
- required videos, feature arrays, sim-state arrays, and recursive artifact hashes are complete.

What CLEAN300 does **not** prove:

- cross-suite Teacher labels;
- correct detector timing transfer;
- attack-effect transfer;
- VIS superiority over RAND;
- universal gripper-duty vulnerability.

Before cross-suite attacks, the next gate is the offline task/mechanism resolver and Teacher-label audit. Do not launch VIS/RAND merely because a clean rollout contains a raw detector emit.

---

## Current research direction after the advisor meeting

The project is no longer prioritizing full reproductions of TMA, UMA, UADA, UPA, FreezeVLA, or other external attacks.

### Main paper-facing direction

Follow the experimental organization of high-quality VLA security papers, especially BadVLA:

1. use a standardized OpenVLA/LIBERO benchmark layout;
2. report clean performance and attacked performance together;
3. separate the main end-to-end benchmark from timing/payload ablations and mechanism analysis;
4. report explicit denominators, per-task statistics, paired conversions, and confidence intervals;
5. include trajectory / qpos / gripper-command visualizations rather than relying on one pooled SR number.

### Planned paper tables

The current schemas are frozen in `tables/paper_table_schemas_20260619/`:

- **Table 1** — end-to-end attack results across suites;
- **Table 2** — detector localization / transfer;
- **Table 3** — visual → OPEN token/command → qpos/contact → failure mechanism;
- **Table 4** — timing and payload ablations;
- **Table 5** — online latency.

### Main benchmark conditions

The paper-facing full benchmark should prioritize:

- CLEAN;
- matched `RAND_T10`;
- frozen legal random-time payload;
- Student-triggered `TRUE_T10`.

Object-only `EARLY_SHIFT`, `SHUFFLED`, and command-level Oracle remain valuable ablation/mechanism evidence. Adapted TMA/UMA remain archival unless the research plan is explicitly reopened.

### Claim change

The project should not claim “outperforms prior SOTA” under incomparable threat models. The intended contribution is:

> **A clean-only causal trigger exposes a timing-sensitive, gripper-selective OpenVLA vulnerability under a short online intervention budget.**

---

## Non-negotiable protocol rules

For all formal runs:

- keep all preregistered parents in the ITT denominator;
- retain no-emission episodes as explicit dispositions;
- never select examples using attack outcomes;
- never tune the detector or attack after observing formal results;
- never use target-suite normalization in a zero-shot claim;
- never use online privileged state or manual anchors for the Student trigger;
- never overwrite a partial or completed formal output directory;
- validate objective, fallback, PGD steps, epsilon, trigger window, and artifact provenance fail-closed;
- use manual video/contact audit before claiming physical gripper-induced failure;
- distinguish `FROZEN`, `AUDIT PASS`, `VALIDATOR PASS`, `CANARY`, and `PREVIEW`.

---

## Repository map

### Current core

- `src/gripper_attack/v2_privileged_teacher.py` — offline privileged Teacher.
- `src/gripper_attack/sc5_schema_adapter_v2.py` — schema normalization and provenance.
- `src/gripper_attack/sc5_event_segmenter_v2.py` — event/mechanism resolver.
- `src/gripper_attack/sc5_streaming_features_v2.py` — online 25D causal features.
- `src/gripper_attack/sc5_detector_runtime.py` — frozen MLP and one-shot trigger FSM.
- `src/gripper_attack/attack_adapter.py` — OpenVLA visual attack adapters.
- `src/gripper_attack/route_contract.py` — strict runtime attack contracts.
- `scripts/stageb/build_sc5_canonical_corpus_v2.py` — canonical clean corpus build.
- `scripts/stageb/train_sc5_v4.py` — frozen-split Student training.
- `scripts/stageb/run_sc5_canonical_replay.py` — offline replay evaluation.
- `scripts/stageb/run_v2_vis_sc5_mlp_bridge.py` — Object detector-triggered VIS/RAND runner.
- `scripts/stageb/run_sc5_cross_suite_clean.py` — suite-agnostic clean collector.
- `scripts/stageb/run_sc5_cross_suite_clean_queue.py` — manifest-driven clean queue.
- `scripts/stageb/audit_cross_suite_clean_300_postrun.py` — clean-corpus audit.

### Current documentation

- `docs/INDEX.md`
- `reports/V2_SC5_C16_CLOSEOUT.md`
- `reports/CROSS_SUITE_CLEAN300_FINAL_ACCEPTANCE_20260619.md`
- `reports/CROSS_SUITE_GENERALIZATION_PLAN_20260619.md`
- `configs/sc5_cross_suite_protocol_v1.yaml`
- `tables/paper_table_schemas_20260619/`

### Legacy / historical

- `scripts/v4_*`
- old Template-B fixed-window commands;
- `docs/claim_and_evidence.md` black-bowl pilot framing;
- `archive/`;
- historical TMA/UMA/SOTA-chain scripts and canary branches.

These are retained for traceability, not as the current paper protocol.

---

## Installation and CPU checks

```bash
conda env create -f environment.yml
conda activate openvla-gripper-attack
pip install -e .
python -m pytest -q tests/
```

GPU execution is environment- and manifest-specific. Do not copy an old fixed-window command from the git history into a formal run. Start from the current protocol/config/report for the intended phase and use a clean synchronized checkout.

Required local data/model paths are documented in `.env.example` and the relevant protocol YAMLs.

---

## Reference papers for experimental design

These papers motivate experimental organization and metrics; their absolute ASR/FR values are not directly comparable to our online `K=10`, dense `L∞` threat model.

- **BadVLA: Towards Backdoor Attacks on Vision-Language-Action Models via Objective-Decoupled Optimization** — clean/triggered performance, multi-suite table design, ablations, robustness, and trajectory presentation.
- **Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics** — task-level FR plus action-level NAD, per-task variance, targeted and untargeted objectives.

---

## Claim boundaries

Allowed at the current evidence level:

- clean-only privileged-to-causal detector training;
- online trigger without privileged state or manual timing;
- strong LIBERO-Object direction and timing specificity;
- 141/141 emitted-condition failure for the frozen Object panel;
- command-level OPEN sufficiency on the matched emitted panel;
- accepted and integrity-audited cross-suite CLEAN300 corpus;
- cross-state / cross-object evidence inside the frozen Object evaluation.

Not yet allowed:

- universal cross-suite detector generalization;
- universal VLA attack;
- cross-model or real-robot generalization;
- complete causal mediation exclusively through the gripper channel;
- direct SOTA superiority over patch/backdoor papers with different threat models;
- cross-suite attack effectiveness before Teacher/mechanism eligibility and paired attack audits.

---

## License

MIT License. See `LICENSE`.
