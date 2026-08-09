# Codex Teacher–Student Detector Completion Plan V1

Status: ACTIVE_CONTROL_PLANE  
Owner: Codex  
Base commit: `7550bf0ae5647df6afb062450b64ef4a3ff1bb9a`  
Branch: `codex/detector-completion-20260726`

## Objective

Complete and reproducibly seal the five-head causal Teacher–Student detector:

1. `physical_criticality`
2. `k10_feasible`
3. `safe_release`
4. `instability`
5. `gripper_closing_state`

The critical path is:

`H0.3-R6 → C3-S3 → C3-G → C3-T0 → C3-T → T2R-D → L23-800 → S0/S1/S2 → I0 → R0`.

No stage may consume artifacts from a failed or superseded stage.

## Universal invariants

- A run is evidence only when launched from a clean detached worktree at a full Git commit SHA.
- Every artifact manifest binds source commit/tree, source-file hashes, environment/container digest, input hashes, exact command, seeds, start/end UTC, exit code, stdout/stderr, hardware/software versions, and final output hashes.
- Receipts never claim a timestamp later than the commit containing them. A receipt that binds an artifact commit is generated in a later transition commit.
- Scientific retries are forbidden. Retry only predeclared infrastructure-invalid runs, retaining the original and using the same config/seed.
- Protected CAL/G10/T2R-D data remain unreadable until the corresponding one-time gate explicitly authorizes access.
- Unknown labels remain unknown; they are never coerced to negative.
- Teacher physical heads cannot consume task success, terminal outcome, attack condition/name, future frames, or Student outputs.
- Student inputs are deployment-available causal streams only. Privileged MuJoCo geometry, Teacher internals, outcome fields, condition IDs, and future frames are forbidden.
- Simulator success rate is diagnostic; frozen manual/event adjudication is authoritative for physical attack outcomes.
- Corrected VIS uses raw gripper OPEN sign `+1.0`. Historical Black-Bowl v1 results remain quarantined.
- Moka is reported as a separate extension stratum unless a preregistered promotion gate is passed.

## Resource policy

- Server ceiling: all 8 A800 GPUs and available multicore CPU/RAM may be used.
- Geometry, Teacher relabeling, and data QA are parallelized by episode/task where deterministic.
- Student search uses the 8 GPUs as independent seeded trials by default; DDP is used only after NCCL/DDP equivalence and throughput smoke tests.
- Before a full batch: one-task CPU smoke, one-GPU smoke, then multi-GPU smoke.
- Automatically stop on NaN/Inf, checksum mismatch, dirty source, schema mismatch, data leakage, non-causal access, or infrastructure-invalid rate above 2%.

## Gates

### H0.3-R6 — Provenance and resolver closure

Purpose: replace the non-consumable R5 evidence package with a role-safe, independently verifiable C1-V2 registry.

Required changes:

- Pass an independent semantic role from parsed BDDL into the resolver: `MANIPULATED_OBJECT`, `OBJECT_TARGET`, or `REGION_TARGET`.
- `REGION_TARGET` may resolve only to `EXACT_SITE`.
- Object roles may resolve only to body/geom or an approved body alias; an exact site is forbidden.
- Alias verification traverses the body ancestor chain. All matched geoms must descend from the accepted root; direct-parent equality is not the definition of hierarchy.
- Add production-path tests for missing region site with same-named body/geom, object/site collision, nested child body, ledger generation, per-task status, totals, and fail-closed exit.
- Run C1 twice into immutable `run_A` and `run_B` directories from the same clean source commit.
- Commit the summaries, alias ledgers, canonical payloads, manifests, logs, and external `SHA256SUMS`; do not submit server paths alone.

Pass:

- 40/40 environments created.
- For supported relations, object and target unresolved/ambiguous/forbidden counts are all zero.
- Run A/B canonical relation payload and alias ledger hashes are identical.
- All contract and integration tests: fail=0, error=0, critical skip=0.
- All self-hashes, external hashes, Git blobs, chronology, and source bindings independently recompute.

Failure: H0.3 remains `HOLD`; every downstream stage stays locked.

### C3-S3 — Geometry observability seal

Purpose: establish which target/support geometry is reconstructable without using outcome labels.

Required evidence:

- Consume only the sealed H0.3-R6 registry.
- Declare quaternion order and coordinate frame for each MuJoCo/model/sidecar source.
- Unit-test identity and known 90-degree rotations plus composition/inversion.
- Classify each fixture/support as static, dynamic reconstructable, articulated unknown, or excluded by preregistered taxonomy.
- Cover every development and eventual 130-episode confirmation mapping row; publish a per-episode observability table.
- Basket and other dynamic supports use all relevant canonical states, not a five-sample spot check.

Pass:

- Mapping completeness 100%.
- Static reconstruction maximum position error ≤1e-6 m and rotation error ≤1e-6 rad.
- Dynamic replay reconstruction p99 position error ≤1e-4 m and rotation error ≤1e-3 rad.
- Zero silent fallback and zero unknown→negative conversion.
- Two deterministic rebuilds have identical canonical hashes.

Failure: stop before C3-G.

### C3-G — Placement geometry evaluator

Purpose: implement causal `In`, `On`, and `Stack` evaluation for fixture and dynamic supports.

Required tests:

- Analytic fixtures for boundary, penetration, contact, height, orientation, and stacking order.
- Development replay tests for fixture, dynamic support, articulated unknown, pre-grasp negatives, release, recovery, and missing telemetry.
- No task success/terminal outcome is an evaluator input.
- Unknown is fail-closed and separately counted.

Pass on frozen non-protected development data:

- Placement episode recall ≥0.90.
- Pre-grasp false-positive rate ≤0.05.
- False-positive rate after failed placement ≤0.05.
- Each supported suite/task stratum recall ≥0.80 or is explicitly underpowered and cannot be pooled to hide failure.
- Evaluator output is deterministic and schema-complete; critical tests all pass.

Failure: one versioned geometry remediation cycle is allowed; no threshold tuning on protected data.

### C3-T0 — Teacher causality

Purpose: remove terminal/success leakage before V23 integration.

Pass:

- Perturbing `task_success`, terminal flag, episode length metadata, condition name, or post-event fields does not change any physical label at the same causal prefix.
- Online streaming and offline prefix outputs match for RF32 and Dual-TCN adapters across empty/short/full histories and left padding.
- Future-frame perturbations never change earlier outputs.
- Test fail=0, error=0, critical skip=0.

Failure: C3-T remains locked.

### C3-T — V23 Teacher freeze

Purpose: integrate the sealed evaluator into one immutable Teacher.

Required outputs:

- Frozen label semantics, unknown policy, persistence/K10 rules, thresholds, schema, source hashes, and reason codes.
- Five heads emitted once per timestep with explicit `value`, `valid_mask`, `reason`, and `confidence`.
- Development audits by task, suite, phase, and reason code.
- Deterministic relabel hash equality on a stratified replay.

Pass:

- Schema and length completeness 100%.
- NaN/Inf=0; identity mismatch=0; unknown→negative=0.
- Placement recall ≥0.90 and pre-grasp FPR ≤0.05 on the frozen development cohort.
- Physical criticality/instability invariance and causal-prefix suites all pass.
- No final-head positive class is absent from the training-eligible development pool; otherwise the head is formally declared non-trainable and the detector scope is revised before unblinding.

Failure: do not access T2R-D.

### T2R-D — One-time Teacher confirmation

Dataset contract: 130 frozen episodes; 13 preregistered articulated/non-placement exclusions; supported placement denominator 117.

Before access, freeze V23 source, container, thresholds, denominators, analysis script, and manifest.

Pass:

- Supported placement recall ≥0.90.
- Pre-grasp false-positive rate ≤0.05.
- No supported suite recall <0.80.
- Unknowns are reported separately and never counted as negatives.
- Zero provenance, schema, or identity violation.

No tuning is permitted after unblinding. Failure produces a preserved failed confirmation result and requires a new Teacher version plus a new untouched confirmation set.

### L23-800 — Full Teacher relabel and seal

Relabel all 800 historical episodes; do not mix V22 and V23 labels.

Pass:

- Exactly 800 source episodes inventoried and hashed.
- Exactly one aligned five-head record per valid timestep.
- Source/label identity, schema, and Merkle checks all pass.
- Two full relabel runs, or one full run plus an independently recomputed stratified 10% audit, have identical canonical labels.
- Every invalid/abstained episode is enumerated with a non-outcome-based reason.
- Publish per-head positive/negative/unknown steps and episodes by train/dev/test grouping.

Split before Student training, grouped by task, initial state/seed, and paired condition so leakage across groups is impossible. Test/held-out groups are sealed before model search.

### S0 — Student data and leakage audit

Pass:

- Feature contract contains only approved causal deployment inputs.
- Normalization/statistics are fitted on train only.
- No duplicate or paired episode crosses a split.
- Label masks and rare-class samplers preserve unknown semantics.
- MLP, RF32, and causal Dual-TCN baselines share identical frozen splits and metrics.

### S1 — Student training and model selection

Compute policy:

- Smoke first; then at most 12 one-seed candidates in parallel across 8 A800s.
- Promote at most four candidates to full data and at least three fixed seeds.
- Primary candidate family is causal Dual-TCN; MLP and RF32 are mandatory comparators.
- Thresholds/calibration are fitted on selection-dev only.
- Selection is lexicographic: satisfy safety/causality/latency constraints, then maximize macro known-mask AUPRC; within one standard error choose the smaller/faster model.

Development pass:

- `physical_criticality`: recall ≥0.90 at frame FPR ≤0.05 and episode alarm FPR ≤0.05.
- `safe_release`: recall ≥0.85 and precision ≥0.80 on known labels.
- Other trainable heads: AUROC ≥0.90 and AUPRC ≥0.75 unless a stronger preregistered per-head baseline applies.
- Macro known-mask AUPRC improves over MLP by ≥0.10 and over RF32 by ≥0.03; episode-cluster 95% lower bound for the primary improvement is >0.
- No primary suite recall <0.80; three-seed primary recall SD ≤0.03.
- Streaming/offline parity exact within declared numerical tolerance.
- p95 latency ≤20 ms and ≤10% of the control period; deployment memory has ≥20% headroom.

If no candidate passes, no detector is shipped.

### S2 — Untouched Student evaluation

Use an independent grouped held-out set never used for Teacher/Student tuning.

Pass:

- Critical-window recall ≥0.90 with 95% CI reported.
- Episode alarm FPR ≤0.05 with 95% CI reported.
- Median lead time ≥3 steps and p10 ≥1 step.
- No primary stratum recall <0.80; worst primary-stratum gap ≤0.15.
- Calibration ECE ≤0.05.
- All five heads and unknown masks are reported; no aggregate may hide a failed head.

After evaluation, the held-out set is considered unblinded and cannot be reused to tune the same claim.

### I0 — Attack/runtime integration

First run detector in shadow mode.

Pass:

- Detector-enabled and disabled deterministic replays have identical actions and episode termination in shadow mode.
- Log completeness ≥99.9%; detector crash rate <0.1%.
- p95 wall-clock overhead ≤5% and stays inside the control deadline.
- Corrected Black-Bowl State5/State7 are primary reports; Moka is separate.
- If active mitigation is added, it requires a separate causal gate: unsafe-event reduction ≥50%, clean success degradation ≤2 percentage points, and unintended clean-step intervention ≤2%.

### R0 — Final reproducibility seal

An independent clean checkout must reproduce:

- H0.3 registry and hashes.
- V23 Teacher confirmation and 800 labels.
- Frozen splits, Student evaluation, export, and integration report.
- Three-seed training within ±2 percentage points on primary metrics and the same selected model family.

Release only with signed manifest, dependency/container lock, commands/configs/seeds, GPU/driver/CUDA inventory, raw logs, exclusions/retries, annotation/evaluation guide, weights, calibration, schema, and known limitations.

## Runtime state at plan freeze

- `7550bf0` resolver improvements: accepted as development progress.
- `H0.3-R5`: HOLD pending R6; not consumable.
- `C3-S3` and later: BLOCKED.
- Protected reads: NOT AUTHORIZED.
- Training/inference/attack: NOT AUTHORIZED until their explicit gates open.
