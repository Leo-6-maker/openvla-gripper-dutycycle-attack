# Repository Hygiene + Paper-Support Execution Plan

Date: 2026-08-24

Status: `PI_REPOSITORY_HYGIENE_PLAN_READY_FOR_CODEX`

This plan authorizes **repository organization, static refactoring, documentation, test/CI hygiene, and paper-support export tooling only**. It does not authorize any new scientific experiment, model inference, GPU execution, simulator creation, `env.step`, V_phys read, Eval160/protected read, or reinterpretation of sealed evidence.

The starting point is the scientific working branch/PR #135 HEAD:

- commit: `7f095965df61e8065d5c76fc2be4504bed4d9ab6`
- Stage Z remains before Z1; the current Z0R2 authority root remains a HOLD until its separately authorized blockers close.

The purpose of this cleanup is not to rewrite project history. The purpose is to make the repository understandable, reproducible, paper-friendly, and safe to maintain without breaking existing evidence authority.

---

# 0. Non-negotiable authority firewall

The repository contains many files whose **path, bytes, SHA256, Git blob, or ancestry are themselves scientific/provenance authority**. Therefore normal software-cleanup rules do not apply blindly.

## 0.1 Never move/delete/rewrite without an authority check

Treat the following as presumptively immutable or path-sensitive until proven otherwise:

- `reports/**`
- `configs/**` that are referenced by a report, root seal, handoff, claim ledger, protocol, or paper authority map
- `docs/handoffs/**`
- `paper/PAPER_V1_*`
- Paper V1 figures/tables/data and final manifests/root seals
- F1/F1T sealed evidence
- Stage X / Stage Z root seals, manifests, ledgers, population freezes, runtime authority maps, protocol freezes
- historical invalid/superseded artifacts explicitly retained for provenance
- any file referenced by exact path/SHA/Git blob in a current scientific handoff or paper claim ledger

For those files, cleanup means **indexing and documenting them**, not relocating them.

## 0.2 No history rewriting

Do not squash/rebase historical scientific PRs, rewrite old artifacts in place, or replace old negative/HOLD results with cleaner new versions.

If a document is stale, add a supersession/index note rather than changing the historical evidence it describes.

## 0.3 No cleanup-driven experiment

Repository organization must not create pressure to rerun anything. If a file can only be classified by running a model/simulator or reading protected outcomes, classify it `UNKNOWN_AUTHORITY` and stop that local action.

---

# 1. Repository cleanup north star

By the end of this work window, a new reader should be able to answer quickly:

1. What is the current scientific story?
2. Which evidence is promotable, negative, diagnostic, engineering-only, invalid/superseded, or pending?
3. Which code is active for current reproducibility?
4. Which scripts are historical/legacy and should not be used for new runs?
5. Which configs are active, immutable evidence, or historical?
6. Which tests/CI jobs protect which code paths?
7. How do I deterministically export paper-ready data from sealed evidence without hand-copying numbers?
8. Which files must never be moved because their path/hash is part of provenance?

The cleanup should optimize for **scientific traceability first, software elegance second**.

---

# 2. Classification vocabulary

Every major file/directory or script family should be assigned one of the following lifecycle classes:

- `ACTIVE_RUNTIME` — code that may be used by a currently authorized or future explicitly authorized execution path.
- `ACTIVE_ANALYSIS` — deterministic analysis/audit code used to read sealed artifacts.
- `ACTIVE_PAPER_EXPORT` — code whose only job is to transform sealed evidence into paper-ready derived data/figures/tables.
- `SCIENTIFIC_AUTHORITY_IMMUTABLE` — artifact/path/bytes must be preserved.
- `HISTORICAL_VALID_EVIDENCE` — valid past evidence, not current runtime.
- `HISTORICAL_INVALID_OR_SUPERSEDED` — preserved only for provenance; must not be used for promotion.
- `LEGACY_CODE_COMPAT` — old code retained only because history/tests/imports depend on it.
- `DEPRECATE_AFTER_COMPAT_AUDIT` — not authority, not needed by active code, but removal requires dependency checks.
- `GENERATED_NONAUTHORITY` — generated files that may be removed/regenerated after verifying they are not evidence.
- `UNKNOWN_AUTHORITY` — fail-closed until resolved.

Do not collapse `HISTORICAL_INVALID_OR_SUPERSEDED` into deletion. Historical failure is part of this project's reproducibility record.

---

# 3. Autonomous cleanup state machine

Codex may progress autonomously through `CODE-R0` → `CODE-R6` if each gate passes. Do not request PI approval between ordinary phases.

Statuses:

- `CODE_R0_REPOSITORY_INVENTORY_PASS`
- `CODE_R1_AUTHORITY_FIREWALL_PASS`
- `CODE_R2_ACTIVE_CODE_SURFACE_PASS`
- `CODE_R3_COMPATIBILITY_SAFE_HYGIENE_PASS`
- `CODE_R4_TEST_CI_DOCUMENTATION_PASS`
- `CODE_R5_PAPER_EXPORT_PIPELINE_PASS`
- `CODE_R6_REPOSITORY_STATIC_AUDIT_PASS`
- terminal: `REPOSITORY_HYGIENE_PAPER_SUPPORT_READY_FOR_PI_REVIEW`

Stop only for the mandatory-stop conditions in Section 11.

---

# 4. CODE-R0 — repository inventory and topology map

## Goal

Build a machine-readable and human-readable map before moving or deleting anything.

## Required outputs

Create:

- `docs/repository/REPOSITORY_MAP_V1.md`
- `docs/repository/REPOSITORY_LIFECYCLE_LEDGER_V1.csv` or `.json`
- `docs/repository/TOP_LEVEL_CLUTTER_AUDIT_V1.md`

## Required inventory

At minimum inventory:

- root-level files
- `.github/workflows/`
- `analysis/`
- `configs/`
- `docs/`
- `docs/handoffs/`
- `paper/`
- `reports/`
- `scripts/`
- `src/`
- `tests/`
- any `n5/`, `v4/`, detector, FEC, pilot, Stage X, or Stage Z families present

For each top-level area record:

- purpose
- lifecycle class
- current scientific role
- current runtime role
- referenced by active tests? yes/no/unknown
- referenced by active report/claim ledger/root seal? yes/no/unknown
- safe to move? yes/no/unknown
- safe to delete? yes/no/unknown
- recommended action

## Special root audit

The root currently contains historical items such as detector completion reports and multiple `N5_LONGRUN_PLAN_V*.json` files. Do not immediately move/delete them. First determine whether they are referenced by current docs/reports/tests. If not current, classify and expose them through an index rather than hiding their history.

## R0 PASS

R0 passes when every major directory and every root-level nontrivial file is classified or explicitly marked `UNKNOWN_AUTHORITY`.

---

# 5. CODE-R1 — authority firewall and immutable-path registry

## Goal

Prevent repository cleanup from corrupting paper provenance.

## Required outputs

Create:

- `docs/repository/IMMUTABLE_AUTHORITY_PATHS_V1.md`
- `docs/repository/IMMUTABLE_AUTHORITY_PATHS_V1.json`
- `scripts/repository/audit_immutable_authority_paths.py`

## Registry construction

Build the registry by scanning current:

- Paper V1 authority map / claim ledger
- Paper V2/F1 delta and F1T authority map
- Stage X/X1R2 handoffs and root seals
- Stage Z root seals/manifests
- root seal artifact manifests
- SHA256 manifests
- docs/handoffs that bind exact artifact paths

Registry rows should include:

- path
- authority type
- referencing artifact(s)
- path-sensitive? yes/no
- byte-sensitive? yes/no
- immutable-by-governance? yes/no
- current SHA/blob if cheap to compute statically

## CI/static guard

Add a lightweight static audit that fails if a cleanup PR:

- deletes a registered immutable path;
- renames/moves it without an explicit approved compatibility mapping;
- rewrites Paper V1 immutable files;
- deletes a root-seal sidecar while keeping the root;
- leaves a manifest pointing to a missing file.

Do not make this guard so broad that normal source-code work becomes impossible.

## R1 PASS

The repository now has an explicit machine-readable firewall protecting evidence paths.

---

# 6. CODE-R2 — active code surface and reproducibility entry points

## Goal

Identify the small set of code a future researcher should actually read first.

## Required outputs

Create:

- `docs/repository/ACTIVE_CODE_SURFACE_V1.md`
- `docs/repository/REPRODUCIBILITY_ENTRYPOINTS_V1.md`
- README files for major active script families if absent

## Required active surfaces

Audit and document at minimum:

### Core library

- `src/gripper_attack/**`

Classify modules into:

- current execution primitives
- current audit/route contracts
- compatibility-only legacy modules
- unused/uncertain modules

### Stage X

- `scripts/stage_x/**`
- associated Stage X configs/tests

Separate:

- Q3/Q3R* engineering history
- E3/E4 structural realizability
- F1 development/execution qualification
- active paper-analysis builders

### Stage Z

- `scripts/stage_z/**`
- Stage Z protocol/config/runtime adapters/tests

Explicitly label Stage Z runner preparation as **not scientific evidence** and preserve execution-disable guards.

### Paper-support tooling

- `scripts/paper/**`
- current Paper V1 builders/audits

Plan a new Paper V2 export surface in R5 instead of mutating immutable V1 builders.

## Entry-point rule

A new user should not have to guess among dozens of historical scripts. Document canonical current commands for:

- static paper evidence audit
- Paper V1 claim audit
- Paper V2 evidence export (after R5)
- Stage X reproducibility/audit-only paths
- Stage Z static authority checks

Do not authorize scientific execution by documenting a command. Mark execution commands as requiring separate PI authorization.

## R2 PASS

Every active/recommended script family has one obvious documented entry point, and legacy alternatives are clearly labeled.

---

# 7. CODE-R3 — compatibility-safe hygiene and deprecation

## Goal

Reduce confusion without breaking provenance or imports.

## Allowed actions

- add package/module docstrings
- normalize naming in newly added files
- add `README.md` files to script/config/test families
- add deprecation notices
- add compatibility wrappers when moving a non-authority source file is clearly beneficial
- remove genuinely generated/non-authority trash after reference audit
- remove duplicate source files only when byte/semantic equivalence and dependency checks are explicit
- update stale root README navigation

## Moving source code

A source file may be moved only if ALL hold:

1. not listed in immutable authority registry;
2. not used as exact source-byte authority by a sealed experiment;
3. all imports/tests/config references are found;
4. compatibility wrapper or import shim is added if historical commands may still rely on old path;
5. tests pass before and after;
6. commit message records old→new mapping.

If any source file was the exact runtime source for a sealed experiment, **do not rewrite that historical byte identity**. A new prospective module can wrap or supersede it, but historical source must remain recoverable.

## Root README rewrite

Update root `README.md` to include:

- current scientific status
- evidence hierarchy
- repository navigation
- current code surfaces
- historical/provenance warning
- paper repository link
- Stage Z current status as pending/HOLD, not result
- protected boundary statement

Do not use README as a new scientific authority; link to sealed artifacts.

## R3 PASS

The repo is easier to navigate while all authority paths and historical reproducibility remain intact.

---

# 8. CODE-R4 — tests and CI map

## Goal

Make CI understandable and reduce accidental coverage gaps.

## Required outputs

Create:

- `docs/repository/TEST_CI_MATRIX_V1.md`

For every `.github/workflows/*.yml`, record:

- workflow purpose
- active source families covered
- test commands
- whether it is historical compatibility or current required CI
- expected runtime/environment
- whether GPU/model/simulator is involved (paper/cleanup CI must be CPU-only)

## Test taxonomy

Organize/document tests by current scientific/software surface:

- core source contract tests
- Stage X engineering/audit tests
- Stage Z static/adapter tests
- paper-analysis/export tests
- legacy compatibility tests

Do not delete a workflow merely because it looks old. First prove no current/legacy protected contract depends on it.

## Cleanup CI

Add a CPU-only repository-hygiene job if useful that checks:

- immutable authority path audit
- JSON parse for new index/ledger files
- Python compile for new repository/paper-support scripts
- targeted unit tests
- `git diff --check`

Do not make this job execute models/simulators.

## R4 PASS

A reader can explain what each CI job protects and no cleanup change silently removes scientific guardrails.

---

# 9. CODE-R5 — deterministic Paper V2 evidence export pipeline

## Goal

Create a clean bridge from the scientific authority repository to the Overleaf-linked manuscript repository, with no hand-copied scientific numbers.

## Architecture

The **main code repository owns scientific source-of-truth and data extraction**.

The **paper repository owns presentation, LaTeX, plotting style, captions, and manuscript layout**.

Do not make the Overleaf/paper repository the primary authority for scientific numbers.

## Required new tooling

Create a new prospective namespace, for example:

- `scripts/paper_v2/`
- `exports/paper_v2/` or `paper/PAPER_V2_EXPORTS/`

Exact naming may be chosen after R0/R2, but do not modify immutable `PAPER_V1_*` bundles.

Provide deterministic exporters for at least:

### Export A — evidence hierarchy

Rows: X0, VI-B2, VII, VIII, IX, E3/E4, F1-B, F1-C4/F1T, Stage Z pending.

Fields:

- stage
- question
- evidence class
- primary unit
- denominator/censoring
- environment exposure
- status
- promotable wording key
- authority path
- authority digest/binding

### Export B — X0 mechanism

Fields needed for:

- T3/T5/T10 source-defined denominators
- raw positive rates
- complete monotone pattern counts
- mechanism-consistent telemetry summaries if already available in sealed paper data

Do not invent iid uncertainty. If bootstrap data are available and intended for plotting, export parent-bootstrap quantities explicitly labeled.

### Export C — timing/generalization cascade

Fields needed for VI-B2, VII, VIII summary.

Preserve censored/non-identifiable cells.

### Export D — Stage IX factorization gap

E0/E1/E3 model-side AUROC and factorized parent-macro AUC plus any selected frozen top-k/LOSO values used in the paper.

### Export E — E3/E4/F1 execution-layer evidence

Keep populations separate:

- E3 12-parent strict single-state realizability
- E4 parent/candidate diagnostic decomposition
- F1-B 24-parent DEV method-development summary
- F1-C4 8-parent fresh executable-qualification summary

Never emit one pooled "funnel success rate".

## Export manifest

Every export bundle must include a machine-readable manifest with:

- source repository HEAD used for export
- exact authority inputs and digests
- generated file paths
- generated file SHA256
- generator script SHA256/blob
- timestamp
- statement that Eval160/protected was not read

## Paper-repo consumption

The paper repository may copy the generated CSV/JSON/TeX data products or reproduce them from the manifest, but any presentation file must retain source metadata in an internal asset manifest.

## R5 PASS

All core paper quantitative content can be regenerated without manually transcribing numbers from prose.

---

# 10. CODE-R6 — final repository static audit

Create:

`docs/repository/REPOSITORY_HYGIENE_FINAL_AUDIT_V1.md`

Verify:

## Authority integrity

- immutable-path registry has no missing files
- all touched authority-adjacent manifests still resolve
- Paper V1 immutable bundle is byte-unchanged
- Stage X/F1/Stage Z sealed evidence touched only if explicitly required by an already authorized scientific gate, not by cleanup

## Software integrity

- relevant CPU CI passes
- targeted tests pass
- Python compile passes for new scripts
- JSON/CSV schemas parse
- no broken imports from compatibility-safe moves

## Documentation integrity

- root README points to current scientific status
- active code surface documented
- historical/legacy status explicit
- paper repo link and export workflow documented

## Paper-export integrity

- export bundle reproducible
- source digests recorded
- no Stage Z result exported unless separately promoted
- no protected reads

Terminal status:

`REPOSITORY_HYGIENE_PAPER_SUPPORT_READY_FOR_PI_REVIEW`

Then STOP. Do not merge automatically.

---

# 11. Mandatory early-stop conditions

Return to PI immediately if:

1. a proposed move/delete touches a path referenced by a root seal/claim ledger/handoff and no compatibility-safe strategy exists;
2. two artifacts disagree on whether a file is current authority;
3. cleanup would require modifying historical sealed bytes;
4. tests reveal that an apparently legacy module is still load-bearing for current scientific execution;
5. a move would change an exact runtime-source identity used by a sealed experiment;
6. paper export requires reading protected/Eval160 data;
7. paper export cannot reproduce a central number from sealed authority;
8. a cleanup change would silently change a scientific denominator, protocol, attack semantics, tokenizer/action authority, OPEN sign, branch estimand, or Stage Z population;
9. a new experiment would be needed to determine whether cleanup is safe;
10. cleanup becomes entangled with reopening F1/F1-D/BRIDGE_V3 or Stage Z Z1 without PI authorization.

Do not stop for ordinary documentation work, adding indices, local static analysis, unit tests, compatibility wrappers, or non-scientific generated-file cleanup.

---

# 12. Explicitly authorized actions

Codex may autonomously:

- inspect the Git tree and history;
- search imports/references;
- add repository maps and lifecycle ledgers;
- add README/index files;
- add static audit scripts;
- add compatibility-safe wrappers;
- improve comments/docstrings;
- add deterministic paper-export scripts;
- add CPU-only tests/CI for cleanup/export tooling;
- remove non-authority generated or duplicate files after documented dependency checks;
- update the root README and contributor navigation;
- commit incremental R0–R6 work on this cleanup branch.

Codex may NOT:

- run models/GPU/simulator/attacks;
- call `env.step`;
- read V_phys/Eval160/protected;
- rerun consumed scientific gates;
- alter scientific protocol semantics;
- weaken fail-closed execution;
- rewrite historical scientific artifacts;
- merge this PR automatically.

---

# 13. Coordination with the paper repository

The paper repository is:

`Leo-6-maker/stage_aware_attack_on_vision_language_model`

Its PR #2 owns the manuscript rewrite and Overleaf-facing assets.

Coordinate as follows:

1. main repo exports evidence + provenance;
2. paper repo imports/consumes exports;
3. paper repo generates final plots/tables/LaTeX presentation;
4. both sides record source digests;
5. scientific wording remains governed by the paper claim ledger;
6. code cleanup must never force a scientific claim rewrite.

The linked Overleaf project is a presentation/synchronization surface, not a new evidence authority.

---

# 14. One-sentence north star

> Make the repository easier to understand and reproduce without changing what the experiments actually were, what they proved, what they failed to prove, or where their authority lives.
