# Top-Level Clutter Audit V1

Status: `CODE_R0_REPOSITORY_INVENTORY_PASS`

Source HEAD: `544307997f7026e5b7b04e1400bbc675a3d63e2e`

This audit is classification-only. It authorizes no move, deletion, or byte rewrite.

## Method

- Enumerated all 1,770 tracked files and confirmed zero untracked files.
- Classified every top-level area and all 15 tracked root files.
- Searched the current tracked tree for each root filename, full file SHA-256, and Git blob ID.
- Read current Paper V1, F1T, and Stage Z authority/status artifacts before assigning scientific roles.
- Treated absence of a current text reference as insufficient proof of safe deletion.

## Root files

| Path | Lifecycle | Current references / role | Move | Delete | Recommendation |
|---|---|---|---|---|---|
| `.env.example` | `ACTIVE_RUNTIME` | Linked by root README; local path template | no | no | Keep at conventional root path |
| `.gitignore` | `ACTIVE_RUNTIME` | Repository hygiene; historical audit references | no | no | Keep |
| `CONTRIBUTING.md` | `ACTIVE_ANALYSIS` | Linked by `docs/INDEX.md`; active governance | no | no | Keep and update only prospectively |
| `environment.yml` | `ACTIVE_RUNTIME` | Linked by README and reproducibility docs | no | no | Keep at conventional root path |
| `pyproject.toml` | `ACTIVE_RUNTIME` | Package/test configuration; loads `src` and `scripts` for pytest | no | no | Keep |
| `README.md` | `ACTIVE_ANALYSIS` | Package metadata and primary navigation; scientifically stale | no | no | Rewrite navigation/status in R3; do not treat as authority |
| `LICENSE` | `ACTIVE_ANALYSIS` | Linked by README/release checklist | no | no | Keep |
| `RELEASE_CHECKLIST_20260510.md` | `HISTORICAL_VALID_EVIDENCE` | Historical public-artifact release receipt; no current pathname reference | no | no | Keep and label historical through navigation |
| `FINAL_DETECTOR_COMPLETION_REPORT.md` | `HISTORICAL_VALID_EVIDENCE` | Referenced by `scripts/detector_v5/run_final_detector_pipeline.py`; contains pre-execution boundaries | no | no | Keep in place; index as historical, not current detector status |
| `N5_LONGRUN_PLAN_V2.json` | `HISTORICAL_INVALID_OR_SUPERSEDED` | V3 predecessor; no current pathname/SHA/blob reference found | no | no | Preserve as superseded plan history |
| `N5_LONGRUN_PLAN_V3.json` | `HISTORICAL_INVALID_OR_SUPERSEDED` | V4 predecessor; explicitly preserved by N5 receipts | no | no | Preserve exact path/bytes |
| `N5_LONGRUN_PLAN_V4.json` | `HISTORICAL_INVALID_OR_SUPERSEDED` | Explicitly preserved by N5 H0/H0.2 records; downstream state is HOLD/provenance-fail | no | no | Preserve exact path/bytes; do not infer current authorization from its internal `RUNNING` field |
| `tmp_a1_target_tests.py` | `DEPRECATE_AFTER_COMPAT_AUDIT` | Ad-hoc action-semantics self-check; no current pathname/SHA/blob reference found | unknown | unknown | Index only; R1/R2 must clear exact-source/import/history use before any deprecation action |
| `tmp_d212_verify.py` | `DEPRECATE_AFTER_COMPAT_AUDIT` | Server-oriented D2.1.2 verification script; no current pathname/SHA/blob reference found | unknown | unknown | Index only; contains historical absolute import setup, so retain for provenance pending audit |
| `tmp_official_eval.py` | `DEPRECATE_AFTER_COMPAT_AUDIT` | Self-contained historical OpenVLA/LIBERO evaluator; no current pathname/SHA/blob reference found | unknown | unknown | Index only; never execute in this lane |

The three `tmp_*.py` files are the only plausible root cleanup candidates. R0 deliberately does nothing to them. A current-tree text search cannot prove that historical commands, sealed source identities, or Git-history references do not depend on their paths or bytes.

## Apparent directory clutter

| Surface | Finding | R0 decision |
|---|---|---|
| `reports/` (402 files) | High cardinality is expected: positive, negative, HOLD, invalid, superseded, manifests, and root seals are all part of reproducibility | Preserve; index by stage/status instead of relocating |
| `configs/` (225 files) | Mixes runtime configs with path/SHA-bound protocol freezes and sidecars | Preserve; R1 registry decides immutable subset |
| `docs/handoffs/` (83 files) | Chronological handoffs include supersessions and failures that must remain recoverable | Preserve all versions; add navigation only |
| `scripts/detector_v5/` (239 files) | Dense mixed history, but tests, CI, reports, and exact-source bindings make bulk cleanup unsafe | Document active/compatibility surfaces; no bulk move |
| `scripts/stage_x/` (64 files) | Includes execution history and current audit/paper builders, some sealed by F1T | Preserve paths; identify canonical audit-only entry points in R2 |
| `n5/` (136 files) | Contains internally superseded/HOLD material plus tests and receipts | Preserve history and label status; never collapse HOLD into PASS |
| `tables/` (17 files) | Historical clean-only tables and placeholder paper schemas are not current Paper V1 authority | Keep as historical; do not populate or promote by cleanup |
| `archive/` (10 files) | Already curated and explained | Keep as-is |

## No-delete decision

No tracked file is classified `GENERATED_NONAUTHORITY` with enough evidence for deletion. R0 therefore approves no move or deletion. The compatibility-safe path is indexing, lifecycle labeling, and later deprecation notices where R1/R2 prove they are safe.

## Pre-existing authority mismatch

The current Stage Z runner-preparation static audit points to a missing
`src/stage_z_preparation/__init__.py` (1,760 bytes; expected SHA-256
`987b755ad6cb613943a9341d160bbcf5faffefa7fd6a913513c2302d61a69cd3`).
The path is absent from current HEAD and from commit `af9f09c`, which introduced
the package and audit. This is not clutter and is not repaired by cleanup.
It is a fail-closed R1 authority-firewall issue.

## Gate result

- every top-level area: classified or explicitly `UNKNOWN_AUTHORITY`
- every tracked root file: classified
- required detector, FEC, pilot, Stage X, Stage Z, `n5`, and `v4` families: inventoried in the machine ledger
- sealed bytes/paths changed: 0
- moves/deletions: 0
- protected/scientific execution: 0

Result: `CODE_R0_REPOSITORY_INVENTORY_PASS`
