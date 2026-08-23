# OpenVLA gripper duty-cycle research artifact

This repository contains the scientific source, sealed evidence, provenance,
contracts, and reproducibility tooling for the OpenVLA/LIBERO gripper
duty-cycle project. The current result is mechanism-first and
factorization-aware; it is not a claim of a universal visual attack or a
universal detector.

This README is navigation, not scientific authority. Resolve claims and status
from the linked authority maps, claim ledgers, manifests, root seals, and
handoffs.

## Current scientific status

- Paper V1 is a sealed mechanism/factorization draft bundle. Its source map is
  [`paper/PAPER_V1_EVIDENCE_AUTHORITY_MAP_V1.json`](paper/PAPER_V1_EVIDENCE_AUTHORITY_MAP_V1.json),
  its wording gate is
  [`paper/PAPER_V1_CLAIM_LEDGER_V1.json`](paper/PAPER_V1_CLAIM_LEDGER_V1.json),
  and its terminal seal is
  [`paper/PAPER_V1_FINAL_ROOT_SEAL_V1.json`](paper/PAPER_V1_FINAL_ROOT_SEAL_V1.json).
- Stage X/F1 is closed under the terminal synthesis
  [`reports/STAGE_X_X1R2_F1T_ROOT_SEAL_V1.json`](reports/STAGE_X_X1R2_F1T_ROOT_SEAL_V1.json).
  Strict selective execution was observed at least once, but reliable sustained
  T5 delivery was not established. BRIDGE/F1-D was not opened.
- Paper V2 is an append-only extension of V1. The existing bounded F1 delta is
  [`paper/PAPER_V2_F1_DELTA_FROM_V1.md`](paper/PAPER_V2_F1_DELTA_FROM_V1.md);
  deterministic, digest-bound CSV/JSON/TeX exports are under
  [`exports/paper_v2/`](exports/paper_v2/) and are checked by
  [`scripts/paper_v2/export_paper_v2_evidence.py`](scripts/paper_v2/export_paper_v2_evidence.py).
- Stage Z is not a scientific result. The controlling root currently records
  `HOLD_STAGE_Z_Z0R2_OFT_CHECKPOINT_AUTHORITY_NOT_ESTABLISHED`, no scientific
  rollout, and zero model/GPU/simulator/environment/protected counters:
  [`reports/STAGE_Z_Z0R2_ROOT_SEAL_V1.json`](reports/STAGE_Z_Z0R2_ROOT_SEAL_V1.json).

## Evidence hierarchy

| Layer | Earned status | Boundary |
| --- | --- | --- |
| X0 physical duty-cycle mechanism | Primary bounded positive | Dose- and phase-dependent OPEN mechanism; descriptive/mechanistic, not formal mediation or universal efficacy. |
| VI-B2, VII, VIII timing-selector line | Valid stage-specific negatives | Frozen held-out/generalization gates only; not proof that every feature or detector is uninformative. |
| IX factorized timing analysis | Model-side factorization evidence | No-environment evidence; not physical attack efficacy. |
| E3/E4 strict selective realizability | Bounded structural evidence | Engineering-parent unit is primary; candidate slots are non-iid diagnostics; no physical efficacy or impossibility claim. |
| F1/F1T execution qualification | Terminal bounded execution-layer evidence | At least one strict execution, but no reliable full-T5 qualification; no BRIDGE/F1-D promotion. |
| Stage Z | Z0R2 authority HOLD | Preparation is not evidence; no Z1 or scientific rollout is authorized by this repository state. |

The canonical source/denominator/claim restrictions are in the Paper V1
authority map and claim ledger, not in this summary.

## Safe repository checks

Create the documented environment if needed:

```bash
conda env create -f environment.yml
conda activate openvla-gripper-attack
pip install -e .
```

The canonical CPU/static checks are:

```bash
python scripts/repository/audit_immutable_authority_paths.py
python scripts/paper/check_paper_v1_claims.py
python scripts/paper_v2/export_paper_v2_evidence.py --check
python -m pytest -q tests/stage_x tests/test_stage_x_primary_matrix_runner.py
python -m pytest -q tests/stage_z/test_stage_z_preparation.py
```

See
[`docs/repository/REPRODUCIBILITY_ENTRYPOINTS_V1.md`](docs/repository/REPRODUCIBILITY_ENTRYPOINTS_V1.md)
for expected markers and exact boundaries. A green check means repository
consistency, not scientific authorization.

## Repository navigation

- [`docs/repository/REPOSITORY_MAP_V1.md`](docs/repository/REPOSITORY_MAP_V1.md):
  top-level ownership, lifecycle, and risk map.
- [`docs/repository/REPOSITORY_LIFECYCLE_LEDGER_V1.json`](docs/repository/REPOSITORY_LIFECYCLE_LEDGER_V1.json):
  machine-readable lifecycle classifications.
- [`docs/repository/ACTIVE_CODE_SURFACE_V1.md`](docs/repository/ACTIVE_CODE_SURFACE_V1.md):
  current execution primitives, audit/contracts, compatibility modules, and
  Stage X/Stage Z/paper surfaces.
- [`docs/repository/IMMUTABLE_AUTHORITY_PATHS_V1.md`](docs/repository/IMMUTABLE_AUTHORITY_PATHS_V1.md):
  protected path/byte firewall and digest rules.
- `paper/`: immutable Paper V1 package plus append-only Paper V2 inputs.
- `reports/`: root seals, manifests, receipts, decisions, and retained failures.
- `configs/`: frozen protocols and runtime/analysis contracts.
- `scripts/paper/`, `scripts/paper_v2/`, `scripts/stage_x/`, `scripts/stage_z/`, and
  `scripts/repository/`: family READMEs identify the one safe current entry
  point and label historical producers/runners.
- `src/gripper_attack/`: shared implementation and compatibility surface.
- `tests/`: CPU/static contract and regression tests.
- `archive/` and `n5/`: retained history; status must be read from evidence,
  not inferred from filenames.

The root `tmp_*.py` files are retained historical diagnostics pending a
separate compatibility decision. They are not current entry points and must not
be executed merely because they remain at top level.

## Historical and provenance warning

The repository intentionally preserves superseded plans, engineering HOLDs,
invalid/non-promotional attempts, exact runtime sources, and historical
receipts. Their presence does not make them current authority. Do not:

- relabel a HOLD as PASS or a runtime/engineering failure as a scientific
  negative;
- rerun, tune, recycle, or top up a frozen cohort to obtain promotion;
- move, rewrite, or delete a sealed path or exact source-byte dependency for
  cleanup;
- substitute clean-only, diagnostic, reserve, or candidate-slot evidence into
  a scientific denominator;
- use green CI as evidence that a scientific gate passed.

Before changing historical material, run the immutable authority audit and
trace imports, subprocess paths, configs, docs, manifests, seals, and Git
history.

## Protected boundary

The controlling Paper V1, F1T, and Stage Z authorities preserve Eval160 and
protected evaluation as `UNREAD`. Repository-hygiene and paper-support work
must perform no model inference, GPU work, simulator use, `env.step`,
adversarial generation/backward pass, physical intervention, `V_phys` read, or
protected read. Any future scientific execution requires separate prospective
PI authorization; this README grants none.

## Main repository and paper repository

This repository owns scientific source-of-truth, evidence extraction, and
deterministic exports. The separate
[`Leo-6-maker/stage_aware_attack_on_vision_language_model`](https://github.com/Leo-6-maker/stage_aware_attack_on_vision_language_model)
repository owns LaTeX, plotting style, captions, and manuscript presentation.
The paper/Overleaf surface consumes digest-bound exports and is not a new
authority for scientific numbers.

## License

MIT License. See [`LICENSE`](LICENSE).
