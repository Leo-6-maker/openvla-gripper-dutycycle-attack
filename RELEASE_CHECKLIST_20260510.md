# Release Checklist 2026-05-10

| Item | Status | Notes |
|---|---|---|
| README updated | PASS | Public title, claim boundary, quickstart, and layout added. |
| environment.yml exactness checked | PASS | Critical deps are exact if discoverable; otherwise marked `# version not pinned, verified working as of 2026-05-10`. |
| docs/reproducibility.md added | PASS | Includes environment, model/data provenance, state/seed list, and six-condition quickstart. |
| docs/claim_and_evidence.md added | PASS | Includes Template B claim, State7 evidence table, and claim boundaries. |
| docs/dataset_diagnostics.md added | PASS | Moka/alphabet soup/drawer documented as diagnostics or exclusions. |
| archive/README.md added | PASS | Maps archive subdirectories to paper narrative. |
| archive script annotations added | PASS | Key Tier1 archive scripts have top-of-file purpose comments. |
| static path scan clean | PASS | No prohibited server/local path hits; curated historical scripts may retain non-secret run IDs for provenance. |
| secret scan clean | PASS | No credential-pattern hits. |
| large artifact scan clean | PASS | No files larger than 1 MB. |
| compileall passes | PASS | `python -m compileall -q src scripts` and Anaconda Python compileall both pass. |
| pytest passes | PASS | 31 passed under Anaconda Python 3.12 with CPU PyTorch. |
| CLI smoke tests pass | PASS | `run_attack_pipeline.py --help`, dry-run, `v4_validate_artifacts.py --help`, `v4_calibrate_triggers_rollout_openvla.py --help`, `v4_run_eval_openvla.py --help`. |
| local sync complete | PASS | Synced to the requested local `outputs/code/openvla_sparse_attack_public` folder. |
| Template B pipeline audit fixes | PASS | Public CLI now dispatches `configs/paper_black_bowl_attack.yaml`, `epsilon=0.10`, `step_size=0.020`, `attack_steps=20`, State7 `75-84`, State5 `78-87`, and constant pregrasp `35-45`; oracle and constant-delta conditions bypass visual PGD; temporal PGD state is reset on non-attack steps. |

## Residual Caveats

- License is MIT and `LICENSE` has been added at repository root.
- Some critical environment dependencies could not be exactly recovered from the available package metadata and are explicitly marked as unpinned but verified working as of 2026-05-10.
- Historical `v4_` names and some non-secret internal run IDs are intentionally retained for legacy/repro scripts; public docs use `scripts/run_attack_pipeline.py`.
