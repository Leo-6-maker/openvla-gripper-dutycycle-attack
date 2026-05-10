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
| static path scan clean | PASS | No server/local path hits. |
| secret scan clean | PASS | No credential-pattern hits. |
| large artifact scan clean | PASS | No files larger than 1 MB. |
| compileall passes | PASS | `python -m compileall -q src scripts` under the verified Python 3.10 OpenVLA environment. |
| pytest passes | PASS | 23 passed, 2 warnings. |
| CLI smoke tests pass | PASS | `run_attack_pipeline.py --help`, dry-run, `v4_validate_artifacts.py --help`, `v4_calibrate_triggers_rollout_openvla.py --help`, `v4_run_eval_openvla.py --help`. |
| local sync complete | PASS | Synced to the requested local `outputs/code/openvla_sparse_attack_public` folder. |

## Residual Caveats

- License is MIT and `LICENSE` has been added at repository root.
- Some critical environment dependencies could not be exactly recovered from the available package metadata and are explicitly marked as unpinned but verified working as of 2026-05-10.
- Historical `v4_` names are intentionally retained for legacy/repro scripts; public docs use `scripts/run_attack_pipeline.py`.
