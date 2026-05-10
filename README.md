# OpenVLA Gripper Duty-Cycle Attack

This repository is a clean research artifact for reproducing an inference-time gripper-targeted visual attack on OpenVLA/LIBERO. The main claim is deliberately narrow: duty-cycle shaping of the gripper-open command during contact-critical windows can induce visible grasp/lift/carry failures in the LIBERO black-bowl task.

## Quick Start

```bash
conda env create -f environment.yml
conda activate openvla-gripper-attack
cp .env.example .env
# edit .env with your local model and LIBERO paths
pip install -e .
python scripts/run_attack_pipeline.py --help
```

Run one paper-facing condition:

```bash
python scripts/run_attack_pipeline.py --task black_bowl --state 7 --seed 1 --condition clean --dry_run
```

Remove `--dry_run` after setting the model/data environment variables.

## Repository Layout

- `src/gripper_attack/`: public reusable attack, trigger, logging, and metric code.
- `scripts/run_attack_pipeline.py`: stable public CLI for the six-condition matrix.
- `scripts/v4_*.py`: legacy/repro entrypoints retained for traceability.
- `docs/reproducibility.md`: exact reproduction instructions and provenance.
- `docs/claim_and_evidence.md`: frozen claim, evidence table, and boundaries.
- `docs/dataset_diagnostics.md`: excluded dataset diagnostics and future-work framing.
- `archive/`: curated historical notes and scripts that explain exploratory branches.

## Claim Boundaries

We do not claim `prev_delta` or margin-objective uniqueness. We do not claim broad cross-dataset generalization. Moka pots, alphabet soup, and open-middle-drawer are documented as diagnostics/future work rather than main evidence.

## License

MIT License. See [LICENSE](LICENSE).
