# Reproducibility Guide

## Environment Setup

Create the conda environment and install the package:

```bash
conda env create -f environment.yml
conda activate openvla-gripper-attack
pip install -e .
```

Critical dependency versions in `environment.yml` are pinned from the verified environment when available. Any unpinned critical dependency is annotated with `# version not pinned, verified working as of 2026-05-10`.

## Model and Data Provenance

Primary task: `libero_spatial_black_bowl`.

Primary model family: OpenVLA LIBERO spatial fine-tune, e.g. `openvla/openvla-7b-finetuned-libero-spatial` or an equivalent local checkpoint. Set paths with:

```bash
export OPENVLA_MODEL_ROOT=/path/to/openvla/models
export OPENVLA_BASE_MODEL_DIR=/path/to/openvla-7b
export OPENVLA_SPATIAL_MODEL_PATH=/path/to/openvla-7b-finetuned-libero-spatial
export LIBERO_DATA_ROOT=/path/to/libero/datasets
export OPENVLA_OUTPUT_ROOT=outputs/repro_black_bowl
```

Evidence states and seeds:

- State7: seeds `1,2,3,4,5`.
- State5: seeds `0,1,2,3,4,5`.

## Six-Condition Quickstart: State7 Seed1

```bash
python scripts/run_attack_pipeline.py --task black_bowl --state 7 --seed 1 --condition clean
python scripts/run_attack_pipeline.py --task black_bowl --state 7 --seed 1 --condition oracle_continuous
python scripts/run_attack_pipeline.py --task black_bowl --state 7 --seed 1 --condition vis_margin_prevdelta
python scripts/run_attack_pipeline.py --task black_bowl --state 7 --seed 1 --condition ctrl_same_gate_zero_margin
python scripts/run_attack_pipeline.py --task black_bowl --state 7 --seed 1 --condition ctrl_random_direction
python scripts/run_attack_pipeline.py --task black_bowl --state 7 --seed 1 --condition constant_delta_pregrasp
```

Condition logic:

- `clean`: baseline rollout.
- `oracle_continuous`: physical upper-bound check that forces gripper opening in the target window.
- `vis_margin_prevdelta`: primary gripper-targeted temporal PGD condition.
- `ctrl_same_gate_zero_margin`: tests whether the effect requires the same temporal initialization/margin setting.
- `ctrl_random_direction`: tests whether arbitrary visual perturbation causes the same failure.
- `constant_delta_pregrasp`: timing control for early contact disturbance rather than release/pre-place failure.
