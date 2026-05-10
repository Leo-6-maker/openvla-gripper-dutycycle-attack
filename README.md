# OpenVLA Gripper Duty-Cycle Attack

This repository is a clean research artifact for reproducing an inference-time gripper-targeted visual attack on OpenVLA/LIBERO. The main claim is deliberately narrow: duty-cycle shaping of the gripper-open command during contact-critical windows can induce visible grasp/lift/carry failures in the LIBERO black-bowl task.

## Prerequisites

- **Python**: 3.10+ (verified with 3.10.18)
- **CUDA**: 12.1+ (verified with 12.1, driver 530)
- **GPU memory**: ~14 GiB for OpenVLA-7B in float16 (single GPU) or 3-4 GPUs
  with ~5 GiB each plus CPU offload
- **Disk**: ~30 GiB for model checkpoints and LIBERO datasets
- **Transformers**: 4.40.x recommended (5.x also works with patching)
- **LIBERO**: datasets must be locally available (MuJoCo rendering uses EGL)
- **OpenVLA model**: not a pip package. Loaded via `trust_remote_code=True`
  directly from the checkpoint directory. The checkpoint must contain
  `modeling_prismatic.py`, `configuration_prismatic.py`, and
  `processing_prismatic.py`. Our runner auto-patches these files via
  `scripts/patch_openvla_compat.py` when `--auto_patch_compat` is passed.

## Quick Start

```bash
# 1. Create and activate environment
conda env create -f environment.yml
conda activate openvla-gripper-attack

# 2. Install the project package
pip install -e .

# 3. Configure paths (copy from example and edit)
cp .env.example .env
# Edit .env — set OPENVLA_MODEL_ROOT, OPENVLA_BASE_MODEL_DIR,
# OPENVLA_SPATIAL_MODEL_PATH, and LIBERO_DATA_ROOT to your local paths.
# Then source the file or export the variables:
source .env   # or: export $(cat .env | xargs)

# 4. Verify the CLI works
python scripts/run_attack_pipeline.py --help
```

### Test the pipeline (no GPU required)

```bash
# Unit tests (31 tests, ~7 seconds)
python -m pytest -q tests/

# Dry-run — prints the dispatched command without executing
python scripts/run_attack_pipeline.py --task black_bowl --state 7 --seed 1 \
    --condition clean --dry_run
```

### Run a real episode (GPU + LIBERO + OpenVLA required)

Before running, ensure the environment variables from `.env` are exported AND
the following are set:

```bash
export MUJOCO_GL=egl                    # headless rendering (required)
export CUDA_VISIBLE_DEVICES=0           # or a comma-separated list of free GPUs
export OPENVLA_CUDA_MAX_MEMORY=10000MiB # per-GPU memory cap for device_map=auto
```

Then run directly via the runner (bypasses the pipeline wrapper, gives full
control over GPU placement):

```bash
python scripts/v4_run_eval_openvla.py \
    --tasks_config configs/v4_tasks_libero.yaml \
    --attack_config configs/paper_black_bowl_attack.yaml \
    --directions_config configs/v4_directions.yaml \
    --task_id libero_spatial_black_bowl \
    --trigger clean --rho 0.0 --seed 1 --episodes 1 \
    --max_steps_override 400 \
    --model_path "$OPENVLA_SPATIAL_MODEL_PATH" \
    --base_model_code_dir "$OPENVLA_BASE_MODEL_DIR" \
    --unnorm_key libero_spatial \
    --camera_obs_key agentview_image \
    --auto_patch_compat \
    --libero_official_preprocess --center_crop --postprocess_gripper \
    --deterministic_init_states --state_ids 7 \
    --render_gpu_device_id 0 --model_gpu_device_id 0 \
    --output_root "$OPENVLA_OUTPUT_ROOT" \
    --run_id my_clean_run --success_metric done
```

Or use the public pipeline wrapper (sets Template B parameters automatically):

```bash
python scripts/run_attack_pipeline.py \
    --task black_bowl --state 7 --seed 1 --condition clean \
    --model_path "$OPENVLA_SPATIAL_MODEL_PATH" \
    --base_model_code_dir "$OPENVLA_BASE_MODEL_DIR"
```

The public entrypoint dispatches the frozen Template B parameters by default:
`epsilon=0.10`, `step_size=0.020`, `attack_steps=20`, State7 window `75-84`,
State5 window `78-87`, and `constant_delta_pregrasp` window `35-45`.
Override them with `--epsilon`, `--step_size`, `--attack_steps`.

### Troubleshooting

- **`ModuleNotFoundError: No module named 'openvla'`**: This is expected.
  OpenVLA is not a pip package. The model code lives inside the checkpoint
  directory and is loaded via `trust_remote_code=True`. Ensure
  `--model_path` points to a checkpoint that contains `modeling_prismatic.py`.

- **CUDA OOM**: Reduce `OPENVLA_CUDA_MAX_MEMORY` or add more GPUs via
  `CUDA_VISIBLE_DEVICES`. The model distributes layers across all visible
  GPUs when `--model_gpu_device_id -1` (auto mode).

- **Transformers 5.x compatibility**: Pass `--auto_patch_compat` to let the
  runner copy and patch the Prismatic model files from the base checkpoint.

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
