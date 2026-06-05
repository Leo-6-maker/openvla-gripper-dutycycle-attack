# Fast VIS Logic Repair — 2026-06-05

## Status

Phase E is still **not** a silver-label generator. This repair only adds the CPU-side scaffolding needed to prevent known invalid Phase E results from entering labels or detector training.

No GPU, rollout, VIS, watcher, detector training, server output mutation, large traces, images, embeddings, or videos were used.

## Why Phase E Failed Before

### v0: `INVALID_ACTION_SPACE_CONFOUNDED`

The original canary passed raw decoded OpenVLA actions directly to `env.step`. It skipped the official gripper action path:

```python
normalize_gripper_action(raw_action, binarize=True)
invert_gripper_action(env_action)
```

That means token-level OPEN did not necessarily become a physical env-space OPEN command.

### v1: `PHASE_MISALIGNED_COMPRESSED_WINDOW`

After the action transform fix, centered L10 compression landed in a natural-open phase. A natural-open window cannot establish low-budget VIS physical transfer from a true closed grasp phase.

### v2: `PHASE_NOT_CAPTURED`

Parent-start aligned L10 still observed qpos around open, so the compressed window did not capture the true closed/contact phase.

## Why Centered Compression Is Invalid

Centered compression assumes phase proxy alignment is stable inside the parent VIS window. Phase E evidence shows that assumption is false: the same parent window can contain closed, transitional, and natural-open physical states. Low-budget windows must be selected from real qpos state, not only from parent-window geometry or `phase_bin_proxy`.

## New Phase-Aligned Selection

`scripts/diagnostics/generate_phase_e_aligned_windows.py` enumerates L8/L10/L12 subwindows at:

- parent_start_aligned
- parent_start_plus_2
- parent_start_plus_4
- centered
- parent_end_aligned

It records qpos source, qpos phase class, denominator status, and phase proxy mismatch. A row is recommended only when it is true_closed or transitional-pre-open, natural-open score is low, denominator is not polluted, provenance is not infra-failed, and phase mismatch is not severe.

Missing qpos is marked `MISSING_QPOS_TRACE` and is never auto-recommended.

## Why MuJoCo Qpos Is Primary

The Phase D/E failures were physical-state failures. MuJoCo joint qpos is the closest available audit of actual gripper state. `obs["robot0_gripper_qpos"]` remains useful as fallback and mismatch audit, but it must not silently replace MuJoCo when MuJoCo qpos is available.

## Mechanism Clean Requirements

`run_phase_e_canary.py` now records:

- action drift: arm/action L2 and Linf summaries
- token/action: token flips, VIS_OPEN count, env gripper OPEN count, raw/env gripper means
- qpos: MuJoCo/obs qpos starts, mins, and opening deltas
- budget: raw eps, processor eps, PGD steps/restarts, step size
- mechanism status and reason

`mechanism_clean` is allowed only when VIS_OPEN and env-action OPEN are present, MuJoCo qpos opening is meaningful, arm drift remains below threshold, provenance is clean, and denominator is not polluted.

## Detector Training Boundary

Phase E rows remain excluded from labels_v2/v3 and detector training. Even `silver_candidate` rows are not train labels. Phase E can only become a silver-candidate generator after:

1. true_closed phase alignment;
2. mechanism_clean audit;
3. positive/negative canary separation;
4. 5-sample smoke pass.

## DeepSeek Next Step

After syncing this commit, DeepSeek should first run the CPU-only aligned-window generator on the server where qpos-bearing trace metadata is available. Only after recommended rows exist should a tiny Phase E canary be launched with:

```bash
python scripts/diagnostics/run_phase_e_canary.py \
  --candidate-csv tables/phaseE_aligned_windows_v0.csv \
  --limit 2 \
  --only-recommended
```

The resulting low-budget CSV must pass `scripts/diagnostics/audit_fast_vis_outputs.py` before any row is discussed as a silver candidate.
