# Feature Binding Protocol V1

Scope: bind frozen Label V2 rows to an existing per-step SC5 25D clean feature CSV.

This layer is CPU-only and metadata-only. It does not build detector rows, train a detector, run OpenVLA/LIBERO, rollout, attack, or use GPU/A800.

Required inputs:

- Label V2 five-file artifact.
- Feature CSV containing `episode_key`, `step`, and the exact 25 `SC5_FEATURES` columns in canonical order.

Checks:

- Label V2 artifact passes the existing five-file loader.
- Feature CSV has exact SC5 feature order, finite values, unique `(episode_key, step)`, no orphan episodes, no missing Label episodes.
- Per-episode steps cover `0..trace_length-1`.
- If feature CSV includes `parent_key`, `suite`, `task_id`, or `trace_length`, those fields must match Label V2.
- Binding manifest records Label artifact, build manifest, manual audit, feature CSV, and feature schema SHA values.

Outputs:

- `binding_manifest.json`: full binding evidence.
- `dataset_manifest.json`, `dataset_statistics.json`, `population_summary.csv`, `feature_summary.csv`: metadata only.

Non-actions:

- `formal_detector_dataset_build = NOT_PERFORMED`
- `training = NOT_PERFORMED`
- `gpu = NOT_PERFORMED`
