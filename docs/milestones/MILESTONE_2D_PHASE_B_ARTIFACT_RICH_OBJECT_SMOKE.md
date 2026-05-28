# Milestone 2D Phase B — Artifact-Rich Object Smoke

## Purpose

Run and validate artifact-rich clean rollouts on three Object-suite tasks (milk, cream_cheese, bbq_sauce) to produce per-timestep step_records, RGB frames, gripper/EEF/action traces, and optional teacher-privileged state. This data enables teacher detector labeling and no-timestep visual/proprio student dataset construction for the final deployment detector.

## Task SR

| Task | N | Success | SR | Avg Steps | Runtime Errors |
|------|---|---------|-----|-----------|----------------|
| milk | 10 | 9 | 0.900 | 163 | 0 |
| cream_cheese | 10 | 10 | 1.000 | 153 | 0 |
| bbq_sauce | 10 | 3 | 0.300 | 250 | 0 |
| **Total** | **30** | **22** | **0.733** | — | **0** |

## Artifact Statistics

| Metric | Value |
|--------|-------|
| Total episodes | 30 |
| Artifact-complete | 30/30 |
| Step records rows | 5,755 |
| Policy RGB frames | 5,355 |
| image_path coverage | 100% |
| gripper/eef/action coverage | 100% |
| Missing manifests | 0 |
| Missing step_records | 0 |
| Missing episode_records | 0 |
| Duplicate episodes | 0 |
| Runtime errors | 0 |
| New Xid during smoke | 0 |

## Environment Fix

The conda env `openvla_official_libero_20260525` had .pth file pollution injecting torch 2.6.0 (from lgzhou) and timm 1.0.26 (from openvla_sparse), which broke `modeling_prismatic.py` vision backbone (timm 1.0.26 `get_intermediate_layers` returns lists, not tuples).

Fix applied:
- Disabled polluting .pth files (z_conda_py310_extra.pth, z_openvla_sparse_extra.pth)
- Installed torch 2.2.0+cu121 and timm 0.9.10 from pre-existing wheelhouse `/data/liuyu/wheelhouse_openvla_official_20260525/`
- Fallback .pth preserves libero/robosuite/mujoco from openvla_sparse at lower priority
- Copied missing deps (matplotlib, scipy, dateutil) from lgzhou env

## Deployment Feature Audit

| Check | Status |
|-------|--------|
| normalized_step absent from deployment features | PASS |
| object/target pose absent | PASS |
| teacher window (as model input) absent | PASS |
| attack outcomes absent | PASS |
| Overall audit | **PASSED** |
| normalized_step_in_deployment_features | false |

## bbq_sauce Note

bbq_sauce remains the most challenging Object task (3/10 = 0.300, consistent with the persistent instability observed across R2 and this smoke). Its episodes are valuable as negative examples and failure-case analysis, not as clean-stable candidates for attack pilot.

## GPUs

| Task | GPU Pair |
|------|----------|
| milk | 4,5 |
| cream_cheese | 2,6 |
| bbq_sauce | 1,3 |

GPU0 (permanent Xid13) and GPU7 (Xid13+Xid31 history) remain quarantined and were not used.

## Output Root

```
/data/liuyu/outputs/milestone_2d_phase_b_artifact_rich_object_smoke_20260527/
```

## Boundary Statement

Milestone 2D Phase B is an artifact-rich clean smoke for data readiness.
It does not run attacks.
It does not train a detector.
It does not use VIS/oracle/random/manual outcomes.
Its purpose is to validate artifact-rich clean trajectories for the final no-timestep visual/proprio online detector.
