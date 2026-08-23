# Stage Z scripts

Current authority status:
`HOLD_STAGE_Z_Z0R2_OFT_CHECKPOINT_AUTHORITY_NOT_ESTABLISHED`.

Canonical CODE-R2 static command:

```text
python -m pytest -q tests/stage_z/test_stage_z_preparation.py
```

This is a synthetic CPU test of action semantics, branch/queue/replan
contracts, fixed population accounting, zero counters, and execution-disable
guards. It is not scientific evidence and does not start Z1.

Lifecycle groups:

- `audit_stage_z_z0_static.py`, `build_common_libero_manifest.py`,
  `build_stage_z_z0r1.py`, and `build_stage_z_z0r2.py` are historical authority
  producers that write reports.
- `build_stage_z_runner_preparation.py` creates runner-preparation material.
  Runner preparation is not scientific evidence, and execution-disable guards
  must remain intact.
- `download_hf_resumable.py`, `resume_sftp_upload.ps1`, and
  `record_stage_z_z0r2_m1_receipt.py` are operational/network/receipt helpers,
  not reproducibility entry points.

Before interpreting Stage Z, run the immutable authority audit and read
`reports/STAGE_Z_Z0R2_ROOT_SEAL_V1.json`. Its next legal action is limited to
repairing the declared authority blocker with no Z1. This README authorizes no
model, GPU, simulator, environment, protected read, or scientific rollout.
