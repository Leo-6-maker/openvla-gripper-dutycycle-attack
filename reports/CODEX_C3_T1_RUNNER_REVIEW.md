# C3-T1 V23 runner and pilot-input review

Code snapshot: `c8632e9af3ab79803cd6a663b8d6d7cec8780076`

## D0-R2P input

The new byte-sealed pilot input was built from the frozen 670-identity DEV_POOL
before opening selected payloads. It contains one identity per task, 40 total,
4 suites, and the three required files per identity:

- `episode_metadata.json`;
- `step_records.jsonl`;
- `privileged_teacher_sidecar.jsonl`.

Root:
`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/v23_dev_pilot_v1_20260727_0315`

Root `SHA256SUMS` SHA256:
`2133ff9475ab8a0b0c425f1a2d56c984633aba3c6c81c8f084fc04dd1d970b8c`

The root checksum command passed after being run from the root. The manifest
records exact episode/step joins, 40/40 payload parses, and protected payload
read `false`. The prior two-file root at
`v23_dev_pilot_v1_20260727_0300` remains preserved as superseded evidence and
is not consumed.

## Runner boundary

`n5/phase2_labels/run_v23_dev_pilot.py` rejects forbidden action/outcome fields
before physical-field projection, consumes only the physical sidecar plus an
independently supplied C3-G geometry case, and requires contiguous `0..T-1`
steps. It does not import the V22 runner and does not load a model or policy.

The preflight now verifies sealed C1-V2 registry closure, an explicitly
allowlisted geometry root, geometry-root file closure, and exact equality
between geometry identities and the frozen 40-episode pilot identities.

## T1B result

`HOLD_RUNNER_GEOMETRY_INPUTS`

The current frozen `C3_S3_ALLOWED_INPUTS_V1` has
`allowed_episode_geometry_roots = []`; therefore the synthetic C3-S3A root is
not an allowed V23 pilot input. When tested with the synthetic allowlist, its
manifest identities are `c3s3a_relation_*`, not the 40 frozen pilot identities,
and the runner returns structured `PREFLIGHT_HOLD` with reason:

`geometry root does not bind exactly the frozen pilot identities`

This is the intended fail-closed result. The synthetic relation fixture is not
used as a pilot geometry substitute.

## Execution boundary

T1C smoke and T1D 40-episode pilot were not run. No protected payload,
OpenVLA/model inference, Student training, rollout, or attack was run.
