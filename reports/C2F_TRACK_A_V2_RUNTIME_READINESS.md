# C2F Track A v2 Runtime Readiness

Date: 2026-07-10

Base commit: `ab9a458c47b8e0a6c2bfcd9fd814bc214e5429bb`

Scope: CPU/static readiness only. No episode was launched.

## Result

Track A completion is now one joint predicate implemented in
`scripts/stageb/audit_c2f_track_a_run.py`. A job is complete only when:

- `episode_metadata.json` exists and parses;
- `runtime_valid is true`;
- `success` is a JSON boolean;
- condition is one of the three frozen Track A conditions and equals the expected job condition;
- `git_commit` is a full 40-character SHA and equals the frozen expected commit;
- protocol is exactly `C2F_TRACK_A_CMDOPEN_ACTION_SPACE` / `2026-07-10.v2`;
- parent key equals the expected job parent;
- `step_records.jsonl` exists, parses, and contains at least one record.

The smoke launcher uses the same predicate for skip/retry decisions. Invalid attempts are archived before retry. The final postrun audit reports metadata files, step-record files/rows, missing and empty step files, commit/protocol mismatches, runtime-invalid jobs, and valid-complete jobs.

## Failure propagation

`run_c2f_track_a_smoke5.sh` now starts with `set -euo pipefail`. A worker error or final audit HOLD therefore returns nonzero. The final audit itself returns `2` unless every expected job satisfies the joint predicate.

## Static gate

```text
SMOKE_EXIT_PROPAGATION = PASS
JOINT_METADATA_STEP_COMPLETION = PASS
COMMIT_PROTOCOL_BINDING = PASS
STATIC_TESTS = PASS
GPU_EPISODES_LAUNCHED = 0
TRACK_A_V2_RUNTIME_READINESS = PASS_STATIC
```

`PASS_STATIC` authorizes review only. Goal smoke, Object replication, Spatial expansion, D7 parity, and all other GPU work remain HOLD.
