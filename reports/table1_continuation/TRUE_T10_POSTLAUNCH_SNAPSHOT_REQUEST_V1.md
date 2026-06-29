# TRUE_T10 Post-Launch Snapshot Request V1

Collect a small read-only snapshot for offline Codex verification. Do not transfer rollout trees, model weights, credentials, videos, or large telemetry directories.

Required metadata:

- full 40-character server commit SHA
- commit parent SHA
- branch name
- `git status --porcelain=v1`
- `git diff`
- `git diff --cached`
- untracked execution-relevant file list
- launch command
- environment variables with secrets redacted
- Python, conda, CUDA, and runtime versions
- worker process list
- GPU, PID, and process mapping
- launch timestamp
- manifest path
- output root
- retry policy
- resource preflight
- worker PID, start time, command line, CWD, source mtime/inode, and process model
- whether the worker is long-lived or spawned per episode
- immutable deployment copy, launch tarball, container/image digest, `.pyc`, or other loaded-byte proof if present

Required copied files:

- worker bound by Spec V2: `worker_41eb3843....py`
- reported disk worker: `worker_e21f7fbe....py`
- bridge
- condition spec V2
- canonical manifest
- designation
- telemetry schema V2
- metric schema
- retry policy
- launch wrapper
- validator
- canary report
- launch logs
- `WORKER_41EB_TO_E21.diff`
- `WORKER_RUNTIME_PROVENANCE.json`
- `WORKER_VALID_ROW_EQUIVALENCE_REPORT.json`
- `GPU2_PROJECT_PROCESS_AUDIT.json`

`WORKER_RUNTIME_PROVENANCE.json` must have one physical-attempt row per observed attempt and exactly one accepted row per canonical manifest `job_key`. Each row must include:

- `job_key`
- `attempt`
- `accepted`
- `pid`
- `start_time`
- `actual_loaded_worker_sha`
- `bridge_sha`
- `manifest_sha`
- `telemetry_schema_sha`
- `provenance_source`
- `provenance_evidence_sha256`

Allowed `provenance_source` values are:

- `immutable_deployment_copy`
- `container_image_digest`
- `launch_bundle_checksum`
- `episode_recorded_worker_sha`
- `verified_pyc_source_pairing`
- `process_specific_deployment_tree`

`WORKER_VALID_ROW_EQUIVALENCE_REPORT.json` must bind old/new worker SHA, manifest SHA, bridge SHA, harness SHA, test-vector inventory SHA, tested/expected valid-row counts, zero diff counts for condition resolution, attack activation, env action, arm lock, termination, retry behavior, and `overall_pass=true`.

Required snapshot control files:

- `FILES.json`
- `SHA256SUMS.txt`
- `BUBBLE_SNAPSHOT_FILES.json`
- `BUBBLE_SNAPSHOT_SHA256SUMS.txt`
- `SERVER_RUNTIME_METADATA.json`

The four reported `docs/gpu/TRUE_T10_BUBBLE_SNAPSHOT*.json` files are `SNAPSHOT_METADATA_ONLY` unless their exact bytes are transferred and the package also contains or immutably binds the files above.

Do not edit Spec V2, manifest, designation, worker, or bridge in place. A corrected spec can only authorize future attempts; it cannot retroactively bind completed episodes.

All paths in `FILES.json` and `SHA256SUMS.txt` must be relative to the snapshot root. Symlinks, path traversal, secret-like files, model weights, and videos must be excluded.
