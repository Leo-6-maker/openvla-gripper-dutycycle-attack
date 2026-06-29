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

Required snapshot control files:

- `FILES.json`
- `SHA256SUMS.txt`
- `BUBBLE_SNAPSHOT_FILES.json`
- `BUBBLE_SNAPSHOT_SHA256SUMS.txt`
- `SERVER_RUNTIME_METADATA.json`

The four reported `docs/gpu/TRUE_T10_BUBBLE_SNAPSHOT*.json` files are `SNAPSHOT_METADATA_ONLY` unless their exact bytes are transferred and the package also contains or immutably binds the files above.

Do not edit Spec V2, manifest, designation, worker, or bridge in place. A corrected spec can only authorize future attempts; it cannot retroactively bind completed episodes.

All paths in `FILES.json` and `SHA256SUMS.txt` must be relative to the snapshot root. Symlinks, path traversal, secret-like files, model weights, and videos must be excluded.
