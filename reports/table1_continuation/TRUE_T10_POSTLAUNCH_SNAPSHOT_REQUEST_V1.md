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

Required copied files:

- worker
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

Required snapshot control files:

- `FILES.json`
- `SHA256SUMS.txt`
- `SERVER_RUNTIME_METADATA.json`

All paths in `FILES.json` and `SHA256SUMS.txt` must be relative to the snapshot root. Symlinks, path traversal, secret-like files, model weights, and videos must be excluded.
