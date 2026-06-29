# Server Runtime Reconciliation V2

Status: `SERVER_SNAPSHOT_REQUIRED`

This report is intentionally not populated with server bytes until a small Bubble snapshot is verified locally. The snapshot verifier must check `SHA256SUMS.txt`, exact `FILES.json` membership, metadata completeness, symlink/path traversal rejection, and forbidden large/secret files. GitHub files must not be overwritten automatically.

| File surface | Server SHA | GitHub SHA | Byte-identical | Semantic-identical | Used by Formal CLEAN | Acceptable for TRUE_T10 | Required action |
|---|---:|---:|---|---|---|---|---|
| worker | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | unresolved | verify snapshot and diff |
| bridge | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | unresolved | verify snapshot and diff |
| runner | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | unresolved | verify snapshot and diff |
| launch/resume script | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | unresolved | verify snapshot and diff |
| protocol/config | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | unresolved | canonical JSON comparison |
| retry policy | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | unresolved | bind retry-policy contract |
| state selection | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | unresolved | bind adapter source SHA |
| Global Freeze | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | unresolved | bind authoritative freeze source SHA |
| victim checkpoint identity | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | unresolved | bind identity artifact only, no weights |
| metric schema | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | unresolved | bind metric schema source SHA |
| gripper semantics | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | unresolved | source diff required |
| environment lock | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | SERVER_SNAPSHOT_REQUIRED | unresolved | package/env lock only, no credentials |
