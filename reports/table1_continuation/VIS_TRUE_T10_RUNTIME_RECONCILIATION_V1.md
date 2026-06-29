# VIS TRUE_T10 Runtime Reconciliation V1

Decision state: `VIS_RUNNING_UNDER_POSTLAUNCH_AUDIT`

Result acceptance: `RESULT_ACCEPTANCE_HOLD`

New condition launch: `NEW_CONDITION_LAUNCH_HOLD`

All server facts below are `REPORTED_UNVERIFIED` until the Bubble snapshot is independently checked. Codex did not connect to the live server, did not stop, resume, migrate, duplicate, or modify any job, and did not inspect aggregate TRUE_T10 outcomes.

## Reported Server State

| Item | Reported value | Verification |
|---|---|---|
| server commit abbreviation | `7b85877` | `SERVER_SNAPSHOT_REQUIRED` |
| CLEAN1500 | `1006/1500` | `SERVER_SNAPSHOT_REQUIRED` |
| TRUE_T10 | launched, 12 workers, `0/162` at report time | `SERVER_SNAPSHOT_REQUIRED` |
| GPUs | `0,1,4,5,6,7`; two workers each | `SERVER_SNAPSHOT_REQUIRED` |
| GPU2 | quarantined | `REPORTED_UNVERIFIED` |
| canary | Fold04, attack_frames=10, open_frames=10, task_success=false | `SERVER_SNAPSHOT_REQUIRED` |

## Reported Runtime SHA

| Artifact | Reported SHA256 | Status |
|---|---:|---|
| bridge | `4ef2a919ee650cf35b35eaa5b9c2152c0d7d18f43710c246ce14dd1c8a83e468` | reported |
| worker | `e21f7fbe7f78003ac2e626bfe9ddb047c194022727bb4d9bc19b9ce0876e337c` | reported |
| condition spec V2 | `7feae7d25a952e7fc018a68205f6cf8e23223ab940b5a897a6734a56563881e1` | reported |
| canonical manifest | `64e20b8ff248fc078d705532aab6d4ec5ea186c143c8b6137fa90d41bdf7a6e4` | reported |
| validator | `921e0e714157c195ed4302bacc003a81daeea22ce13556f3c99b36a89cc003cd` | reported |
| telemetry schema V2 | `b94194659274d7020e35cf69aca3f3054c5f6fb1d0c8921e6b02f93f213d8d8a` | reported |

## GitHub Commit Reconciliation

`git ls-remote origin` matched PR #42 head `a8baa87b87ebe8d841eb187bb89197c9fc282426` and did not show any ref containing reported abbreviation `7b85877`.

This does not prove the server commit does not exist locally; it only means Codex did not find it in current GitHub refs. Full commit identity, parent SHA, dirty status, and diffs require the read-only server snapshot.

## Worker Drift Audit

Previously frozen worker SHA: `41eb3843eb4c6414068cfca3be9dc2bb730b49684832a1ddc333d92589e7dceb`

Reported running worker SHA: `e21f7fbe7f78003ac2e626bfe9ddb047c194022727bb4d9bc19b9ce0876e337c`

Byte-level and semantic diff: `SERVER_SNAPSHOT_REQUIRED`

Required equality checks:

- canary worker SHA == running worker SHA: `SERVER_SNAPSHOT_REQUIRED`
- actual worker SHA == spec bound worker SHA: `SERVER_SNAPSHOT_REQUIRED`
- actual worker SHA == all 162 manifest-row worker SHAs: `SERVER_SNAPSHOT_REQUIRED`

If any equality fails after snapshot verification, state becomes `VIS_RUNTIME_QUARANTINE_HOLD`.

## Manifest / GPU / Running Integrity

The reported manifest SHA is `64e20b8ff248fc078d705532aab6d4ec5ea186c143c8b6137fa90d41bdf7a6e4`. Exact row validation, launch-time output-directory existence, GPU/PID/process mapping, CLEAN1500 overlap, and GPU2 quarantine verification all require the snapshot.

## Current State

`VIS_RUNNING_UNDER_POSTLAUNCH_AUDIT`

`RESULT_ACCEPTANCE_HOLD`

`NEW_CONDITION_LAUNCH_HOLD`

No result-driven tuning was performed by Codex. No job was replaced with a different state or seed by Codex. GPU2 quarantine is reported but unverified. No new Table 1 condition was launched by Codex.
