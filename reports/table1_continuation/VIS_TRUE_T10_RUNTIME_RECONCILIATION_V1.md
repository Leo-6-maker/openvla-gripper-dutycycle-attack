# VIS TRUE_T10 Runtime Reconciliation V1

Decision state: `VIS_RUNNING_UNDER_POSTLAUNCH_AUDIT`

Worker runtime binding: `RUNTIME_BINDING_P0_HOLD`

Result acceptance: `RESULT_ACCEPTANCE_HOLD`

New condition launch: `NEW_CONDITION_LAUNCH_HOLD`

GPU2 status: `GPU2_QUARANTINE_ACTIVE`

All server facts below are `REPORTED_UNVERIFIED` until the Bubble snapshot is independently checked. Codex did not connect to the live server, did not stop, resume, migrate, duplicate, or modify any job, and did not inspect aggregate TRUE_T10 outcomes.

## Reported Server State

| Item | Reported value | Verification |
|---|---|---|
| server commit abbreviation | `7b85877` | `SERVER_SNAPSHOT_REQUIRED` |
| CLEAN1500 | in progress | `REPORTED_UNVERIFIED` |
| TRUE_T10 | running under post-launch audit | `REPORTED_UNVERIFIED` |
| GPUs | `0,1,4,5,6,7`; two workers each | `SERVER_SNAPSHOT_REQUIRED` |
| GPU2 | quarantine active; project job count reported zero; requalification not run; external `isaac-gr00t-n1.7` workload reported | `REPORTED_UNVERIFIED` |
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

## Worker Runtime Binding P0

Spec-bound worker SHA: `41eb3843eb4c6414068cfca3be9dc2bb730b49684832a1ddc333d92589e7dceb`

Reported disk worker SHA: `e21f7fbe7f78003ac2e626bfe9ddb047c194022727bb4d9bc19b9ce0876e337c`

Byte-level and semantic diff: `SERVER_SNAPSHOT_REQUIRED`

The mismatch is a P0 hold because current disk bytes do not prove which worker each episode loaded. The original spec, manifest, and designation must remain unchanged. Post-hoc spec rebinding is prohibited and cannot retroactively prove preregistration.

Required equality checks:

- canary worker SHA == running worker SHA: `SERVER_SNAPSHOT_REQUIRED`
- actual worker SHA == spec bound worker SHA: `SERVER_SNAPSHOT_REQUIRED`
- actual worker SHA == all 162 manifest-row worker SHAs: `SERVER_SNAPSHOT_REQUIRED`

Decision tree:

- all formal jobs used `41eb...`: spec binding may remain valid, but future retry/restart must use exact `41eb...` bytes or a new non-mixed spec/manifest.
- all formal jobs used `e21...` and valid-row behavior is proven equivalent: preserve original spec/manifest, create `TRUE_T10_POSTLAUNCH_RUNTIME_DEVIATION_V1.json`, and leave result acceptance to independent review.
- mixed worker versions, unknown loaded bytes, or any valid-row behavioral difference: `VIS_RUNTIME_QUARANTINE_HOLD`.

## Manifest / GPU / Running Integrity

The reported manifest SHA is `64e20b8ff248fc078d705532aab6d4ec5ea186c143c8b6137fa90d41bdf7a6e4`. Exact row validation, launch-time output-directory existence, GPU/PID/process mapping, CLEAN1500 overlap, and GPU2 quarantine verification all require the snapshot.

## Required Next Artifacts

- `BUBBLE_SNAPSHOT_SHA256SUMS.txt`
- `BUBBLE_SNAPSHOT_FILES.json`
- `worker_41eb3843....py`
- `worker_e21f7fbe....py`
- `WORKER_41EB_TO_E21.diff`
- `WORKER_RUNTIME_PROVENANCE.json`
- `WORKER_VALID_ROW_EQUIVALENCE_REPORT.json`
- `GPU2_PROJECT_PROCESS_AUDIT.json`
- `TRUE_T10_POSTLAUNCH_RUNTIME_DEVIATION_V1.json` only for Case B

## Current State

`VIS_RUNNING_UNDER_POSTLAUNCH_AUDIT`

`RUNTIME_BINDING_P0_HOLD`

`RESULT_ACCEPTANCE_HOLD`

`NEW_CONDITION_LAUNCH_HOLD`

`GPU2_QUARANTINE_ACTIVE`

No result-driven tuning was performed by Codex. No job was replaced with a different state or seed by Codex. GPU2 quarantine is reported active but still pending offline snapshot verification. No new Table 1 condition was launched by Codex.
