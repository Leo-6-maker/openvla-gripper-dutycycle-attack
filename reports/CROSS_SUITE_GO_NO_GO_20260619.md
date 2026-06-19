# Cross-Suite GO / NO-GO 2026-06-19

## Current Decision

`NO_GO_FOR_GPU`

CPU-only protocol and manifest work is complete enough for review, but new rollout execution is not authorized.

## GO Items

- Latest local `origin/main` identified: `75dc110b558ee8c879ed66f2ba2e6b2f243157c5`.
- Isolated local feature worktree created.
- Suite-matched checkpoint directories exist for Spatial, Goal, and LIBERO-10.
- Object-trained detector checkpoint SHA and dataset SHA recorded.
- Static LIBERO benchmark task lists read from official environment.
- Protocol YAML freezes detector, thresholds, 25D features, normalization source, K, VIS epsilon, PGD steps, target token, and leakage flags.
- CPU readiness audit passes.

## NO-GO / Blockers Before Clean Smoke

- Active GPU process PID `24990` is still using GPU1/5.
- The server dirty checkout must not be used for development or launch.
- Server local `origin/main` is stale; fetch timed out during audit.
- Current Object runner hard-codes Object checkpoint, suite, unnorm key, action stats, and object-site parsing.
- The current active video/batch state does not match the expected 18-video queue; human review is needed before scheduling new GPU work.

## Required Approval Before 18-Clean Smoke

Human approval must confirm:

1. active GPU process has finished or target GPUs are explicitly allocated;
2. server-side launch checkout/path is clean and based on the approved branch;
3. suite checkpoint SHA recording method is accepted;
4. Object-specific runner changes are reviewed;
5. no current output directory will be reused or overwritten.

## First Approved GPU Step Once Unblocked

Run only the 18 clean-smoke entries from `tables/cross_suite_smoke_manifest.csv`.

Do not run VIS/RAND until each candidate parent has:

- clean task success;
- `invalid_feature_steps == 0`;
- mechanism eligibility;
- legal detector trigger or abstain;
- complete telemetry/video outputs.
