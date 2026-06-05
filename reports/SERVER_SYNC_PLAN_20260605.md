# Server Sync Plan 2026-06-05

Purpose: make the reviewed Codex branch visible and usable on the server without disturbing the current checkout or any watcher/job that might depend on it.

No handoff self-reference update is needed. Use `git log -1 -- reports/HANDOFF_20260605_WINDOW_COMPRESSION_AND_DETECTOR.md` when the current committed handoff revision is needed.

## Current Version State

| Item | Value |
|---|---|
| Local reviewed branch | `exp/vis-prefix-margin-repair-20260603` |
| Local HEAD | `a74eaead95fc139548ee2e39b0ec1c40bf254c96` |
| Remote | `origin https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack.git` |
| Remote branch HEAD | `a74eaead95fc139548ee2e39b0ec1c40bf254c96` |
| Remote visibility | PASS: `git ls-remote origin exp/vis-prefix-margin-repair-20260603` matches local HEAD |
| Server repo | `/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524` |
| Server current branch | `exp/vis-payload-upgrade-validation-20260601` |
| Server current HEAD | `653ed33d78578aa0f0af96539a9c8b4c2a6d4c08` |
| Server has local reviewed HEAD | No |

## Required Sync Action

Server must be synced to the reviewed branch before DeepSeek runs label merge, schema audit, or detector v2 training. Current server checkout is stale relative to this review package.

DeepSeek detector v2 training remains **BLOCKED** until:

1. Server can access the reviewed branch at `a74eaead95fc139548ee2e39b0ec1c40bf254c96`.
2. Label builder supports Batch3b/Batch3c.
3. `tables/object_phase_response_labels_v2.csv` is generated.
4. `scripts/diagnostics/audit_label_schema.py --labels-csv tables/object_phase_response_labels_v2.csv` passes.

## Option A - Safe Worktree, Recommended

Use this when the current checkout may be referenced by existing outputs, watchers, terminals, or DeepSeek scripts. It avoids disrupting the old branch.

```bash
cd /data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524
git fetch origin

git worktree add \
  /data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605 \
  origin/exp/vis-prefix-margin-repair-20260603

cd /data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605
git rev-parse HEAD
git branch --show-current
ls scripts/diagnostics/audit_label_schema.py
ls scripts/diagnostics/generate_window_compression_candidates.py
ls reports/CODEX_PARALLEL_REVIEW_SUMMARY_20260605.md
```

Expected HEAD:

```text
a74eaead95fc139548ee2e39b0ec1c40bf254c96
```

Note: a detached worktree from `origin/...` is acceptable for read-only audit/training preparation. If DeepSeek needs local branch commits, create a local branch in the worktree after verifying HEAD.

## Option B - In-Place Checkout, Only If No Watcher/Jobs Depend On Current Repo

Use this only after confirming no active watcher, terminal, or output provenance depends on the current checkout.

```bash
cd /data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524
git fetch origin
git status --short
git switch exp/vis-prefix-margin-repair-20260603
git pull --ff-only
```

Before Option B, inspect the untracked files currently present in the server checkout:

```bash
git status --short
```

Current read-only verification showed untracked server copies of:

```text
scripts/diagnostics/finalize_phase_response_labels.py
scripts/train_vulnerability_ready_detector_v1.py
```

Do not overwrite or implicitly rely on those untracked scripts without reviewing them.

## Recommendation

Use **Option A**. It creates a clean reviewed worktree and keeps the old checkout available for provenance or active-process dependencies.

Do not train detector v2 from the old server checkout.
