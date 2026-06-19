# Cross-Suite Server Audit 2026-06-19

## Scope

Read-only audit before any cross-suite SC5 Layer 1 -> 2 -> 3 work. No OpenVLA model load, PGD, rollout, training, process kill, cleanup, stash, reset, or output modification was performed.

## Server Snapshot

- host: `klfy-SYS-4028GR-TR2`
- audit time: `2026-06-19T00:43:05+08:00`
- user: `liuyu`
- data disk: `/data` 1.8T total, 625G used, 1.1T available
- root disk: `/` 916G total, 310G used, 559G available

## Current Dirty Checkout

- path: `/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605`
- branch: `exp/l12-production-streaming-adapter-20260615`
- checkout HEAD: `52bdc33507a584e158581f9eded6dc657ac71029`
- server local `origin/main`: `99a51fb6a66e168967fb5cc5e9a1cdc5325233e1`
- fresh local GitHub `origin/main` from development machine: `75dc110b558ee8c879ed66f2ba2e6b2f243157c5`
- server `git fetch origin --prune`: timed out under audit command, so the server local remote ref is stale relative to the development machine.

Dirty state:

- tracked modification: `scripts/stageb/run_d4_clean_shadow.py`
- untracked files: 40
- detailed inventory: `tables/cross_suite_dirty_inventory_20260619.csv`

The dirty checkout must not be reset, cleaned, stashed, restored, or used as the development checkout for this work.

## Active GPU / Process State

`nvidia-smi` showed active compute on physical GPUs 1 and 5:

- PID: `24990`
- command: `/data/aviary/envs/openvla_official_libero_20260525/bin/python scripts/stageb/run_v2_vis_sc5_mlp_bridge.py --condition TRUE_T10 --task_idx 0 --state_id 0 --anchor 86 --render_gpu 5 --seed_id 1 --mlp_path /data/liuyu/repos/sc5_census_freeze_cc356f3_20260618/outputs/sc5_canonical_eng/sc5_mlp_s2.pt --output_dir /data/liuyu/outputs/vtest2/VIS/soup_s0 --save_video --video_fps 10`
- cwd: `/data/liuyu/repos/sc5_census_freeze_cc356f3_20260618`
- `CUDA_VISIBLE_DEVICES`: `1,5`
- output directory: `/data/liuyu/outputs/vtest2/VIS/soup_s0`

Parent shell PID `24804` included a historical `pkill -9 -u liuyu python` command before launching the active job. This was observed only; no signal was sent and no remediation was attempted.

## 18-Video Batch Status

The expected "18-video batch" was not observed as an active 18-job queue. Instead:

- active process count matching the bridge runner: 1
- active pair: `TRUE_T10 soup_s0`, `task_idx=0`, `state_id=0`
- `/data/liuyu/outputs/vtest2/VIS/soup_s0`: directory existed but no files were observed during audit
- completed recent video artifacts were observed under:
  - `/data/liuyu/outputs/layer123_videos/TRUE_T10/cream_s0`
  - `/data/liuyu/outputs/layer123_videos/TRUE_T10/soup_s0`

This mismatch is a risk item. It does not block CPU-only planning, but it blocks any new GPU rollout until the active job/batch is externally reviewed.

## Environment And Checkpoints

Python environment used by active process:

- `/data/aviary/envs/openvla_official_libero_20260525`

Available suite-matched OpenVLA checkpoints:

- `/data/aviary/models/openvla/openvla-7b-finetuned-libero-object`
- `/data/aviary/models/openvla/openvla-7b-finetuned-libero-spatial`
- `/data/aviary/models/openvla/openvla-7b-finetuned-libero-goal`
- `/data/aviary/models/openvla/openvla-7b-finetuned-libero-10`

Frozen detector checkpoint:

- path: `/data/liuyu/repos/sc5_census_freeze_cc356f3_20260618/outputs/sc5_canonical_eng/sc5_mlp_s2.pt`
- checkpoint SHA256: `66ec2d487ef4b4c673cb2c7c147c7f64c6e27c3e1eb6ced4470bf18466c11628`
- dataset SHA256: `f942f4b0856d3449fa4e98f6d6e74ac8d5e8e9af7082373f961f79b0a6930cd9`

## Safe Development Worktree

Created locally, not on the dirty server checkout:

- path: `D:\vla_attack\repo_work\sc5_cross_suite_generalization_20260619`
- branch: `feature/sc5-cross-suite-generalization-20260619`
- base: latest local `origin/main`
- base SHA: `75dc110b558ee8c879ed66f2ba2e6b2f243157c5`

Server worktree creation is deferred because the current server checkout is dirty and GPU1/5 has an active process.

## Risks / NO-GO Items

- Server local `origin/main` is stale because fetch timed out.
- Active GPU process is using GPU1/5 and must not be interrupted.
- Current active process parent command previously included `pkill -9 -u liuyu python`; batch provenance needs human review.
- Current Object runner is suite-specific and must not be used directly for cross-suite rollouts.
- Any new GPU smoke requires explicit approval after protocol review.
