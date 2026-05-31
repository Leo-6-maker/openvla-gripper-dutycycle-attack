# Codex Autonomous Phase 0 Preflight

Date: 2026-05-31

## Branch / Commit

- Server branch: `exp/codex-next-experiment-harness-20260531`
- Server starting commit: `8ff150d8d11e4e4001d1a3fc786ea09738471ae5`
- GitHub authoritative parent branch: `exp/vis-token-prefix-redecode-and-crosssuite-audit-20260531`
- GitHub authoritative parent HEAD: `b1445ad06abcda91d76b1b1009fa98cb52892ea6`

The server branch is patch-equivalent history from the prior VIS branch, not the GitHub source-of-truth SHA. The clean commit/push should be made from the GitHub-based checkout.

## GPU Status

`nvidia-smi` completed.

- GPU0 had an active non-Codex process using about 2055 MiB: `.../RoboTwin/bin/python`.
- GPU1, GPU2, GPU3, GPU4, GPU5, GPU6, GPU7 were effectively idle at about 3 MiB each.
- This task did not use GPU0 and did not start rollout or training.

## Xid Status

`dmesg -T | grep -i "NVRM|Xid" | tail -120` showed historical GPU0 Xid 13 and Xid 43 events on 2026-05-29. No new rollout was launched in this phase.

## Disk

`df -h /data/liuyu`:

- size: 1.8T
- used: 611G
- available: 1.2T
- use: 36%

## Active Jobs

`screen -ls` reported no sockets for user `liuyu`.

## Dirty / Untracked Files

The server worktree already contained many unrelated untracked files from earlier milestone work and backups. I did not delete or stage them.

Rule for this branch:

- do not use `git add -A`
- use explicit `git add` paths only
- do not commit backups, output roots, checkpoints, videos, or rollout artifacts

## Output Root Plan

External output root reserved for optional generated artifacts:

```text
/data/liuyu/outputs/codex_autonomous_vis_crosssuite_20260531
```

The committed outputs in this branch are small diagnostics/reports/tables only. No large output root contents are committed.
