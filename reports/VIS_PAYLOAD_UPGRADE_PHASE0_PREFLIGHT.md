# VIS Payload Upgrade Phase 0 Preflight

Date: 2026-06-01

Branch: `exp/vis-payload-upgrade-validation-20260601`
SHA: `8ff150d8d11e4e4001d1a3fc786ea09738471ae5`

## Working Tree

- Tracked dirty files at branch creation:
  - `scripts/run_official_eval_artifact_rich.py`
  - `src/gripper_attack/attack_adapter.py`
- Many VIS/CrossSuite files are present as untracked server worktree files. They are treated as real local worktree context, not assumed committed upstream.
- No untracked files larger than 5 MB were found in the repo scan.

## GPU / Runtime

- GPUs 0-7 were idle by `nvidia-smi` at preflight.
- GPU7 maps to PCI bus `00000000:0F:00.0` / UUID `GPU-da4d4ba8-f4f9-7513-4dc9-87f4c7033d7f`.
- Kernel log shows fresh 2026-06-01 10:00:34 Xid 13 and Xid 43 on PCI `0000:0f:00`, python pid 37076.
- Policy for this branch: avoid GPU7; avoid GPU0 by default; prefer GPU4/5 for short clean-only/no-rollout diagnostics after each preflight check.
- `screen -ls` reported no active sessions for `liuyu`.

## Disk / Sync

- `/data/liuyu`: 1.8T total, 612G used, 1.2T available, 36% used.
- GitHub sync is not clean from the server: HTTPS fetch failed with certificate host mismatch (`rukita.co` vs `github.com`), and SSH `ls-remote` did not complete during preflight.
- GitHub PR #5 comments were used only as orientation. The server worktree is the source of truth for executable files in this continuation.

## Gate

- Proceed only with clean-only frame planning/collection and no-rollout diagnostics.
- Do not run VIS rollout, detector-triggered rollout, sus30, CrossSuite training, or production-line changes.
- Treat any new Xid/OOM/illegal memory access as a stop signal for the affected GPU and quarantine related output.
