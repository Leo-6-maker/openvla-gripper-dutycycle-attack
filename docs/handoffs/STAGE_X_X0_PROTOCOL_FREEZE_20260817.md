# Stage X X0 protocol freeze — 2026-08-17

Stage X is a read-only mechanism audit of already-consumed, non-protected
Stage V and Stage VI-B2 formal evidence. It does not create a detector, train
Teacher/Student, run PGD, execute a new environment step, or read Eval160.

The frozen protocol is:

`configs/STAGE_X_X0_DUTY_CYCLE_MECHANISM_PROTOCOL_V1.json`

The input populations are bound to the immutable Stage V M4 labels and branch
evidence, including the explicit retained bridge parent, and the Stage VI-B2
aggregate labels/observations/branches. Abstained or censored branches remain
ABSTAIN and are never converted to negative labels.

The first server action after CI is a CPU/read-only mediator-availability
audit. It must enumerate all 40 Stage V parents and all 16 Stage VI-B2
parents, bind the exact source files, and mark any unavailable mediator as
`NOT_AVAILABLE`. Only exact telemetry may be analyzed; no field may be
reconstructed from outcomes.

X0 reports dose response, paired parent-level uncertainty, monotonicity,
temporal heterogeneity, and an explicitly descriptive mechanism chain. X0
does not authorize physical PGD. X2 requires both X0=A and the independently
gated clean no-environment X1 sequential-PGD diagnostic.

Stage IX remains immutable with terminal status
`STAGE_IX_NO_MODEL_SIDE_TIMING_SIGNAL`; its F0 root is a negative reference,
not a Stage X input population.

Official environment for any later authorized runtime is
`/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`. The server
base worktree is dirty and is out of scope; all Stage X runtime work uses a
separate clean worktree. Foreign GPU processes are not modified.

Protected boundary: `Eval160=UNREAD`, `protected evaluation=UNREAD`, and all
protected counters remain zero.
