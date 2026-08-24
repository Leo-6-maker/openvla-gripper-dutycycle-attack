# Stage IX F0 contract freeze

Status: `FROZEN_BEFORE_F0_EXECUTION`

Stage VIII remains immutable with final status
`STAGE_VIII_R1_NO_GENERALIZABLE_RELATIVE_SELECTOR`. This branch starts the
new namespace `STAGE_IX_FACTORIZED_PGD_TIMING_UTILITY`; it does not train or
modify another passive vulnerability detector.

The canonical attack was resolved from the current public paper quickstart:

- victim: `/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10`;
- `token_prefix_pgd`, strict no-fallback route;
- `gripper_logit_margin_cw`, target raw gripper `1.0`, official postprocess;
- `epsilon=0.10`, `step_size=0.020`, `PGD-20`, `cw_margin=5.0`;
- processor-space Linf projection around the exact clean `pixel_values`, with
  fp32 projection and dtype-budget correction;
- target action token is derived from the current clean arm action and the
  model's norm statistics/bin centers; no remembered fixed token ID is used.

The old `.25/.05/60` legacy protocol and historical `.03/.06` sweeps are not
the Stage IX primary contract. The exact implementation/config hashes are in
`configs/STAGE_IX_CANONICAL_PGD_CONTRACT_V1.json`.

`G_t` is frozen as a clean-only physical-opportunity gate: finite valid clean
state, raw gripper `<0.5`, and LIBERO env gripper `>0.5`. It is explicitly an
opportunity gate, not a causal V_phys detector, and consumes no privileged
geometry, future state, outcome, or protected data.

F0 uses only consumed non-protected Stage V/VI exact sealed snapshots and T5
labels for a development-only factorized proxy. It performs forward/backward
passes and in-memory image perturbations only; it does not call `env.step`,
modify a rollout, or create physical labels. The three frozen candidates are
E0 clean margin, E1 one-step margin gain, and E3 three-step margin gain. The
full 20-step no-environment result is model-side targetability reference only.

The frozen promotion gate is recorded in
`configs/STAGE_IX_F0_VIS_EXPLOITABILITY_PROTOCOL_V1.json`. If no candidate
passes, the required terminal seal is `STAGE_IX_NO_MODEL_SIDE_TIMING_SIGNAL`;
no physical PGD timing matrix or protected evaluation may follow.

Protected counters are frozen at zero and `Eval160`/protected evaluation
remain unread.
