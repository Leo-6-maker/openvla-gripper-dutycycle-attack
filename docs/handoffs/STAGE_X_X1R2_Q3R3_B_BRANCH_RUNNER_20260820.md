# Stage X1R2 Q3R3-B handoff

Status: `STAGE_X1R2_Q3R3_BRANCH_RUNNER_STATIC_PASS`

Q3R3-B is CPU/mock engineering evidence only. No real OpenVLA inference,
Student inference, simulator construction, env.step, PGD, physical
intervention, V_phys, Eval160, or protected read occurred.

## Source and evidence

- Implementation source commit: `b8b4fe5d588a5ef6e8ffa32ba44f8c993c7d26be`
- Implementation source tree: `3a82fb5a6e6d2c35fb898bad8378a5d9fb49382d`
- Durable output root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/stage_x1r2_q3r3_b_20260820`
- Static audit SHA256: `16c0835f687c80a2cb0d3ad55a76c557612873fb52c37228d706b84aa1943878`
- Official-environment pytest: `5 passed`

## Structural checks

- Reference Student timing is one-shot.
- Prefix replay contains exactly steps `[0, 1, 2]` for the mock `t_emit=3` fixture.
- No pre-branch model or Student callback is available or invoked.
- Common first observation bytes are required and mismatches fail closed.
- Attack authorization before the branch step or before structural gates fails
  closed.
- Branch-state comparison binds all required fields and freezes `atol=1e-12`,
  `rtol=0`, with no widening.
- Protected counters are all zero.

The implementation is a pure-Python contract; it does not import torch,
transformers, or construct an environment. The old Q3R2-C visual-prefix HOLD
remains unchanged.

## Next gate

`STAGE_X1R2_Q3R3_FOUR_SUITE_BRANCH_REPLAY_PASS` is the next authorized gate.
It may use only unstarted identities from the already permanently excluded
Q3R2 engineering pool. It is not scientific efficacy evidence. Q3R3-D,
scientific population selection, PGD, V_phys, Eval160, and protected
evaluation remain unauthorized.
