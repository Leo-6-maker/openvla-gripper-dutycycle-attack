# Stage X1R2 Q3R3-D2 exact arm-isolation hold — 2026-08-20

Status: `OWNER_REVIEW_X1R2_EXACT_ARM_ISOLATION_FEASIBILITY_REQUIRED`

This is an engineering qualification result, not X1R2 efficacy evidence. The
four D2 fixtures are permanently engineering-only and are consumed. They must
not be rerun, replaced, or promoted into a scientific population.

## Immutable bindings

- D2 source commit: `b86006d95b20b82b1dbdf91d159e8269c112b6fa`
- D2 source tree: `1e10664e02541fac5287c36e6514f3d5df2c71eb`
- D2 protocol SHA256: `e2d53e32d4091cf5b8abc233fdc38a78b877fed52452869b5d3f4799cde8db94`
- Q3R3-C2 root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/stage_x1r2_q3r3_c2_20260820`
- Q3R3-C2 root-seal SHA256: `fe23991c94852bc65269d740a6c67f782350dc61e600f766fdfbf85461abbdcc`
- D2 durable root: `/mnt/sdc/dty_user/openvla_attack_outputs/n5/stage_x1r2_q3r3_d2_20260820`
- Official environment: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`

## D2 execution

The four fixtures were selected from the new C2 branch-replay PASS in the
existing frozen engineering order. C2 reference observations were loaded from
the single sealed `REFERENCE_CLEAN` trajectory; random-time replay image
equality was diagnostic only, while branch-state equality remained required.

| suite | fixture | t_emit | CLEAN arm | TRUE arm |
|---|---|---:|---|---|
| libero_10 | `Q3R2-LIBERO_10-07` | 266 | PASS | `STRUCTURAL_INVALID_NO_SELECTIVE_CANDIDATE` |
| libero_goal | `Q3R2-LIBERO_GOAL-08` | 52 | PASS | `STRUCTURAL_INVALID_NO_SELECTIVE_CANDIDATE` |
| libero_object | `Q3R2-LIBERO_OBJECT-12` | 155 | PASS | `STRUCTURAL_INVALID_NO_SELECTIVE_CANDIDATE` |
| libero_spatial | `Q3R2-LIBERO_SPATIAL-06` | 65 | PASS | `STRUCTURAL_INVALID_NO_SELECTIVE_CANDIDATE` |

The same strict candidate-audit failure occurred in all four suites after the
shared `TokenPrefixPGDAttacker` repair. The traceback reaches
`_select_strict_arm_candidate`; it is not the earlier missing
`region_token_ids`/`None.detach` implementation error. No candidate satisfying
all frozen gates was accepted.

The frozen attack semantics were not changed:

- processor-space `epsilon=0.03`, step size `0.006`, five steps;
- canonical strict route, no fallback, no re-encode, no actuator overwrite;
- native checkpoint-local `NATIVE_OPEN` primary class;
- exact direct-generated arm-token equality on dimensions `0..5`;
- token `31745` secondary diagnostic only.

The four TRUE-arm receipt SHA256 values are:

- L10: `a159bc833f1346d50e804ab9b7027139b0067b3db185f58d5322ff45cb19b6e5`
- Goal: `f8c9fb1b373717bd69ee03b060f1a4779f749d7ed3a1bc3c2ac98dde9a40af1a`
- Object: `db8495f709dfc7d01e5cc516cbe9bf5a3a6b4c576f30187f23dfee5c8696379e`
- Spatial: `2eaa09dfa282452bba9caf4c1b4a96218840e2fe34a43035d28b535c63999b5b`

## Boundary and accounting

The TRUE arms failed before an accepted attack result or attacked
`env.step`. The failure receipts report zero `attacked_env_steps`, physical
interventions, V_phys reads, attack-outcome reads, protected reads, and
Eval160/protected reads. The runner's `pgd_calls` counter is not a proof that
no attack invocation occurred: it increments only after a successful return,
whereas this failure occurs inside the strict candidate audit. The correct
classification is therefore **invoked but not accepted/materialized**, with
zero attacked environment steps.

The D2 aggregate audit is intentionally a HOLD because no suite could produce
a complete five-arm receipt:

- audit status: `HOLD_Q3R3_D_ENGINEERING_MATRIX`;
- audit SHA256: `daa74f4279ac4672adc7b233b2c7c72ea444aa9e13f24c0d4c03b8c2fb6617ca`;
- next gate in the durable audit: `OWNER_REVIEW_Q3R3_D_ENGINEERING_HOLD`.

## Scientific boundary

- historical Q3R3-D hold remains immutable;
- D2 fixtures remain engineering-only and are not efficacy data;
- R0/R1/R2 scientific population selection has not started;
- no X1R2 scientific parent was selected;
- no V_phys was read;
- Eval160 remains `UNREAD`;
- protected evaluation remains `UNREAD`.

Because four independently suite-matched current-runtime fixtures fail the
same exact selective-candidate gate after the shared implementation repair,
this is the frozen method-level branch. Do not launch more engineering
fixtures, tune the attack, weaken arm isolation, or start R0/R1/R2 until PI
reviews whether exact gripper-selective feasibility is compatible with the
frozen attack estimand.

Next gate: `OWNER_REVIEW_X1R2_EXACT_ARM_ISOLATION_FEASIBILITY_REQUIRED`.
