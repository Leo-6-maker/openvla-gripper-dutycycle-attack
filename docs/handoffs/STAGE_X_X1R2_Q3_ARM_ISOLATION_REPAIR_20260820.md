# Stage X1R2 Q3 arm-isolation repair handoff

Status: `OWNER_REVIEW_Q3_ARM_ISOLATION_REPAIR_REQUIRED`

This stacked change is engineering-only. It does not rerun historical `Q3-F01`, start `Q3-F02` through `Q3-F04`, select a scientific X1R2 parent, read `V_phys`, or read Eval160/protected evaluation.

## Immutable historical boundary

The real-model `Q3-F01` exposure in PR #134 remains immutable and consumed as runtime-invalid after `ARM_TOKEN_ISOLATION_FAIL`. No attacked `env.step` started. `Q3-F02` through `Q3-F04` remain sealed and not started.

The failure is not relabeled as a validator defect. The current diagnosis is an unresolved autoregressive prefix/optimization spillover hypothesis: the visual perturbation is allowed to change the greedy arm prefix, while the old soft teacher-forced arm penalty did not enforce the final discrete direct-generated prefix.

## Frozen gripper-selective contract

A genuine visual gripper-selective attack is now defined as:

1. clean and adversarial actions are produced by the same official deterministic `model.generate` path from the same prompt/input IDs;
2. both direct-generated sequences contain exactly seven action tokens;
3. direct-generated arm token IDs at dimensions `0..5` are exactly equal;
4. the clean direct-generated gripper token is not in the suite-local native `OPEN` class;
5. the adversarial direct-generated gripper token is in the suite-local native `OPEN` class and has changed from clean;
6. only the directly generated adversarial action may reach `env.step`, after all gates pass.

The native `OPEN` class is primary semantic authority; token `31745` remains only a secondary diagnostic. No action-to-token re-encoding, actuator overwrite, fallback, or arm-gate weakening is allowed.

## Prospective engineering repair

`STRICT_CANDIDATE_AUDIT_V1` audits, in frozen order, `delta0` followed by PGD iterations `1..5`. It selects the first candidate satisfying every structural gate, including the clean-non-OPEN to adversarial-native-OPEN transition. If no candidate qualifies, it raises `STRUCTURAL_INVALID_NO_SELECTIVE_CANDIDATE` before attacked `env.step`.

The selected candidate pixels are reused byte-for-byte by the final direct-generation and execution path. Candidate selection cannot use task outcome, Student score, `V_phys`, attack effect, manual outcome, or protected data.

## New permanently excluded fixture

The complete remaining engineering-only candidate universe is deterministically ranked. `Q3-AR-F01` is the first rank:

`libero_10/task_09/state_43` (`M012`, ordinal `10`)

It is already permanently excluded by the frozen manual `FAIL / PRECONTACT_OR_APPROACH` disposition and has no prior attack exposure. The fixture is not a replacement or scientific top-up.

## Pre-GPU evidence

- runtime repair source commit: `b7237611c466077a9a7e6f0b1102e9176cfa2c88`
- runtime repair source tree: `fd5eeef98480b4c608ebd4eafb8e325afa8cd17a`
- the final GitHub PR seal commit/tree is documentation provenance and must not be substituted for the runtime source binding above
- static audit: `PASS_Q3_ARM_REPAIR_STATIC_AUDIT_PRE_GPU`
- targeted regression: `15 passed`
- Python compilation: pass
- working-tree whitespace check: pass
- model inference: `0`
- env steps: `0`
- PGD calls: `0`
- physical interventions: `0`
- Eval160: `UNREAD`
- protected evaluation: `UNREAD`

Next legal action is one real-model, five-arm run of the new permanently excluded `Q3-AR-F01` fixture under the official A800 environment, after this branch has an immutable GitHub source binding and CI pass. This is still an engineering qualification only; it does not authorize X1R2 scientific execution.
