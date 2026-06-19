# Contributing

This repository separates maintainable code from experiment evidence.

## Code and Protocol Changes

- Put reusable code under `src/`, `scripts/`, `configs/`, and `tests/`.
- Keep frozen protocols and claim boundaries in compact docs.
- Add CPU tests for new audit, collector, queue, schema, and parser logic.

## Generated Outputs

Do not commit large generated evidence outputs. Keep these out of git:

- raw server output roots
- full episode ledgers
- videos and tensors
- large audit snapshots
- model checkpoints

Use `generated/`, `audit_outputs/`, `paper/generated/`, or external artifact storage. Final evidence bundles should include `SHA256SUMS`, an artifact manifest, and reproduction commands.

## Scientific Claims

Clean-only data supports clean denominator, success-rate, feature-validity, and detector-emission accounting only. It does not establish Teacher timing, VIS/RAND superiority, physical gripper causality, or closed-loop attack success.

Every experiment-result PR must state:

- denominator
- seeds and controls
- invalid and infra-failed runs
- exact source commit
- output root or release artifact SHA
- allowed and forbidden claims
