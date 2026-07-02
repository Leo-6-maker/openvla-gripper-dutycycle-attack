# RAND_T10 Direction Control Audit 2026-07-02

Source: server episode_summary.json files under the authoritative Object evidence root.

## Accounting

- RAND_total: 162
- RAND_emitted: 141
- RAND_no_emission: 21
- RAND_attack_applied: 0
- RAND_attack_not_applied: 162
- RAND_success_total: 162
- RAND_success_attacked: 0
- RAND_success_unattacked: 162
- RAND_failure_total: 0
- RAND_failure_attacked: 0
- RAND_failure_unattacked: 0

## Finding

`RAND_T10` is not clean fallback in the server summaries: attacked rows have `attack_applied=True`/`attack_frames=10`.
The recomputed accounting preserves the reported 162/162 task-success total while showing 141 attacked/emitted rows and 21 unattacked/no-emission rows.

The exact random-direction implementation remains protocol-provenance work: this audit recovered row-level attack application from summaries, but did not seal launch/config code as proof of random pixel/sign/target semantics.
