# M3 v2 seed81 trajectory feasibility audit

## Status

`M3_V2_SEED81_INTERMEDIATE_FEASIBILITY: NOT ESTABLISHED`

The existing v2 seed81 artifact cannot be used to promote an intermediate
iterate into a v2.1 candidate.  Deterministic replay did not reproduce the
frozen final tensor hashes or final official margin, and a direct rerun of the
original fixed-frame canary runner also failed to bitwise-reproduce the frozen
terminal margin/hash.

This is a fixed-frame audit only.  No LIBERO rollout, multi-frame panel,
rescue experiment, or held-out transfer was launched.

## Evidence

| Artifact | Result |
| --- | --- |
| `tables/m3_v2_seed81_trajectory_telemetry.csv` | Existing debug has surrogate-only per-step margin/prefix/gradient telemetry. |
| `tables/m3_v2_seed81_full_trajectory_official_audit.csv` | Deterministic replay official-decodes delta0 and post-update iterations 1-20, but reconstruction is invalid. |
| `tables/m3_v2_seed81_original_canary_rerun_condition_results.csv` | Original runner rerun emits the same final token and arm match class, but terminal margin/hash differ from the frozen artifact. |
| `tables/m3_v2_seed81_replay_reproducibility_audit.csv` | Frozen, original-runner-rerun, and replay terminal artifacts compared side by side. |
| `tables/m3_v2_seed81_trajectory_claim_matrix.csv` | Claim boundary for this audit. |

## Key Observations

- Offline surrogate telemetry suggested early iterations with clean arm prefix `6/6` and log-ratio margin above `6.0`, but those rows are not official generation evidence.
- The replay delta0 hash matched the frozen artifact, so the frozen input, seed, and random start were recovered.
- The deterministic replay terminal official tokens matched the frozen token sequence and kept arm match at `2/6`, but final tensor hashes and official margin did not match.
- A direct original-runner rerun at commit `284eeaf56640c51a8a96bf5842e3b746d390ea40` also did not reproduce the frozen terminal hash/margin.  It produced the same terminal token sequence and arm match count, but margin `29.99901008605957` rather than frozen margin `29.249469757080078`.

## Interpretation

The v2 seed81 terminal behavior remains `ARM_NONSELECTIVE`: token `31744` is
reached, but arm prefix match is only `2/6`.  The attempted intermediate audit
does not establish a valid selective iterate because the underlying 20-step
trajectory is not reconstructable to the required tensor/hash and official
margin standard.

## Allowed Claim

The v2 seed81 trajectory contains surrogate-only early steps that appear
selective, but deterministic replay did not validate the frozen trajectory.
Therefore v2.1 promotion from this artifact is not supported.

## Forbidden Claim

Do not claim a feasible v2.1 intermediate, fixed-frame Layer3 success, or
closed-loop effect from this audit.  Do not use the surrogate-only telemetry as
official-token evidence.

## Next Action

Proceed to the next preregistered route: design a new arm-constrained objective
or penalty from the current audited base.  Do not run multi-frame or LIBERO
experiments from v2 seed81.
