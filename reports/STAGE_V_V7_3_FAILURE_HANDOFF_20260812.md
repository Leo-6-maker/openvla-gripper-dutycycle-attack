# Stage V V7.3 Fresh Qualification Handoff

Date: 2026-08-12
Status: `FAIL_SEALED_NON_CONSUMABLE`
Protocol: `STAGE_V_V7_FRESH_QUALIFICATION_V1.3`

## Decision

V7.3 is sealed as a fresh-clean qualification quota failure. The frozen
qualification target was 10 qualified parents per suite. The run exhausted all
60 `libero_goal` candidates without producing a qualified parent, so V7 is not
authorized to produce the formal 24/8/8 parent split. No V7 rows, V_phys labels,
Teacher, Student, Scheduler, or protected evaluation artifacts are consumable.

This is a qualification failure, not a refutation of the vulnerability or
Teacher–Student hypotheses: the downstream scientific stages were not run.

## Bound provenance

- Runtime source commit: `f91182c5650fc0a2339b5e68f8cfb78c69231c6c`
- Runtime source tree: `8b6246726d0b23fde42d51729b193f5522ac9534`
- Candidate manifest:
  `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_V7_FRESH_CLEAN_QUALIFICATION_CANDIDATE_MANIFEST_V4_POST_V7_FAIL3_20260812T033000Z.json`
  - SHA256: `643b6093074b0a1856a8a26da5fec1a77d9ef5e8475f6374079c0d4ae90c5b51`
- Run root:
  `/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_V7_FRESH_QUALIFICATION_V1_3_20260812T034000Z`
- Execution used GPUs `0–7`, eight project workers, `MODE_B_THROUGHPUT_SCIENCE`,
  eager attention, and the authorized 20 GiB minimum free-memory gate. Foreign
  processes were not terminated.

## Results

| Suite | Candidate identities attempted | Report-qualified | Independent recomputation | Frozen target |
|---|---:|---:|---:|---:|
| `libero_10` | 30 | 10 | 12 | 10 |
| `libero_goal` | 60 | 0 | 0 | 10 |
| `libero_object` | 20 | 10 | 15 | 10 |
| `libero_spatial` | 20 | 10 | 10 | 10 |
| **Total** | **130** | **30** | **37** | **40** |

The independent audit found 260/260 queue entries `DONE_VALID`, no duplicate
parent keys, and no worker errors. The failure is therefore the exhausted
`libero_goal` quota, not an unresolved queue or runtime-integrity error.

Run receipts:

- Report: `CONTROL_QUALIFICATION_REPORT.json` — SHA256
  `ac3784eccd11968ac9e9e7454f34197718ed4dd1fb108dafcbeacc1bff656b68`
- Independent audit: `CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json` — SHA256
  `347b0208c66facbd086aa111655a221d7f8d2f97da5abb3e8aea9be3de8312fa`
- Failure seal: `V7_FAILURE_SEAL.json` — SHA256
  `ffce29d29ecfdb73a8d3645e6575608c30b3d42ff5d115233108825a06c21774`
- Failed-attempt exclusion: `V7_FAILED_CLEAN_ATTEMPT_EXCLUSION.json` — SHA256
  `8275790089bf081e72238b748511cb92aba73240645e7cddea2ecc7a195e7140`

## Exclusion and safety boundary

The 130 attempted identities were added to the cumulative clean-attempt
exclusion with zero overlap against prior exclusions:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_V_CUMULATIVE_CLEAN_ATTEMPT_EXCLUSION_V7_POST_V7_FAIL4_20260812T212000Z.json`

SHA256: `b6576ca6f3ea6ebee9b6a2159018ead81eda5eef3bec2f016d5786ad3c05f918`
Union status: `PASS`; parent count: `887`; overlap count: `0`.

The failure seal is non-consumable, forbids resume/rerun of the same root, and
does not permit threshold relaxation or pool manipulation. `eval160_reads=0`,
`protected_eval_reads=0`, and `vis_pgd_attack_rollouts=0`. No M4, Teacher,
Student, Scheduler, or attack rollout was launched.

M3.5 V1.4.2 remains the last valid upstream gate. Any further qualification
attempt requires a new prospective protocol/decision and must use the updated
cumulative exclusion; this handoff does not authorize one.

## Post-seal implementation audit

The sealed root remains immutable and non-consumable. A subsequent source audit
found that the V7.3 runtime helper at `f91182c5` did not implement the frozen
qualification contract: both the producer and independent auditor treated
`terminal_outcome` and `terminal_state_sha256` equality as hard gates even
though `terminal_hash_equality_hard_gate` was frozen as `false`. The helper also
placed the initial-state identity in that same loop; initial identity remains a
hard gate, but terminal fields are descriptive only.

All 60 `libero_goal` rows recorded terminal-state mismatches. A read-only
recomputation that removes only those out-of-contract terminal gates yields 38
protocol-relevant rows before the frozen per-suite cap of 10; this is diagnostic
evidence only, not a qualification result, because V7.3 ran under the wrong
implementation. Therefore the original seal's quota-failure classification
must not be used as scientific evidence. A new source commit, protocol, root,
fresh candidate universe, and authorization are required before any retry.
