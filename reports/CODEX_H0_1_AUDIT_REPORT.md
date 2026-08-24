# CODEX H0.1 Independent Evidence Audit

Date: 2026-07-26  
Audit branch: `codex/h0-1-independent-audit-20260726`  
Source repository: `Leo-6-maker/openvla-gripper-dutycycle-attack`  
Audit scope: H0 evidence integrity, receipt provenance, C1 source binding, causal-contract tests

## Verdict

```text
H0.1                           = PROVENANCE_FAIL
C3-S2 / C3-G permission        = NOT GRANTED
T2R-D unblinding               = NOT GRANTED
C3-S2/C3-G student work       = NOT STARTED
CAL / G10 / T2R-D data read    = 0
training / inference / attack  = NOT STARTED
```

The H0 receipt's underlying server artifacts are present and their reported
server SHA256 values match the receipt. The H0 receipt itself is not a
consumable proof: its self hash is not reproducible under the explicit H0.1
algorithm, its upstream repository-file SHA bindings are stale, its source
commit is not the current HEAD, and the C1 producer source cannot be tied to
the claimed commit on the server.

## Locked source facts

| Item | Observation |
|---|---|
| Local audit source | `00c54f93c1dbd00b2ea42e34de7cdfbbad9d3756` |
| Local source branch | `deepseek/integration-final-detector-20260724` |
| Audit worktree | clean, independent worktree from the exact HEAD |
| GitHub branch | `deepseek/integration-final-detector-20260724` at `00c54f93...` |
| GitHub PR | no PR returned for this head branch |
| GitHub checks | no check-runs/status contexts returned for this SHA; commit prose claiming CI green is not a current GitHub check record |
| Server host/user | `pm-364c0001` / `dty_user` |
| Server repo HEAD | `68a8af0dc73ddb54c31fef57fa49597200b09533` |
| Server worktree | dirty; unrelated untracked/modified files observed |
| Server Python | 3.10.16 in `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800` |
| GPU task started by this audit | 0 |

The dirty DeepSeek checkout was not edited. All audit changes are isolated
to the audit worktree.

## Receipt integrity

H0.1 defines the new receipt algorithm as:

```text
remove top-level self_sha256
json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
SHA256(UTF-8 bytes)
```

For the historical `H0_RECEIPT.json`:

```text
claimed self_sha256       = fc3206f240e29fbaeb1e5acd77be907993ac4041b3c73cfbf95ae09d685d0a2b
H0.1 recomputed SHA       = f1f51afec7f4d4ef065a9dbedb3ddfcc122364a5b9c19e10f1ff9da619ecca7b
algorithm field present   = no
result                    = FAIL / historical algorithm unspecified
```

The receipt's upstream file bindings also do not match the files at the
current source commit:

| Binding | Receipt SHA | Current file SHA | Match |
|---|---|---|---|
| `N5_LONGRUN_PLAN_V4.json` | `745b1bf9...` | `f893fa6f...` | no |
| `C1_RECEIPT.json` | `1d9183b3...` | `6e2d5e71...` | no |
| `C2_FINAL_RECEIPT.json` | `b6ab9748...` | `3cf31165...` | no |
| `C2_TASK09_FORENSIC_RECEIPT.json` | `d63f247f...` | `ce5c3014...` | no |
| `C3_S_V1_RECEIPT.json` | `4199f5f9...` | `0ed382c5...` | no |

The four underlying server artifact hashes do match the receipt's server
bindings:

```text
ENTITY_REGISTRY_SUMMARY.json = c1ebad4cfc9dc44db84a2f89423c39540edb70bc2ba711d427bf0148bfcbad77
C2R0_REPORT.json             = f7de7c4413a2f5a92d5374938b4a0ec7da2982540294c6ea234c295547a93c76
C2R02_REPORT.json            = 4de9c3edf5c069faccbd46f7e4336f2c69c3b48a3633532e53d889d5d28fd166
STATIC_FIXTURE_SEAL.json    = 01474c45604d6c0195f5c9aaded4fbd80b143da2d150fa5a6f6a71673cde0182
```

That is content availability, not a valid source-to-receipt closure.

## C1 forensic result

The server C1 summary contains:

```text
n_tasks                   = 40
n_ok                      = 40
n_env_errors              = 0
n_blocked_resolutions     = 0
n_unresolved_supported    = 0
n_supported_placement    = 38
n_articulated_unsupported = 2
top-level status           = FAIL
```

The producer source computes:

```python
all_pass = (... and n_ok + n_articulated == 40)
```

With the observed values this is `40 + 2 == 40`, hence false. The per-task
rows are a plausible content-valid diagnostic, but the status is not a valid
PASS. More importantly, the receipt claims source commit `b5c9634`, while:

```text
server repo HEAD                   = 68a8af0...
b5c9634 resolvable in server repo = no
server copied source SHA256        = b0567f3d6fc2eee2af116a8b587471d05a65c634e6ab08a6d50832b04a2af603
local b5c9634 source SHA256        = 1d9218afc6f8f223a3129934fddaacf0634337936bf92c0d487ae35570322b1c
```

The C1 artifact must remain `C1_PROVENANCE_UNRESOLVED` until reproduced on a
clean, source-bound checkout. No identities or labels need to change for
that rerun.

## Protocol and state-machine result

`PROTOCOL_AMENDMENT_V3.json` declares the 800-identity allocation to be
pre-committed, but the repository chronology only shows the V3 amendment
being first committed at `09faff3` and amended at `205eaf9`; it does not
independently bind a pre-training allocation receipt and source/manifest
closure before the relevant training event. That statement is therefore a
claim, not a proven chronology.

The current plan still says `current_gate=H0_EVIDENCE_BASELINE`, while the
historical H0 receipt says `status=PASS` and `next_gate=C3-S2`. There is no
separate immutable transition receipt. A PASS receipt cannot auto-advance the
plan until its own bindings pass independent review.

The correction is recorded without modifying V3 in:

`reports/PROTOCOL_AMENDMENT_V4_PROVENANCE_CORRECTION.json`

## Strengthened tests

Added, without replacing the historical V1 suite:

`n5/phase3_student/tests/test_h0_contracts_v2.py`

It adds strict duplicate-key parsing, explicit receipt hashing, stale-binding
detection, current-HEAD binding checks, AST ordering for pre-action C2
capture, and numerical causal-prefix coverage for RF32/Dual lengths
`1,2,8,16,31,32,33,64,78,127,128,129` when torch is available. It also
contains the success/terminal perturbation test required to prevent terminal
success from changing a physical Teacher label.

Local results:

```text
historical V1 suite: 18 tests, 1 failure, 10 dependency skips
  failure: H0_RECEIPT upstream C1_RECEIPT SHA mismatch
independent V2 suite: 8 tests, 6 executed, 2 dependency skips
  strict receipt/stale binding checks: detected
  numerical prefix parity: not executed (local torch unavailable)
  dynamic Teacher perturbation: not executed (local numpy unavailable)
```

The V2 suite's receipt tests pass because they assert that the invalid
historical receipt is rejected; they do not promote it.

## Required resolution

1. Reissue H0 evidence with an explicit self-hash algorithm and current file
   bindings, or mark the old H0 receipt historical/non-consumable.
2. Reproduce C1 on a clean checkout and bind source HEAD, worktree status,
   full source-script SHA, input SHA, and output SHA.
3. Add an immutable plan-transition receipt from H0.1 to C3-S2.
4. Run the numerical prefix and Teacher perturbation tests in the official
   environment before any C3-S2 work.

Until these are complete:

```text
H0.1 = PROVENANCE_FAIL
C3-S2 / C3-G = HOLD
C3-S2 PASS = not proven
T2R-D reveal = forbidden
student training/inference = not authorized
```

## Audit outputs

```text
reports/CODEX_H0_1_AUDIT_REPORT.md
reports/C1_PROVENANCE_HOLD_PACKET.json
reports/PROTOCOL_AMENDMENT_V4_PROVENANCE_CORRECTION.json
n5/phase3_student/tests/test_h0_contracts_v2.py
```

No production artifacts, protected split data, model outputs, or server
worktree files were modified.
