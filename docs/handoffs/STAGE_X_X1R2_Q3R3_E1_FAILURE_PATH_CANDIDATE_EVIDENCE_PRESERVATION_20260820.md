# STAGE X X1R2 Q3R3 E1 failure-path candidate evidence preservation — 2026-08-20

## Decision

Status: `STAGE_X_X1R2_Q3R3_E1_FAILURE_PATH_CANDIDATE_EVIDENCE_PRESERVATION_PASS`

E1 repaired only the failure-path evidence observability gap. It did not alter
candidate generation, candidate order, selector gates, attack objective,
projection, budget, route, or any scientific estimand.

E0 remains exactly:

`HOLD_Q3R3_E0_FEASIBILITY_EVIDENCE_INSUFFICIENT`

Strict arm-isolation feasibility remains unresolved. E1 is not a feasibility
result and does not authorize a new TRUE arm or any scientific escalation.

## E0 publication

The existing E0 artifacts were added to PR #135 append-only. Their raw SHA-256
values matched the pre-publication files and the existing E0 root seal:

- `reports/STAGE_X_X1R2_Q3R3_TAKEOVER_READBACK_V1.json` — `d947f3905a862e5d1223411184fafe114ec5897a490a9598b84dad8a834d19d0`
- `reports/STAGE_X_X1R2_Q3R3_E0_EXACT_ARM_ISOLATION_FEASIBILITY_AUDIT_V1.json` — `4aabadbaf8a262ae39b349fb40905f10e9560cb5f1e5de23411d90723f0e349e`
- `reports/STAGE_X_X1R2_Q3R3_E0_CANDIDATE_MATRIX_V1.csv` — `b911043ab64746f3f3af367f2fae00bba22918a05dc42b014bf64d0f5d7511fc`
- `docs/handoffs/STAGE_X_X1R2_Q3R3_E0_EXACT_ARM_ISOLATION_FEASIBILITY_20260820.md` — `3456b4019c8a87dcac877ea5cf639f70695d734af40e55cb97e1a1b3dc2dddc7`
- `reports/STAGE_X_X1R2_Q3R3_E0_ROOT_SEAL_V1.json` — `50bcab9abfb262c5c942593a8eefc01993cfebe201a6447e4d81a91b7ac24db1`

The E0 root seal still binds four scientific artifacts; its sidecar was also
published unchanged.

## Minimal implementation change

`src/gripper_attack/failure_evidence.py` now performs compact, standard-library
only failure serialization. It captures:

- exception `diagnostics`;
- adapter `last_attack_diagnostics` as an independent source;
- semantic equality of both sources;
- six ordered rows `delta0 -> pgd_iteration_5`;
- direct 7-token IDs, arm equality/mismatch, native-open booleans, gripper
  change, and processor-input SHA;
- selected candidate fields, including null on total failure.

The receipt is flushed and `fsync`'d before the exception is re-raised. If the
two diagnostic sources disagree, the receipt becomes an explicit structural
HOLD and does not choose one source silently. Early failures with no candidate
audit remain `candidate_audit_complete=false`; no candidate rows are invented.

The runner invokes this helper only in its existing exception path. The frozen
`_select_strict_arm_candidate` implementation is byte-for-byte unchanged from
the pre-E1 commit. The D2 protocol and selective attack contract are unchanged.

## Verification

- Five focused pure-CPU/mock E1 tests: `5/5 PASS`.
- Standard-library AST parse of changed source/tests: `PASS`.
- Existing selector regression confirms the first valid candidate remains
  selected even when a later valid candidate also exists.
- PR #135 CI: `source-registry`, `detector-v5-cpu`, and `stageb-cpu`: all
  `SUCCESS`.
- No GPU worker, OpenVLA inference, simulator, PGD/backward, new fixture,
  V_phys, Eval160, protected, or attack-outcome read occurred.

Source used for CPU tests and code/E0 publication:

- commit: `94971bd595243a6615d9bb418d841a2962e74cf3`
- tree: `4d483781c3f5b58fb931965a024b592d99f1ba5e`

The later append-only E1 report/root-seal commit will be separately live-
verified; it must not be conflated with the source commit used for CPU tests.

## Mandatory stop

E1 is complete. Stop here.

E1 PASS does not authorize E2, a new permanently-excluded fixture, GPU/model
inference, simulator execution, R0/R1/R2, RAND/SHUFFLED/random-time, V_phys,
Eval160, or protected evaluation. Return to Owner/PI for a separate decision on
whether prospective six-row feasibility evidence may be acquired.
