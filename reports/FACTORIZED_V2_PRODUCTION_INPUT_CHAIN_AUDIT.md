# Factorized V2 production-input chain audit

This is a read-only blocker receipt for the Exact-W32 source roots audited on
the Official V3 A800 server. It is not a production bundle and does not grant
inference, calibration, training, or attack authorization.

The final machine-built audit root is
`FACTORIZED_V2_PRODUCTION_INPUT_AUDIT_CODEX_20260722_V3` with
`SHA256SUMS` digest
`eb15ef2e6f0355fe667cd29166d594e0fb4b74fdc71771401b73d1c940480e71`.

## Observed source roots

- Exact-W32 child coordinates: `o0_i0` through `o3_i2` (12/12).
- Prediction child seals: 12/12 PASS.
- Run/checkpoint child seals: 12/12 PASS.
- Inner-CV split root seal: PASS, SHA256SUMS digest `6689c7...`.
- Factorized Teacher root seal: PASS, SHA256SUMS digest `c97cc6...`.
- W32 parent root: HOLD; top-level `SHA256SUMS` is absent.
- S1 FIT root: PASS, SHA256SUMS digest
  `db5ea2c8a4a24bd50e032e44f4cb54089d131b7497daf4aa731d625b536cb93f`.

## Exact joins

The read-only audit found 200/200 held-out identities and exact prediction,
Student, clean-runtime, and Teacher step joins for every split (12/12).
Every split also has clean action coverage from:

`clean/<identity>/step_records.jsonl:clean_action_raw_7d[6]`

The exact feature-order SHA is
`3d1101d26567a41bf688587a70c5100a3629ad62f12f9568947ee178a8a63366` for all
12 splits. The action field is the direct clean OpenVLA raw action. The
canonical contract is
raw value `< 0.5` = CLOSE, `> 0.5` = OPEN, and the boundary is abstention.
The action field is not postprocessed or attacked. The S1 Student rows and
the Factorized prediction rows do not contain a raw action field; the clean
source must therefore remain an explicitly bound external runtime source.

## Identity and bundle status

The current mounted roots do not provide independent calibrator-fit and
policy-selection identity manifests plus unique sealed OOF prediction roots.
No OOF assignment is invented. The strict identity verdict is therefore
`BLOCKED_MANIFEST_INCOMPLETE` until those sources are mounted and audited.

No canonical runtime, calibration, policy-selection, or evaluation production
bundle was emitted from this blocked state. V3.2 handoff generation is
fail-closed and rejects the current audit.

```text
CODEX_PRODUCTION_INPUT_CHAIN = HOLD
CANDIDATE_CLOSE              = DIRECTLY_AVAILABLE_FROM_CLEAN_RUNTIME
IDENTITY_AUDIT               = BLOCKED_MANIFEST_INCOMPLETE
RUNTIME_BUNDLE               = BLOCKED_ROOT_SEAL
CALIBRATION_BUNDLE           = BLOCKED_INDEPENDENT_IDENTITY_SOURCES
POLICY_SELECTION_BUNDLE      = BLOCKED_INDEPENDENT_IDENTITY_SOURCES
OFFLINE_EVALUATION_BUNDLE    = BLOCKED_ROOT_SEAL
AUTHORITATIVE_L3             = HOLD
FULL_FIT                     = HOLD
ATTACK                       = HOLD
MODEL_INFERENCE              = NOT EXECUTED
TRAINING                     = NOT EXECUTED
ROLLOUT                      = NOT EXECUTED
``` 
