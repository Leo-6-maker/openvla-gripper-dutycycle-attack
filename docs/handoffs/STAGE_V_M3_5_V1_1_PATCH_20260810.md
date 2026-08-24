# Stage V M3.5 V1.1 prospective patch — 2026-08-10

## Decision

GPT review found static P0s in the V1 drafts. They are patched prospectively
in V1.1. V1 remains historical draft material; V6 remains immutable FAIL.

```text
V6 CLOSEOUT                         PASS
V6 FORENSIC INTERPRETATION          PASS
M3.5 SCIENTIFIC DIRECTION           PASS
M3.5 OPERATIONAL CONTRACT           FROZEN_PROSPECTIVE / NO RUNTIME AUTHORIZATION
M3.5 DIAGNOSTIC RUNTIME             NOT AUTHORIZED
V7                                  NOT AUTHORIZED
M4                                  BLOCKED
```

## New frozen prospective files

- `configs/STAGE_V_M3_5_LABEL_AND_QUALIFICATION_CONTRACT_V1_1.json`
- `configs/STAGE_V_FRESH_SCIENCE_PARENT_QUALIFICATION_CONTRACT_V1_1.json`
- `configs/STAGE_V_M3_5_DIAGNOSTIC_PROTOCOL_V1_1.json`
- `src/gripper_attack/stage_v_m3_5_phase_classifier.py`
- `scripts/detector_v5/build_stage_v_m3_5_probe_plan.py`
- `reports/STAGE_V_M3_5_DIAGNOSTIC_PARENT_SELECTION_V1.json`
- `reports/STAGE_V_M3_5_STATIC_INDEPENDENT_AUDIT_V1_1.json`

They are frozen before any M3.5 intervention runtime. A runtime validation
receipt must preserve the contract SHA; a scientific change becomes V1.2.

## P0 closure

1. Freshness binds the mechanically generated V4 exposure union (`53`, manifest
   SHA `b234c896b4eeee862914a717e56a79cefa3ee8ba43fdd4b8c7aafb027ec0a612`,
   union-key SHA `62012ba0246d123fb6caa59a8dbc80bc3bd4fb27ea32a6a6c4f87bb3c111dc4a`)
   and V2 cumulative clean-attempt union (`357`, manifest SHA
   `a8be0582c5cbe3ae2224cf6786ac18a0558f2105962aef73ea69fffd462e81e6`,
   union-key SHA `bbe427d645efcaa683d7cfb305014333317ed9cd1e369a1825406b08c2e6302a`).
   The old 50/117 component manifests and V3/V1 aggregate receipts are
   provenance only, not authoritative V7 inputs.
2. Accounting is explicit: 24 shared control executions plus 72 treatment
   executions equals 96 physical branch executions per parent, while only 72
   treatment rows receive labels. Formal totals are 2,880 label rows and 3,840
   physical branch executions.
3. The 24 probes are selected deterministically from clean-only phase strata
   (`6 × PRE_CONTACT`, `CONTACT_MANIPULATION`, `ENGAGED_LIFT`, `CARRY`) using a
   frozen hash rank. The executable classifier is fail-closed on missing
   geometry/contact telemetry, and the probe builder has no outcome-informed
   backfill.
4. Horizon is dose-specific: treatment delivery, 10-step physical observation,
   and independent official task-consequence horizons are recorded separately.
5. During T3/T5/T10 only the gripper component changes; after treatment each
   branch resumes its own closed-loop policy from its own state. Control replay
   after treatment is forbidden.
6. OPEN/aperture response is a treatment/mediator receipt, not physical
   vulnerability. Physical failure classes now have executable predicates and
   explicit abstains.
7. Repeatability is a hard gate: three valid repetitions must emit the same
   registered class and all treatment repetitions must be compliant. Discordance
   emits `HOLD_STOCHASTIC_INTERVENTION_OUTCOME`.
8. All four suites must have diagnostic coverage; no positive-balance tuning is
   permitted.

## Static audit and runtime boundary

The independent server-side static audit passed `54/54` checks with receipt
SHA `7b46c1feb30cf072a03b71c34a349a1a786e7632fe06f1d484a6f1e70cfa3fbf`.
It was run with exact source commit/tree
`c8f81820a345aa98ff5c386b82bdc67d8333b76b` /
`c9b04bd00911ff8b98623d4347e1205db61e6479` and Python
`/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python`.
The diagnostic parent selection is frozen at 8 exposed parents, 2 per suite,
with zero outcome reads.

No GPU diagnostic has started under V1.1. The diagnostic protocol is frozen for
validation but has `runtime_authorized=false`; its intervention runner is not
yet bound, so no M3.5 runtime is executable under this freeze. No V7 candidate
selection, fresh qualification, formal split, M4 map, Teacher, Student, Stage
O, VIS, or protected Eval160 access is authorized. GPU3 foreign workloads
remain outside the project's control boundary.

The frozen V1.1 protocol SHA is
`604e55408f15f04d9481dd85bf8ee8dd4772e1b5f62d8021b904a60498b34738` and is
recorded in `reports/STAGE_V_M3_5_PROTOCOL_V1_1_SHA256SUMS.txt`. After any separately
authorized runtime, the independent audit must verify the compliance receipts,
physical predicates, 3/3 repeatability, and all-suite coverage without changing
the frozen SHA.
