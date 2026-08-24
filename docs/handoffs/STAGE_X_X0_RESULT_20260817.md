# Stage X X0 result — 2026-08-17

X0 status: `STAGE_X_PHYSICAL_DUTY_CYCLE_MECHANISM_SUPPORTED`.

The sealed read-only result root is:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_X_DUTY_CYCLE_MECHANISM/STAGE_X0_RESULT_20260817T095900Z`

Its result `ROOT_SEAL.json` binds summary SHA256
`ff2e18c905a108cb51dbecf82473d1cd4e301e02a86c5e87bf39aab723fd35af` and
`SHA256SUMS` SHA256
`fb8da5b1f9ce30bef7884563ec9579c1c9c7d3c2e3fb749efc89c56be3d6fbd1`.
`PROVENANCE.json` is present. The exact mediator availability root is:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_X_DUTY_CYCLE_MECHANISM/STAGE_X0_MEDIATOR_AVAILABILITY_20260817T094900Z`

The availability audit covers 40 Stage V parents plus 16 Stage VI-B2 parents,
5,376 four-arm branches, and 1,344 four-arm probe groups. M1 commanded open
fraction, M2 aperture excess over exact relative-step overlap, M3 contact loss,
and M4 object displacement are exact and available. The frozen downstream
task-failure taxonomy is `NOT_AVAILABLE`; it was not reconstructed from
`V_phys`.

Among 1,344 dose rows per dose, consumable `V_phys` rows are T3=1,245,
T5=1,191, and T10=1,126. Their raw positive rates are 0.39438, 0.67758,
and 0.87300. Complete three-dose probe patterns contain 1,126 rows and are
all monotone (`000/001/011/111`); no nonmonotone pattern was observed.
Parent-bootstrap uncertainty uses 2,000 replicates with seed 20260817; no
iid-row confidence interval is used.

The descriptive mechanism chain is consistent with dose escalation:
command delivery is exact for eligible rows, aperture excess increases from
T3 to T5 to T10, contact-loss incidence increases, and object displacement
increases. This is descriptive evidence only, not a formal mediation claim.

No new environment step, PGD, physical intervention, or protected read was
performed. `Eval160=UNREAD`, `protected evaluation=UNREAD`, and all protected
counters are zero.

X0=A permits the frozen clean no-environment X1 sequential-PGD diagnostic.
It does not authorize X2 physical PGD. Teacher/Student, Stage IX negative
conclusions, and all prior estimands remain unchanged.

Bound source for the X0 analyzer:

- commit `7c36489a262ee5da4826936da4520608eb30fe46`;
- tree `f2c9d01d72bcded05fbf887cdf2b4cfe96f18dbb`;
- official environment `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`.
