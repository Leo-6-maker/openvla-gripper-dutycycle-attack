# Stage V M4 stale-authorization supersession

Status: `PASS_SAFETY_HARDENING_FORMAL_M4_STILL_HELD`

The historical matched-action protocol and authorization remain immutable
evidence, but they are no longer a consumable formal-M4 authority under the
new governance path. The append-only machine gate binds:

- historical M4 protocol SHA256 `b70a8614938a33a7e8c3ba19de6aa8b37fede67f109a66405071d675cf92c795`;
- historical M4 authorization SHA256 `91a62f446fde33d0fd1aae32f6150e83da8ef371231f49eef2416cc571f52dbd`;
- formal A/B reconciliation SHA256 `1de1d0d4ca1be7cb8e1acc82d5033f0c8521885525493381bc8ff065d831162`;
- status `HOLD_FORMAL_M4_CORRIDOR_INSUFFICIENT`;
- stable count `29`, required count `40`, exact stable keys, the 10-per-suite
  and 24/8/8 split contracts, and zero protected counters.

Machine gate:
`configs/STAGE_V_M4_FORMAL_AUTHORIZATION_SUPERSESSION_HOLD_V1.json`
(SHA256 `6677d3e847b4913944afa008a9d32f17ba8893dd6ff03e0ab3d5a04a31fd4f13`).

Both the formal parent runner and formal-M4 authorization issuer now require a
new `PASS_FORMAL_M4_CORRIDOR` receipt bound to the final 40-parent manifest,
split, corridor protocol, corridor authorization, reconciliation, source
commit/tree, and zero protected counters. The old protocol is rejected as
`M4_FORMAL_AUTHORIZATION_SUPERSEDED_BY_CORRIDOR_HOLD`; the static audit receipt
is `FAIL_STATIC_CONTRACT` with `formal_corridor_gate_bound=false`.

Static negative-audit receipt:
`reports/STAGE_V_M4_STALE_AUTHORIZATION_STATIC_AUDIT_V1.json`
(SHA256 `328284ff296d0c09bce3a80c590167e35cdde74d58595b73d948b6dde10504d2`).

No formal intervention, label generation, protected read, Teacher, Student,
Scheduler, Timing, or VIS rollout was started. Verify the current PR tip from
Git metadata; this handoff does not self-reference a future tip.

Next legal action: rebuild the prospective freshness inventories, freeze the
outcome-blind replenishment pool, and obtain fresh V7-style qualification plus
independent corridor A/B PASS evidence. The malformed reserve A3/B3 root
metadata remains non-consumable and must not be repaired or rerun in place.
