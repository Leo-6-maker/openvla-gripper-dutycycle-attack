# Stage V M3.5 V1.4 Gate-B provenance HOLD — 2026-08-12

## Decision

```text
M3_5_LABEL_VALIDATION = HOLD
V7 / M4 / Teacher / Student = BLOCKED
Gate-B partial roots = NON_CONSUMABLE
Eval160 / protected reads = 0
```

The first eight-parent Gate-B launch was stopped before any receipt was
written. Its frozen protocol still bound one historical Gate-A receipt and
audit (`6ce999…` / `fe990b…`), while the newly selected eight Gate-A roots had
different exact hashes. That violates the V1.4 requirement that Gate-B
authorization bind the actual Gate-A PASS receipt and independent audit for
each parent.

The eight runner PIDs were reaped by exact PID only; foreign CAMEF/GR00T
processes were preserved. The partial roots and launch logs remain forensic
artifacts and are not labels.

## Corrective contract

`STAGE_V_M3_5_V1_4_GATE_B_PROTOCOL_FROZEN.json` now uses
`PER_PARENT_EXACT_SHA256` bindings for all eight selected diagnostic parents.
The Gate-B runner and independent auditor both fail closed unless the supplied
Gate-A receipt and independent-audit hashes match the frozen per-parent map.

The corrected run requires a new source commit/tree, exact regression, static
audit, runtime authorization, and a fresh Gate-B root. No tolerance, retry-to-
pass, dose change, horizon change, or partial-label consumption is allowed.
