# Execution Authorization V1

Status: PLANNING_ONLY

## Current State

```text
GATE_A = HOLD
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```

## Gates

| Gate | Requirement |
|---|---|
| GATE_A | planning protocols and branch ancestry complete |
| GATE_B | Label V2 provenance, leakage, and manual audit pass |
| GATE_C | detector preregistered thresholds pass |
| GATE_D | exact-prefix and telemetry smoke pass |
| GATE_E | parent panel, conditions, seeds, and statistics frozen |
| GATE_F | explicit experiment authorization |

## First Authorizable Work

After Gate A review, the first possible execution scope is Label V2 builder and
CPU validation. Detector training, rollout, attack launch, and server GPU jobs
remain unauthorized.
