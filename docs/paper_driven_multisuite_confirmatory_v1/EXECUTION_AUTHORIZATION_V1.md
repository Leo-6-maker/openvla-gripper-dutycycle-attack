# Execution Authorization V1

Status: PLANNING_ONLY

## Current State

```text
GATE_A1_LABEL_SPEC = PASS
LABEL_V2_IMPLEMENTATION_AUTHORIZATION = AUTHORIZED_CPU_ONLY
LABEL_V2_BUILD_EXECUTION_AUTHORIZATION = NOT_AUTHORIZED
GATE_A2_DETECTOR = HOLD
GATE_A3_ATTACK = HOLD
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```

## Gates

| Gate | Requirement |
|---|---|
| GATE_A1_LABEL | source manifest, Label V2 schema, coordinate semantics, builder root, manual audit sampling, leakage checks |
| GATE_A2_DETECTOR | unique main detector, populations, splits, loss, normalization, threshold, baselines, seeds, Gate C criteria |
| GATE_A3_ATTACK | exact-prefix snapshot, atomic matrix, baselines, CQ formulas, artifact schema, compute budget, retry/abort |
| GATE_B | Label V2 provenance, leakage, and manual audit pass |
| GATE_C | detector preregistered thresholds pass |
| GATE_D | exact-prefix and telemetry smoke pass |
| GATE_E | parent panel, conditions, seeds, and statistics frozen |
| GATE_F | explicit experiment authorization |

## Authorization Template

Each authorization must specify:

```text
authorized_commit_sha
input_manifest_sha
source_roots
output_root
allowed_commands
cpu_limit
gpu_limit
maximum_jobs
maximum_runtime
maximum_storage
retry_eligibility
terminal_failure_rule
abort_conditions
authorization_expiry
authorization_scope
```

## First Authorizable Work

After Gate A1 review, the first possible execution scope is Label V2 builder
and CPU validation. Detector training, rollout, attack launch, and server GPU
jobs remain unauthorized.

## Authorization Records

- `LABEL_V2_IMPLEMENTATION_ONLY_AUTHORIZATION_V1.md`
