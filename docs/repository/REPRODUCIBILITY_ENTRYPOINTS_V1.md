# Reproducibility entry points V1

Status: `CODE_R2_REPRODUCIBILITY_ENTRYPOINTS_PASS`

Run these commands from the repository root. They are CPU/static and
read-only with respect to sealed scientific artifacts. Passing them establishes
repository consistency only; it is not scientific execution authorization or
a new scientific result.

## Canonical current commands

### Static paper evidence and immutable-path audit

```text
python scripts/repository/audit_immutable_authority_paths.py
```

Expected terminal marker:
`CODE_R1_AUTHORITY_FIREWALL_PASS entries=88 sidecar_pairs=14 git_objects=2`.

This checks committed and working-tree Git blobs, historical declared digest
bases, sidecars, required Git objects, ancestry, and protected paths. It does
not reinterpret claims.

### Paper V1 claim audit

```text
python scripts/paper/check_paper_v1_claims.py
```

Expected terminal marker:
`PAPER_V1_CLAIM_AUDIT_READ_ONLY_PASS claims=48 eval160=UNREAD protected=UNREAD`.

This reconstructs claims from the manuscript and compares them with the sealed
ledger. It does not call the historical V1 sealer or write under `paper/`.

### Stage X reproducibility and audit-only contracts

```text
python -m pytest -q tests/stage_x tests/test_stage_x_primary_matrix_runner.py
```

These tests use CPU tensors, fixtures, and mocks. They do not load OpenVLA,
invoke a simulator, generate attacks, or call a real environment. Historical
Stage X `run_*`, `freeze_*`, `repair_*`, `seal_*`, and artifact-producing
`audit_*` scripts are not canonical commands and remain governed by their
sealed protocols and terminal HOLD/PASS boundaries.

### Stage Z static authority/preparation contracts

```text
python -m pytest -q tests/stage_z/test_stage_z_preparation.py
```

This synthetic test verifies the 7-D gripper-only intervention contract,
OPEN semantics, queue/replan handling, fixed population accounting, zero
execution counters, and execution-disable guards. It does not run a policy,
model, GPU, simulator, `env.step`, Z1, or scientific rollout.

Also run the immutable-path audit above before interpreting Stage Z authority.
The current root remains a Z0R2 authority HOLD, not a result.

### Paper V2 evidence export

Status: `NOT_AVAILABLE_UNTIL_CODE_R5`.

CODE-R5 will add a new deterministic export command and output namespace. This
document must be updated when that command exists and passes its golden test.
Until then, do not substitute a Paper V1 builder or rerun the immutable F1T
synthesis producer.

## Evidence reading order

1. Verify bytes and paths with the immutable authority audit.
2. Read `paper/PAPER_V1_EVIDENCE_AUTHORITY_MAP_V1.json` and the source root
   seals it names.
3. Verify Paper V1 wording with the read-only claim checker.
4. Read `reports/STAGE_X_X1R2_F1T_ROOT_SEAL_V1.json` for the terminal Stage X
   boundary.
5. Read `reports/STAGE_Z_Z0R2_ROOT_SEAL_V1.json` before any Stage Z discussion.
6. Treat CI and these commands as engineering checks, never as replacements
   for sealed scientific gates or PI authorization.

## Explicitly outside these entry points

Do not use this document to authorize model loading, GPU work, simulator use,
`env.step`, adversarial generation, `V_phys`, Eval160/protected reads, Stage Z
Z1, F1/F1-D/BRIDGE reopening, scientific reruns, or modification of sealed
authority paths. Those operations require a separate prospective authority and
are prohibited in the repository-hygiene lane.
