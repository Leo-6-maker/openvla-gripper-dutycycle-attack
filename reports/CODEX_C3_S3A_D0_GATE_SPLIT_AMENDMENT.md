# C3-S3A / D0 gate split amendment

This amendment separates fresh geometry numerical validity from Official V3
Clean2000 consumability before any new geometry run.

## C3-S3A — fresh geometry numerical validity

Input is restricted to a newly generated, sealed
`C3_S3_GEOMFIT_SYNTHETIC_V1` root. It contains no Clean2000 payload, episode
history, protected split, policy output, Teacher label, Student output or
attack result.

The fixture must contain exactly 44 supported relations:

- 11 static relations, at least 10 configurations each;
- 31 dynamic relations, at least 100 samples each;
- 2 articulated relations, at least 100 samples each.

The source chain is `OBSERVABLE_LOCAL_POSE_RECONSTRUCTION`. The reference
chain is an independent synthetic MuJoCo world-body pose chain. Articulated
rows are resolved synthetic dynamic rows; `UNKNOWN_ARTICULATED` is not an
acceptable result.

Frozen numerical thresholds remain those in
`configs/C3_S3_NUMERICAL_THRESHOLDS_V1.json`:

- static position maximum: `1e-6 m`;
- static rotation maximum: `1e-6 rad`;
- dynamic position p99: `1e-4 m`;
- dynamic rotation p99: `1e-3 rad`.

The required sequence is static/dynamic/articulated smoke, then independent
`run_A` and `run_B`, then canonical comparison. C3-S3A requires 44/44 exact
coverage, nonzero denominators, zero articulated unknowns, zero protected
reads, identical canonical A/B results, and sealed outputs.

## D0 — Official V3 Clean2000 consumability

`D0 = HOLD` remains unchanged. This amendment does not authorize reading or
relabeling the Official V3 Clean2000 payload, resolving the missing V22-800
intersection, proving FIT-only membership, or entering C3-S3-I3/I4 for the
real dataset.

## Boundaries

The C3-S3A implementation does not import OpenVLA, policy, Teacher, Student,
rollout or attack code. A C3-S3A PASS would authorize only the separately
scoped first-stage C3-G-DEV implementation/testing requested by the gate; it
does not authorize C3-T, Clean2000 relabeling, Student training or attack.
