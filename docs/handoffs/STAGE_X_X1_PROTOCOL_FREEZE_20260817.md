# Stage X X1 protocol freeze — 2026-08-17

X0 is sealed as `STAGE_X_PHYSICAL_DUTY_CYCLE_MECHANISM_SUPPORTED`. This
checkpoint freezes the next diagnostic only:

`configs/STAGE_X_X1_SEQUENTIAL_PGD_PROTOCOL_V1.json`

X1 is a clean-observation, no-environment-step sequential PGD persistence
audit. It binds the Stage IX canonical PGD contract byte-for-byte and forbids
fallback routes, hyperparameter changes, repeated-frame padding, attacked
frames, physical intervention, and protected reads.

The input audit must first establish whether the consumed clean snapshot roots
contain exact consecutive images/processor tensors from each probe time. The
`clean_reference_action_window` is an action record, not an observation; it
cannot be counted as a sequence of frames. Missing consecutive frames are
`NOT_IDENTIFIABLE`, never synthetic repeated frames or negative PGD outcomes.

For each valid clean sequence, X1 records per-frame clean/attacked margins,
target rank, targeted OPEN success, decoded action, norm, iteration count, and
process/GPU identity. Q1/Q2/Q3 are evaluated for d={3,5,10}, with T5 primary
and T3/T10 supporting. All uncertainty is parent-resampled; no iid-row CI is
used.

X1 may only unlock a fresh X2 authorization if its frozen gate passes and X0
remains A. This protocol itself does not authorize X2, physical PGD, Eval160,
or protected evaluation.
