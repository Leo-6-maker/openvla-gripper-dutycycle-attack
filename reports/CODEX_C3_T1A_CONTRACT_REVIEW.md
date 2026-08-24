# C3-T1A V23 semantic contract review

Status: `PASS`

Code snapshot: `c8632e9af3ab79803cd6a663b8d6d7cec8780076`

The V23 contract now uses `safe_release_computed` as the sole K10 input. The
right-censor helper separately consumes `observed_future_steps_available` and
returns `UNKNOWN` for null, negative, boolean, string, NaN, Inf, or insufficient
observed future steps. Protocol horizon boundaries 0/9/10 and the
safe-release/K10 invariant are covered.

Validation:

- local official-contract and V23-runner tests: `16/16 PASS`, `0 fail`, `0 error`;
- official A800 environment (`Python 3.10.16`, `torch 2.2.2+cu118`): `16/16 PASS`;
- Python compilation: `PASS`;
- JSON parsing: `PASS`;
- legacy V22 runner: not imported or executed.

No episode payload, model, OpenVLA, rollout, or attack was executed.
