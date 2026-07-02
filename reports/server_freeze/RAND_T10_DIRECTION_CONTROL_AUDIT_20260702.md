# RAND_T10 Direction Control Audit 2026-07-02

Source: server episode_summary.json files under the authoritative Object evidence root.

## Accounting

- RAND_total: 162
- RAND_emitted: 141
- RAND_no_emission: 21
- RAND_attack_applied_field_true: 0
- RAND_attack_frames_positive: 141
- RAND_success_total: 162
- RAND_success_with_attack_frames_positive: 141
- RAND_success_without_attack_frames_positive: 21
- RAND_failure_total: 0

## Finding

The raw server summaries contain a field-level ambiguity:

- `attack_applied` is `False` for all 162 RAND_T10 summaries.
- `attack_frames=10` is present for 141 RAND_T10 summaries.
- `mlp_triggered=True` is present for the same 141-row emitted set.

Therefore this audit does **not** mark RAND_T10 protocol semantics sealed. The accounting supports 162/162 task success and 141 rows with positive attack-frame telemetry, but the implementation definition still requires config/manifest/launch evidence before calling it a sealed random-direction control.
