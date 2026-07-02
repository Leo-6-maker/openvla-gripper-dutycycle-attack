# RAND_T10 Direction Control Audit 2026-07-02

Final classification: RAND_FIELD_CONFLICT_BOUNDED.

RAND_T10 row accounting from frozen Object artifacts:

- total: 162
- emitted: 141
- no-emission: 21
- raw `attack_applied=True`: 0
- `attack_frames>0`: 141
- multi-source attack confirmed: 0
- frames-only supported: 141
- task success: 162
- task failure: 0

Recovered candidate protocol evidence:

- `object_breadth_120.jsonl` contains candidate RAND values: `condition=RAND_T10`, `method=RAND`, `objective=random_linf_control`, `timing=student`, epsilon `0.023529411764705882`, `pgd_steps=20`, `K=10`, and `target_token=31744`.
- Those values are now recorded as `CANDIDATE_PROTOCOL_VALUE_UNBOUND` unless formal Object artifact binding is proven.

Boundary:

RAND_T10 has attack-frame schedule evidence for 141 emitted rows, but no sealed independent numeric perturbation evidence in the current ledger. It is not a fully sealed direction/selectivity control.

Artifact: `tables/server_freeze/rand_t10_episode_accounting.csv`.
