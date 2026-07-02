# RAND_T10 Direction Control Audit 2026-07-02

Final classification: RAND_PERTURBATION_EXECUTION_CONFIRMED_PROTOCOL_PARTIAL.

RAND_T10 row accounting from frozen Object artifacts:

- total: 162
- emitted: 141
- no-emission: 21
- raw `attack_applied=True`: 0
- `attack_frames>0`: 141
- multi-source attack confirmed: 141
- task success: 162
- task failure: 0

Recovered protocol evidence:

- Existing breadth manifest entries bind `condition=RAND_T10`, `method=RAND`, `objective=random_linf_control`, `timing=student`, epsilon `0.023529411764705882` (6/255), `pgd_steps=20`, `K=10`, and `target_token=31744`.
- Frozen formal summaries and telemetry support attack execution for the 141 emitted rows and no attack execution for the 21 no-emission rows.
- The raw `attack_applied` field conflicts with attack-frame/telemetry evidence and is not used as the acceptance criterion.

Boundary:

RAND_T10 is preserved as a historical perturbation control with confirmed emitted-row perturbation execution, but it is not upgraded to a fully sealed direction/selectivity control because the formal Object protocol binding remains partial and the raw writer semantics are not sealed.

Artifact: `tables/server_freeze/rand_t10_episode_accounting.csv` and `tables/server_freeze/object_frozen_master_ledger.csv`.
