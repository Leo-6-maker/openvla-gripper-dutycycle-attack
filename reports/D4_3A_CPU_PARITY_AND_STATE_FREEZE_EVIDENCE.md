# D4.3A CPU Parity and State Freeze Evidence

## Execution Provenance

- EXECUTION_BASE_HEAD: 84f11908f1456fba428f03b919199424de6b54dc
- EXECUTION_EFFECTIVE_PATCH_COMMIT: acd8bc52e870f22b0e3dd8741be53a4118670f16
- FINAL_VERIFIED_CLEAN_HEAD: acd8bc52e870f22b0e3dd8741be53a4118670f16
- CLEAN_HEAD_E1_RERUN: PASS (105/105)
- SERVER: klfy-SYS-4028GR-TR2
- PYTHON: /home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python

## Results

- CPU_TESTS: PASS (105/105)
- D4.2c_PARITY: PASS (46/46 ALL GATES)
- STATE_FREEZE: PASS (402+98=500, 10x50, 4 canary, 30 panel)

## D4.2c Parity Details

- step_mismatches: 0
- feature_mismatches: 0
- norm_feat_mismatches: 0
- max_norm_feat_diff: 0.00e+00
- mlp_score_mismatches: 0
- max_mlp_score_diff: 5.00e-07
- emit_mismatches: 0
- skipped: 0
- n_traces: 46
- ALL GATES: PASS

## Frozen Canary States

- milk_s23
- salad_dressing_s19
- bbq_sauce_s36
- tomato_sauce_s6

## Frozen Panel States

10 tasks x 3 states = 30 states (see d4_shadow_state_manifest.csv)

## Status

- LIVE_CANARY: NOT RUN
- PANEL: NOT RUN
- ATTACK: NOT RUN
- ONLINE_TRAINING: NOT RUN
- EARLY_TRIGGER_SAFETY: NOT ESTABLISHED
