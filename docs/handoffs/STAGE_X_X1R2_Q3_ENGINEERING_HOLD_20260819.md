# Stage X1R2 Q3 engineering hold

Status: `STAGE_X_X1R2_Q3_HOLD_ARM_TOKEN_ISOLATION_AFTER_EXPOSURE`

Q3-F01 (`M007`, `libero_10/task_04/state_30`, ordinal 5) was run on physical GPU1 with the official A800 environment and exact source `10b5d24ca206f8d42b93950d7db5b46a7c5625ec` / tree `106482d00584073252fb48e03895ea4712499cd0`.

The clean engineering episode materialized. The fixed `TRUE_PGD_T5_ENGINEERING` branch then performed model inference and one accepted PGD result with six loss forwards and five backward calls. At step 90 the runner raised `ARM_TOKEN_ISOLATION_FAIL:90`. The attacked action was materialized, but no attacked `env.step()` started or completed. No V_phys, physical intervention, attack-outcome read, Eval160 read, or protected read occurred.

This is a genuine structural failure after exposure. Q3-F01 is permanently consumed as runtime-invalid for this qualification. It is not rerunnable, and Q3-F02 through Q3-F04 are sealed as not started. No Q3 engineering result is consumable, and no X1R2 scientific population or formal attack matrix is authorized.

The durable evidence remains under `/llm_jzm/dty_user/openvla_attack_x1r2_q3_20260819/fixtures/Q3-F01`; exact artifact hashes are recorded in `reports/STAGE_X_X1R2_Q3_ENGINEERING_HOLD_V1.json`. Eval160 and protected evaluation remain unread.
