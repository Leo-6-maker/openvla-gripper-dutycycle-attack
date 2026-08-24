# Stage X1R2 Q3 arm-repair hold

Status: `OWNER_REVIEW_Q3_ARM_REPAIR_HOLD_CLEAN_EMIT_REPLAY_MISMATCH`

The new permanently excluded engineering fixture `Q3-AR-F01` was started once under the approved repair source and stopped fail-closed before any attack arm. The clean replay returned `PASS_SCREENING_CLEAN_EPISODE` and `clean_success=true`, but its Student trace had no first emit (`None`) while the frozen fixture receipt required `133`:

`Q3_STUDENT_EMIT_REPLAY_MISMATCH:None!=133`

This is a post-exposure runtime/clean-replay mismatch. It does not validate or invalidate the arm-isolation repair because TRUE/RAND/SHUFFLED arms never started.

## Durable evidence

- source commit/tree: `b7237611c466077a9a7e6f0b1102e9176cfa2c88` / `fd5eeef98480b4c608ebd4eafb8e325afa8cd17a`
- physical GPU: `2`; mount free memory: `31270 MiB`
- output root: `/llm_jzm/dty_user/openvla_attack_x1r2_q3_arm_repair_20260820`
- parent receipt SHA256: `ed7143d6bd1c50655fb5b61295ca2ed9ced965069251b366a3ad154600912e38`
- episode manifest SHA256: `80408f3389a43a77d4029155e3dd87c437ea5f8e2f4b4b2a8e765b31b11c90fe`
- clean telemetry SHA256: `7742ac082e772be424aa8c173480f6edf6e64a58a234a2748c1568dc1eb45a3e`
- clean video SHA256: `9945ece6501d22cdfd353786178cff1ff35e70fc70bc4cf7312b19972e642a4a`

Only clean artifacts exist. There is no attack arm directory, attack tensor, or branch receipt.

## Exposure accounting

`openvla_model_inference_calls=256`, `env_step_calls=266`, `pgd_calls=0`, `attack_invocation_count=0`, `attack_backward_calls=0`, `attacked_env_steps=0`, `physical_interventions=0`, `V_phys=0`, and all protected counters remain zero. External GPU processes were not modified.

## Required stop

Do not rerun `Q3-AR-F01`, do not select a replacement, do not start `Q3-F02` through `Q3-F04`, and do not select an X1R2 scientific population. The next action requires PI review of the clean Student emit replay mismatch and its root cause. Eval160 and protected evaluation remain unread.
