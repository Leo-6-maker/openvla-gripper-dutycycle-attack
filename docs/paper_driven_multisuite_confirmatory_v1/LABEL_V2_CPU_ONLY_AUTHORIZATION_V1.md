# Label V2 CPU-Only Authorization V1

Status: PLANNING_ONLY_AUTHORIZATION_RECORD

```text
GATE_A1_LABEL = PASS
AUTHORIZED_SCOPE = CLEAN2000 Label V2 builder + CPU validation + manual audit preparation
EXPERIMENT_AUTHORIZATION_STATUS = NOT_AUTHORIZED
```

| Field | Value |
|---|---|
| authorized_commit_sha | `3bcc0685f2921e77f17f1973e020acd9247389d6` |
| input_manifest_path | `tables/server_freeze/clean2000_teacher_source_availability.csv` |
| input_manifest_sha | `09cefaa3f50d552adde6f3040fe25e11e295b8f97f71f05745d8cec710b5d962` |
| source_roots | read-only `/mnt/sdc/dty_user/openvla_attack/evidence/CLEAN2000_CANONICAL_V1`; verified backup `/data/liuyu/openvla_gripper_freeze/20260702_codex_verified_v3` |
| output_root | new empty directory under `outputs/clean2000_teacher_labels_v2/` |
| allowed_commands | `python tools/multisuite_detector/build_clean2000_label_v2.py ... --dry-run`; CPU validation commands for row count, SHA, leakage, and crosstab only |
| cpu_limit | CPU only |
| gpu_limit | 0 GPUs |
| maximum_jobs | 1 builder process plus CPU validators |
| maximum_runtime | 2 hours |
| maximum_storage | 2 GB |
| retry_eligibility | retry only after deleting the new failed output directory |
| terminal_failure_rule | any SHA mismatch, row-count mismatch, leakage finding, or unexplained row aborts authorization |
| abort_conditions | source mutation, symlink/path traversal, non-empty output root, GPU allocation, OpenVLA inference, rollout, attack code invocation |
| authorization_expiry | next protocol revision or 2026-07-10, whichever comes first |
| authorization_scope | Label V2 builder, CPU validation, manual audit sample materialization |

Not authorized: detector training, OpenVLA inference, simulator rollout, attack
execution, GPU jobs, source artifact mutation.
