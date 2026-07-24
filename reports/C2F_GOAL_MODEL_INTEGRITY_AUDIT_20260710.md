# C2F Goal Model Integrity Audit - 2026-07-10

STATUS: PASS_C2F_GOAL_MODEL_INTEGRITY_AUDITED

CPU/static audit only. No LIBERO env, rollout, intervention, attack, or GPU episode was launched.

- model_path: `/mnt/sdc/dty_user/openvla_attack/models/libero-goal`
- file_count: 19
- referenced_shards: 4
- missing_referenced_shards: 0
- processor_class: `PrismaticProcessor`
- model_class: `OpenVLAForActionPrediction`
- norm_stats_keys: `['libero_goal']`
- resolved unnorm_key: `libero_goal`

Manifest: `artifacts/goal_model_manifest.json`
