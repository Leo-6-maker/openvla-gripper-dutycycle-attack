# C3-S3-I1 Audit Contract Hardening

Status: `PASS`

Code snapshot: `f63486394212ad25ec0e9cd4d2c5ee0de94d2c31`

The I1 gate is closed on the official A800 environment with:

```text
49 passed
0 failed
0 errors
0 skipped
```

The test run used `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800` and covered the C3-S3 geometry contract, H0 contracts, fail-closed allowlist handling, sealed-root closure, exact step joins, independent computation-chain declarations, quaternion sign equivalence, relation coverage, numerical denominators, threshold violations, NaN rejection, and articulated-unknown HOLD behavior.

The frozen numerical contract is `configs/C3_S3_NUMERICAL_THRESHOLDS_V1.json`, sourced from `CODEX_DETECTOR_COMPLETION_PLAN_V1.md`. The final numerical gate directly evaluates:

- static position maximum `<= 1e-6 m`;
- static rotation maximum `<= 1e-6 rad`;
- dynamic position p99 `<= 1e-4 m`;
- dynamic rotation p99 `<= 1e-3 rad`;
- all static and dynamic denominators `> 0`;
- exactly 44 supported `(task_key, relation_index)` keys with no missing or extra key.

I1 consumed no episode semantics and did not access CAL, G10, T2R-D, FIT-DEV, model inference, training, rollout, or attack data. I2 metadata-only FIT input census is authorized by the gate sequence.
