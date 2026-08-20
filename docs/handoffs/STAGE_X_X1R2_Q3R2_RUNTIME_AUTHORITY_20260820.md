# Stage X X1R2 Q3R2 Runtime Authority — 2026-08-20

## Decision

`STAGE_X_X1R2_Q3R2_RUNTIME_AUTHORITY_PASS`

This is a static provenance pass only. It does not authorize a scientific
population, model inference, simulator construction, `env.step`, PGD, physical
intervention, V_phys read, Eval160 read, or protected evaluation.

## Exact bindings

- GitHub PR: `#135`
- reviewed runtime-code source: commit `85fa8e678ca599f21f5a69d180c7179f9ef99478`, tree `f6555a5d49dda45f29ef64ca8ae4b65b7b08d3f9`
- deployment checkout observed during audit: commit `3e3be4b678ac6e41071698488aebe3ece10099ed`, tree `fbfee8c0fe967f0c5f9aa2666979dfd4ff6b9633`
- runtime-code files were checked by exact Git blob against the reviewed source identity
- victim contract: `configs/STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1.json`, Git blob `b46af63d37ae7e26bf095c8492084cf9860e64a4`, server raw SHA256 `37f5d4ea205835fedc8f4d02c3e722c80581aa5c04d9349c7eed0dd4f6364265`

The deployment checkout contains only authority artifacts after the reviewed
runtime source; the runtime semantics remain bound to the exact reviewed
runtime-code commit/tree above. Any later executable runner must declare its
own exact source commit/tree and runner blob.

## Model and Student closure

All four suite model directories matched the frozen contract by file count,
total bytes, canonical tree SHA, and key-file SHA:

| suite | files | bytes | tree SHA |
|---|---:|---:|---|
| libero_10 | 18 | 15085093727 | `4a83f512232909d34ec2f835acf492713b4c174f0b016ac00cbb330ed5ff8dbd` |
| libero_goal | 19 | 15085095390 | `5354cfe948abd56789ea3b50976fb3693d68a8b617771ca0db8fee368dfd542d` |
| libero_object | 19 | 15085095882 | `f3e5c61db14bd2670e98ea742bfec6baace25533ce8ad2c11685d68e20957f6c` |
| libero_spatial | 19 | 15085095735 | `b3faccff2e0c1b401973aca6e12e98ae23482441d85199ae9507251ac1dea1b5` |

Frozen Student bindings remain unchanged: 25D input, hidden 64, causal RFs
32/128, runtime heads `physical_criticality`, `k10_feasible`,
`safe_release`, `instability`, `gripper_closing_state`, thresholds 0.55/0.80,
and one-shot scheduling. The historical alias remains
`k10_feasibility -> k10_feasible`; no detector or threshold change was made.

## Environment facts sealed, not normalized away

Invocation remains:

`/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python -s`

The executable resolves to `/home/sz/miniconda3/envs/hallo/bin/python3.10`,
with Python `3.10.16`, official prefix, `include-system-site-packages=true`,
Torch `2.2.2+cu118`, Transformers `4.40.1`, NumPy `1.26.4`, and pip-freeze
SHA `8572a1662921ab187d43b7d6353b7bb0b918ddb57c5d358885cf5c1626df3deb`.
The project editable source path is present in `sys.path`; the clean runner's
repo-first insertion rule is part of the bound runtime and must be preserved.
Robosuite's private macros file is absent and its import warning is retained
as an explicit environment fact.

## Audit receipts

- durable PASS report:
  `/mnt/sdc/dty_user/openvla_attack_outputs/n5/stage_x1r2_q3r2_runtime_authority_20260820/STAGE_X_X1R2_Q3R2_RUNTIME_AUTHORITY_AUDIT_V1.json`
- PASS report SHA256:
  `c5047677081a5e1ace586b3f10efb9b7e98fe1b9ab9c79a10d27759d94a79ecd`
- first corrected-run HOLD report is preserved, not deleted:
  `FIRST_RUN_HOLD_RUNTIME_SOURCE_BINDING.json`, SHA256 `2e89ebd70f12480bb087c8f638cc0ea64d6557aa928d8485080bd67a56901ee8`
- final audit errors: `[]`

Protected counters remain zero. Eval160 and protected evaluation remain
`UNREAD`.

## Next legal gate

Proceed to `STAGE_X1R2_Q3R2_ENGINEERING_FIXTURE_POOL_FREEZE`: freeze a new,
outcome-blind, permanently excluded multi-candidate fixture pool for the four
suites. Do not use Q3-F01/Q3-AR-F01, any scientific parent, Student score,
clean outcome, attack result, or protected identity for fixture selection.
