# Stage IX F0 result

Status: `STAGE_IX_NO_MODEL_SIDE_TIMING_SIGNAL`

The frozen Stage IX F0 protocol was executed without environment steps or
physical intervention. The authoritative aggregate root is:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_IX_FACTORIZED_PGD_TIMING_UTILITY/STAGE_IX_F0_VIS_EXPLOITABILITY_20260816T172453Z`

The root contains 1,344/1,344 rows, eight 168-row worker shards, the frozen
canonical PGD contract, opportunity gate, F0 protocol, provenance, SHA256
sidecars, and a valid `ROOT_SEAL`. The eight worker shards were copied
byte-for-byte from the completed no-env run at
`STAGE_IX_F0_VIS_EXPLOITABILITY_20260816T171025Z`; no worker was rerun for
aggregation.

Bound source:

- commit: `98838b91a34c134c24e430c6b660dfcb33ba6137`;
- tree: `daea7d0eabb89027218869c6b5c1b4035b187f31`;
- F0 runner SHA256: `d335a9d41bf4e28572e6f2146f73fc0cb5ed4c1aa526cab9379ce41d9f630862`.

All three prospective scores failed the frozen gate:

| score | DEVTEST model AUROC | model top-1 lift | factorized parent-macro AUC | factorized top-1 lift | factorized top-3 lift | LOSO mean / worst |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 0.870743 | 1.034704 | 0.483698 | 0.969748 | 0.915874 | 0.484166 / 0.223790 |
| E1 | 0.900510 | 1.034704 | 0.521112 | 0.969748 | 0.969748 | 0.559237 / 0.312899 |
| E3 | 0.897157 | 1.034704 | 0.523390 | 0.969748 | 0.969748 | 0.600697 / 0.236271 |

The model-side targetability signal is therefore not sufficient for the
factorized timing utility. No score satisfies the required factorized
parent-macro AUC, top-1 lift, top-3 lift, suite, and LOSO requirements.

Protected-boundary audit:

- `env_step_with_perturbed_action=false`;
- `physical_intervention=false`;
- `attack_rollouts=0`;
- `vis_pgd_attack_rollouts=0`;
- `env_steps_with_perturbed_action=0`;
- `eval160_reads=0`;
- `protected_reads=0`;
- `Eval160` and protected evaluation remain unread.

This is a valid negative Stage IX development conclusion. Do not create a
physical PGD timing matrix, scheduler, new passive detector, or protected
evaluation from this result. Stage VIII remains immutable with
`STAGE_VIII_R1_NO_GENERALIZABLE_RELATIVE_SELECTOR`; Teacher/Student and all
prior frozen artifacts remain unchanged.
