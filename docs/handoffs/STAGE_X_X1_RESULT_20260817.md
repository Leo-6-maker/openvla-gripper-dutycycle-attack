# Stage X X1 result — 2026-08-17

X1 status: `STAGE_X_SEQUENTIAL_PGD_SIGNAL_WEAK` (protocol outcome `X1=C`).

The final sealed result root is:

`/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_X_DUTY_CYCLE_MECHANISM/STAGE_X1_SEQUENTIAL_PGD_20260817T101200Z_RESEALED_V1`

The root seal binds:

- result summary SHA256 `0677a2488d409beb5e5d7f590f1765413e8a8536bed7b97cd668eb453fa91808`;
- `SHA256SUMS` SHA256 `eb389626222c763ab9a7d397c6b481919dedac6ecdf604cab2e9493f4fe36a7e`;
- `ROOT_SEAL.json` SHA256 `d83c4dc71817c39403829176578e574a9290834502d8928daf68e057e19eb680`.

The initial execution root
`.../STAGE_X1_SEQUENTIAL_PGD_20260817T101200Z` is preserved as a superseded
sealing diagnostic. Its scientific files matched, but its live aggregate log
was still being appended when the first `SHA256SUMS` was produced. No worker
evidence was changed or rerun; the re-sealed root excludes that live log.

The frozen X1 protocol and exact source binding are unchanged:

- official environment `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`;
- source commit `d5919552990bc98eb85ee2fbc45715e314d4ef81`;
- source tree `0bcb4f87e1fbb9699039b3708df95988d790c53d`;
- protocol SHA256 `c8fe0fc47040784733645e7dbe4699549bef591904fc7b21775c2b727164b805`;
- source script SHA256 `6bc5046a335d2b372a53f5cd10db58da915397301b6adb9ca0fc5a3c75649fe3`;
- clean-sequence audit SHA256 `d60971b5a70490939b145005b87ee4567d42c75c5e1a0daa2ab6e02604cff301`.

The run used six admitted physical GPUs (`0,1,2,3,6,7`), one project worker
per GPU. Worker rows were `22+11+30+32+16+10=121`, matching the 121 exact
clean sequence starts. The aggregate contains 577 frame results. The two
other GPUs were below the strict free-memory admission threshold and were
not used; foreign workloads were not modified.

The diagnostic reused the frozen Stage IX canonical PGD contract and exact
single-frame reference map. It performed only sequential clean replay:
recorded clean snapshots, targeted OPEN PGD inference, and no environment
step. The primary endpoint was T5; T3/T10 were supporting doses. No candidate
among the identifiable Q1/Q2/Q3 metrics at horizons 3, 5, or 10 passed the
full frozen gate. Q1/Q3 at horizon 10 were not identifiable because no exact
10-frame DEVTEST rows were available. This is a weak/non-passing diagnostic
result, not evidence authorizing physical PGD.

The official interpreter was invoked with `PYTHONNOUSERSITE=1` because the
account user-site caused default interpreter startup to hang. This is the
same official environment path; no package, model, PGD contract, or semantic
parameter was changed.

All worker and aggregate protected counters are zero:

- `protected_reads=0`;
- `eval160_reads=0`;
- `attack_rollouts=0`;
- `vis_pgd_attack_rollouts=0`;
- `env_steps_with_perturbed_action=0`.

`physical_intervention=false`, `Eval160=UNREAD`, and
`protected evaluation=UNREAD`. Therefore X2 physical PGD is not authorized,
and no physical timing/selectivity matrix is authorized. Preserve this
re-sealed X1 root as the terminal Stage X diagnostic evidence pending owner
review.
