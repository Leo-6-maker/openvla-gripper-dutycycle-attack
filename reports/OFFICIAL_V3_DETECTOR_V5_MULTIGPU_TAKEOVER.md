# Official V3 Detector V5 Multi-GPU Takeover Audit

Date: 2026-07-18 23:32 CST
Branch: `codex/official-v3-detector-v5-20260718`
Required starting HEAD: `9c13a1fcb2b16a0651f1c474afbd424865df9e11`
Current development HEAD: `4a217211f058103d05a8ba6699b3f3ffc781a8a4`

## R0 decision

`R0_READ_ONLY_TAKEOVER = PASS_WITH_GPU_HOLD`

The repository and sealed evidence are identifiable. The server shared checkout is a dirty historical checkout and no A800 is currently safe for a new process because every device has an existing process or substantial allocation. No GPU task was started.

## GitHub source state

- Repository: `Leo-6-maker/openvla-gripper-dutycycle-attack`
- PR: `#87`, open, Draft, mergeable
- HEAD at initial takeover: `9c13a1fcb2b16a0651f1c474afbd424865df9e11`
- Current branch HEAD after R1/CI fix: `4a217211f058103d05a8ba6699b3f3ffc781a8a4`
- Base: `archive/official-v3-b3-25d-execution-5e27d7c`
- Required archive commit: `5e27d7c4b1a188bc6a78555f94d2571222587805`
- Checks at the initial takeover HEAD: `detector-v5-cpu = SUCCESS`, `source-registry = SUCCESS`, `stageb-cpu = SUCCESS`
- The first R1 push added the strict evaluator and working-point review. Its detector-v5 check exposed a test import-path defect; a minimal fix is now in `4a21721` and is awaiting the replacement CI run.
- The PR body still needs a current-HEAD handoff update after the replacement CI completes.
- No merge, Ready-for-review transition, main-branch change, force-push, review, or comment was performed.

## SSH and server state

SSH used the existing configured key with `BatchMode` and `IdentitiesOnly`; no key material was printed or copied.

- Host: `pm-364c0001`
- User: `dty_user`
- Repository path: `/mnt/sdc/dty_user/openvla_attack`
- Shared checkout HEAD: `ace1876281a9ad6ed68e1229a6e17346356766e9`
- Shared checkout branch: `feature/sc5-abstention-v2-20260622`
- Shared checkout: dirty, behind its origin by 54 commits, with unrelated tracked and untracked changes
- Consequence: this checkout is not used for V5 development. An isolated clean worktree at the required V5 HEAD is still needed before code execution.

Official environment was verified without starting a CUDA task:

- Prefix: `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800`
- Python: `3.10.20`
- PyTorch: `2.2.0+cu121`
- CUDA runtime: `12.1`; `torch.cuda.is_available() = True`; 8 devices visible
- Transformers: `4.40.1`

All eight A800s have other-user processes or substantial allocations:

| GPU | Memory used / total | Utilization | Observed process class | Decision |
|---:|---:|---:|---|---|
| 0 | 60835 / 81920 MiB | 83% | CAMEF, isaac-gr00t | hold |
| 1 | 66479 / 81920 MiB | 85% | CAMEF | hold |
| 2 | 54681 / 81920 MiB | 41% | CAMEF | hold |
| 3 | 45039 / 81920 MiB | 100% | CAMEF | hold |
| 4 | 46189 / 81920 MiB | 0% | vllm process | hold; memory allocated |
| 5 | 46185 / 81920 MiB | 0% | vllm process | hold; memory allocated |
| 6 | 41883 / 81920 MiB | 100% | mmunlearner | hold |
| 7 | 33168 / 81920 MiB | 59% | pi0 process | hold |

`GPU_TASKS_STARTED = 0`. No process was stopped or altered. The user authorized all GPUs in principle, but current census still fails the safe-idle condition; authorization does not permit killing or sharing another process's GPU allocation.

## Sealed inputs and existing evidence

All paths below are read-only inputs. The listed SHA is the current `SHA256SUMS` file SHA, not a claim that a new audit has rewritten the root.

- Policy-intent root: `.../ops/OFFICIAL_V3_V5_POLICY_INTENT_BINDING_V1_20260718_02`
  - `SHA256SUMS`: `d0a534da50df1f0e341c06d649cd8f52b89707d50b88ff56e02bb2b234451123`
- Privileged physics source: `.../ops/OFFICIAL_V3_PRIVILEGED_PHYSICS_TEACHER_AUDIT_V1_20260718_02`
  - `SHA256SUMS`: `abd25de6dcf18d5c6ca198f49d337e8598b46317107eb5940e1fd7322709bf08`
- Existing utility Teacher: `.../ops/OFFICIAL_V3_DETECTOR_V5_TEACHER_UTILITY_V3_c823653_20260718`
  - `SHA256SUMS`: `60a7def4ae35d760f10515af1cc134cc7aa423442538e5a7bb9d156da8fb56aa`
- Existing Teacher audit: `.../ops/OFFICIAL_V3_DETECTOR_V5_TEACHER_UTILITY_AUDIT_c1f955b_20260718`
- Formal registry: `.../ops/OFFICIAL_V3_CAMPAIGN_REGISTRY_V1_d31187f`
- FIT S1 root: `.../ops/OFFICIAL_V3_S1_FIT_V1_d31187f`
- FIT fold root: `.../ops/OFFICIAL_V3_FIT_FOLDS_V1_d31187f`

Existing A/B development evidence remains present and was not modified:

- A checkpoint: `OFFICIAL_V3_DETECTOR_V5_A_PROPRIO_MATCHED_CONTROL_F0_S20260717_REVIEW4_20260718`
  - checkpoint SHA: `f142a5ba09ae2945966fb76199f18ae1764c62a1883198bd6995d14236298d17`
  - root `SHA256SUMS` SHA: `d87b7f61082ce1323648d2f9acb8d60aa99b2ee4933a6407056936cd4293f0e6`
- B checkpoint: `OFFICIAL_V3_DETECTOR_V5_B_POLICY_INTENT_MATCHED_SMOKE_F0_S20260717_REVIEW5_20260718`
  - checkpoint SHA: `0feec139dfde75603ab24191f93ccbf4396eca43b180f480b28afef8433a99a9`
  - root `SHA256SUMS` SHA: `c985edf6657aeec9539c5ef6a9c68bfe0b24d6c0916b03e9e64266f43efb6ca4`
- A causal replay root `SHA256SUMS` SHA: `d9309551fb23c8640034eff9ebd2038c6ee8a84574a26a175fb11077c71c9f22`
- B causal replay root `SHA256SUMS` SHA: `dc770dc750f6bb4c0ec04ec92a9a41bad5ef124688d1eda2e4ae47f9cd0f3143`

The C2F search found candidate clean observation trees under `/mnt/sdc/dty_user/openvla_attack_evidence/c2f/`, including `clean2000_obs_clean_36712cc` and prior RGB/embedding experiment roots. The candidate observation root currently exposes shard/log directories but no top-level `SHA256SUMS` in the shallow inventory. It is therefore not accepted as an Official RGB sidecar until R4 proves exact trajectory binding and verifies a complete seal.

## R1 read-only re-evaluation result

The existing A/B matched-smoke checkpoints were replayed on CPU from the clean isolated worktree at `6424a96` using new, non-overwrite output roots. The old roots were not modified.

- A output root: `.../OFFICIAL_V3_DETECTOR_V5_A_PROPRIO_R1_WORKING_POINT_6424a96_20260718`
- B output root: `.../OFFICIAL_V3_DETECTOR_V5_B_POLICY_INTENT_R1_WORKING_POINT_6424a96_20260718`
- A prediction audit: `PASS`; `formal_training_authorized=false`; `formal_attack_authorized=false`
- B prediction audit: `PASS`; `formal_training_authorized=false`; `formal_attack_authorized=false`
- Both runs: 200 validation identities, 126 true-mixed episodes, causal-anchor top-1 `112/126 = 0.8888888889`
- A: 172 raw online emits, 5 outside-rankable scheduler events, 1 release trigger, 9 regrasp triggers, pure-negative abstention `0/3`
- B: 175 raw online emits, 5 outside-rankable scheduler events, 1 release trigger, 7 regrasp triggers, pure-negative abstention `1/3`
- Working-point status: `HOLD` for both; no threshold in the fixed grid met critical-window recall `>= 0.95`
- A/B review root: `.../OFFICIAL_V3_DETECTOR_V5_AB_WORKING_POINT_REVIEW_6424a96_20260718`
- A/B disagreement counts: causal `12`, emit `21`, scheduler `33`, release `2`, regrasp `4`

These figures are sealed development diagnostics. `112/126` is causal-anchor argmax, not scheduler selection accuracy; no formal model-selection or attack authorization follows.

## Confirmed gaps carried into R1--R4

1. `evaluate_v5_causal_online.py` currently loads the checkpoint with `strict=False`; future evaluation must use `strict=True`.
2. Current summaries mix causal-anchor argmax diagnostics with actual one-shot scheduler selection; these must be separate fields.
3. A/B working-point selection and exact disagreement identity lists are not yet sealed under the required maximum-threshold rule.
4. Policy redundancy and fusion diagnostics are not yet sealed.
5. Runtime intent causality must be read from the actual runner; it must not be inferred from a report.
6. Physics task/object-state slices have not yet been formally decoded for all 40 tasks; no physics Teacher may be generated by guessing.
7. C2F observation/RGB artifacts are not yet proven `EXACT_TRAJECTORY_BOUND`.
8. No new GPU run is authorized until the R1/R2/R3/R4 CPU/I/O gates are complete and a GPU is demonstrably idle; the current all-device census has not met that condition.

## Planned allocation and stop gates

When an unoccupied GPU is verified, use one process per physical GPU via `CUDA_VISIBLE_DEVICES`, with a distinct output root, log, PID record, and seal. Do not use DDP.

1. R1: re-evaluate sealed A/B roots; no training.
2. R2: strengthen checkpoint/prediction auditors; no training.
3. R3: decode all 40 task physics layouts, then build and independently seal Physics Teacher V2.
4. R4: bind C2F observations to Official FIT trajectories; visual work remains HOLD unless all required identities are exact.
5. R5: extract causal visual features only after R4 exact binding.
6. R6: run matched 80/200 Physics smoke on available GPUs, within the fixed protocol.
7. R7: full Fold-0 only after R6 scientific and engineering gates pass.
8. R8: four-fold, one-seed only after R7; stop after the four-fold result for review.

Protected splits (FIT-DEV, CAL, CHECK, and states 30--49 semantics) were not read in this takeover. Attack results, Direct-open, canary, CS200, and rollout roots were not read.

## Mutation declaration

- GitHub mutations in this takeover: `0`
- Server code/evidence artifact mutations in this takeover: `0`
- Server Git metadata: one clean detached worktree was created at `/tmp/codex_v5_physics_6424a96`; the dirty shared checkout was not changed.
- CLEAN/S1/old Teacher/old A/B roots modified: `0`
- Protected split semantic reads: `0`
- Attack data or rollout started: `0`
- GPU tasks started: `0`

This report records the R0 takeover only. It does not authorize formal training, model selection, FIT-DEV/CAL/CHECK access, or attack execution.
