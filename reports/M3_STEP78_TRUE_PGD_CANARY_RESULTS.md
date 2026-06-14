# M3 Step78 True-PGD Canary Results

## Result

`RANDOM_NOT_BEATEN`

The single-frame Tomato step78 canary passed route, optimization, and official-transfer gates, including an official token flip from `31872` to `31744`. It failed the random superiority gate because `RAND20` reached the same official target margin and also emitted `31744`.

No fixed-frame panel, full-window rollout, critical-close rescue, held-out parent test, or Layer3 success claim is authorized from this result.

## Run Context

- Branch: `exp/m3-step78-true-pgd-canary-20260615`
- Experiment commit: `af545e1c5eb1012ad5dc8b8872e50596315bd4d5`
- Report/follow-up commit: `6d5abca43dd6dae3487d547c20ff617a242e1abe`
- Base merge commit: `a47f0a9ddd00ad61b47a16e439aea4c9c3f8d7e7`
- Server clean clone: `/data/liuyu/repos/m3_step78_current`
- Python environment: `/data/aviary/envs/openvla_official_libero_20260525`
- Model: `/data/aviary/models/openvla/openvla-7b-finetuned-libero-object`
- GPU mapping: `CUDA_VISIBLE_DEVICES=2,6`
- No LIBERO attack rollout was launched.

## Inputs

The raw step78 input was captured with a clean-only run because no raw pre-processor observation artifact was found. Video frames were not used as model input.

- Input directory: `/data/liuyu/outputs/m3_step78_true_pgd_20260614/capture_step78_f18537d_r2`
- Raw image SHA256: `0b0f5f99432600f2d5b6c52987bda628bb17c3fa2dc549080041ef87070e558a`
- Processor tensor SHA256: `bf0dc0840b05e42f1806771d72966121f07a5c412a9625d7be56e64e1076b939`
- Clean exact tokens: `[31900, 31870, 31915, 31882, 31862, 31913, 31872]`
- Clean score row: `31744=41.0`, `31872=41.25`, margin `-0.25`

## Preflight

Initial zero-step preflight exposed a cache-path mismatch: no-cache full-context surrogate matched `generate(use_cache=False)` but not official default cached generation on `PGD_DELTA0`.

The runner was fixed to use explicit `cached_autoregressive_generate_v1` for the M3 target-token surrogate. After that fix:

- CLEAN: `SURROGATE_OFFICIAL_SCORE_PATH_MATCH`
- PGD_DELTA0: `SURROGATE_OFFICIAL_SCORE_PATH_MATCH`

## Primary Metrics

| Condition | Official Tokens | Gripper | Official 31744 Margin | Surrogate Margin | Arm Match |
| --- | --- | ---: | ---: | ---: | ---: |
| CLEAN | `[31900, 31870, 31915, 31882, 31862, 31913, 31872]` | 31872 | -0.25 | -0.25 | n/a |
| PGD_DELTA0 | `[31900, 31870, 31915, 31882, 31862, 31913, 31872]` | 31872 | -0.5 | -0.5 | n/a |
| TRUE_PGD_FINAL | `[31900, 31870, 31915, 31882, 31862, 31913, 31744]` | 31744 | 5.75 | 5.75 | 6/6 |
| RAND20 | `[31900, 31870, 31915, 31882, 31891, 31834, 31744]` | 31744 | 5.75 | 5.75 | 4/6 |
| SHUFFLED_GRAD_PGD20 | `[31900, 31870, 31915, 31882, 31891, 31834, 31744]` | 31744 | 5.0 | 5.0 | 4/6 |

## Gate Assessment

- INFRA: PASS
- OPTIMIZATION: PASS
- OFFICIAL TRANSFER: PASS
- RANDOM SUPERIORITY: FAIL
- ARM SELECTIVITY: PASS for TRUE_PGD

`TRUE_PGD_FINAL` did not exceed the `RAND20` best official margin. Therefore this canary does not establish `TRUE_PGD > matched random controls`.

The selected `RAND20` best was arm-nonselective relative to TRUE_PGD: its arm prefix matched the clean arm prefix in `4/6` positions, while TRUE_PGD matched `6/6`. This is an important post-hoc audit observation, but it does not change the preregistered primary result. The selected RAND20 control remains in the original primary comparison, and the stage result remains `RANDOM_NOT_BEATEN`.

## RAND20 Candidate Distribution

The 20-candidate random control distribution is summarized using surrogate margins only. It is not an official flip-rate distribution because only the selected best candidate was officially decoded in the canary.

- candidate count: `20`
- surrogate margin > 0: `12/20`
- surrogate margin >= 5: `2/20`
- maximum surrogate margin: `5.75`
- second-best surrogate margin: `5.5`
- median surrogate margin: `0.25`
- selected candidate id: `0`
- selected candidate seed: `94127280`

## CW V1 Saturation

The v1 objective uses:

```text
relu(best_competitor - target31744 + margin)
margin = 5
```

Both TRUE_PGD and the selected RAND20 candidate reached target margin `5.75`, so both entered the zero-loss margin set. The terminal `5.75` tie is evaluated according to the preregistered gate and remains `RANDOM_NOT_BEATEN`, but the v1 hinge cannot continue ranking candidates once margin is at least `5`.

## Artifact SHA256

- Canary condition CSV: `66dbf7fa1ce5bbf71d5fcf4f630ffdf4d32342a2fc17642c752595497b6f0d9e`
- Canary route audit CSV: `be9542bdbaafab0709b4310c6c609b3b2a5f983648517643e8bc1519ec6f0995`
- Canary candidate controls CSV: `8b54e1175882c73524adf68464d5bc155dbf7e21c042eb082c8763ebbeb28215`
- Canary debug JSON: `07a80d9c7e068c5f319ab737a8188862b0687c911a1aa93f5125aa82d6417258`
- Preflight JSON: `3820091a8dc27f7810f0129cdfe7306e63bcc01cceed0d168c0586a262f2a4c7`

Additional provenance is recorded in `tables/m3_step78_canary_manifest_af545e1_seed80.csv`, including model directory fingerprint, GPU mapping, driver version, runner SHA256, adapter SHA256, route-contract SHA256, config SHA256, and input hashes.

## Allowed Claim

On the preregistered Tomato step78 fixed frame, the strict true-PGD route was executed and produced official transfer to token `31744`, but it did not outperform the matched 20-candidate random control.

## Forbidden Claim

Do not claim `TRUE_PGD > random`, closed-loop critical-closure disruption, paired task effect, held-out transfer, or a solved Layer3 pipeline from this result.
