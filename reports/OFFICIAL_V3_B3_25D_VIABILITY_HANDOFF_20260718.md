# Official V3 B3-25D viability handoff

This report records the first real FIT viability execution. It is a results handoff, not a model-selection or attack authorization.

## Execution binding

| Item | Value |
|---|---|
| Execution checkout | `/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/trainer_checkout_f98715b` |
| Execution code HEAD | `5e27d7c4b1a188bc6a78555f94d2571222587805` |
| Archive ref | `archive/official-v3-b3-25d-execution-5e27d7c` |
| Execution parent/tree | `0b053945...` / `1a8634cd...` |
| Execution worktree | clean; no untracked files |
| GitHub PR #82 HEAD at handoff | `87a4561af8883baacf161961c2fb9a114ff459c6` |
| Python | 3.10.20 |
| PyTorch | 2.2.0+cu121 |
| Transformers | 4.40.1 |
| Environment | `/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800` |
| OpenVLA/source path | NVIDIA A800, CUDA/BF16-capable |
| B3 detector training/evaluation dtype | float32 |
| Threshold | fixed provisional threshold `0.5`; no post-hoc retuning |

The execution HEAD is recorded separately because the server checkout was detached and is not the same ref as the current GitHub PR #82 head. This handoff does not claim that PR #82 itself contains the execution commit.

The execution commit is now reachable from the archive ref and its commit URL is recorded in [the source-closure manifest](manifests/OFFICIAL_V3_EXECUTION_SOURCE_CLOSURE_20260718.json). The server bundle used to recover the object has SHA256 `0dc3d6d1a05f28d25288c510b661a0605d76fa6881b64cb1d4c7979fa74f7a85`.

## Data and S1 evidence

- CLEAN V3: 2000/2000 sealed; 1384 task successes and 616 valid task failures.
- FIT registry: 800 identities, 4 suites × 200, 40 tasks × 20.
- S1 root: `OFFICIAL_V3_S1_FIT_V1_5e27d7c`; independent root audit: `PASS`, 800 identities.
- Teacher aggregate: `PASS`, 800/800 identities, 0 invariant violations, 0 duplicate identities.
- Nondegeneracy gates: every suite has known T10 positives; every task has known T10; L10 later-event and later known-T10 coverage are present.
- Campaign-bounded provenance remains explicit: 361 FIT artifacts retain direct start binding and 439 remain instance-start unresolved; no artifact was rewritten or remapped in this handoff.

Evidence bindings:

```text
CLEAN snapshot SHA256SUMS       a7806ae1bb9f7dc9183ff9d7ee96a6393b6e4b028593d6291729afd9f746e8ff
FIT registry root                b42cc794bcf9e837106ecb54f99d70d85e2f47f8d44b1ce08862aebf9ef892f7
FIT registry CSV                 09f71b3a9b8250c80735382ba5deab6dbcadfa21b645e4a981eefb114b236af5
FIT registry summary             744811f880491d7d4dadf9aa796429e87e7ceadf3f23177707c48c5d54d6e266
S1 root                          15c97212fde19682a9e3042d6d051c51606b0989881d471cb8eb80f22354b0cf
S1 independent audit JSON        ca3beb80c8bdff31bf438a8a87d5063075f2d0a82f35af51a006ce5d23ddf4b6
Teacher aggregate JSON           35969682c487bc7b515089a3523b2506e7561ecccdcb859d160efcbd05347844
Provisional viability JSON       0bccf84eeb23d32aba58f4461b35a8dba6e8359b7c7ffd63f1b4843b386eb78d
Training protocol                52c8a1be592ea605cccc8b530a4c1b85334d66ccfe64dbf85e117570674358a8
Viability evaluator              ab1ea5780ad7c0a9c916bbc946aaed6ea91bdf8f6064c1f211abfc3ed7139a66
Viability aggregator             1f385cb305a9ede1c07f17137821d6fa6f4ea94b3a7d58c0cbe60137dc3fa296
Prediction builder               155e2217fbbd1498790de1c7800a2026e79d52b90fd133ba1cd7635a7ef9bb59
```

The complete source/config status is in [the execution-input manifest](manifests/OFFICIAL_V3_B3_25D_EXECUTION_INPUTS_20260718.json). A separately sealed decision-config file was not recovered because the formal PASS/HOLD decision CLI was not run for this B3_25D-only provisional report. This is recorded as `NOT_RECOVERED`, not silently treated as a PASS.

## Training execution

The B3_25D fold candidate matrix completed 12/12 runs: four state-block folds × three fixed seeds. Each run used 600 training episodes and produced a sealed `FIT_FOLD_TRAINED_CANDIDATE` checkpoint bundle. All 12 held-out prediction bundles contain 200 validation identities and are sealed.

The checkpoints are candidates only:

```text
eligible_for_model_selection = false
formal_attack_authorized     = false
full-FIT refit               = not started
```

The complete per-run checkpoint/prediction digests, input-chain hashes, record counts and training losses are in [the fold/seed table](tables/OFFICIAL_V3_B3_25D_FOLD_SEED_METRICS_20260718.csv) and [the per-run evidence index](manifests/OFFICIAL_V3_B3_25D_RUN_EVIDENCE_INDEX_20260718.json). Baseline-only values are duplicated in [the baseline table](tables/OFFICIAL_V3_B3_25D_BASELINE_COMPARISON_20260718.csv).

## Viability result

The sealed provisional report is `B3_25D_ONLY_PROVISIONAL`. It is not the formal 24-coordinate decision because the B3_25D9D bundles were not contract-valid under the campaign-bounded source layout.

| Metric | Value | Frozen/provisional gate | Status |
|---|---:|---:|---|
| Mean full-T10 event hit | 0.9572 | ≥ 0.8000 | PASS |
| Mean later-event hit | 0.8821 | ≥ 0.5000 | PASS |
| Release overlap | 0 | = 0 | PASS |
| Mean negative-episode any-emit | 0.1345 | ≤ 0.1000 | FAIL |
| Maximum negative-episode any-emit | 0.1875 | ≤ 0.1000 | FAIL |
| Close-only baseline full-T10 | approximately 0.988–1.000 | B3 not worse | FAIL |
| Time-since-close baseline full-T10 | approximately 0.992–0.996 | diagnostic comparator only | diagnostic |

Interpretation: B3_25D covers most positive retention windows, including later events, but has not demonstrated selective triggering under the frozen false-emit and baseline-comparison gates. This is a negative viability result, not an attack result.

No threshold was retuned after held-out inspection. No seed was selected from these results. CAL and CHECK inputs were not read. The reported rates are unweighted arithmetic means across the 12 fold-seed runs; maxima are the maximum run rate. Micro-pooled rates were not used to alter the gate.

## CS200 preparation

The main-table parent manifest is prepared but not executed:

```text
parent_count                 = 200
task_count                   = 40
parents_per_task             = 5
state_range                  = 30–49
selection                    = canonical_parent_key ascending, first 5 successful CLEAN parents per task
check_status                 = NOT_RUN
attack_execution_authorized  = false
rollout artifacts            = 0
selection manifest SHA        da2da73c1fe3fadd5197fb10d6a6a3064720d0acb7cc90874badcc86f7eb90e3
parent rows SHA               2e70f74a94f1613dc2656b00cbe66d9a4944a32dc99ba635ec66c943ba4f8088
CS200 SHA256SUMS SHA          1e12c4c2e791a9665455390838b7e3a5298d1f6219f00979a8e3d103ac6b89c8
```

## Gate status

```text
CLEAN_2000                         = PASS
FIT/S1 source and Teacher audit   = PASS
B3_25D fold candidates            = 12/12 sealed
B3_25D held-out predictions       = 12/12 sealed
B3_25D viability                  = HOLD / pre-registered gate failed
B3_25D9D formal matrix            = NOT RUN / HOLD
FULL_FIT REFIT                    = HOLD
FIT-DEV / CAL / CHECK             = HOLD
CS200 parent preparation          = PASS, preparation only
ATTACK ROLLOUT                    = NOT STARTED
```

The next permissible step is an independently frozen protocol or metric-semantics review. Reusing this held-out set for post-hoc threshold rescue, selecting the best seed, full-FIT refitting, or starting the CS200 attack would invalidate this handoff.

## DeepSeek takeover notes

Start with the GitHub archive ref and the server evidence root; do not infer current state from PR #82's original body.

```text
GitHub repo: Leo-6-maker/openvla-gripper-dutycycle-attack
Execution source: archive/official-v3-b3-25d-execution-5e27d7c
Server root: /mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716
Python: /mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python
FIT registry: ops/OFFICIAL_V3_CAMPAIGN_REGISTRY_V1_d31187f
S1 root: ops/OFFICIAL_V3_S1_FIT_V1_5e27d7c
Viability report: ops/OFFICIAL_V3_FIT_VIABILITY_B3_25D_PROVISIONAL_5e27d7c
CS200 preparation: ops/OFFICIAL_V3_CS200_PARENT_PREPARATION_V1_5e27d7c
```

Safe takeover order:

1. Verify the archive commit, this report's source-closure manifest, and the server root hashes.
2. Read the execution-input manifest before interpreting thresholds or baseline status.
3. Treat all 12 checkpoints as candidates, never as selected production models.
4. Keep `B3_25D9D`, full-FIT refit, FIT-DEV, CAL, CHECK and attack blocked.
5. If changing metric semantics or thresholds, create a new protocol/config version and a new independent validation identity; do not rewrite this result.
6. Before any new server run, verify the official environment and obtain explicit authorization for that new protocol.

Known traps:

- `formal_training_authorized=true` in a fold execution authorization only means that the authorized fold candidate was allowed to train; it does not make the checkpoint model-selection eligible.
- `FIT_FOLD_TRAINED_CANDIDATE` and `eligible_for_model_selection=false` are the authoritative checkpoint lifecycle fields.
- The 439 unresolved campaign-bound start UUIDs are not evidence of corrupted data, but they are also not exact per-artifact provenance.
- Do not copy manifests into artifact roots or recompute old artifact checksums.
- Do not use the CS200 preparation manifest as an attack authorization; it explicitly has `check_status=NOT_RUN`.
- The raw checkpoint and prediction bundles remain on the server only; GitHub contains reports, hashes and source references, not binaries or raw prediction records.
