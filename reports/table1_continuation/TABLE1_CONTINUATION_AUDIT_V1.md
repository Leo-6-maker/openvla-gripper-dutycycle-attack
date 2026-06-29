# Table 1 Continuation Audit V1

Audit date: 2026-06-29

Scope: read-only repository and live-server audit for Issue #41 handoff. No rollout, VIS, random, oracle, benchmark, attack, cleanup, or frozen-artifact mutation was launched by this audit.

## A. Verified GitHub facts

- Local audit branch: `codex/table1-continuation-audit-v1`
- Local audit HEAD: `93e2373bc06db7598eeb8f36ec0a0fbde349d319`
- Base branch checked: `origin/experiments/cross-suite-generalization-v1`
- Merge commit containment: PASS. `origin/experiments/cross-suite-generalization-v1` contains `93e2373bc06db7598eeb8f36ec0a0fbde349d319`; at audit time it resolves to that commit.
- Relevant commit graph:
  - `93e2373 docs(table1): add SOTA comparison plan and GPT continuation handoff`
  - `2a6a9c9 freeze(loto): 30/30 checkpoint Global Freeze V1 - all SHA256 verified`
  - `9425643 fix(cross-suite): collector - only refuse COMPLETE/SCHEMA_FAIL, not any file`
  - `1583813 feat(cross-suite): 1500-job manifest generator + queue worker + canary freeze`
- `gh issue view 41` could not be used locally because GitHub CLI is not authenticated. The user-provided Issue #41 handoff text was used as the issue entrypoint.

Key committed file SHA256:

| File | SHA256 | Status |
|---|---:|---|
| `docs/handoff/GPT_TABLE1_HANDOFF_20260629.md` | `f81d6d846efa929f24eeb5de5a501f50ecea67084c7eeac60e1550c817713fa4` | present |
| `docs/table1/TABLE1_SOTA_COMPARISON_AND_EXECUTION_PLAN_V1.md` | `bcb7de1d5b3df85ef0e0c032494421a55b1cb893f6a15f74130ce7fe60ac9922` | present |
| `docs/TABLE1_HANDOFF_INDEX_20260629.md` | `565c19e67638713e37956bdd7e7eb273d2dcdc43fc4296304d804f804196599f` | present |
| `docs/handoff/NEW_GPT_START_PROMPT_20260629.md` | `b5d864cf97b3a8d160e6edcefcc5c62ce4fef055035ebbcf4f450dfc5d1fadcd` | present |
| `docs/table1/LOTO_TABLE1_BASELINE_PREREGISTRATION_V1_DRAFT.json` | `e765982e91622b69847333216e9a624ec127954e7656f9a2ec4f9abe371430bd` | present, `DRAFT_NOT_AUTHORIZED` |
| `reports/phase7_table1/final_v3/TABLE1_FINAL_V3_4.md` | `c35f67f778384b3a84c9c30b585fed3f009f6a9bfdc3d6764ac8fbfcdd013e51` | legacy only |
| `docs/gpu/LOTO_GLOBAL_FREEZE_V1.json` | `d8c7b8426705a12f22541884edc25981ed9262481a45047fc1e9164d56f1403b` | committed freeze |
| `docs/gpu/LOTO_METRIC_SCHEMA_V2.json` | `785c66dae1195aec05d884d792f9c5b05270cc62a60ed15c47516e1e1167b86f` | committed schema |
| `reports/CROSS_SUITE_CLEAN1500_CANARY_FREEZE_V1.json` | `6e629827f78230d92d210049e490cf81e8fd98dcc5dd25e4d104b5bd438edbd1` | committed canary |

Verified committed facts:

- Phase B pooled counts are documented in the handoff and Table 1 plan, but the raw committed table needed to recompute them was not found on the GitHub branch during this audit. Marked `UNVERIFIED_FROM_COMMITTED_RAW_TABLES`.
- Expected Phase B freeze SHA `89911600e0bdc08e46e21f1ebfa85ab37c1d16fb741d4fe7c52409d7e76cd241` matches the live server artifact, not a committed file under that exact path.
- VIS engineering canary facts are documented in handoff prose and live server freeze artifact. The live server artifact SHA is `91f313ef365bd1c86b1ef921fdb79bc8110e9f87f20cf9e5bf46690755fa77e5`.
- Legacy Table 1 `TABLE1_FINAL_V3_4.md` uses N=24/27-style earlier panels. It is superseded context and must not feed the LOTO 162-row formal matrix.

Missing from GitHub branch:

- `LOTO_PHASE_B_RESULTS_FREEZE_V1.json`
- `LOTO_VIS_STATE_SELECTION_V1.json`
- `LOTO_VIS_ENGINEERING_CANARY_V1.json`
- Formal CLEAN closure bundle
- Any condition-specific freeze bundle for current Table 1 baselines
- Final non-draft baseline preregistration

## B. Verified server facts

Server: `dty_user@10.60.2.56 -p 33571`

Live repo:

- Path: `/mnt/sdc/dty_user/openvla_attack`
- Branch: `feature/sc5-abstention-v2-20260622`
- HEAD: `ace1876281a9ad6ed68e1229a6e17346356766e9`
- This live repo does not contain commit `93e2373bc06db7598eeb8f36ec0a0fbde349d319`.
- The live repo is dirty and lacks the handoff/Table 1 files listed in section A.

Golden bundle:

- Path: `/mnt/sdc/dty_user/openvla_project/freeze/loto_phase_b_v1`
- `sha256sum -c SHA256SUMS.txt`: all listed files OK.
- Files verified: `LOTO_GLOBAL_FREEZE_V1.json`, `LOTO_GLOBAL_FREEZE_V1_VERIFY.json`, `LOTO_TEST_OPEN_EVENT_V1.json`, `LOTO_PHASE_B_RESULTS_FREEZE_V1.json`, `LOTO_VIS_STATE_SELECTION_V1.json`, `LOTO_VIS_ENGINEERING_CANARY_V1.json`.

Live LOTO root:

- Path: `/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1`
- Phase B results SHA: `89911600e0bdc08e46e21f1ebfa85ab37c1d16fb741d4fe7c52409d7e76cd241`
- VIS canary SHA: `91f313ef365bd1c86b1ef921fdb79bc8110e9f87f20cf9e5bf46690755fa77e5`
- VIS state selection SHA: `457aadd2c93192418f0558249423889bb8def9e27198699dbaf39820187b5719`

Formal CLEAN:

- Path: `/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/CLEAN`
- Manifest SHA: `105f25a3b4bf1681eb4959ce63e17dec62296c0abb12db8ade969773ab375bc4`
- Manifest rows: 162
- Parent groups: 54 by `(fold,state_id,detector_seed)`, each with 3 perturbation replicates
- Unique output dirs: 162
- `episode_summary.json` present: 107
- Missing `episode_summary.json`: 55
- `task_success=True`: 107/107 observed summaries
- Malformed JSON: 0 observed in CLEAN JSON scan
- Zero-byte files: 0 observed in CLEAN tree scan
- Worker logs include failed jobs, for example GPU1 early failures and GPU2 odd failures.
- Formal CLEAN workers were still active during audit.
- Required freeze files missing: `MANIFEST.sha256`, `accepted_job_keys.txt`, `RESULT_INVENTORY.json`, `PROVENANCE_AUDIT.json`, `PAIRING_AUDIT.json`, `ARTIFACT_SHA256SUMS.txt`, `CONDITION_RESULTS.json`, `CONDITION_FREEZE.json`, `README_RESTORE.txt`.

CLEAN1500:

- Path: `/mnt/sdc/dty_user/openvla_attack_evidence/sc5_cross_suite_clean1500_v1`
- Status JSON scan: 1470 `COMPLETE`
- `episode_summary.json` count: 736
- Suite summary files observed: `libero_spatial=292`, `libero_goal=248`, `libero_10=190`
- Malformed JSON: 0 in scanned JSON
- Active CLEAN1500 processes were present during audit.

Provenance SHA observed on live server:

| Artifact | SHA256 |
|---|---:|
| `scripts/stageb/run_vis_formal_worker.py` | `a7bdd8601a32fe835ced3af681d036861ba246c19ad2edcfea5797749890e0928` |
| `scripts/stageb/run_v2_vis_sc5_mlp_bridge.py` | `cf125d2393f2ca0a5ec1b62610b22b8d5c17733a647b9824edd2aab19995daa6` |
| `scripts/stageb/run_cross_suite_clean_v3.py` | `e98754bdcf766356f8c07eb8d87224e4eea6d8ccbc90613c665591c9bb788b06` |
| `configs/cross_suite_clean1500_protocol_v1.json` | `449da97b2a0bcb339b19b629e19ffbab7590bf1bed0c4410b8fb114861633ce0` |
| `configs/cross_suite_clean1500_protocol_v1_gpu7_goal.json` | `bb6f4033a80fef6e6884aac2013880409d7bfd777731a87c9412f13a9609dc99e` |
| `configs/cross_suite_object_target_registry_v1.json` | `b0c0ec6fb33dbc2b066f9c759b1762d5a2ebada7cce2f9559a99150e0a9ec750` |
| `COLLECTOR_ARCHIVE_e98754bd.py` | `e98754bdcf766356f8c07eb8d87224e4eea6d8ccbc90613c665591c9bb788b06` |

Server resource state:

- `/`: 97% used at audit command time; login banner earlier reported 100%.
- `/mnt/sdc`: 88% used, 340G available.
- `/llm_jzm`: 100% used, 7.7G available.
- Inodes were not pressured: `/` 11%, `/mnt/sdc` 3%, `/llm_jzm` 8%.
- GPU processes were active on all 8 A800 GPUs; Formal CLEAN and CLEAN1500 processes shared GPUs with other users' jobs.
- Login banner reported 255 zombie processes.

## C. Current status matrix

| Activity | Status | Verified evidence | Missing requirement | Next gate | Launch authorized |
|---|---|---|---|---|---|
| Phase B detector evaluation | PASS | Live Phase B freeze SHA matches handoff | committed raw recomputation table | none for audit | no new compute |
| Post-Phase-B detector tuning | PROHIBITED | handoff, prereg draft, freeze docs | none | never tune from formal outcomes | no |
| VIS engineering canary | PASS | live canary SHA `91f313ef...` | committed raw recomputation table | keep separate from formal | no new canary needed |
| Formal CLEAN | IN PROGRESS | 162-row manifest, 107 summaries, active workers | 55 summaries and freeze bundle | close and freeze | only current CLEAN workers observed; no new attack |
| Formal Prefix VIS | HOLD | prereg draft exists | Formal CLEAN closure and prereg freeze | condition freeze | no |
| RAND | HOLD | old RAND implementation evidence exists | LOTO manifest/freeze/prereg | condition freeze | no |
| Shuffled gradient | HOLD | bridge supports `SHUFFLED_T10` | LOTO canary/freeze/prereg | engineering check | no |
| UMA | HOLD | untargeted CE objective exists | exact LOTO condition spec | prereg freeze | no |
| UADA | HOLD | audit says no true action-discrepancy objective | implementation or rename | objective decision | no |
| UPA | HOLD | no faithful implementation found | implementation audit | objective decision | no |
| Adapted TMA-OPEN | HOLD | old TMA CE path exists | matched LOTO freeze/prereg | objective freeze | no |
| Original-protocol TMA | HOLD | no original reproduction found | separate protocol reproduction | prereg | no |
| Adapted FreezeVLA | HOLD | no implementation found | four-fold canary plus adaptation spec | canary | no |
| Timing panel | HOLD | old random/early/student evidence exists | LOTO timing manifests | prereg freeze | no |
| ArmLock ablation | HOLD | old ArmLock evidence exists | LOTO ablation plan if needed | Table 2 decision | no |
| Cross-suite VIS | HOLD | handoff says future line | Object Table 1 incomplete | finish Object Table 1 | no |
| CLEAN1500 | IN PROGRESS | live status scan, active processes | final freeze/audit | continue background monitoring | current collection only |
| Baseline preregistration | HOLD | draft is `DRAFT_NOT_AUTHORIZED` | null fields | fill and freeze | no |
| Final Table 1 aggregation | HOLD | no frozen condition bundles | all condition bundles | frozen analysis script | no |

## D. Mismatches

| item | source_a | source_b | impact | severity | resolution_required | launch_blocking |
|---|---|---|---|---|---|---|
| live repo branch | GitHub base `experiments/cross-suite-generalization-v1 @ 93e2373` | server `/mnt/sdc/dty_user/openvla_attack` is `feature/sc5-abstention-v2-20260622 @ ace18762` | live workers are not running from the handoff branch | P0 | bind live worker SHA to accepted freeze or redeploy reviewed branch before new formal baselines | true |
| handoff files | present on GitHub branch | missing in live repo | server cannot be assumed to know Issue #41 rules | P1 | sync docs or record server-side command bundle | true for new launches |
| Formal CLEAN status | handoff says audit/complete-then-freeze | live has 107/162 summaries and active workers | not closed | P0 | wait for terminal outcomes and freeze | true |
| freeze bundle | required per condition | CLEAN has only `MANIFEST.jsonl` | no immutable clean baseline | P0 | generate freeze bundle after completion | true |
| preregistration | draft has many null fields | live has some concrete SHAs | not frozen, values not registered | P1 | fill from verified sources and freeze | true |
| legacy Table 1 | `TABLE1_FINAL_V3_4.md` N=24/27 | current LOTO design N=162/condition | denominator mismatch | P0 if mixed | keep legacy context only | true |
| UADA semantics | Table 1 plan requires action-discrepancy objective | existing audit says only CE untargeted exists | baseline taxonomy gap | P1 | implement or rename | true for UADA |
| storage | expected stable run environment | `/llm_jzm` 100%, `/` near full | worker instability risk | P1 | capacity audit before more launches | true for expansion |

## E. Risks

P0:

- New formal baseline launch from live server would use a branch that does not contain the audited handoff commit.
- Formal CLEAN is not closed: 55/162 summaries missing during audit.
- CLEAN freeze bundle is absent.
- Legacy Phase 7 Table 1 could contaminate LOTO Table 1 if copied into the new matrix.
- Any analysis that treats current 107 completed CLEAN summaries as complete would invalidate denominators.

P1:

- Baseline preregistration remains `DRAFT_NOT_AUTHORIZED` with null path/SHA/budget/manifest fields.
- UADA and UPA are not verified as faithful implementations.
- Server storage pressure and zombie-process count raise reliability risk.
- CLEAN1500 has multiple count surfaces: 1470 COMPLETE by status JSON scan, 736 `episode_summary.json`; the final audit must define the authoritative ledger.
- Current live repo is dirty, so script SHA rather than commit alone must be used for provenance.

P2:

- GitHub CLI is unauthenticated locally, so Issue #41 metadata was not connector-verified.
- Original-protocol TMA and Adapted FreezeVLA can wait until Batch A gates close.

## F. Formal CLEAN closure

FORMAL_CLEAN_NOT_CLOSED

Checklist:

| Closure item | Status | Evidence |
|---|---|---|
| 162/162 legal terminal outcomes | HOLD | 107 `episode_summary.json`; 55 missing |
| 54 unique parent groups | PASS | manifest groups by `(fold,state_id,detector_seed)` = 54 |
| 3 replicates per parent | PASS in manifest | each parent has 3 perturbation seeds |
| no illegal replacements | UNVERIFIED | no accepted freeze/parent map yet |
| no duplicate units | PASS in manifest | 162 unique output dirs |
| replicate consistency | UNVERIFIED | incomplete |
| clean qualification rule frozen | HOLD | no condition freeze |
| runner provenance frozen | HOLD | SHA observed, no freeze bundle |
| protocol provenance frozen | HOLD | SHA observed, no condition freeze |
| checkpoint provenance frozen | HOLD | global freeze exists, condition binding absent |
| manifest frozen | HOLD | `MANIFEST.jsonl` present, no `MANIFEST.sha256` file |
| freeze bundle generated | HOLD | required bundle files missing |
| SHA256SUMS complete | HOLD | missing |
| inventory complete | HOLD | missing |
| restore README complete | HOLD | missing |
| independent backup complete | UNVERIFIED | golden bundle exists; independent backup still not established for formal CLEAN |

## G. Baseline implementation inventory

See `tables/table1_baseline_inventory_v1.csv`.

Summary:

- `CLEAN`, `RAND_LINF`, `SHUFFLED_GRADIENT`, `UMA_UNTARGETED_CE_PGD`, `ADAPTED_TMA_OPEN`, and `PREFIX_LOG_RATIO_OPEN` have partial or legacy code paths.
- `UADA_DOF1_3`, `UPA_DOF1_3`, `ADAPTED_FREEZEVLA`, and `ORIGINAL_PROTOCOL_TMA` are not verified as faithful implementations.
- No condition has a current LOTO formal freeze bundle or complete preregistration.

## H. Minimum next actions

1. Let current Formal CLEAN finish; do not start attacks while it is incomplete.
2. Run a read-only closure validator over the completed CLEAN root: 162 summaries, 54 parents, 3 reps each, no malformed/zero-byte output, no duplicate output dirs, no replacement states.
3. Generate the Formal CLEAN freeze bundle only after closure passes.
4. Resolve live-code mismatch: either sync the audited `93e2373` lineage to server or explicitly freeze the live script SHAs and document why `ace18762` is accepted for the already-running CLEAN line.
5. Fill preregistration nulls from verified files only; keep unverified fields as HOLD.
6. Freeze Batch A condition specs before any baseline launch: `PREFIX_STUDENT`, `RAND_LINF_STUDENT`, `ADAPTED_TMA_OPEN_STUDENT`, `PREFIX_RANDOM_TIME`.
7. Defer UADA, UPA, FreezeVLA, original-protocol TMA, cross-suite VIS, defense, real-robot, and detector changes.

## I. GO/HOLD verdict

GO/HOLD verdict: HOLD for all new formal attack/baseline launches.

Allowed now:

- Continue observing the already-running Formal CLEAN and CLEAN1500 workers.
- Perform read-only closure audits.
- Generate deterministic validators, reports, and freeze bundles after completion.

Not authorized:

- Any new formal baseline launch.
- Any attack interpretation from incomplete Formal CLEAN.
- Any detector retuning.
- Any legacy Phase 7 number copied into LOTO Table 1.
