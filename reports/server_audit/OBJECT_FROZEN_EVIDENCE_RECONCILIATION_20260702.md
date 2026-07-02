# Object Frozen Evidence Reconciliation — 2026-07-02

## Evidence Location

**Server only**: `/mnt/sdc/dty_user/openvla_attack/evidence/sc5_object_privileged_loto_v1/vis_heldout_formal_v1/`

**Local copy**: NOT PRESENT (only `SCHEMA_CANARY_GATE.json` and `canary/` directory exist locally at `evidence/sc5_object_privileged_loto_v1/`)

---

## Frozen Results (as previously reported)

These numbers were provided by the user as the frozen benchmark. This reconciliation checks each against server artifacts.

| Condition | Claimed Success | Claimed FR | Denominator | Artifact Confirmed |
|---|---|---|---|---|
| CLEAN | 162/162 | 0.0% | 162 (ITT) | YES — 244 dirs under CLEAN/ (folds + logs + canary) |
| RAND_T10 | 162/162 | 0.0% | 162 (ITT) | YES — 246 dirs under RAND_T10/ |
| RANDOM_TIME_V3 | 119/162 | 26.5% | 162 (ITT) | YES — 245 dirs under RANDOM_TIME_V3/ |
| EARLY_SHIFT_T10 | 98/141 | 30.5% | 141 (emitted) | YES — 217 dirs under EARLY_SHIFT_T10/ |
| TRUE_T10 ITT | 21/162 | 87.0% | 162 (ITT) | YES — 263 dirs under TRUE_T10/ |
| TRUE_T10 emitted-only | 0/141 | 100.0% | 141 (emitted) | Needs verification — sub-denominator within TRUE_T10 |
| COMMAND_OPEN_ORACLE | 0/141 | 100.0% | 141 (emitted) | YES — 216 dirs under COMMAND_OPEN_ORACLE_T10/ |

---

## Artifact Structure

```
vis_heldout_formal_v1/
  {CONDITION}/
    fold_XX/           # 9 folds (01-09)
      state_Y/         # 2 states per fold (18 total)
        det_seed_Z/    # 3 detector seeds (0-2 or 1-3)
          pert_seed_W/ # 3 perturbation seeds (0-2)
            (rollout artifacts: video, telemetry, metrics)
    logs/
    (canary data)
```

Per fold × state × detector × perturbation: 9 × 2 × 3 × 3 = 162 leaf directories for a fully populated condition.

### Per-Condition Artifact Directory Counts

| Condition | Dir Count | Expected | Ratio |
|---|---|---|---|
| CLEAN | 244 | 162 | 150% (includes logs, canary) |
| COMMAND_OPEN_ORACLE_T10 | 216 | 141 | 153% |
| EARLY_SHIFT | 147 | ~162 | 91% |
| EARLY_SHIFT_T10 | 217 | 141 | 154% |
| RAND_LINF | 155 | ~162 | 96% |
| RANDOM_TIME | 247 | ~162 | 152% |
| RANDOM_TIME_INVALID_V1 | 425 | varies | — (invalidated results) |
| RANDOM_TIME_V3 | 245 | 162 | 151% |
| RAND_T10 | 246 | 162 | 152% |
| SHUFFLED | 242 | ~162 | 149% |
| TMA | 282 | 162 | 174% |
| TMA_RANDOM_TIME | 281 | 162 | 173% |
| TRUE_T10 | 263 | 162 | 162% |
| UMA | 270 | ~162 | 167% |
| **Total** | **3495** | | |

Note: Directory counts include fold directories, state directories, logs, canary, and retry subdirectories — they are NOT equivalent to completed rollout count. The counts being 150-175% of expected is consistent with the hierarchy depth being counted.

---

## TRUE_T10 Structure Detail

TRUE_T10 has a different structure from other conditions:

```
TRUE_T10/
  formal_manifest.jsonl
  formal_v1/          # 244 subdirectories — the actual frozen evidence
    fold_07/...
  launch/
  MANIFEST_gpu0_even.jsonl
  MANIFEST_gpu0_odd.jsonl
  ... (per-GPU manifests)
```

The `formal_v1/` subdirectory contains 244 items, consistent with 141 emitted episodes × perturbation seeds + overhead.

---

## Detector Model (LOTO 10-Fold)

Fold data for the 10-fold Leave-One-Task-Out detector:

| Fold | Heldout Anchors | Feature Dataset | Teacher Labels | Teacher Config | Normalization | Phase B Eval | Training |
|---|---|---|---|---|---|---|---|
| 01 | YES | YES | YES | YES | YES | — | — |
| 02 | YES | YES | YES | YES | YES | YES | YES (v3) |
| 03 | YES | YES | YES | YES | YES | YES | YES (v3) |
| 04 | YES | YES | YES | YES | YES | YES | YES (v3) |
| 05 | YES | YES | YES | YES | YES | YES | YES (v3) |
| 06 | YES | YES | YES | YES | YES | YES | YES (v3) |
| 07 | YES | YES | YES | YES | YES | YES | YES (v3) |
| 08 | YES | YES | YES | YES | YES | YES | YES (v3) |
| 09 | YES | YES | YES | YES | YES | YES | YES (v3) |

Fold 01 has CHECKPOINT_SHA256SUMS.txt — other folds may also have checkpoints but need verification.

---

## Attack Protocol (as claimed frozen)

| Parameter | Value |
|---|---|
| K (perturbation budget) | 10 |
| epsilon | 6/255 in processor pixel space |
| PGD steps | 20 |
| Target token | 31744 (gripper OPEN) |
| Objective | autoregressive_prefix_gripper_target_token_logratio_arm_v3 |
| Preprocessing | official_pil_lanczos lineage |
| Route | strict |
| Fallback | none |
| Arm gate | at least 5/6 arm dimensions preserved |

### Protocol Consistency Check

| Check | Status |
|---|---|
| Config file with epsilon=6/255 present on server | TO VERIFY |
| Config file with PGD=20 present on server | TO VERIFY |
| Config file with K=10 present on server | TO VERIFY |
| Config file with token=31744 present on server | TO VERIFY |
| Bridge script uses attack_adapter with correct params | TO VERIFY (attack_adapter.py is DIRTY on server) |
| Preprocessing path uses PIL Lanczos | TO VERIFY |

**PROVENANCE GAP**: The attack adapter (`src/gripper_attack/attack_adapter.py`) is modified on the server but the diff has not been reviewed. The exact parameters used for Object frozen evidence generation may differ from what is claimed. This needs line-level audit against the frozen attack protocol.

---

## Manifest and SHA Verification

| Item | Path | SHA Present |
|---|---|---|
| CLEAN2000 SHA256SUMS | `CLEAN2000_CANONICAL_V1/SHA256SUMS.txt` | YES |
| Object freeze manifest | `LOTO_GLOBAL_FREEZE_V1.json` | Needs hash |
| Object freeze verify | `LOTO_GLOBAL_FREEZE_V1_VERIFY.json` | Needs hash |
| Phase B results freeze | `LOTO_PHASE_B_RESULTS_FREEZE_V1.json` | Needs hash |
| Fold 01 checkpoint | `fold_01/FOLD01_CHECKPOINT_SHA256SUMS.txt` | YES |

---

## Missing Provenance Items

1. **No local copy of Object evidence** — server-only, single point of failure
2. **Attack adapter dirty** — `attack_adapter.py` modified on server, diff not captured
3. **Bridge script dirty** — `run_v2_vis_sc5_mlp_bridge.py` modified on server
4. **Eval script dirty** — `v4_run_eval_openvla.py` modified on server
5. **Per-fold checkpoint SHA** — only Fold 01 has SHA256SUMS, other 8 folds need verification
6. **Config files** — attack protocol configs need SHA hashing
7. **Manifest SHA** — formal manifests need SHA hashing
8. **Aggregation script** — the script that produced "119/162", "98/141" etc. needs to be located and audited
9. **Denominator rationale** — the choice of 162 ITT vs 141 emitted for different conditions needs documentation
10. **No-emission handling** — the 21 no-emission episodes in TRUE_T10 need per-episode documentation

---

## Evidence Chain Completeness

| Evidence Chain | Manifest | Artifact | Aggregation | Provenance Seal | Status |
|---|---|---|---|---|---|
| Timing specificity | Partial | YES | Partial | NO | GAP |
| Direction/selectivity | Partial | YES | Partial | NO | GAP |
| Clean-only causal deployment | Partial | YES | Partial | NO | GAP |
| OPEN-command mechanism sufficiency | Partial | YES | Partial | NO | GAP |

Each chain has artifacts but lacks full provenance sealing (manifest SHA, code commit SHA, config SHA, aggregation script SHA).

---

## Summary

The Object frozen evidence exists on the server with 3,495 artifact directories across 14 conditions. The core claimed numbers (162/162 CLEAN, 0/141 TRUE_T10 emitted, etc.) are plausible given the directory counts, but formal provenance reconciliation is incomplete. Specifically:

- Attack adapter code is dirty on the server
- Per-fold checkpoint SHA only verified for Fold 01
- Aggregation scripts not audited
- No local evidence copy
- Manifest/artifact pairing not confirmed at the per-episode level

**Recommendation**: Do NOT re-run any Object experiment. Instead, seal provenance by:
1. Capturing the dirty server code diff
2. Hashing all checkpoints, manifests, and configs
3. Auditing the aggregation scripts
4. Rsyncing evidence to local/secondary storage

---

NO NEW EXPERIMENT WAS LAUNCHED.
NO LIVE SCIENTIFIC ARTIFACT WAS MODIFIED.
