# FIT670 V2 Formal Runbook

Legacy V1 canary/output roots are development-only. Do not resume them.
Run every command from one clean detached worktree at the final reviewed commit.
Do not commit or modify source between canary and formal collection.

## 0. Stop legacy workers and preserve old roots

Terminate only processes running `run_fit670_atomic_worker.py` from a pre-V2
commit. Preserve their outputs, but mark them `LEGACY_V1_NONCONSUMABLE`.

## 1. Rebuild a sealed 8-shard plan

The old `/tmp/fit670_shard_plan.json` is rejected because it has no immutable
root seal.

```bash
python n5/phase2_labels/build_fit670_shard_plan.py \
  --identity-allowlist \
  /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fit670_allowlist/FIT670_IDENTITY_ALLOWLIST.json \
  --n-shards 8 \
  --out /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fit670_shard_plan_v2
```

Use:

```bash
export FIT670_SHARD_PLAN=/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fit670_shard_plan_v2/FIT670_GPU_SHARD_PLAN.json
```

## 2. Build a canary-only V2 transition

Choose a new empty canary root and a new transition root. Use the exact clean
upstream and LIBERO repositories actually imported by the worker.

```bash
CANARY_ROOT=/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fresh670_v5_v2_canary
CANARY_TRANSITION=/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fit670_transition_v2_canary

python n5/phase2_labels/build_fit670_transition_v2.py \
  --mode canary \
  --identity-allowlist \
  /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fit670_allowlist/FIT670_IDENTITY_ALLOWLIST.json \
  --shard-plan "$FIT670_SHARD_PLAN" \
  --model-path /mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10 \
  --official-worker /mnt/sdc/dty_user/openvla_attack_official_v3_20260716/scripts/official_clean_worker.py \
  --registry-summary /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c1_v2_r7/run_A/ENTITY_REGISTRY_V2_SUMMARY.json \
  --alias-ledger /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c1_v2_r7/run_A/ALIAS_LEDGER.json \
  --upstream-root "$FIT670_UPSTREAM_ROOT" \
  --libero-root "$FIT670_LIBERO_ROOT" \
  --r5e-comparison-root /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r5e_r1/comparison \
  --physical-gpus 0,1,2,3,4,5,6,7 \
  --allowed-output-root "$CANARY_ROOT" \
  --out "$CANARY_TRANSITION"
```

## 3. Run and validate exactly 8 canary episodes

```bash
bash n5/phase2_labels/run_fit670_v2.sh \
  canary "$CANARY_TRANSITION" "$CANARY_ROOT" 0,1,2,3,4,5,6,7

python n5/phase2_labels/validate_fit670_canary_v2.py \
  --output-root "$CANARY_ROOT" \
  --identity-allowlist \
  /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fit670_allowlist/FIT670_IDENTITY_ALLOWLIST.json \
  --shard-plan "$FIT670_SHARD_PLAN" \
  --transition-receipt "$CANARY_TRANSITION"
```

Proceed only if the sealed review reports:

```text
status = PASS_ENGINEERING_CONSUMABLE_INPUT_GATE
n_episodes = 8
n_shards = 8
```

## 4. Build the formal V2 transition

Use a different, new empty output root. Formal transition generation will
fail unless it can recursively verify and bind the canary review.

```bash
FORMAL_ROOT=/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fresh670_v5_v2_formal
FORMAL_TRANSITION=/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fit670_transition_v2_formal

python n5/phase2_labels/build_fit670_transition_v2.py \
  --mode formal \
  --canary-review-root "$CANARY_ROOT/CANARY_REVIEW_V2" \
  --identity-allowlist \
  /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fit670_allowlist/FIT670_IDENTITY_ALLOWLIST.json \
  --shard-plan "$FIT670_SHARD_PLAN" \
  --model-path /mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10 \
  --official-worker /mnt/sdc/dty_user/openvla_attack_official_v3_20260716/scripts/official_clean_worker.py \
  --registry-summary /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c1_v2_r7/run_A/ENTITY_REGISTRY_V2_SUMMARY.json \
  --alias-ledger /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/c1_v2_r7/run_A/ALIAS_LEDGER.json \
  --upstream-root "$FIT670_UPSTREAM_ROOT" \
  --libero-root "$FIT670_LIBERO_ROOT" \
  --r5e-comparison-root /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/r5e_r1/comparison \
  --physical-gpus 0,1,2,3,4,5,6,7 \
  --allowed-output-root "$FORMAL_ROOT" \
  --out "$FORMAL_TRANSITION"
```

## 5. Run 670 and finalize

```bash
bash n5/phase2_labels/run_fit670_v2.sh \
  formal "$FORMAL_TRANSITION" "$FORMAL_ROOT" 0,1,2,3,4,5,6,7

python n5/phase2_labels/finalize_fit670_collection_v2.py \
  --output-root "$FORMAL_ROOT" \
  --identity-allowlist \
  /mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/fit670_allowlist/FIT670_IDENTITY_ALLOWLIST.json \
  --shard-plan "$FIT670_SHARD_PLAN" \
  --transition-receipt "$FORMAL_TRANSITION"
```

The only release condition is a sealed:

```text
FINALIZATION_V2/GLOBAL_MANIFEST.json
status = PASS_CONSUMABLE
n_identities_found = 670
```

Any worker nonzero exit, missing/duplicate identity, broken recursive seal,
entity without `logical_name`/`alias_to`, contact truncation, import-origin
mismatch, dirty source/upstream/LIBERO tree, or staging residue stops the run.
