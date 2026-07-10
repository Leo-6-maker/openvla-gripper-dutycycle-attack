# C2g Detector-v2 — S1 Remediation and Bounded Resume

Date: 2026-07-10

## Repository state

```text
source_pr = #58 (Draft, unmerged)
remediation_pr = #59 (Draft, unmerged)
server_branch = codex/c2g-strict-server-smoke-20260710
server_fix_base = 3eb55e6fc6849b740f6c5395ca3679048a7f6b09
scientific_contract_changes = none
D7_TABLE1 = STILL_FROZEN
```

Always fetch and bind the exact current remote server-branch head. Do not hard-code the
head recorded in this document after the branch moves.

## Audit of the first Codex return

The first stop decision was correct. S0 passed, while S1 had three independent
fail-closed blockers:

1. the audited Goal manifest no longer matched shards 1–3;
2. official BDDL uses compact `turnon` syntax while the canonical resolver supported
   `turn_on`;
3. official Panda XML exposes numbered jaw names `finger_joint1[_tip]` and
   `finger_joint2[_tip]`.

The two-line Codex engineering fix at `3eb55e6f` is accepted. It preserves package
imports for the release job builder and restores suite provenance from the frozen
five-part parent-key namespace when compact audit rows omit the duplicate suite field.

## Implemented semantic remediation

### BDDL syntax

`turnon` and `turnoff` are explicit syntax aliases for the existing canonical
`turn_on` and `turn_off` operators. This does not add a new Teacher label rule and does
not use language-only inference. Unknown predicates remain fail-closed.

Collection canonicalizes the aliases before target resolution. The strict read-only
inventory accepts only these explicit aliases and records the alias contract in its
report.

### Panda jaw identity

The numbered Panda components are treated as two deterministic jaws:

```text
finger_joint1 / finger_joint1_tip -> left jaw identity
finger_joint2 / finger_joint2_tip -> right jaw identity
```

The downstream bilateral-contact test is symmetric, so this naming convention does not
assert a geometric handedness claim. It only prevents the two physically distinct jaws
from being collapsed into an unresolved class.

## Goal provenance policy

Do not edit the old manifest in place and do not silently bless the changed shards.
Use the following decision order.

### Option A — restore or locate the frozen bytes

Search read-only server roots for candidate files matching the old manifest hashes:

```text
model-00001-of-00004.safetensors = d76243af9294ac2f069aef97de74ad4f84c17cb6fae0bda72169ae198f2c3bd8
model-00002-of-00004.safetensors = 74eed479d3a21e2a24cb21a0bdf565e52ea8f8c55f60237d74e657b7bea223c2
model-00003-of-00004.safetensors = 12b45ae85535565148907e379d6f35bcba83b9497006ff3b66658fa412ffac85
model-00004-of-00004.safetensors = d3b8e759db56b86709c56f8d156952415dd64e101feccfd01e701487041dd8c1
```

Prefer a complete alternate model directory. Do not overwrite the current model folder
until the candidate directory itself passes the full strict model audit.

### Option B — explicit C2g-only rebase of current Goal bytes

Use this only when no complete frozen copy is available.

First run the new read-only static integrity audit:

```bash
python tools/multisuite_detector/audit_c2g_goal_model_integrity_v2.py \
  --model-path /mnt/sdc/dty_user/openvla_attack/models/libero-goal \
  --previous-manifest artifacts/goal_model_manifest.json \
  --output-report "$WORK_ROOT/s1/goal_model_static_integrity_v2.json"
```

Required static gate:

```text
status = PASS_C2G_GOAL_MODEL_STATIC_INTEGRITY_V2
all index-referenced shards exist
safetensors headers parse
index tensor set equals shard-header tensor set
config architecture is compatible
libero_goal normalization statistics exist
prior byte mismatches remain explicitly listed
OpenVLA model loads = 0
LIBERO rollouts = 0
```

Then run exactly one Goal load-only validation and finalize a new external manifest:

```bash
python scripts/stageb/finalize_c2g_goal_model_manifest_v2.py \
  --static-report "$WORK_ROOT/s1/goal_model_static_integrity_v2.json" \
  --model-path /mnt/sdc/dty_user/openvla_attack/models/libero-goal \
  --output-manifest "$WORK_ROOT/s1/goal_model_manifest_v2.json" \
  --device cuda:0 \
  --rebase-approval C2G_GOAL_MODEL_REBASE_20260710
```

The token authorizes only a new **C2g Detector-v2 Goal baseline** after static and
load-only PASS. It does not claim continuity with the old Goal bytes, does not modify
D7, and does not authorize a LIBERO rollout or attack.

Required final manifest gate:

```text
status = PASS_C2G_GOAL_MODEL_INTEGRITY_AUDITED_V2
provenance_mode = EXPLICIT_REBASE_CURRENT_BYTES or RESTORED_FROZEN_BYTES
load_only_validation.status = PASS_C2G_GOAL_MODEL_LOAD_ONLY
parameter_count > 0
token_semantics_sha256 is nonempty
LIBERO environments created = 0
LIBERO rollouts launched = 0
attacks launched = 0
```

Set:

```bash
export GOAL_MODEL_MANIFEST="$WORK_ROOT/s1/goal_model_manifest_v2.json"
```

The strict model-map validator now recomputes every file hash in any real manifest file
ledger. A stale manifest therefore cannot pass merely because its top-level status says
PASS.

## Required rerun sequence

### R0 — fetch and rerun S0

```bash
git fetch origin --prune
git checkout codex/c2g-strict-server-smoke-20260710
git reset --hard origin/codex/c2g-strict-server-smoke-20260710
export C2G_HEAD="$(git rev-parse HEAD)"
git status --short
git diff --check

python -m py_compile \
  src/gripper_attack/c2g_semantic_aliases.py \
  src/gripper_attack/c2g_bddl_metadata.py \
  src/gripper_attack/c2g_teacher_v2_contact_identity.py \
  tools/multisuite_detector/audit_c2g_static_assets_strict.py \
  tools/multisuite_detector/audit_c2g_goal_model_integrity_v2.py \
  scripts/stageb/finalize_c2g_goal_model_manifest_v2.py \
  scripts/stageb/build_c2g_suite_model_map.py

python -m unittest discover -s tests -p 'test_c2g*.py' -v
bash -n scripts/stageb/run_c2g_clean_window_pipeline.sh
bash -n scripts/stageb/run_c2g_clean_window_pipeline_strict.sh
```

R0 must preserve a clean worktree.

### R1 — rerun alias-aware static asset inventory

Use the same official roots as the first S1 run:

```bash
python tools/multisuite_detector/audit_c2g_static_assets_strict.py \
  --bddl-root /ABSOLUTE/OFFICIAL/BDDL/ROOT \
  --xml-root /ABSOLUTE/OFFICIAL/XML/ROOT \
  --output-json "$WORK_ROOT/s1/static_asset_inventory_strict.json"
```

Required:

```text
40 BDDL files parsed
107 XML files parsed
parse errors = 0
unsupported operators = []
unresolved finger candidates = []
left and right jaw aliases nonempty
turnon appears only in semantic_alias_contract, not unsupported_operators
```

### R2 — resolve Goal provenance

Perform Option A first. If unavailable, perform Option B exactly as specified above.
Do not proceed with a merely edited copy of the legacy manifest.

### R3 — strict model map

With the restored or v2 external Goal manifest:

```bash
bash scripts/stageb/run_c2g_clean_window_pipeline_strict.sh models
```

Required:

```text
PASS_C2G_STRICT_SUITE_MODEL_MAP
PASS_C2G_STRICT_SUITE_MODEL_VERIFICATION
all four suites present
all weight shards hash-bound
Goal manifest current-byte verification PASS
```

### R4 — bounded continuation

After R0–R3 pass, resume the original strict handoff at S2. Keep all existing caps:

```text
new CLEAN training episodes <= 40
CLEAN evaluation parents <= 4
training epochs = 1
matched online parents <= 1
attacked runs for the selected parent = 4
full matrix = 0
counterfactual replays = 0
D7 modifications = 0
```

Stop at the first later fail-closed gate and return exact evidence.

## Required resumed report

```text
REMOTE_HEAD
EXECUTED_HEAD
NEW_FIX_COMMITS
S0_TEST_COUNT
S0_STATUS
STRICT_STATIC_ASSET_STATUS
UNSUPPORTED_OPERATORS
UNRESOLVED_FINGER_ALIASES
GOAL_PROVENANCE_MODE
GOAL_STATIC_REPORT_SHA256
GOAL_FINAL_MANIFEST_SHA256
GOAL_OPENVLA_LOADS
GOAL_LIBERO_ROLLOUTS
S1_STATUS
S2_AND_LATER_STAGE_STATUS
CLEAN_TRAIN_EPISODES_LAUNCHED
CLEAN_EVAL_EPISODES_LAUNCHED
ATTACKED_EPISODES_LAUNCHED
TRAINING_EPOCHS
P0_FINDINGS
P1_FINDINGS
SCIENTIFIC_CONTRACT_CHANGES = NONE / HOLD
D7_TABLE1 = STILL_FROZEN
GO_HOLD_NEXT_STAGE
```
