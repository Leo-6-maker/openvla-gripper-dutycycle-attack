# C2g Static Task/Asset Inventory Readiness — 2026-07-10

## Status

```text
STATIC_INVENTORY_TOOL = PASS_LOCAL_SYNTHETIC
LIVE_BDDL_OPERATOR_CENSUS = NOT_RUN
LIVE_MUJOCO_NAME_CENSUS = NOT_RUN
LIBERO_ENVIRONMENT_CREATED = NO
SIMULATOR_STARTED = NO
OPENVLA_LOADED = NO
GPU_USED = NO
```

## Purpose

`tools/multisuite_detector/audit_c2g_static_assets.py` closes the remaining read-only vocabulary gap before any Teacher-v2 replay work. It scans existing files only and does not import or create a LIBERO environment.

The tool performs two inventories:

1. BDDL/PDDL/Lisp task files
   - parses S-expressions;
   - extracts operators from `:goal` forms;
   - counts operators globally and per task;
   - compares observed operators with the Teacher-v2 target resolver registry;
   - reports unsupported operators and parse failures.

2. MuJoCo XML assets
   - records body, geom, joint, and site names;
   - inventories finger/gripper/jaw-like names;
   - resolves left/right aliases using the Teacher-v2 contact helper;
   - reports unresolved finger candidates and XML parse failures.

Every scanned file is bound by relative path, size, and SHA256. The aggregate manifest hash is independent of absolute mount paths.

## Fail-closed behavior

The overall status is `HOLD_WITH_GAPS` when any required condition fails:

```text
NO_TASK_FILES
TASK_PARSE_ERRORS
UNSUPPORTED_GOAL_OPERATORS
NO_XML_FILES
XML_PARSE_ERRORS
UNRESOLVED_FINGER_ALIASES
NO_LEFT_FINGER_ALIAS
NO_RIGHT_FINGER_ALIAS
```

The command exits `2` on HOLD and `0` only on PASS.

## Server-side read-only command template

Paths must be resolved from the installed LIBERO package without creating an environment.

```bash
python tools/multisuite_detector/audit_c2g_static_assets.py \
  --bddl-root /path/to/libero/task_or_bddl_root \
  --xml-root /path/to/libero/assets_or_mujoco_root \
  --output-json /mnt/sdc/dty_user/openvla_attack_evidence/c2g/static_asset_inventory_<commit>.json
```

Multiple `--bddl-root` and `--xml-root` arguments are allowed.

Required review fields:

```text
status
task_inventory.file_count
task_inventory.parse_errors
task_inventory.observed_operators
task_inventory.unsupported_operators
xml_inventory.file_count
xml_inventory.parse_errors
xml_inventory.finger_candidates
xml_inventory.finger_aliases
xml_inventory.unresolved_finger_candidates
artifact_manifest_sha256
```

## Local synthetic validation

The focused test suite covers:

- logical and quantified BDDL goals;
- comment removal and unbalanced syntax;
- unsupported operator reporting;
- resolved left/right finger aliases;
- unresolved gripper-name reporting;
- missing-root fail-closed behavior;
- deterministic aggregate manifests and report SHA sidecars.

Executed locally:

```text
python -m unittest tests.test_c2g_static_asset_inventory -v
Ran 5 tests
OK
```

The server-mounted real inventory remains required before `NONPLACEMENT_TARGET_RESOLUTION` or `ROLE_AWARE_CONTACT_IDENTITY` can be promoted beyond static/synthetic status.

## Boundaries

This tooling does not authorize:

- deterministic restore;
- counterfactual replay;
- causal-label materialization;
- detector training;
- LIBERO rollout;
- OpenVLA inference;
- GPU experiments;
- D7 changes.
