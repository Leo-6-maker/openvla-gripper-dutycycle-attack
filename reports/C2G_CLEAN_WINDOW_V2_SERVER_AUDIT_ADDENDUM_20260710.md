# C2g Clean-Window Detector v2 — Server Audit Addendum

This addendum supplements `C2G_CLEAN_WINDOW_V2_IMPLEMENTATION_HANDOFF_20260710.md` with the executable CPU-only dry-audit path added after the initial handoff was written.

## New server-audit entry point

```text
tools/multisuite_detector/audit_c2g_clean_window_v2.py
```

The command:

- recursively discovers existing `episode_metadata.json` + `step_records.jsonl` pairs;
- derives mechanism type only from explicit/structured task metadata;
- never guesses mechanism from task index or natural-language keywords;
- selects up to one eligible and one boundary/unsupported episode per suite before filling any remaining dry-run slots;
- builds clean-only Teacher-v2 labels without OpenVLA, LIBERO, attack execution, or GPU use;
- freezes a SHA256 manifest of every metadata/step artifact read;
- audits known/null semantics, target/distractor identity, release-safe veto, fixed-B start uniqueness, and absolute-EEF-z shortcut violations;
- exits `0` only on PASS and `2` on HOLD/error;
- refuses to write audit outputs inside the repository worktree.

## Exact Codex command

Run from a clean checkout of the reviewed head:

```bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
INPUT_ROOT="/ABSOLUTE/READ_ONLY/PATH/TO/CLEAN_EPISODES"
AUDIT_ROOT="/ABSOLUTE/EXTERNAL/PATH/c2g_clean_window_v2_audit_$(date +%Y%m%d_%H%M%S)"

python tools/multisuite_detector/audit_c2g_clean_window_v2.py \
  --input-root "$INPUT_ROOT" \
  --output-dir "$AUDIT_ROOT" \
  --repo-root "$REPO_ROOT" \
  --episodes-per-suite 2 \
  --burst-length 10 \
  --strict-four-suites
```

Expected output files:

```text
clean_window_v2_audit_report.json
clean_window_v2_episode_summary.jsonl
clean_window_v2_dry_labels.jsonl
clean_window_v2_input_manifest.jsonl
clean_window_v2_read_errors.jsonl
clean_window_v2_violations.jsonl
```

## Required report checks

```text
status = PASS_C2G_CLEAN_WINDOW_V2_DRY_AUDIT
missing_suites = []
read_error_count = 0
violation_count = 0
uses_attack_outcome = false
openvla_inference_runs = 0
libero_rollouts_launched = 0
gpu_episodes_launched = 0
detectors_trained = 0
datasets_materialized = 0
```

For every selected mechanism-eligible episode, `known_rows` must be nonzero. Unsupported or unresolved episodes may remain fully unknown and are treated as abstention/boundary evidence, not negatives.

## Structured mechanism routing

`src/gripper_attack/c2g_clean_mechanism.py` provides the fail-closed route:

```text
structured target + destination             -> pick_place_transfer
multiple structured targets/subgoals        -> multi_object_transfer
structured articulated operator/fixture     -> articulated_object
structured hold/lift constraint              -> constrained_manipulation
ambiguous/unresolved/language-only metadata  -> unsupported_or_unknown
```

An explicit unknown mechanism value raises an error instead of silently creating a new class.

## Current authorization boundary

Passing this tiny CPU audit authorizes only a review of whether a small clean-label materialization plan is scientifically and operationally safe. It does not authorize:

```text
full CLEAN2000 materialization
model training
OpenVLA loading
LIBERO rollout
GPU execution
counterfactual replay
attack experiment
D7 replacement or modification
```
