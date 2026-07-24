# Codex Formal 25D Feature Extraction Handoff V1

Status: READY_FOR_REVIEW_SYNTHETIC_ONLY

Tasks:

```text
C2_FX_01 Source record schema audit
C2_FX_02 25D feature extraction interface
C2_FX_03 initial_state_hash provenance binding interface
C2_FX_04 feature extraction validator
C2_FX_05 formal extraction dry-run / synthetic e2e tests
```

## Scope

This batch adds a repository implementation path for producing the future
formal 25D per-step feature CSV. It does not execute formal extraction and does
not construct the formal detector dataset.

```text
Real feature extraction: NOT_PERFORMED
Formal detector dataset build: NOT_PERFORMED
Detector training: NOT_PERFORMED
OpenVLA/LIBERO: NOT_PERFORMED
Attack: NOT_PERFORMED
GPU/A800: NOT_PERFORMED
```

## Entry Points

```text
tools/multisuite_detector/extract_formal_25d_features_v1.py
tools/multisuite_detector/validate_formal_25d_features_v1.py
tools/multisuite_detector/audit_clean_rollout_feature_sources_v1.py
```

The extractor accepts only an existing clean source CSV with this exact schema:

```text
episode_key,parent_key,suite,task_id,initial_state_hash,trace_length,step,
source_record_path,source_condition,initial_state_hash_provenance,
<SC5_FEATURES[0..24]>
```

It emits the future C2 feature contract:

```text
episode_key,parent_key,suite,task_id,initial_state_hash,trace_length,step,
<SC5_FEATURES[0..24]>
```

## Boundaries

The extractor fails closed when:

- `initial_state_hash` is absent, malformed, or derived from episode key,
  parent key, or source path;
- source provenance is not one of the reviewed reset-state provenance values;
- source condition is not clean;
- source path is outside the approved clean-record root;
- source rows contain teacher, label, split, attack, adversarial, or future
  columns;
- feature order differs from `SC5_FEATURES`;
- feature values are missing, NaN, or Inf;
- step coverage is incomplete or duplicated;
- Label V2 episode set or parent/suite/task/trace identity does not match.

## Real-Data Status

The current server decision remains:

```text
REAL_FEATURE_ARTIFACT_BINDING = HOLD_NO_BINDABLE_ARTIFACT
recommendation = REEXTRACT_25D_FEATURES
```

Formal execution still requires a separate server authorization that binds:

- frozen clean rollout source records;
- approved clean-record root;
- initial-state provenance source;
- output path;
- exact extractor commit and SHA;
- independent validator command.

## Tests

Synthetic tests cover positive extraction/validation and fail-closed cases for
missing/reordered features, NaN/Inf, duplicate/missing steps, trace and identity
mismatches, invalid or forbidden initial-state hashes, attack/future columns,
SHA tamper, manifest path tamper, and validator mutation detection.
