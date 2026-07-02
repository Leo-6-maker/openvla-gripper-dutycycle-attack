# Object Actual Protocol Audit 2026-07-02

Source: authoritative server Object evidence root, independently parsed from `episode_summary.json`.

## Recovered

- preprocessing_backend_requested: `['upstream_tf_jpeg']`
- preprocessing_backend_resolved: `['upstream_tf_jpeg']`
- preprocess_uses_jpeg: `['True']`
- actual_attn: `['eager']`
- actual_dtype: `['bfloat16']`
- checkpoint SHA values: 27 unique
- dataset SHA values: 9 unique

## Unresolved

The following fields remain `UNVERIFIED` because this pass did not recover sealed config/command evidence:

- epsilon
- epsilon_space
- PGD steps
- step size
- K
- target token
- objective
- strict route
- fallback policy
- arm gate
- attack config version
- launch command
- aggregation script

## Classification

`FROZEN_EMPIRICAL_RESULTS_PROTOCOL_PARTIAL`

Existing Object artifacts reproduce the reported outcome totals under an empirically recovered legacy preprocessing pipeline using `upstream_tf_jpeg`. The remaining attack hyperparameters and historical execution provenance are not fully sealed.
