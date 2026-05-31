# VIS Contact-Frame Selection Audit

Date: 2026-05-31

## Scope

Read-only audit. No rollout, no training, no environment stepping, and no model
execution were run.

The purpose was to verify whether existing saved frames used by VIS no-rollout
diagnostics are actual contact/carry/pre-place frames.

## Output

```text
tables/vis_contact_frame_selection_audit.csv
tables/vis_contact_frame_selection_summary.csv
```

## Selector

Implemented:

```text
scripts/diagnostics/select_vis_contact_frames.py
```

The selector scores existing `step_records.jsonl` rows using clean artifact
fields such as:

- `phase`
- `policy_step_idx`
- `proxy_lift_carry_gate_active`
- `proxy_grasp_gate_active`
- `proxy_lift_carry_closed`
- `proxy_lift_carry_eef_z_delta_from_min`
- saved frame availability under `frames/step_XXXX.png`

It does not use VIS/oracle/sus30 outcomes as labels and does not select windows
from attack success.

## Audit Result

| Metric | Value |
| --- | ---: |
| rows total | 7 |
| frame available but not contact | 4 |
| contact candidate missing frame | 3 |
| contact candidate with frame available | 0 |

## Key Finding

The `milestone_2i_visual_fusion_online_detector_pilot_20260530` saved frames
used in previous one-frame VIS diagnostics are available, but they are all
`wait` / pre-policy frames:

```text
phase = wait
policy_step_idx = -1
```

Therefore the previous VIS no-rollout positives should be interpreted as
engineering/token-flip evidence, not contact/pre-place evidence.

## Contact Candidates Found

The selector found contact/carry candidates in existing Object clean artifacts,
but matching frame images are missing:

| Run | Candidate Step | Status |
| --- | ---: | --- |
| `obj_ketchup_s0` | 98 | candidate_missing_frame |
| `obj_tomato_sauce_s0` | 134 | candidate_missing_frame |
| `obj_cream_cheese_s0` | 143 | candidate_missing_frame |

These runs also do not contain videos or image files at shallow depth, so there
is no existing visual frame to feed into the OpenVLA re-decode diagnostic.

## Gate Decision

VIS contact-frame confirmation is blocked by missing verified contact-frame
images.

Do not run forced-window VIS micro.

## Next Required Action

Collect or reconstruct visual observations for the selected contact/carry steps:

1. Prefer artifact-rich clean rerun with frame dumps enabled for the selected
   Object tasks/states.
2. Minimum no-attack collection:
   - ketchup state0
   - tomato_sauce state0
   - cream_cheese state0
   - clean only
   - dump `frames/step_XXXX.png`
   - preserve full step records
3. After frames exist, rerun no-rollout VIS diagnostics on selector-chosen
   contact/carry steps.

No VIS rollout should be proposed until verified contact-frame no-rollout
evidence is available.
