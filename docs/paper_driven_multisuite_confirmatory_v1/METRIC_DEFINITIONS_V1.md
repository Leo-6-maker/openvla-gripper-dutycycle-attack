# Metric Definitions V1

Status: PLANNING_ONLY

## Primary Endpoints

- `ITT_FR`: failures over all preregistered parent episodes.
- `CQFR`: contact-quality failures over all preregistered parent episodes.
- `official_SR`: simulator official success rate.
- `no_emit_rate`: detector no-emission episodes over eligible episodes.

## Detector Metrics

- `event_recall`: positive episodes with at least one valid detector trigger.
- `window_recall`: positive episodes with trigger inside teacher window.
- `recall_pm10`: trigger within 10 steps of teacher anchor.
- `false_trigger_rate`: no-event episodes with any trigger.
- `false_triggers_per_episode`: count of triggers on no-event episodes.
- `median_abs_timing_error`: median absolute trigger-anchor step gap.

## Mechanism Metrics

- `delta_open_duty`: attacked open-command duty cycle minus matched clean duty.
- `arm_NAD`: normalized arm action deviation versus exact-prefix clean replay.
- `Linf_actual`: observed per-frame perturbation Linf, not requested epsilon.

## Statistical Reporting

Every reported result must include numerator, denominator, 95% CI, and the
paired contrast used for the table. The primary endpoint is ITT; emitted-only
subgroups are descriptive.

