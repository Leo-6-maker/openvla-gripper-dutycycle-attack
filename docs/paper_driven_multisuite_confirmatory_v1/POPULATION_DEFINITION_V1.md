# Population Definition V1

Status: PLANNING_ONLY

## Populations

| population_id | Definition | Allowed use |
|---|---|---|
| PRIMARY_ATTACK | clean success AND mechanism eligible AND V2 positive event AND exact-prefix reproducible | confirmatory attack denominator |
| ELIGIBLE_NO_EVENT | mechanism eligible AND no valid event | detector false-trigger and abstention evaluation |
| MECHANISM_INELIGIBLE | articulated, planar, unsupported, or mechanism absent | abstention and boundary evaluation |
| MULTI_EVENT | multiple object-transfer events | event-level exploratory evaluation |
| DETECTOR_ELIGIBLE | mechanism-eligible positives plus mechanism-eligible no-event rows | detector recall, precision, false-trigger, and no-emit metrics |
| DETECTOR_SAFETY | mechanism-ineligible and unsupported rows | correct abstention and safety metrics |
| DETECTOR_MULTI_EVENT | multi-event rows scored at event level | exploratory event-level detector analysis |

Cross-suite attack claims apply only to `PRIMARY_ATTACK`. Unsupported tasks must
not be used to fill attack denominators.

## Parent Selection

For each suite, target 20 primary parents, maximum 3 parents per task, and at
least 5 eligible tasks where available. If a suite has fewer than 20 legal
parents, use all legal parents and report the shortfall.
