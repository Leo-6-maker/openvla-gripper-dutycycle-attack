# Artifact Schema V1

Status: PLANNING_ONLY

## Per Episode

- manifest row JSON;
- episode summary JSON;
- runtime telemetry JSONL;
- exact-prefix identity record;
- perturbation numeric telemetry;
- detector emit record;
- metric record;
- video or contact-quality audit pointer when required.

## Per Condition

- frozen manifest;
- SHA256SUMS;
- aggregate metrics;
- denominator ledger;
- retry/quarantine ledger when applicable;
- provenance envelope with code, config, dataset, checkpoint, and schema SHAs.

## Rule

No table cell may cite a result without a denominator ledger and a source
artifact SHA.

