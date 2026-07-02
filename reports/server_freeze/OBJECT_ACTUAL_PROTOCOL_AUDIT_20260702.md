# Object Actual Protocol Audit 2026-07-02

Final classification: FROZEN_EMPIRICAL_RESULTS_PROTOCOL_PARTIAL_BOUNDED.

Review6 correction: weak negative-search claims are demoted. Fields without formal value/binding evidence now use:

```text
UNRESOLVED_NO_FORMAL_BINDING_EVIDENCE
```

rather than `UNRECOVERABLE_AFTER_TARGETED_SEARCH`, because the committed ledger does not contain reproducible file counts, exact command ledger, candidate paths, or script SHA sufficient for a stronger negative proof.

Resolution status counts:

```json
{
  "CANDIDATE_PROTOCOL_VALUE_UNBOUND": 5,
  "RECOVERED": 30,
  "UNRESOLVED_NO_FORMAL_BINDING_EVIDENCE": 115
}
```

Binding status counts:

```json
{
  "CANDIDATE_VALUE_NOT_FORMALLY_BOUND": 5,
  "FORMAL_ARTIFACT_BOUND": 30,
  "NO_VALUE_TO_BIND": 115
}
```

Checkpoint/dataset rows now include actual SHA sets via:

- `unique_sha_count`
- `sha_value`
- `episode_count`

Additional artifacts:

- `tables/server_freeze/object_protocol_provenance_ledger.csv`
- `tables/server_freeze/object_protocol_negative_search_ledger.csv`
- `tables/server_freeze/object_protocol_sha_sets.csv`
