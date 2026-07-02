# Object Actual Protocol Audit 2026-07-02

Final classification: FROZEN_EMPIRICAL_RESULTS_PROTOCOL_PARTIAL_BOUNDED.

Revision 5 correction: source attribution is split into value source and formal binding source. Candidate values from `object_breadth_120.jsonl` are not called recovered unless binding to frozen formal Object artifacts is proven.

Resolution status counts:

```json
{
  "CANDIDATE_PROTOCOL_VALUE_UNBOUND": 5,
  "RECOVERED": 30,
  "UNRECOVERABLE_AFTER_TARGETED_SEARCH": 115
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

Rules now used:

- `FORMAL_ARTIFACT_BOUND`: value came from frozen formal episode summaries or per-row Object ledger SHA.
- `CANDIDATE_PROTOCOL_VALUE_UNBOUND`: value came from a candidate manifest, but formal binding is not proven.
- `NO_VALUE_TO_BIND`: targeted search found no usable value source.

Negative search ledger:

- `tables/server_freeze/object_protocol_negative_search_ledger.csv`

Claim boundary:

The Object results remain empirical legacy evidence. They are not promoted to a freshly preregistered fully recovered protocol. No Object rerun was started.
