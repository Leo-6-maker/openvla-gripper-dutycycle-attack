# Object Actual Protocol Audit 2026-07-02

Final classification: FROZEN_EMPIRICAL_RESULTS_PROTOCOL_PARTIAL_BOUNDED.

Recovered fields include the Object runtime commit where present, checkpoint SHA, dataset SHA, preprocessing backend from episode summaries, and RAND_T10 breadth-manifest protocol values where the manifest binds the historical Object output roots.

Resolution status counts:

```json
{
  "RECOVERED": 35,
  "UNRECOVERABLE_AFTER_TARGETED_SEARCH": 115
}
```

Unrecoverable boundary:

Fields still marked `UNRECOVERABLE_AFTER_TARGETED_SEARCH` were checked against artifact metadata, condition roots, manifests, launch/log/script candidates, output-root string search, and git/runtime inventory evidence. They are closed as negative evidence for this audit, not silently inferred from current code or later PR drafts.

Claim boundary:

The Object results remain empirical legacy evidence. They are not promoted to a freshly preregistered fully recovered protocol. No Object rerun was started.

Artifact: `tables/server_freeze/object_protocol_provenance_ledger.csv`.
