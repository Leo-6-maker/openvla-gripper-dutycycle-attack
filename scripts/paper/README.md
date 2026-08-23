# Paper-support scripts

Canonical read-only commands:

```text
python scripts/repository/audit_immutable_authority_paths.py
python scripts/paper/check_paper_v1_claims.py
```

`check_paper_v1_claims.py` reuses the historical claim parser and hard checks
without invoking any write/seal function. The repository audit verifies the
sealed byte/path chain.

Lifecycle of the older V1 scripts:

- `audit_paper_v1_claims.py` is the historical claim-ledger and final-bundle
  producer. It writes immutable Paper V1 files; do not run it for hygiene.
- `build_paper_v1_figures_tables.py` is an immutable V1 figure/table producer;
  do not rerun it.
- `audit_paper_v1_authority_map.py` and `audit_paper_v1_supplement.py` are
  historical byte-sensitive audits. The V1 authority chain contains declared
  hashes from both canonical Git and historical CRLF materializations, so they
  are not the cross-platform canonical entry point.

Paper V2 receives a deterministic CSV/JSON/TeX export surface in CODE-R5. Do
not mutate V1 builders or use the presentation repository as scientific source.
Its canonical command and lifecycle notes are in `scripts/paper_v2/README.md`.
