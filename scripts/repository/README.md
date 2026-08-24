# Repository audits

Canonical command:

```text
python scripts/repository/audit_immutable_authority_paths.py
```

The audit verifies the machine-readable authority registry in
`docs/repository/IMMUTABLE_AUTHORITY_PATHS_V1.json`. It is read-only and
standard-library-only. A PASS protects path and byte identity; it does not
authorize execution or establish a scientific claim.

Update the registry and auditor together only when a prospective governance
change has explicit authority. Never use this family to rewrite historical
seals, sidecars, manifests, claim ledgers, or handoffs.
