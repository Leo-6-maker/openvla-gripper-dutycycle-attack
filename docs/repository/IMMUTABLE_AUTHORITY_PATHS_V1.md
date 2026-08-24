# Immutable Authority Paths V1

Status: `CODE_R1_AUTHORITY_FIREWALL_PASS`

Registry source:

- branch: `codex/repository-hygiene-paper-support-20260824`
- source HEAD: `a8f5a068e58f2aeef735d9e39966da8cffed6ce6`
- source tree: `950eeb6ce5491ad1fcfc442cb1bdd4cdf16521b5`
- machine registry: `docs/repository/IMMUTABLE_AUTHORITY_PATHS_V1.json`
- local immutable entries: 86
- immutable historical Git objects: 2
- root/sidecar pairs: 14
- merged reference observations: 120

This registry is an immutable firewall, not a cleanup allowlist. A path absent from the registry is not automatically safe to move, delete, or rewrite.

## Authority derivation

The 88 rows come from actual authority relations in:

- `paper/PAPER_V1_EVIDENCE_AUTHORITY_MAP_V1.json`
- `paper/PAPER_V1_FINAL_BUNDLE_MANIFEST_V1.json`
- `paper/PAPER_V1_FIGURE_TABLE_MANIFEST_V1.json`
- `reports/STAGE_X_X1R2_F1T_EVIDENCE_AUTHORITY_MAP_V1.json`
- `reports/STAGE_X_X1R2_F1T_ROOT_SEAL_V1.json`
- `reports/STAGE_Z_Z0R2_ROOT_SEAL_V1.json`
- `reports/STAGE_Z_Z0R2_ARTIFACT_MANIFEST_V1.json`
- the current Stage Z scientific-review handoff

Each machine row records its path, authority type(s), referencing artifact(s), path/byte/governance immutability flags, canonical SHA-256, Git blob, declared digest(s), and any historical digest basis.

The registry includes Paper V1 data, figures, tables, manuscript, claim ledger/audit, authority map, manifests/root seals; Paper V1 source handoffs and the two immutable Stage VIII Git objects; E3/E4 sources; F1/F1T authority and root members; the Paper V2 F1 delta; and the current Z0R2 protocol, manifest, authority maps, ledgers, and root seal.

## Digest contract

`current_sha256` is always computed from canonical Git blob bytes. The working tree is checked with Git's path filters, so a normal Windows CRLF checkout does not look like a scientific byte rewrite.

Paper V1's sealed authority map historically mixes two digest bases:

- canonical Git/LF bytes;
- CRLF materialization of the same tracked text.

V1 records the basis per declaration and verifies both deterministically. It does not rewrite the immutable Paper V1 map to normalize this historical convention.

## Static guard

Run:

```bash
python scripts/repository/audit_immutable_authority_paths.py
```

The audit fails closed when:

- a registered path is missing, renamed, or deleted;
- committed or working-tree canonical bytes differ from the registered Git blob;
- canonical or historically declared SHA-256 values differ;
- a historical Git object is unavailable or changed;
- a root-seal sidecar is missing or does not bind its registered artifact;
- an authority source is absent from the registry;
- the registry source HEAD is no longer an ancestor;
- an unimplemented compatibility mapping is introduced.

Current result:

```text
CODE_R1_AUTHORITY_FIREWALL_PASS entries=88 sidecar_pairs=14 git_objects=2
```

There are no approved move/rename compatibility mappings in V1.

## Deliberate non-authority exclusion

`reports/STAGE_Z_MULTI_MODEL_RUNNER_PREPARATION_STATIC_AUDIT_V1.json` is preserved but excluded. It is a generated engineering snapshot outside the `b16f1df`-bound Z0R2 root, no current authority artifact binds it, and its listed untracked `src/stage_z_preparation/__init__.py` is absent from Git. Exclusion means non-promotional, not deletable.

## R1 disposition

- registered immutable path/byte changes: 0
- missing registered files: 0
- digest conflicts after explicit Git/CRLF basis resolution: 0
- broken root/sidecar pairs: 0
- protected/scientific execution: 0

Result: `CODE_R1_AUTHORITY_FIREWALL_PASS`
