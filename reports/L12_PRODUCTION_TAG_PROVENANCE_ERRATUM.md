# Production Tag Provenance Erratum

**Date:** 2026-06-17

## Tag Identity

```
Tag:      l12-d5-v1-production-20260617
Points to: 593ffad (bundle packaging commit)
Branch:   exp/l12-production-streaming-adapter-20260615
```

## Commit Distinctions

Three distinct commits appear in the release chain:

| Role | Commit | Description |
|------|--------|-------------|
| **Tag target** | `593ffad` | Commit the tag points to. Contains the final bundle JSON v1.0.0 with server-verified SHA256 values. |
| **Bundle `source_code_commit`** | `d1463d5` | The commit at which the bundle JSON was authored. Differs from tag target because the bundle SHA was updated post-authoring (evidence SHAs computed from server-side files). |
| **Post-tag branch HEAD** | `4239a12` | Final acceptance report, timing handoff v1, and alignment tool committed AFTER tag creation. |

## Immutability Confirmation

- Tag `l12-d5-v1-production-20260617` has NOT been moved or overwritten since creation.
- `git show l12-d5-v1-production-20260617:configs/d5_v1_production_bundle.json` produces the frozen bundle content.
- Bundle SHA (git object): `446f8aa98c22bb1367d18936642159535339601d`

## Clarification

Earlier project records referenced `4239a12` as the tag commit. This was incorrect — `4239a12` is the post-tag branch HEAD containing non-bundle documentation commits. The tag itself points to `593ffad`.
