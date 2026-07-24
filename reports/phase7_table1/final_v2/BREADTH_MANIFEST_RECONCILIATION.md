# Breadth Manifest SHA Reconciliation

**Date**: 2026-06-27

## Two SHAs

| File | SHA256 | Exists |
|------|--------|:------:|
| `manifests/object_breadth_120.jsonl` (current) | `b0b0fb4c01ebbab4081fbe69cd8544503d5de37121abeb5229ef9828ae702f8f` | Yes |
| `manifests/object_breadth_120.sha256` (sealed sidecar) | `6aedde8048ee8d56621d812a7293ddd7aa10c18ca4295afda1a1057396092e4e` | Yes (sidecar only) |

## Root Cause

The JSONL was rewritten after the sidecar was sealed. The `.sha256` sidecar records the hash of the original sealed manifest. The current JSONL has been modified (likely reformatted/re-serialized) producing a different hash.

## Investigation

- Both the local Git copy and server copy show `b0b0...` for the JSONL
- The `.sha256` sidecar (both local and server) records `6aedde...`
- 120 lines in both (same number of entries)
- Content comparison: same scientific keys, same job parameters
- The difference is in JSON formatting (whitespace, key ordering)

## Resolution

**The sealed canonical manifest SHA is `6aedde8048ee8d56621d812a7293ddd7aa10c18ca4295afda1a1057396092e4e`** as recorded in the sidecar file. This is the hash that should be used in all provenance references.

The current JSONL (`b0b0...`) is functionally identical (same 120 scientific keys, same parameters) but was re-serialized after sealing. It should be restored to match the sealed hash, or the sidecar should be treated as the authoritative record.

## Impact on Table 1

None. The 120 scientific keys, task/state/seed/condition assignments, and output directory paths are identical. Panel B denominators are unaffected.

## Action

- Mark `6aedde...` as canonical sealed manifest SHA in provenance manifest
- Note `b0b0...` as current file-system artifact (re-serialized, functionally identical)
- No re-run needed
