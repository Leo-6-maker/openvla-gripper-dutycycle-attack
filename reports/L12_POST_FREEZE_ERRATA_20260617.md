# Post-Freeze Erratum

**Date:** 2026-06-17

## Tag Identity

| Item | Value |
|------|-------|
| Production tag | `l12-d5-v1-production-20260617` |
| Tag commit | `593ffad` |
| Tag message | "D5 v1 production release — Layer 1/2 sealed" |

## Commit Reference Clarification

Three distinct commits appear in the release artifacts:

1. **Tag commit** (`593ffad`): The commit the tag points to. Contains the final
   bundle v1.0.0 with server-verified SHAs and the GPU regression result.

2. **Bundle `source_code_commit`** (`d1463d5`): The commit at which the bundle
   was authored. This differs from the tag commit because the bundle SHA was
   updated after initial creation (evidence SHAs computed from server-side files).

3. **Post-tag branch HEAD** (`4239a12`): The final acceptance report, timing
   handoff, and alignment tool were committed AFTER the tag was created.

The previous final acceptance report referenced commit `593ffad` which IS the
correct tag commit. No correction needed.

## Immutability Confirmation

- Tag `l12-d5-v1-production-20260617` has NOT been moved or overwritten.
- All work from this point forward is on branch `exp/l12-postfreeze-timing-panel-20260617`.
- D5 v1 checkpoint, tau, adapter, runtime, and detector are frozen and unchanged.
