# Q3R2 Pool SHA Correction — 2026-08-20

The frozen 48-row engineering pool and its ordering are unchanged. The first
clean-run attempt correctly stopped before model loading because the protocol
used a Windows raw-file SHA while the Linux checkout materializes the same Git
blob with different line endings.

Canonical binding is now the cross-platform Git blob SHA:

`reports/STAGE_X_X1R2_Q3R2_ENGINEERING_FIXTURE_POOL_V1.json`

- Git blob SHA256: `2f567ccf16fdd42479a566b48a9ac9893fd027cb`
- observed Linux raw SHA256: `9177b54262acb969619b577f50c8e4aa58dfde83ce13a40434b8ab26386460b1`
- earlier Windows raw SHA256: `4a703459336c7fa1c93d7e8dc7fe6c9391ac9c3c2986bd5bf443d083ef7fa0cb`

The runner now verifies `git rev-parse HEAD:reports/...` against the Git blob,
so the authority is line-ending independent without weakening identity. The
failed attempt produced zero model inference, zero simulator steps, zero
Student forwards, zero attack calls, and zero exposed fixtures. The original
HOLD log remains preserved.
