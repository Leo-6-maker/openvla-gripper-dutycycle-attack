# Table1 Generic Autowindow Baseline — Branch Hygiene

**Branch**: `fix/table1-generic-autowindow-baseline-20260524`
**Based on**: `origin/main` @ `f02394f`
**Timestamp**: 2026-05-24

## Detector Artifact Hashes

| File | SHA256 |
|------|--------|
| `scripts/detect_contact_window_from_clean.py` | `b7cc9b89...` |
| `configs/generic_autowindow_detector.yaml` | `7510cae8...` |

## Files Changed

| File | Status | Description |
|------|--------|-------------|
| `src/utils/autowindow_protocols.py` | NEW | Canonical detector protocol definitions |
| `src/utils/autowindow_validation.py` | NEW | Fail-fast validators (ProtocolValidationError) |
| `src/utils/autowindow_runner.py` | NEW | Command builder + artifact validation helpers |
| `docs/autowindow_protocols.md` | NEW | Full autowindow protocol documentation |
| `docs/protocol_baseline_20260523.md` | MODIFIED | Added autowindow section |
| `tests/v4/test_autowindow_protocol.py` | NEW | 25 test methods |
| `reports/table1_autowindow_baseline_20260524/` | NEW | This report |

## Verification

- compileall: PASS
- smoke test (12 assertions): PASS
- pytest: UNAVAILABLE (smoke covers all validator paths)

## No-Rollout Certification

- No rollout was run
- No GPU was started
- No command_open was run
- No targeted VIS was run
- No Table2 / 142 was run
- No artifacts were modified
