# [DeepSeek] H0.2 Provenance Remediation Report

Date: 2026-07-26
Branch: `deepseek/h0-2-remediation-20260726`
Parent commit: `00c54f93c1dbd00b2ea42e34de7cdfbbad9d3756`
Producer: DeepSeek

## Verdict

```text
H0.2                            = HOLD_REVIEW
C1 PROVENANCE                   = UNRESOLVED
C3-S2 / C3-G authorization      = NOT GRANTED
T2R-D unblinding                = NOT AUTHORIZED
training / inference / attack   = NOT AUTHORIZED
```

## Phase 1: Codex Audit Verification

All five Codex H0.1 audit files verified against published SHAs:

| File | SHA256 | Match |
|------|--------|-------|
| `CODEX_H0_1_AUDIT_REPORT.md` | `26b932a9...` | YES |
| `CODEX_H0_1_AUDIT_RECEIPT.json` | `9f4c7175...` | YES |
| `C1_PROVENANCE_HOLD_PACKET.json` | `bcccc9e9...` | YES |
| `PROTOCOL_AMENDMENT_V4_PROVENANCE_CORRECTION.json` | `7ff76196...` | YES |
| `test_h0_contracts_v2.py` | `080026e0...` | YES |

Audit commit `c6fdadbed2894aa6f9e5983fde19fa4075a6cfa8` verified.
Receipt self_sha256 `c1a7ce9a...` verified under canonical algorithm.

## Phase 2A: Receipt Self-Hash Algorithm

**Resolution**: All H0.2 receipts use the Codex canonical algorithm:

```
remove top-level self_sha256
json.dumps(obj, sort_keys=true, separators=(",", ":"), ensure_ascii=false)
SHA256(UTF-8 bytes)
```

Historical H0 receipts retained unmodified in `n5/phase3_student/h0_evidence_baseline/`.

## Phase 2B: C1 Provenance

**Result: PROVENANCE_FAIL**

### Script Version Audit

Three distinct C1 script versions identified:

| Version | SHA256 | Origin | Produces |
|---------|--------|--------|----------|
| Claimed b5c9634 | `3360bb17...` | Local repo at commit b5c9634 | Unknown (not runnable on server) |
| Server original | `b0567f3d...` | `/mnt/sdc/.../t2rc1_full_registry.py` | 40/40 artifact with status:FAIL (bug) |
| Current (00c54f9) | `953e2b6e...` | This remediation branch | 26/40, 15 unresolved |

### Server Rerun

- Server host: `pm-364c0001`
- Server repo HEAD: `68a8af0` (dirty, unrelated to any known integration commit)
- Claimed source commit `b5c9634`: **NOT RESOLVABLE** in server repo
- Python: 3.10.16, NumPy: 1.26.4, MuJoCo: 3.9.0

Rerun output (fresh directory `c1_h0_2_rerun_v2/`):
- 26/40 tasks OK
- 38 supported placement, 2 articulated
- 15 unresolved (all libero_spatial tasks)
- 0 blocked, 0 env errors
- status: FAIL

### Root Cause

Libero_spatial tasks have BDDL targets like `plate_1` and `akita_black_bowl_1` that do not EXACT-match MuJoCo body names (`plate_1_main`, `akita_black_bowl_1_main`). The STRICT resolution policy correctly rejects substring/body fallback, but the naming gap means these targets cannot be resolved without an explicit BDDL-to-MuJoCo body name alias map.

### Required Resolution

Either:
1. Restore the original working script (SHA `b0567f3d...` or `3360bb17...`) with verified SHA and documented resolution policy, OR
2. Accept 26/40 with documented naming gap and create explicit BDDL-to-MuJoCo body name alias map

## Phase 2C: SHA Binding

All H0.2 receipts bind to explicit commit `00c54f93c1dbd00b2ea42e34de7cdfbbad9d3756`. Server artifact SHAs recorded where available. File bindings are to committed blobs, not floating HEAD references.

## Phase 2D: Protocol Provenance

Codex PROTOCOL_AMENDMENT_V4_PROVENANCE_CORRECTION adopted. V3 preserved unmodified. Key disclosure: the 800-identity allocation was operationally used but lacks independent pre-training chronological proof in the repository.

## Phase 2E: V2 Server Tests

**NOT EXECUTED**. Blocked by C1 PROVENANCE_FAIL. Per protocol, any critical runtime test that cannot be executed means H0.2=HOLD. The V2 tests require a full server environment with numpy, torch, LIBERO, MuJoCo, and project imports.

Tests to execute after C1 resolution:
1. import/path validation
2. receipt self-hash verification
3. C2 pre-action AST/order
4. Teacher success/terminal invariance perturbation
5. N5 full/prefix/streaming logit parity (RF32/Dual, 12 prefix lengths)
6. short sequence non-blocked by RF128

## Phase 2F: Runtime State

`N5_RUNTIME_STATE_V1.json` created as a separate immutable state file. V4 plan preserved unmodified as historical document.

## Deliverables

| File | Status |
|------|--------|
| `H0_2_REMEDIATION_REPORT.md` | Written |
| `H0_2_REMEDIATION_RECEIPT.json` | Written |
| `C1_RERUN_RECEIPT.json` | Written |
| `TEST_H0_V2_RUN_RECEIPT.json` | Written (NOT_EXECUTED) |
| `N5_RUNTIME_STATE_V1.json` | Written |
| `H0_2_HOLD_PACKET.json` | Written |
| `PROTOCOL_AMENDMENT_V4_PROVENANCE_CORRECTION.json` | Adopted from Codex |
| `test_h0_contracts_v2.py` | Adopted from Codex |

## Next Step

This remediation branch is submitted for Codex independent H0.2 review. **Do not merge. Do not self-declare Joint PASS.** Await Codex verification before any C3-S2, C3-G, or T2R-D work.
