# Git Provenance Blob Equivalence Audit

**Generated**: 2026-05-29 20:20 CST
**Server**: 10.60.133.4
**Repo**: /data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524

---

## Summary

| Field | Value |
|-------|-------|
| Server HEAD | `c62214fabe3d9991029a3a450a9a1f0f4de75f14` |
| Remote audited freeze | `087044390498f271fddfe009d95e6701fc9450fd` |
| Branch | `eval/official-libero-clean-20260525` |
| Working tree | Clean (only untracked) |

## Blob Equivalence Results

All 6 pilot-critical core files compared via `git hash-object` against blob SHAs from remote audited freeze commit `0870443`:

| File | Expected Blob | Actual Blob | Match |
|------|--------------|-------------|-------|
| `scripts/run_official_eval_artifact_rich.py` | `78e9b0db` | `78e9b0db` | ✅ |
| `scripts/launch_object_matched_attack_pilot_gpu26.sh` | `68ec0223` | `68ec0223` | ✅ |
| `scripts/aggregate_object_attack_pilot.sh` | `2e740891` | `2e740891` | ✅ |
| `scripts/evaluate_cq_object_attack_pilot.sh` | `8ad331a5` | `8ad331a5` | ✅ |
| `src/gripper_attack/triggers.py` | `761d01be` | `761d01be` | ✅ |
| `tests/v4/test_success_predicate_regression.py` | `733d10c0` | `733d10c0` | ✅ |

**ALL_MATCH = TRUE** — 6/6 files blob-identical.

## Supplementary Checks

| Check | Result |
|-------|--------|
| py_compile `run_official_eval_artifact_rich.py` | ✅ OK |
| py_compile `triggers.py` | ✅ OK |
| unittest `test_success_predicate_regression.py` | ✅ 6/6 OK |
| bash -n `launch_object_matched_attack_pilot_gpu26.sh` | ✅ OK |
| bash -n `aggregate_object_attack_pilot.sh` | ✅ OK |
| bash -n `evaluate_cq_object_attack_pilot.sh` | ✅ OK |

## Final Decision

**final_state = content_equivalent_to_remote_freeze**

**P0 Git blocker RESOLVED.**

The server commit `c62214f` differs in SHA from remote audited freeze `0870443` because it includes a non-overlapping change (git branch command compatibility fix in launch script). However, all 6 pilot-critical core files are blob-identical to the remote audited freeze. No code difference affects correctness, success predicate, attack logic, or provenance.

### Manifest Recording Rules

For all future pilot v2 run manifests:
- `runner_commit = c62214fabe3d9991029a3a450a9a1f0f4de75f14`
- `audited_remote_equivalent_commit = 087044390498f271fddfe009d95e6701fc9450fd`
- `blob_equivalence_audited = true`
- `blob_equivalence_files = 6/6`

The SHAs are NOT identical. The pilot-critical file contents ARE identical.
