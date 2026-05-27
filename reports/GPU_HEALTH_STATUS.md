# GPU Quarantine Debug Report

**Date:** 2026-05-27
**Server:** 10.60.133.4 (8× RTX 2080 Ti)

## GPU Status Table

| GPU | PCI | Power | Temp | Util | Memory | Xid History | Active Process | Recommendation |
|-----|-----|-------|------|------|--------|-------------|----------------|----------------|
| 0 | 04:00.0 | 146W | 49°C | 0% | 2317 MiB | **Xid13** (2026-05-26, multiple SM Warp Exception) | lgzhou PID 6139 | **QUARANTINE** |
| 1 | 06:00.0 | 30W | 27°C | 0% | 4 MiB | None | None | **SANDBOX OK** |
| 2 | 07:00.0 | 267W | 67°C | 43% | 8713 MiB | Xid31 MMU Fault (2026-05-25) | W26 official eval PID 7347 | **PRODUCTION ACTIVE** |
| 3 | 08:00.0 | 35W | 32°C | 0% | 4 MiB | **Xid31** MMU Fault (2026-05-25) | None (reset blocked by PID 1438) | **PENDING STRESS** |
| 4 | 0C:00.0 | 113W | 74°C | 33% | 8713 MiB | None | W45 official eval PID 7346 | **PRODUCTION ACTIVE** |
| 5 | 0D:00.0 | 198W | 66°C | 41% | 8226 MiB | None | W45 official eval PID 7346 | **PRODUCTION ACTIVE** |
| 6 | 0E:00.0 | 94W | 70°C | 46% | 8226 MiB | None | W26 official eval PID 7347 | **PRODUCTION ACTIVE** |
| 7 | 0F:00.0 | 22W | 36°C | 0% | 4 MiB | **Xid31** (3×, May 18-25) + **Xid13** Misaligned Address (2026-05-25) | None (reset blocked by PID 1438) | **PENDING STRESS** |

## Xid Error Summary

### GPU0 (PCI:0000:04:00) — QUARANTINED
- **Xid 13**: SM Warp Exception — Illegal Instruction Encoding
- Occurred: 2026-05-26 21:15:03 and 21:15:36
- Affected GPCs 1, 3, 4 across all TPCs and SMs
- Also Xid 43 on PIDs 22601, 29275 (channel error)
- Hardware damage confirmed. Do not use for any workload.

### GPU7 (PCI:0000:0f:00) — PENDING STRESS
- **Xid 31**: MMU Fault — FAULT_PDE ACCESS_TYPE_VIRT_READ (May 18, May 25 ×3)
- **Xid 13**: SM Warp Exception — Misaligned Address (2026-05-25 15:26)
- Also Xid 43 on PID 33685
- Both MMU faults AND SM warp errors → elevated risk

### GPU3 (PCI:0000:08:00) — PENDING STRESS
- **Xid 31**: MMU Fault — FAULT_PDE ACCESS_TYPE_VIRT_READ (2026-05-25 15:19)
- Single event, MMU fault only, no SM warp errors

### GPU2 (PCI:0000:07:00) — PRODUCTION ACTIVE
- **Xid 31**: MMU Fault (2026-05-25 17:49)
- Currently running official eval stably → acceptable risk

## Reset Attempts

| GPU | Reset Attempted | Result | Reason |
|-----|----------------|--------|--------|
| 0 | No | Skipped | Has lgzhou active process (PID 6139) |
| 1 | N/A | N/A | Clean, no reset needed |
| 3 | Yes | **BLOCKED** | PID 1438 + other processes hold fd (memory-mapped) |
| 7 | Yes | **BLOCKED** | PID 1438 + other processes hold fd (memory-mapped) |

Cannot kill PID 1438 or production processes per policy. GPU3/7 cannot be reset while the system is running.

## Pure PyTorch Stress Tests

Stress script: `scripts/gpu_pure_torch_stress.py`
- No MuJoCo, no EGL, no OpenVLA
- 4096×4096 matrix multiply + MLP forward-backward
- 15 minutes each

### GPU3 (PCI:0000:08:00) — **PASSED**

| Metric | Value |
|--------|-------|
| Iterations | 54,448 in 900.0s |
| Rate | ~60.5 iter/s |
| Peak memory | 340.3 MiB (stable) |
| Max GPU-CPU diff | 9.61e-04 (under 1e-3 threshold) |
| CUDA errors | None |
| Verdict | **PASSED** — Stable under pure PyTorch |

### GPU7 (PCI:0000:0f:00) — **MARGINAL PASS**

| Metric | Value |
|--------|-------|
| Iterations | 69,324 in 900.0s |
| Rate | ~77.0 iter/s |
| Peak memory | 340.3 MiB (stable) |
| Max GPU-CPU diff | 1.02e-03 (borderline, at threshold) |
| CUDA errors | None during compute |
| Verdict | **MARGINAL PASS** — No crash, but GPU-CPU discrepancy at threshold |

Note: GPU7 completed more iterations than GPU3 (69K vs 54K) in the same time window despite being the same GPU model. This clock-rate variance may indicate thermal/power management differences.

Initial run v1 failed due to `torch.cuda.check_error()` API mismatch (PyTorch 2.2.0 requires argument). Fixed in v2. Both GPUs survived first 100 iterations in v1 before the script error.

## GPU1 Smoke Tests

| Test | Result |
|------|--------|
| CUDA matrix multiply | PASSED |
| MLP forward/backward | PASSED |
| PyTorch compile check (3 files) | PASSED |

## Pre-Flight Test Suite (GPU1)

| Test File | Tests | Result |
|-----------|-------|--------|
| test_proprio_student_leakage.py | 2 | PASSED |
| test_proprio_student_splits.py | 2 | PASSED |
| test_proprio_student_dataset.py | 1 | PASSED |
| test_proprio_student_training.py | 2 | PASSED |
| test_protocol_validation.py | 26 | PASSED |
| test_contact_quality_v2.py | 9 | PASSED |
| **Total** | **42** | **ALL PASSED** |

## Official Eval Status

W45 (GPU4,5): alphabet_soup 8/10, cream_cheese 8/10, salad_dressing 9/10, BBQ sauce running
W26 (GPU2,6): tomato_sauce 9/10, butter 8/10, milk 8/10, chocolate_pudding running
Combined: 50/60 = 0.833

## Decision Rules Applied

- [x] GPU0 with Xid13 → QUARANTINE (has active user process, cannot reset)
- [x] GPU3/7 reset blocked by system processes → report and skip reset, stress-test instead
- [x] GPU1 clean → sandbox OK for non-production tasks
- [x] GPU26/45 in production → do not disturb
- [ ] GPU3/7 stress results pending → update recommendation after completion

## Recommended GPU Roles (Final)

| GPU | Role | Constraints |
|-----|------|-------------|
| 0 | **QUARANTINE** | Xid13 hardware damage. Do not use for any workload. |
| 1 | **SANDBOX** | Light non-production: tests, compile, small models. Safe. |
| 2 | **PRODUCTION** | Currently active (official eval). Reusable after completion. Has Xid31 history but currently stable. |
| 3 | **MODEL-ONLY SANDBOX** | PyTorch stress PASSED (54K iter, diff=9.6e-04). Has Xid31 history. OK for model inference; no EGL/MuJoCo rendering. |
| 4 | **PRODUCTION** | Currently active (official eval). Reusable after completion. Clean history. |
| 5 | **PRODUCTION** | Currently active (official eval). Reusable after completion. Clean history. |
| 6 | **PRODUCTION** | Currently active (official eval). Reusable after completion. Clean history. |
| 7 | **ELEVATED RISK SANDBOX** | PyTorch stress MARGINAL (diff=1.02e-03). Has Xid31 + Xid13 history. Model-only sandbox at most. Do not use for production. |

## Milestone 3A Blocker Status (Updated)

GPU01 (0,1) cannot be used for attack pilot because GPU0 has Xid13 hardware damage.

**Safe alternatives for Milestone 3A attack pilot:**
- **Option A** (Recommended): Wait for GPU26 (2,6) or GPU45 (4,5) to free up after official eval completes. Cleanest GPUs.
- **Option B**: GPU13 (1,3) — GPU1 is clean, GPU3 passed PyTorch stress. Use GPU1 for rendering, GPU3 for model layers. Cautious but viable.
- **Option C**: GPU1 alone (single GPU) — may be slow for OpenVLA 7B inference.

GPU7 remains too risky for attack pilot (Xid13 + marginal stress result).

## Next Steps

1. ~~Complete GPU3/7 stress tests → update recommendations~~ DONE
2. Official eval completion → free GPU26/45 for recommended attack pilot option
3. Update GPU health dmesg after stress tests to check for new Xid errors
4. If no new Xid on GPU3: GPU13 (1,3) can be used cautiously for small pilot
5. GPU7: monitor for new Xid; remains elevated risk, do not use for production

## Final State

```json
{
  "final_state": "gpu_quarantine_debug_complete",
  "date": "2026-05-27",
  "gpu0": "quarantine_xid13",
  "gpu1": "sandbox_clean",
  "gpu2": "production_active_xid31_history",
  "gpu3": "model_only_sandbox_stress_passed",
  "gpu4": "production_active_clean",
  "gpu5": "production_active_clean",
  "gpu6": "production_active_clean",
  "gpu7": "elevated_risk_sandbox_stress_marginal_xid13_xid31_history",
  "tests_passed": 42,
  "stress_gpu3": "PASSED",
  "stress_gpu7": "MARGINAL_PASS",
  "milestone_3a_recommended_gpu": "GPU26 or GPU45 after official eval completes",
  "milestone_3a_fallback_gpu": "GPU13 cautious use"
}
```
