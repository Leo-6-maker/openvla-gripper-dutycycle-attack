# Stage-B RC1a S6 Summary: K5c Expansion and Selector v0.3

**Date**: 2026-06-09
**Commit**: ca3a97e
**Branch**: exp/vis-prefix-margin-repair-20260603

## Executive Summary

S6 completed K5c targeted expansion (16 parents × K=5 = 160 jobs) and re-ran the leakage-free selector on the combined stable pool v2 (40 parents).

### Key Results

- **rand_sensitive**: 6 → 16 (target 10, significantly exceeded)
- **cmd_specific**: 11 → 15 (modest increase)
- **stable_negative**: 5 → 5 (unchanged)
- **strict_vis_phys**: 5 → 5 (no expansion — phys enrichment failed)
- **stable pool v2**: 40 parents across 9 tasks

### Old Label Corrections

K-repeat protocol corrected 2 old single-shot label errors:
1. milk[235,245]: old=NEG → K5=cmd_specific_borderline (VIS=11/11, one RAND outlier)
2. alphabet_soup[65,75]: old=CMD → K5=rand_sensitive (RAND dominates)

### Selector v0.3

| Strategy | rand_hit | cmd_hit | yield |
|----------|----------|---------|-------|
| Random | 0.38 | 0.50 | 0.30 |
| **Abstain(CleanRand)+Random** | **0.12** | **0.62** | **0.60** |
| Abstain(CleanRand)+TaskRank | 0.12 | 0.50 | 0.38 |
| Oracle UB | 0.00 | 1.00 | 1.00 |

- **Layer-1 abstain**: ROBUST — rand_hit consistently reduced to 0.12
- **Layer-2 cmd ranking**: DEGRADED — TaskRank now worse than Random after abstain (0.50 vs 0.62)
- **Best strategy**: Abstain(CleanRand)+Random
- CleanCmd OOF ranking remains WIP (cannot exceed task identity baseline)

### GPU Health

- 3 CUDA illegal memory access errors on worker_26 (GPU 2,6)
- All 3 retried successfully on healthy GPU pairs
- GPU 2,6: degraded pending reboot and health check
- GPU 3,7: permanently blacklisted (Xid31 MMU fault)

### Artifacts

```
tables/stageb_v1_1_k5c_queue_rc1a_ca3a97e.csv
tables/stageb_v1_1_k5c_job_audit_rc1a_ca3a97e.csv
tables/stageb_v1_1_k5c_retry_audit_rc1a_ca3a97e.csv
tables/stageb_v1_1_k5c_parent_probability_labels_rc1a_ca3a97e.csv
tables/stageb_v1_1_stable_parent_pool_k5_k5b_k5c_rc1a_ca3a97e.csv
tables/selector_v0_3_leakage_free.csv
reports/STAGEB_RC1A_CA3A97E_K5C_GPU_FAILURE_AUDIT.md
scripts/stageb/postprocess_k5c.py
scripts/diagnostics/run_selector_v0_3.py
scripts/diagnostics/audit_milk_anomaly.py
```

### Next

- Reboot server, GPU 2/6 health check
- Action-hidden sidecar for Layer-2 cmd ranking improvement
- Cross-suite only after Layer-0 mechanism eligibility
