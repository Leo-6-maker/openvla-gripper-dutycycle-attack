# K5c GPU Failure Audit — Worker_26 (GPU 2,6)

**Date**: 2026-06-09
**Commit**: ca3a97e
**Server**: klfy-SYS-4028GR-TR2

## Failed Jobs

| job_id | pair_id | condition | atk | original_n_steps | original_error | retry_gpu | retry_n_steps | retry_open | retry_valid |
|--------|---------|-----------|-----|-----------------|----------------|-----------|---------------|-------------|-------------|
| 520060 | k5c_cmd_milk_neg | vis_pgd | 0 | 116 | CUDA illegal memory access | 1,0 | 250 | 11 | valid |
| 520065 | k5c_cmd_milk_neg | random_linf | 2 | 123 | CUDA illegal memory access | 4,5 | 250 | 11 | valid |
| 520090 | k5c_cmd_alpha | vis_pgd | 0 | 70 | CUDA illegal memory access | 4,5 | 80 | 4 | valid |

All 3 failed jobs were on **worker_26 (GPU 2,6)**.
All 3 were **retried successfully** on healthy GPU pairs (4,5 or 1,0).
None of the original failed jobs are counted in pV/pR calculations.

## dmesg Analysis

- All Xid31 MMU Fault entries in dmesg are on **PCI:0000:07:00 (GPU 7)**, already blacklisted
- **No Xid errors on PCI:0000:02:00 (GPU 2) or PCI:0000:06:00 (GPU 6)**
- GPU 2,6 CUDA errors may be transient driver issues, not hardware MMU faults

## Disposition

- GPU 2,6: **degraded pending reboot and health check**
- GPU 3,7: **permanently blacklisted (Xid31 MMU fault)**
- GPU 1,0: healthy, no errors
- GPU 4,5: healthy, no errors (GPU 4 had one Xid43 at boot, no runtime impact)

## Retry Audit

All 3 retries produced valid results:
- 520060: retried on GPU 1,0 → open=11, n_steps=250, matches other VIS runs
- 520065: retried on GPU 4,5 → open=11, n_steps=250 (attack seed variability)
- 520090: retried on GPU 4,5 → open=4, n_steps=80, matches other VIS runs

**0 scientific failures — all failures are infra, all resolved by retry.**

## Retry GPU Provenance Note

The committed retry shell script (`run_k5c_retry_failed.sh`) uses GPU 4,5 for all 3 retries.
In execution, job 520060 was retried separately on GPU 1,0 (a manual retry command run after
the initial GPU 4,5 retry produced prefix mismatch and early termination for that specific job).
Jobs 520065 and 520090 were retried on GPU 4,5 as committed in the batch script.

The static retry audit CSV (`tables/stageb_v1_1_k5c_retry_audit_rc1a_ca3a97e.csv`)
accurately records the actual retry GPU pair used for each job.
