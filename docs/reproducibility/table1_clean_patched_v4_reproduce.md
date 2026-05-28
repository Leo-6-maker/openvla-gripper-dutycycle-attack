# Reproducing Table1 Clean Patched v4

This document describes how to reproduce the Table1 clean patched-v4 results. The eval uses the corrected official-eval-aligned runner (`tmp_official_eval.py`) with PIL Lanczos preprocessing.

## Official-Eval-Aligned Runner Command Template

```bash
CUDA_VISIBLE_DEVICES=<GPU_PAIR> MUJOCO_GL=egl python /tmp/official_eval.py \
  --model_path /data/aviary/models/openvla/openvla-7b-finetuned-libero-<suite> \
  --task_suite_name libero_<suite> \
  --num_trials_per_task 10 \
  --num_steps_wait 10 \
  --center_crop \
  --seed 7 \
  --attn_impl eager \
  --cuda_visible_devices <GPU_PAIR> \
  --render_gpu_device_id <RENDER_GPU_ID> \
  --output_root /data/liuyu/outputs/table1_<suite>_<date> \
  --run_id_prefix <suite>_official \
  --worker_id <suite>_<worker_id>
```

Per-suite model paths:
- Spatial: `openvla-7b-finetuned-libero-spatial`
- Object: `openvla-7b-finetuned-libero-object`
- Goal: `openvla-7b-finetuned-libero-goal`
- LIBERO-10: `openvla-7b-finetuned-libero-10`

Per-suite max steps (hardcoded in runner):
- Spatial: 220
- Object: 280
- Goal: 300
- LIBERO-10: 520

## Recovery Aggregator

After all workers complete, reconstruct final Table1 SR from worker stdout logs:

```bash
python scripts/recover_table1_from_logs.py
python scripts/generate_table1_final_report.py
```

The recovery aggregator parses `Task SR: X/Y = Z` patterns from worker log files. It does NOT trust stale partial CSV files, which may have been overwritten by parallel workers.

## Worker-Safe Shard Naming

As of commit ddfc219, the runner writes worker-specific shards:

```
tables/shards/manifest_{suite}_worker_{worker_id}.csv
tables/shards/summary_{suite}_worker_{worker_id}.csv
```

Each worker must use a unique `--worker_id`. Parallel workers writing to the same output root will NOT overwrite each other's data.

## Warnings

1. **Do not trust stale partial CSVs.** If any worker crashed or ran with incorrect GPU indices, its partial output may be present in the output directory but should be excluded from final aggregation.

2. **Always check worker logs**, not CSV files, for the authoritative SR. The log contains the `=== FINAL ===` marker and per-task SR lines that the recovery aggregator uses.

3. **GPU topology matters.** GPU indices shift after PCIe rescans. Always verify `nvidia-smi` and `nvidia-smi topo -m` before starting a run. The correct PCI mapping for the production server is:
   - GPU0: 04:00.0, GPU1: 06:00.0, GPU2: 07:00.0, GPU3: 08:00.0
   - GPU4: 0C:00.0, GPU5: 0D:00.0, GPU6: 0E:00.0, GPU7: 0F:00.0

4. **GPU0 and GPU7 are quarantined.** GPU0 has permanent Xid13 hardware damage. GPU7 has Xid13+Xid31 history. Do not use them for production runs.

## Protocol Constraints

- No attack conditions are enabled.
- No VIS/oracle/random/manual outcomes are used.
- Action path is `generate_manual_decode` (honest labeling).
- Preprocessing backend is `official_pil_lanczos`.
- JPEG round-trip is disabled.

## Boundary Statement

This reproduction procedure generates a clean denominator baseline for detector development and attack evaluation.
It does not run attacks.
It does not train detectors.
It does not use attack/oracle/random/manual outcomes.
