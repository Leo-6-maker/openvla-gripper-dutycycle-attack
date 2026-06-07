# Experiment Status Snapshot Auto

CPU-only status snapshot. No GPU, rollout, VIS, watcher, or detector training was run.

## Artifact Status

- Batch3b / labels_v2: MISSING
- labels_v3 candidate: present (1 rows)
- Batch4 precheck: MISSING
- Batch4 VIS summary: present (0 rows)
- Phase E qpos audit: present (8 rows)
- detector v2 metrics: MISSING
- detector v3 metrics: MISSING

## GPU Blacklist

- GPU3 and GPU7 remain blacklisted.

## Next Actions

- Wait for Batch4 full VIS outputs, then run closeout and labels_v3 builder.
- Run Phase E qpos cache audit before aligned-window generator or canary.