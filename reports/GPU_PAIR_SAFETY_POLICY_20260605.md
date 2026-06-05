# GPU Pair Safety Policy 20260605

## Current Blacklist

- GPU3: Xid31 reproduced; permanently blacklisted.
- GPU7: Xid31 reproduced; permanently blacklisted.
- Pair 2,3: disabled because GPU3 is blacklisted.
- Pair 6,7: disabled because GPU7 is blacklisted.

## Reserved Pairs

- Pair 1,0: reserved for DeepSeek Batch3b/c gold VIS mainline.
- Pair 4,5: reserved for DeepSeek Batch3b/c gold VIS mainline.
- Pair 2,6: allowed only for Fast cascade after smoke. Codex must not run this pair.

## CUDA_VISIBLE_DEVICES Rule

Do not use:

```bash
CUDA_VISIBLE_DEVICES=2,6 ... --gpu_pair 2,6
```

or:

```bash
CUDA_VISIBLE_DEVICES=2,6 ... --gpu-pair 2,6
```

When `CUDA_VISIBLE_DEVICES=2,6` is set, CUDA remaps visible devices to logical IDs `0,1`.
If that mode were used, the internal argument would need to be `--gpu_pair 0,1`, but this is not recommended for the current Fast cascade plan because it is easy to confuse physical and logical IDs in logs.

Recommended Fast cascade convention:

- Do not set `CUDA_VISIBLE_DEVICES`.
- Pass physical pair `--gpu_pair 2,6` or `--gpu-pair 2,6` only from the server-side scheduler after smoke approval.
- Record the physical pair explicitly in every output CSV.

## Codex Boundary

Codex must not start GPU jobs, rollout, VIS, watcher, or detector v2 training in this phase.
Codex may only perform code audit, schema audit, dry-run interface checks, and small patches that prevent unsafe scheduling.
