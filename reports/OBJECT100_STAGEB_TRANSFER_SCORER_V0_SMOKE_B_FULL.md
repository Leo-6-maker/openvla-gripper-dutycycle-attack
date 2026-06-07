# Stage-B Transfer Scorer — Smoke-B (100 paired)

**EXPLORATORY ONLY — USES 1R PROVISIONAL LABELS — NOT FINAL DETECTOR**

**Paired**: 100, **Features**: 26, **Matched**: 100
**qpos**: FIXED for future runs / BLOCKED for existing 194 traces
**physical_response**: NOT TRAINED (qpos not reliable for existing data)

## Label Distribution

| Label | Count |
|---|---|
| cmd_susceptible | 27 |
| random_confounded | 16 |

## Leave-Task-Out AUROC

| Label | Model | AUROC (mean±std) |
|---|---|---|
| cmd_susceptible | LR | 0.4210 ± 0.1360 |
| cmd_susceptible | RF | 0.5596 ± 0.0838 |
| cmd_susceptible | TaskOnly | 0.5000 ± 0.0000 |
| cmd_susceptible | Shuffle | 0.4850 ± 0.2298 |

## Per-Task AUROC

| Task | LR | RF | TaskOnly | Shuffle |
|---|---|---|---|---|
| alphabet_soup | 0.537 | 0.463 | 0.5 | 0.4815 |
| bbq_sauce | 0.625 | 0.625 | 0.5 | 0.0 |
| butter | 0.4 | 0.5 | 0.5 | 0.525 |
| cream_cheese | 0.2 | 0.4857 | 0.5 | 0.5714 |
| ketchup | nan | nan | nan | nan |
| milk | nan | nan | nan | nan |
| orange_juice | 0.3393 | 0.6964 | 0.5 | 0.6071 |
| salad_dressing | 0.425 | 0.5875 | 0.5 | 0.725 |
| tomato_sauce | nan | nan | nan | nan |
