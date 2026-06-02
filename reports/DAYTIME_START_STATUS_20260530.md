# Daytime Start Status 2026-05-30

## GPU: All idle (0 jobs). No fresh Xid. Disk 35%.

## Completed Datasets
- Full10 Oracle: 100/100 (10 tasks × 5 states × clean/oracle)
- Detector-Clean Prep: 50/50 (10 tasks × 5 states × clean)
- Pilot V2: 24/24
- Oracle Expansion: 12/12

## Detector Ablation Readiness
- ProprioNoStep checkpoint: EXISTS (SHA: 4b3f3d47...)
- VisualNoStep checkpoint: EXISTS (SHA: 2d6defaa...)
- VisualProprioNoStep checkpoint: EXISTS (SHA: e496a4bf...)
- Object-100 labeled data: EXISTS
- Object-100 visual features: EXISTS
- Full10 step_records: EXISTS (150 episodes)

## Oracle Sensitivity (10 tasks)
| Rank | Task | Failure Rate | Class |
|------|------|-------------|-------|
| 1 | tomato_sauce | 4/5 | HIGH |
| 2 | cream_cheese | 4/5 | HIGH |
| 3 | butter | 3/5 | HIGH |
| 4 | chocolate_pudding | 3/5 | MED |
| 5 | bbq_sauce | 4/5 | MED |
| 6 | alphabet_soup | 3/5 | MED |
| 7 | milk | 2/5 | LOW |
| 8 | orange_juice | 1/5 | LOW |
| 9 | ketchup | 0/5 | ROBUST |
| 10 | salad_dressing | 0/5 | ROBUST |

## Current Claims
- detector-selected candidate windows show task-dependent oracle sensitivity
- sensitivity mediated by feedback burst, not trigger timing
- HIGH tasks avg burst 95 vs ROBUST avg burst 47

## Next: Detector Ablation Evaluation
Evaluate ProprioNoStep vs VisualNoStep vs VisualProprioNoStep on attack-relevance metrics using Full10 oracle outcomes.
