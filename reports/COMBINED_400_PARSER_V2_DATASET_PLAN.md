# Combined 400-Episode Parser V2 Dataset Plan

## Suites
| Suite | Episodes | Parser V2 Status | Privileged Coverage | Teacher Labels |
|-------|----------|-----------------|--------------------|-----------------|
| Object-100 | 100 | Done | 100% | Done (v1=v2 verified) |
| Spatial-100 | 100 | Old labels diagnostic-only | ~high | Needs rerun |
| Goal-100 | 100 | **Running (19/100)** | Pending audit | Pending |
| L10-100 | 100 | **Running (9/100)** | Pending audit | Pending |

## Combined Dataset Plan

### Phase 1: Complete Suite Refreshes
- [ ] Goal-100 parser-v2 rerun finishes
- [ ] Goal-100 postprocess (audit → teacher → labeled)
- [ ] L10-100 parser-v2 rerun finishes
- [ ] L10-100 postprocess (audit → teacher → labeled)

### Phase 2: Spatial Decision
Option A: Use Spatial-100 old labels (privileged coverage was high, parser patterns robust).
Option B: Rerun Spatial-100 with parser v2 for full audit trail.
Decision: TBD after Goal/L10 audit. If Goal/L10 privileged coverage meets expectations, Spatial old labels are likely acceptable.

### Phase 3: Visual Features
- [ ] Object-100: Done (18,875 images, 2176-dim)
- [ ] Spatial-100: Done (13,773 features on GPU7)
- [ ] Goal-100: Extract on GPU7 after rollout completes
- [ ] L10-100: Extract on GPU7 after rollout completes

### Phase 4: Combined Dataset Assembly
1. Merge all 4 suite no_timestep datasets.
2. Unify feature schema:
   - 13 proprioceptive channels
   - 2176-dim frozen visual features
   - mechanism_type (from parser v2)
   - task_suite label
3. Leakage audit on combined schema.
4. Train/val/test split: stratify by suite + mechanism.

### Phase 5: Universal Detector Training
1. Universal ProprioNoStep (baseline)
2. Universal VisualNoStep
3. Universal VisualProprioNoStep
4. Suite-specific upper bounds
5. Leave-one-suite-out generalization

### Blockers
- Goal-100 and L10-100 must complete (ETA ~4-5h from 2026-05-29 10:30)
- Spatial decision needed before combined training
- GPU7 extraction queue: Goal (~2h) → L10 (~2h) after rollouts complete
