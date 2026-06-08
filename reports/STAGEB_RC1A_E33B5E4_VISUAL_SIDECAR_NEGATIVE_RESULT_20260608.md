# Stage-B RC1a e33b5e4 — Visual Sidecar Negative Result

**Date**: 2026-06-08
**Code commit**: e33b5e4
**Data anchor**: d4a3827
**Matched**: 69/72 windows with DINOv2-equivalent (OpenVLA SigLIP) embeddings

## Method

- OpenVLA built-in SigLIP vision backbone (2176-dim)
- Clean rollout frames at window_start, window_center, window_end
- RC1a-aligned: official task language, gripper postprocess, rot180 frames, AutoProcessor pixel_values
- GroupKFold by task_state_seed, fold-specific StandardScaler
- LogisticRegression, class_weight=balanced

## Results

### Head A: cmd_specific (N=54, pos=18)

| Feature Group | AUROC | P@5 | Enrich@5 |
|--------------|-------|-----|----------|
| **TaskOnly** | **0.695** | **1.00** | **3.0x** |
| CleanNoTaskNoTiming | 0.594 | 0.40 | 1.2x |
| VisualOnly | 0.611 | 0.40 | 1.2x |
| Visual+CleanNoTask | 0.625 | 0.40 | 1.2x |
| Visual+Clean+Task | 0.625 | 0.40 | 1.2x |

**TaskOnly > VisualOnly. Visual does NOT reduce task bias.**

### Head C: abstain_any (N=44, pos=8)

| Feature Group | AUROC | P@5 | Enrich@5 |
|--------------|-------|-----|----------|
| TaskOnly | 0.642 | 0.00 | 0.0x |
| **CleanNoTaskWithTiming** | **0.889** | **0.80** | **4.4x** |
| VisualOnly | 0.694 | 0.40 | 2.2x |
| Visual+CleanNoTask | 0.701 | 0.40 | 2.2x |

**Clean+Timing > Visual. Visual adds nothing to abstain prediction.**

### Head D: shared_qpos (N=42, pos=6)

All models near chance (AUROC 0.19–0.54). No signal.

## Conclusion

**Global OpenVLA SigLIP clean-frame embeddings do not reduce task bias for cmd_specific.**
Clean timing/proprio remains the strongest signal for abstain_any.
Visual sidecar is an offline negative ablation — not a deployed detector module.

This does NOT mean "visual features are useless." It means:
1. Global frozen SigLIP embeddings from a single center frame do not help under the current 69-window setting.
2. Future visual work should explore mechanism-specific features (gripper-object crop, action-token hidden states, OpenVLA decoding representations) rather than global scene embeddings.

## Allowed claims

- Global frozen visual embedding did not improve cmd_specific or abstain_any over clean proprio+timing features
- Abstain_any remains the strongest and most trustworthy detector signal (AUROC=0.889)
- Task bias in cmd_specific prediction is not resolved by current visual features

## Forbidden claims

- Visual features are useless for vulnerability detection (only this specific frozen-global approach failed)
- Detector is solved
- cmd_specific can be reliably predicted from clean features alone
