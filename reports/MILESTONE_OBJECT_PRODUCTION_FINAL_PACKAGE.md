# Object Production Final Package

**Date**: 2026-06-01 | **Status**: Production-ready for group meeting / paper

## Core Result

ProprioNoStep + sustained_command_open_proxy_30 selectively disrupts high oracle-sensitive Object tasks while preserving robust controls.

| Class | Tasks | sus30 Success |
|-------|-------|--------------|
| High | cream_cheese, tomato_sauce | **0/10** |
| Robust | ketchup, salad_dressing | **10/10** |
| Medium | alphabet_soup, bbq_sauce, butter, chocolate_pudding | 6/20 |
| Low | milk, orange_juice | 7/10 |

**Selectivity**: 100 percentage point gap (High 0% vs Robust 100%).

## Mechanism

1. **Detector**: ProprioNoStep — 13-dim proprio/action CausalTCNDetector. Fires at contact/transport/placement phase (step 100-160). Proprioceptive signal naturally encodes gripper-object contact dynamics.

2. **Attack**: sustained_command_open_proxy_30 — command-layer grip override. When detector triggers, gripper action set to fully open for 30 steps.

3. **Selectivity mechanism**: Detector fires uniformly; task dynamics determine failure vs survival. cream_cheese (deformable) and tomato_sauce (round, rolls) fail. ketchup (flat, stable) and salad_dressing survive.

## Ablation Summary

| Attempt | Result |
|---------|--------|
| ProprioNoStep standalone | Production — contact-phase |
| VisualNoStep V6 (frozen) | Pre-contact, non-selective |
| VisualNoStep_v2 (trained) | Step 4, 100% universal trigger |
| Proprio+Visual re-ranker | Scores zero at contact |
| Contact-aware delta training | Step 4, pre-contact |
| VIS PGD on verified frames | CE drops, gripper unchanged (0 token flips) |

## Why Proprio Works, Visual Doesn't

Proprio signal (gripper force, EEF velocity) changes WHEN contact begins. Visual signal (DINOv2+SigLIP scene appearance) peaks at episode start ("scene novelty") and decays monotonically. Contact-phase timing requires motion/force signal, not static appearance.

## Cross-Suite

Object-ProprioNoStep zero-shot transfer limited by workspace distribution shift (eef_z 5.7x). Spatial 65% partial transfer; Goal 8% insufficient.

## Valid Claims

- ProprioNoStep + sus30 selectively disrupts high-sensitive Object tasks
- Robust controls preserved
- Effect is Object-suite validated
- Static visual fails contact-phase timing across multiple attempts
- VIS token-prefix PGD does not flip decoded gripper token at ε≤8/255 on verified frames
- Cross-suite transfer is limited

## Forbidden Claims

- VIS attack successful / universal attack / detector oracle-optimal
- ProprioNoStep universal across LIBERO / cross-suite attack ready
- Command-layer sus30 equals VIS
