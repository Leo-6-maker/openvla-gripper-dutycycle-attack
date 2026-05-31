# Group Meeting Final Outline

**Date**: 2026-06-01

## Slide Outline

### 1. Problem: Gripper Duty-Cycle Vulnerability
- OpenVLA controls gripper through 7-dim action tokens
- Sustained open-grip commands during contact phase can disrupt task completion
- Question: can we selectively exploit this?

### 2. Production Detector: ProprioNoStep
- 13-dim proprio/action CausalTCNDetector (38,602 params)
- No RGB, no privileged state, no oracle labels
- Fires at contact/transport/placement phase (step 100-160)

### 3. Oracle Sensitivity Spectrum
- Full10 oracle: 100 rollouts across 10 Object tasks
- Task-dependent sensitivity: cream_cheese+tomato (high), ketchup+salad (robust)
- Vulnerability is not universal

### 4. Sustained Proxy Design
- sustained_command_open_proxy_30: burst_steps=30, hold_mode=fixed
- Command-layer proxy — not VIS, not PGD
- Only activates when ProprioNoStep triggers consecutively

### 5. Full10 Result: High 0/10, Robust 10/10
- 100 percentage point selectivity
- All high tasks hit max_steps (290) without placement
- All robust tasks complete early (step 143-202)

### 6. Why Proprio Works: Contact Dynamics
- Proprio signal (gripper position, EEF velocity, action forces) changes WHEN contact begins
- Visual signal (DINOv2+SigLIP) peaks at episode start and decays
- This is not about model complexity — it's about input domain

### 7. Why Static Visual Failed (4 attempts)
- VisualNoStep V6: pre-contact, non-selective
- VisualNoStep v2: step 4 universal trigger
- Re-ranker: scores zero at contact
- Contact-aware delta: step 4 universal trigger

### 8. VIS PGD Result: Loss Moves, Action Does Not
- White-box PGD confirmed working (dtype fix, re-decode path)
- On verified contact frames: CE drops 34→0, gripper stays 0.996
- Token-level attacks at ε≤8/255 cannot flip discrete token decisions

### 9. Cross-Suite Limitation
- Object-ProprioNoStep zero-shot: Spatial 65%, Goal 8%
- Root cause: eef_z distribution shift 5.7x
- Cross-suite sus30 blocked

### 10. Next: Paper Writing + Future Directions
- Paper: selective sustained proxy on Object
- CrossSuite-v2: relative EEF features
- VIS: larger ε or alternative objectives (future work)
