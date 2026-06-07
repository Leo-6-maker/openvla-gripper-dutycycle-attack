# Current VIS Attack Scope Audit

**Date**: 2026-06-05  
**Source**: `src/gripper_attack/attack_adapter.py` (TokenPrefixPGDAttacker), `scripts/vis_rollout_adaptive_v3.py`  
**Auditor**: Claude (server-side execution)

---

## 1. Current PGD Spatial Scope

**Full 224×224 image, no localization.**

`TokenPrefixPGDAttacker._build_inputs_and_labels()` (line 198-211):
- Takes raw observation → `prepare_openvla_image_for_attack()` → OpenVLA processor
- Processor outputs `pixel_values` tensor: full processed image
- PGD optimizes the ENTIRE `pixel_values` tensor (all spatial positions, all channels)
- Budget enforced via `_project_pixel_master()` (line 213-214): `clamp(adv, x_orig ± epsilon)` in processor pixel space

**What is NOT masked/constrained:**
- No object ROI / bounding box
- No gripper ROI
- No segmentation mask
- No patch constraint (e.g., 32×32 patch)
- No spatial attention gating
- No per-pixel weight map

**Code evidence** (line 693-704):
```python
x_orig_model = x0.detach()       # full pixel_values
x_orig = x_orig_model.detach().float()
delta = torch.zeros_like(x_orig) # full-tensor perturbation
adv = self._project_pixel_master(x_orig + delta, x_orig)
```

The perturbation `delta` has the same shape as `pixel_values` — every pixel of the processed 224×224 image is independently perturbable within the Linf budget.

---

## 2. Current Temporal Scope

**Dense attack within perturb window only.**

`scripts/vis_rollout_adaptive_v3.py` controls temporal scope:
- `--perturb_start`, `--perturb_end`: 18-step window (default)
- Attack applied at EVERY step within the window under `strategy=full`
- No attack before or after window
- No sparse/intermittent scheduling by default

**Per-step PGD**: Each in-window step runs 40 PGD iterations with 3 restarts = up to 120 forward/backward passes per step.

---

## 3. Current PGD Budget

| Parameter | Default | Source |
|-----------|---------|--------|
| `eps_raw_pixels` | 6 | CLI → converted to processor space |
| `pgd_steps` | 40 | per in-window step |
| `pgd_restarts` | 3 | best restart selected |
| `objective` | `prefix_locked_gripper_open_margin` | gripper OPEN margin loss |
| `step_size` | `max(epsilon / num_steps, 1e-4)` | computed automatically |
| `random_start` | `False` | temporal_init default = "none" |

**Loss function** (line 443-448):
```python
log_open = torch.logsumexp(gripper_row[region_token_ids], dim=0)
max_non_open = gripper_row[non_open_mask].max()
gripper_loss = F.relu(max_non_open - log_open + margin)
```

Gripper OPEN logit margin loss + arm-preserve CE — forces gripper to OPEN while keeping arm trajectory close to clean action.

---

## 4. Current Workload Estimation

Per full VIS candidate (18-step window, 40 steps, 3 restarts):
- PGD passes: 40 × 3 = 120 per window step
- Forward passes: 120 × 18 = 2,160
- Backward passes: same as forward (1 grad per step)
- Model: OpenVLA 7B (bf16), ~14 GB on 2 GPUs

**Observed runtime** (Batch3b Wave 1):
- VIS jobs running 25+ minutes and still in progress
- Clean rollout: ~3-4 min
- Random rollout: ~3-6 min
- Full VIS: est. 25-40 min per candidate

**Total wall-clock for 9 candidates**: ~90-120 min with 3 healthy GPU pairs.

---

## 5. Deployment Concern

1. **Full-image dense PGD is useful as gold vulnerability discovery** — it confirms whether ANY visual perturbation can flip the gripper.
2. **Not suitable as scalable large-sweep default** — 25-40 min/candidate is too slow for 50+ candidate screening.
3. **Not directly realistic for real-robot perturbation** — real physical attacks are more likely localized (patch, sticker, lighting change).
4. **Spatial over-parameterization** — 224×224×3 = 150,528 free parameters for 18 steps is far more than needed to flip one token.

---

## 6. Recommendation

| Tier | Method | Use Case |
|------|--------|----------|
| **Gold** | Full VIS (eps=6, steps=40, restarts=3) | Final confirmation, disputed samples |
| **Silver** | Low-budget VIS (eps=4, steps=10, restarts=1, L10) | Scalable screening |
| **Screen** | Policy-only VIS audit | Pre-filter candidates without env stepping |
| **Proxy** | Command forced-open | Upper-bound physical/task susceptibility |
| **Future** | ROI/masked VIS | Realism improvement for physical attacks |

---

## 7. Code Locations

| Component | File | Key Lines |
|-----------|------|-----------|
| PGD attacker class | `src/gripper_attack/attack_adapter.py` | 113-759 (TokenPrefixPGDAttacker) |
| Loss computation | `src/gripper_attack/attack_adapter.py` | 394-489 (`_loss`) |
| OPEN region | `src/gripper_attack/attack_adapter.py` | 276-378 (`get_gripper_region_by_decoded_action`) |
| PGD loop | `src/gripper_attack/attack_adapter.py` | 723-751 |
| Wrapper | `scripts/vis_phase_conditioned_attack.py` | 1-248 |
| Rollout | `scripts/vis_rollout_adaptive_v3.py` | 1-643 |
| Preprocessing | `src/gripper_attack/attack_adapter.py` | 44-58 |
