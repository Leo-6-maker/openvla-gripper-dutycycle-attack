# Next Session Handoff — After VisualNoStep V6 Pilot

**Date**: 2026-05-30 | **Status**: production_ready_for_group_meeting | **Prepared for**: Next session / collaborator handoff

## Server Connection

```
ssh -o "ProxyCommand=ssh -i ~/.ssh/id_ed25519_vla -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null scene@10.60.133.3 nc 10.60.133.4 22" -i ~/.ssh/id_ed25519_vla -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null liuyu@10.60.133.4
```

Or use SSH config alias `vla` if configured (see `~/.ssh/config`):
```
Host vla-jump
    HostName 10.60.133.3
    User scene
    IdentityFile ~/.ssh/id_ed25519_vla
Host vla
    HostName 10.60.133.4
    User liuyu
    IdentityFile ~/.ssh/id_ed25519_vla
    ProxyCommand ssh vla-jump nc %h %p
```

## Environment

- **Repo**: `/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524`
- **Conda env**: `official_libero_20260525`
- **Python**: `/data/aviary/envs/openvla_official_libero_20260525/bin/python`
- **OpenVLA**: `/data/aviary/models/openvla/openvla-7b-finetuned-libero-object`

## Production Branch

- **Branch**: `exp/sustained-proxy-burst-control-20260530`
- **Server HEAD**: `703c172` (local SHA from `git am` of hardening patch)
- **Remote hardening**: confirmed — `attack_burst_steps` guarded to `sustained_command_open_proxy` only; other attack conditions use `det_out["trigger_duration"]`; legacy `VIS_targeted` removed
- **Remote-visible lineage**: `e7e5bd1` (sustained proxy) → `07e13a0` (hardening) or equivalent patches applied
- **Remote**: `git@github.com:Leo-6-maker/openvla-gripper-dutycycle-attack.git` (push from local Windows due to server SSL proxy intercepting GitHub HTTPS)

## Production Detector

- **Model**: `/data/liuyu/outputs/milestone_2e3_object100_visual_proprio_no_step_20260527/models/ProprioNoStep_baseline.pt`
- **Architecture**: CausalTCNDetector(in_dim=13, h_dim=64, n_ph=8, n_l=3)
- **Input**: 13-dim proprio/action features, 16-step history
- **Status**: Production — validated through Full10 sus30 (50 episodes)

## Production Attack

- **Mechanism**: `sustained_command_open_proxy`
- **Parameters**: `--attack_burst_steps 30 --attack_hold_mode fixed --detector_trigger_duration 5 --detector_hazard_threshold 0.1`
- **How it works**: When ProprioNoStep triggers consecutively for 5 steps, gripper action is overridden to fully open (1.0) for 30 steps. This is a command-layer sustained proxy.

## Full10 Oracle Sensitivity (100 episodes)

- **High sensitive**: cream_cheese, tomato_sauce
- **Medium**: butter, chocolate_pudding, bbq_sauce, alphabet_soup
- **Low**: milk, orange_juice
- **Robust**: ketchup, salad_dressing

## Full10 sus30 — Sustained Proxy (50 episodes)

| Class | Success Rate | Key Tasks |
|-------|-------------|-----------|
| High | 0/10 | cream_cheese 0/5, tomato_sauce 0/5 |
| Robust | 10/10 | ketchup 5/5, salad_dressing 5/5 |
| Medium | 6/20 | mixed |
| Low | 7/10 | mixed |

**Selectivity**: 100 percentage points (High 0% vs Robust 100%)

## VisualNoStep V6 (24 episodes, threshold=0.05)

| Task | clean | sus30 |
|------|-------|-------|
| cream_cheese | 3/3 | 1/3 |
| tomato_sauce | 3/3 | 1/3 |
| ketchup | 3/3 | 0/3 (robust broken) |
| salad_dressing | 3/3 | 2/3 |

**Key finding**: VisualNoStep triggers ~100 steps earlier than ProprioNoStep (step 14-63 vs 120-161).

**Mechanism**: Proprio signal encodes contact dynamics (gripper force, velocity, position) — it naturally fires at contact/transport/placement phase. Visual signal encodes scene/object appearance — it fires on "this looks like a difficult task/scene" rather than "contact is now established." The current attack needs contact-phase timing for selective disruption; VisualNoStep V6 turns it into non-selective grasp blocking.

## Key Claim Boundaries

### VALID
1. ProprioNoStep is the production online detector.
2. sustained_command_open_proxy_30 selectively causes failures on high oracle-sensitive tasks.
3. Selectivity: High 0% vs Robust 100%.
4. VisualNoStep triggers but lacks contact-phase selectivity.
5. Proprioceptive signal naturally encodes gripper-object contact dynamics — this is why ProprioNoStep works.

### FORBIDDEN
1. VIS attack successful / failed
2. Universal attack
3. Detector is oracle-optimal
4. All Object tasks vulnerable
5. Visual information useless
6. VisualNoStep production-ready

## GPU Status (as of 2026-05-30)

| GPU | Status |
|-----|--------|
| 0 | QUARANTINED — lgzhou RoboTWin (2055 MiB), Xid13 history (5/29) |
| 1-6 | IDLE — available for OpenVLA rollouts |
| 7 | IDLE — lightweight tasks only (OOM risk for OpenVLA 7B) |

**dmesg**: Last Xid13 on 5/29 14:03. No fresh Xid events. GPU0 quarantine stands.

## Active Jobs

None. All GPUs idle. No screen sessions.

## Next Recommended Actions

1. **Immediate**: Manual frame-level audit of priority episodes (see `reports/MANUAL_AUDIT_FINAL_GUIDE.md`)
2. **Medium**: Visual v2/re-ranker training (requires approval — see `reports/VISUAL_DETECTOR_V2_TRAINING_PLAN_AFTER_V6.md`)
3. **Medium**: Cross-suite generalization evaluation (LIBERO Spatial, Goal)
4. **Long**: Defense/mitigation study
5. **Long**: Group meeting presentation preparation

## Key Output Directories

| Purpose | Path |
|---------|------|
| Full10 Oracle | `milestone_2f_object_oracle_sensitivity_full10x5_20260529/` |
| Full10 sus30 | `milestone_2h_sustained_proxy_full10x5_sus30_20260530/` |
| Detector-clean | `milestone_2f_object_detector_clean_full10x5_prep_20260529/` |
| Visual features | `milestone_2f_full10_frozen_visual_features_20260530/` |
| Visual V6 pilot | `milestone_2j_visual_fusion_online_pilot_v6_20260530/` |
| Visual v2 training | `milestone_2j_visual_detector_v2_training_20260530/` (failed) |
| Proprio model | `milestone_2e3_object100_visual_proprio_no_step_20260527/` |

## Important Reports (local)

- `reports/MILESTONE_SELECTIVE_SUSTAINED_PROXY_FINAL.md`
- `reports/GROUP_MEETING_SUMMARY_SUSTAINED_PROXY_20260530.md`
- `reports/FINAL_DETECTOR_STATUS_AFTER_VISUAL_FAILURE.md`
- `reports/VISUAL_NOSTEP_V6_NONPRODUCTION_FREEZE.md`
- `reports/MANUAL_AUDIT_FINAL_GUIDE.md`
- `reports/VISUAL_DETECTOR_V2_TRAINING_PLAN_AFTER_V6.md`
- `tables/manual_audit_final_priority.csv`
- `tables/visual_nostep_v6_online_summary.csv`
- `tables/visual_nostep_v6_vs_proprio_selectivity.csv`
