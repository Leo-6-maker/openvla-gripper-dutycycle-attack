#!/usr/bin/env python3
"""Patch vis_rollout_adaptive_v3.py to add ProprioNoStep online CPU detector.
Appends detector class, CLI flags, and per-step inference to the rollout loop.

Usage: python patch_proprionostep_detector.py
Then: CUDA_VISIBLE_DEVICES=2,6 python vis_rollout_adaptive_v3.py --use_proprionostep_detector ...
"""
import os, sys

REPO = '/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605'
path = os.path.join(REPO, 'scripts/vis_rollout_adaptive_v3.py')
with open(path) as f:
    content = f.read()

# ═══════════════════════════════════════════════════════════════════
# PATCH 1: Add CLI flags (after existing controller args, around line 102)
# ═══════════════════════════════════════════════════════════════════
old_p1 = """    ap.add_argument('--controller', choices=['fixed','open_streak_stop','open_count_stop','qpos_safety_stop','min_hold_qpos_cap','streak_with_qpos_cap'], default='fixed')"""
new_p1 = """    ap.add_argument('--controller', choices=['fixed','open_streak_stop','open_count_stop','qpos_safety_stop','min_hold_qpos_cap','streak_with_qpos_cap'], default='fixed')
    ap.add_argument('--use_proprionostep_detector', action='store_true', help='Enable ProprioNoStep online CPU phase detector')
    ap.add_argument('--proprionostep_model_path', type=str, default='/data/liuyu/outputs/proprionostep_cpu_20260602/proprio_no_step_tcn_cpu.pt')
    ap.add_argument('--proprionostep_hazard_threshold', type=float, default=0.5, help='Hazard score threshold for trigger')
    ap.add_argument('--proprionostep_trigger_duration', type=int, default=5, help='Min steps to sustain trigger')
    ap.add_argument('--proprionostep_cooldown', type=int, default=20, help='Cooldown steps after trigger ends')
    ap.add_argument('--proprionostep_device', type=str, default='cpu', help='Device for ProprioNoStep (must be cpu)')
    ap.add_argument('--proprionostep_history_len', type=int, default=32, help='History buffer length for TCN')"""

if old_p1 in content:
    content = content.replace(old_p1, new_p1)
    print('Patch 1 OK: CLI flags added')
else:
    print('Patch 1 FAILED: CLI pattern not found')

# ═══════════════════════════════════════════════════════════════════
# PATCH 2: Add ProprioNoStepDetector class (after imports, around line 250)
# ═══════════════════════════════════════════════════════════════════
detector_class = """

# ── ProprioNoStep Online CPU Detector ──────────────────────────────
FEATURE_ORDER = [
    'gripper_command', 'gripper_qpos', 'gripper_width',
    'eef_x', 'eef_y', 'eef_z',
    'eef_vx', 'eef_vy', 'eef_vz',
    'action_dx', 'action_dy', 'action_dz',
    'action_gripper',
]
PHASE_NAMES = ['far_closed','near_closed','pre_lock_closed','grasp_formation','stable','natural_open','uncertain','invalid']

class ProprioNoStepDetector:
    def __init__(self, model_path, device='cpu', history_len=32,
                 hazard_threshold=0.5, trigger_duration=5, cooldown=20):
        self.device = torch.device(device)
        self.history_len = history_len
        self.hazard_threshold = hazard_threshold
        self.trigger_duration = trigger_duration
        self.cooldown = cooldown

        # Load model weights
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        self.proj_weight = state['proj.weight'].to(self.device)
        self.proj_bias = state['proj.bias'].to(self.device)
        self.convs = torch.nn.ModuleList()
        for i in range(3):
            conv = torch.nn.Conv1d(64, 64, 3, padding=1)
            conv.weight.data = state[f'convs.{i}.weight'].to(self.device)
            conv.bias.data = state[f'convs.{i}.bias'].to(self.device)
            self.convs.append(conv)
        self.phase_head = torch.nn.Linear(64, 8)
        self.phase_head.weight.data = state['phase_head.weight'].to(self.device)
        self.phase_head.bias.data = state['phase_head.bias'].to(self.device)
        self.hazard_head = torch.nn.Linear(64, 1)
        self.hazard_head.weight.data = state['hazard_head.weight'].to(self.device)
        self.hazard_head.bias.data = state['hazard_head.bias'].to(self.device)
        self.release_head = torch.nn.Linear(64, 1)
        self.release_head.weight.data = state['release_head.weight'].to(self.device)
        self.release_head.bias.data = state['release_head.bias'].to(self.device)
        self.eval()

        # State
        self.history = []  # list of 13-dim feature vectors
        self.trigger_active = False
        self.trigger_counter = 0
        self.cooldown_counter = 0
        self.last_hazard = 0.0
        self.last_release = 0.0
        self.last_phase_idx = -1
        self.last_phase_conf = 0.0

    def eval(self):
        self.proj_weight.requires_grad_(False); self.proj_bias.requires_grad_(False)
        for c in self.convs: c.eval()
        self.phase_head.eval(); self.hazard_head.eval(); self.release_head.eval()

    def extract_features(self, obs, action_vec):
        '''Extract 13-dim feature vector from observation and action.'''
        # Get qpos from MuJoCo (obs qpos always 0)
        try:
            gripper_qpos = float(env.sim.data.qpos[-2:].mean())  # finger joints
        except:
            gripper_qpos = 0.0
        gripper_width = obs.get('robot0_gripper_qpos', [0.0])[0] if hasattr(obs, 'get') else 0.0
        eef_pos = obs.get('robot0_eef_pos', np.zeros(3))
        eef_vel = obs.get('robot0_eef_vel', np.zeros(3)) if 'robot0_eef_vel' in (obs if isinstance(obs, dict) else {}) else np.zeros(3)

        feats = np.array([
            float(action_vec[-1]) if len(action_vec) > 0 else 0.0,  # gripper_command (from action)
            float(gripper_qpos),
            float(gripper_width) if not isinstance(gripper_width, (list, np.ndarray)) else float(np.mean(gripper_width)),
            float(eef_pos[0]) if len(eef_pos) > 0 else 0.0,
            float(eef_pos[1]) if len(eef_pos) > 1 else 0.0,
            float(eef_pos[2]) if len(eef_pos) > 2 else 0.0,
            float(eef_vel[0]) if len(eef_vel) > 0 else 0.0,
            float(eef_vel[1]) if len(eef_vel) > 1 else 0.0,
            float(eef_vel[2]) if len(eef_vel) > 2 else 0.0,
            0.0, 0.0, 0.0,  # action_dx/dy/dz (from previous step, approximate with zeros)
            float(action_vec[-1]) if len(action_vec) > 0 else 0.0,  # action_gripper
        ], dtype=np.float32)
        return feats

    def step(self, obs, action_vec):
        # Run one inference step. Returns dict with detector outputs.
        feats = self.extract_features(obs, action_vec)
        self.history.append(feats)
        if len(self.history) > self.history_len:
            self.history = self.history[-self.history_len:]

        result = {
            'hazard_score': 0.0, 'release_safe_score': 0.0,
            'phase_idx': -1, 'phase_confidence': 0.0,
            'trigger_now': False, 'trigger_reason': '',
        }

        if len(self.history) < 8:  # need minimum history
            return result

        # Build input tensor [1, 13, T]
        x = torch.as_tensor(np.stack(self.history, axis=0), dtype=torch.float32, device=self.device).T.unsqueeze(0)
        # Project
        x = torch.relu(torch.matmul(x.transpose(1, 2), self.proj_weight.T) + self.proj_bias).transpose(1, 2)
        # TCN layers
        for conv in self.convs:
            x = torch.relu(conv(x))
        # Global average pool
        x_pooled = x.mean(dim=-1)  # [1, 64]

        phase_logits = self.phase_head(x_pooled)
        phase_probs = torch.softmax(phase_logits, dim=-1)
        phase_idx = int(phase_probs.argmax(dim=-1).item())
        phase_conf = float(phase_probs.max(dim=-1).values.item())

        hazard = float(torch.sigmoid(self.hazard_head(x_pooled)).item())
        release = float(torch.sigmoid(self.release_head(x_pooled)).item())

        self.last_hazard = hazard
        self.last_release = release
        self.last_phase_idx = phase_idx
        self.last_phase_conf = phase_conf

        result['hazard_score'] = round(hazard, 6)
        result['release_safe_score'] = round(release, 6)
        result['phase_idx'] = phase_idx
        result['phase_confidence'] = round(phase_conf, 6)

        # Trigger logic
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.trigger_active = False
            return result

        if hazard >= self.hazard_threshold:
            self.trigger_counter += 1
            if self.trigger_counter >= self.trigger_duration and not self.trigger_active:
                self.trigger_active = True
                result['trigger_now'] = True
                result['trigger_reason'] = 'hazard_%.3f_sustained_%d' % (hazard, self.trigger_counter)
        else:
            if self.trigger_active:
                self.cooldown_counter = self.cooldown
            self.trigger_active = False
            self.trigger_counter = 0

        return result
"""

# Insert after the controller state initialization (around line 449-460)
old_p2 = "# Adaptive controller state"
if old_p2 in content:
    content = content.replace(old_p2, detector_class + "\n" + old_p2)
    print('Patch 2 OK: ProprioNoStepDetector class added')
else:
    print('Patch 2 FAILED: insertion point not found')

# ═══════════════════════════════════════════════════════════════════
# PATCH 3: Initialize detector after args parsing (around line 458)
# ═══════════════════════════════════════════════════════════════════
old_p3 = """if args.controller != 'fixed' and args.condition == 'vis_pgd':
    print(f'    Adaptive controller: {args.controller} K={args.K} Q={args.Q} max_dur={ctrl[\"max_dur\"]}')"""
new_p3 = """# ── ProprioNoStep detector init ──
proprionostep_detector = None
if args.use_proprionostep_detector:
    if args.proprionostep_device != 'cpu':
        print('WARNING: ProprioNoStep must run on CPU; forcing cpu')
        args.proprionostep_device = 'cpu'
    proprionostep_detector = ProprioNoStepDetector(
        model_path=args.proprionostep_model_path,
        device=args.proprionostep_device,
        history_len=args.proprionostep_history_len,
        hazard_threshold=args.proprionostep_hazard_threshold,
        trigger_duration=args.proprionostep_trigger_duration,
        cooldown=args.proprionostep_cooldown,
    )
    print(f'    ProprioNoStep detector loaded: hazard_thr={args.proprionostep_hazard_threshold} '
          f'trig_dur={args.proprionostep_trigger_duration} cooldown={args.proprionostep_cooldown}')

if args.controller != 'fixed' and args.condition == 'vis_pgd':
    print(f'    Adaptive controller: {args.controller} K={args.K} Q={args.Q} max_dur={ctrl[\"max_dur\"]}')"""

if old_p3 in content:
    content = content.replace(old_p3, new_p3)
    print('Patch 3 OK: detector init added')
else:
    # Try alternate pattern
    print('Patch 3: trying alternate pattern...')
    alt = """if args.controller != 'fixed' and args.condition == 'vis_pgd':
    print(f'    Adaptive controller: {args.controller} K={args.K} Q={args.Q} max_dur={ctrl[\"max_dur\"]}')"""
    if alt in content:
        content = content.replace(alt, new_p3)
        print('Patch 3 OK (alternate)')
    else:
        print('Patch 3 FAILED')

# ═══════════════════════════════════════════════════════════════════
# PATCH 4: Run detector at each step (after action decode, around line 520)
# ═══════════════════════════════════════════════════════════════════
old_p4 = """                # Update attack/controller audit state using causally available qpos."""
new_p4 = """                # ── ProprioNoStep detector inference ──
                proprionostep_result = {}
                if proprionostep_detector is not None:
                    try:
                        proprionostep_result = proprionostep_detector.step(obs, clean_action_vec)
                        # Override attack window: if detector triggers, start attacking from current step
                        if proprionostep_result.get('trigger_now') and args.condition == 'vis_pgd':
                            if not in_window:
                                in_window = True
                                cfg['perturb_start'] = t
                                cfg['perturb_end'] = min(t + args.max_duration - 1, cfg.get('episode_length', 300))
                    except Exception as _de:
                        proprionostep_result = {'hazard_score': -1.0, 'error': str(_de)[:80]}

                # Update attack/controller audit state using causally available qpos."""

if old_p4 in content:
    content = content.replace(old_p4, new_p4)
    print('Patch 4 OK: per-step detector inference added')
else:
    print('Patch 4 FAILED: insertion point not found')
    for i, line in enumerate(content.split('\n')):
        if 'Update attack/controller audit state' in line:
            print('  Found at L%d: %s' % (i+1, line.strip()[:80]))
            break

# ═══════════════════════════════════════════════════════════════════
# PATCH 5: Add detector fields to trace row (around line 612)
# ═══════════════════════════════════════════════════════════════════
old_p5 = """        'ctrl_qpos_delta': round(ctrl['qpos_delta_online'], 6), 'ctrl_attacks': ctrl['attacks_applied'],"""
new_p5 = """        'ctrl_qpos_delta': round(ctrl['qpos_delta_online'], 6), 'ctrl_attacks': ctrl['attacks_applied'],
        'proprionostep_hazard_score': proprionostep_result.get('hazard_score', 0.0),
        'proprionostep_release_safe_score': proprionostep_result.get('release_safe_score', 0.0),
        'proprionostep_phase_idx': proprionostep_result.get('phase_idx', -1),
        'proprionostep_phase_confidence': proprionostep_result.get('phase_confidence', 0.0),
        'proprionostep_trigger_now': int(proprionostep_result.get('trigger_now', False)),
        'proprionostep_trigger_reason': proprionostep_result.get('trigger_reason', ''),"""

if old_p5 in content:
    content = content.replace(old_p5, new_p5)
    print('Patch 5 OK: detector trace fields added')
else:
    print('Patch 5 FAILED: trace fields not found')

# ═══════════════════════════════════════════════════════════════════
# PATCH 6: Add detector summary to final summary (around line 648-680)
# ═══════════════════════════════════════════════════════════════════
old_p6 = "    'window_token_flips': n_flip, 'avg_arm_l2': avg_al, 'total_dt_s': round(total_dt, 1),"
new_p6 = """    'window_token_flips': n_flip, 'avg_arm_l2': avg_al, 'total_dt_s': round(total_dt, 1),
    'proprionostep_triggered': int(any(r.get('proprionostep_trigger_now', 0) for r in window_rows if isinstance(r, dict))),
    'proprionostep_trigger_count': sum(1 for r in window_rows if isinstance(r, dict) and r.get('proprionostep_trigger_now', 0)),
    'proprionostep_hazard_mean': round(float(np.mean([r.get('proprionostep_hazard_score', 0) for r in window_rows if isinstance(r, dict)])), 6) if window_rows else 0.0,"""

if old_p6 in content:
    content = content.replace(old_p6, new_p6)
    print('Patch 6 OK: detector summary fields added')
else:
    print('Patch 6 FAILED: summary fields not found')

# ═══════════════════════════════════════════════════════════════════
# Write
# ═══════════════════════════════════════════════════════════════════
with open(path, 'w') as f:
    f.write(content)

# Quick syntax check
import py_compile
try:
    py_compile.compile(path, doraise=True)
    print('\n=== ALL PATCHES APPLIED, SYNTAX OK ===')
except py_compile.PyCompileError as e:
    print('\n=== SYNTAX ERROR ===')
    print(str(e))
