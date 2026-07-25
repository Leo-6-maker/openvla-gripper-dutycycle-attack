"""Thin wrapper around frozen N4 Detector. Does NOT reimplement any Detector logic.

Processes step-by-step with full trajectory prefix (W128 causal history preserved).
Must pass 300/300 P4 emit parity before FEC rollout.
"""
import json, os, numpy as np, torch, torch.nn as nn

# ── Frozen constants ──
PLATT_A = 0.5190011735319306
PLATT_B = 0.812702331013635
TAU = 0.855
D_PERSIST = 6
HIDDEN = 64
CKPT_PATH = '/mnt/sdc/dty_user/openvla_attack_evidence/formal_v23_student_training_v1/o0_i0/checkpoint.pt'


def _load_encoder():
    import sys
    sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack/src')
    from gripper_attack.v6_critical_student import CausalTCNEncoder

    class N4Encoder(nn.Module):
        def __init__(self, base_dim=43, proxy_dim=8, hidden=64, short_rf=32, long_rf=128, dropout=0.1):
            super().__init__()
            self.short_tcn = CausalTCNEncoder(base_dim + proxy_dim, hidden, short_rf, dropout)
            self.long_tcn = CausalTCNEncoder(base_dim + proxy_dim, hidden, long_rf, dropout)
            self.fusion = nn.Linear(hidden * 2, hidden)

        def forward(self, x):
            return self.fusion(torch.cat([self.short_tcn(x), self.long_tcn(x)], dim=-1))

    return N4Encoder


def _compute_proxies(f25d, p9d, g9d, T):
    """8 causal response proxies. EXACT copy from frozen train_v23_split.py."""
    proxies = np.zeros((T, 8), dtype=np.float32)
    cmd = f25d[:, 0]; qpos = f25d[:, 1]
    proxies[:, 0] = cmd - qpos
    proxies[:, 1] = (cmd < 0).astype(np.float32)
    proxies[1:, 2] = np.diff(qpos); proxies[0, 2] = 0
    dur = 0; cd = np.zeros(T)
    for t in range(T):
        if cmd[t] < 0: dur += 1
        else: dur = 0
        cd[t] = dur
    proxies[:, 3] = cd
    proxies[:, 4] = np.sqrt(f25d[:, 6] ** 2 + f25d[:, 7] ** 2 + f25d[:, 8] ** 2)
    for t in range(T):
        w_s = max(0, t - 4); w_e = min(T, t + 1)
        proxies[t, 5] = np.var(qpos[w_s:w_e]) if w_e - w_s > 1 else 0
    proxies[:, 6] = g9d[:, 0]; proxies[:, 7] = g9d[:, 7]
    return np.nan_to_num(proxies, 0).astype(np.float32)


def _calibrated_prob(raw_logit):
    xc = np.clip(PLATT_A * np.array(raw_logit) + PLATT_B, -50, 50)
    return 1.0 / (1.0 + np.exp(-xc))


class N4DetectorAdapter:
    """Thin wrapper: loads frozen checkpoint, runs Detector step-by-step.

    Accumulates history buffer and passes full [0:t+1] trajectory prefix
    at each step, exactly matching the runtime parity streaming implementation.
    """

    def __init__(self, device='cuda:0', norm_data_path=None):
        self.device = torch.device(device)
        N4Encoder = _load_encoder()

        ckpt = torch.load(CKPT_PATH, map_location=self.device, weights_only=False)
        self.encoder = N4Encoder().to(self.device)
        self.head = nn.Linear(HIDDEN, 1).to(self.device)
        self.encoder.load_state_dict(ckpt['enc'])
        self.head.load_state_dict(ckpt['head'])
        self.encoder.eval()
        self.head.eval()

        if norm_data_path and os.path.isfile(norm_data_path):
            norms = torch.load(norm_data_path, map_location='cpu', weights_only=False)
            self.n25d_m = norms['n25d_m'].to(self.device)
            self.n25d_s = norms['n25d_s'].to(self.device)
            self.np9d_m = norms['np9d_m'].to(self.device)
            self.np9d_s = norms['np9d_s'].to(self.device)
            self.ng9d_m = norms['ng9d_m'].to(self.device)
            self.ng9d_s = norms['ng9d_s'].to(self.device)
        else:
            raise ValueError("Normalization parameters required.")

        self.reset_episode()

    def reset_episode(self):
        self.persistence_counter = 0
        self.latch = False
        self.emit_step = None
        self._t = 0
        # History buffers
        self.hist_f25d = []
        self.hist_p9d = []
        self.hist_g9d = []
        self.trajectory_logits = []
        self.trajectory_cal_probs = []
        self.trajectory_cc = []

    def step(self, features_25d, policy_9d, gripper_9d, candidate_close):
        """Process one step. Passes full [0:t+1] trajectory prefix to CausalTCN.

        This is the EXACT same computation as the runtime parity streaming inference:
        - Accumulate history
        - Compute proxies over full history
        - Normalize full history
        - Forward pass on full history
        - Take last-step output as current prediction
        """
        # Accumulate
        self.hist_f25d.append(features_25d)
        self.hist_p9d.append(policy_9d)
        self.hist_g9d.append(gripper_9d)
        T = len(self.hist_f25d)

        f25d_arr = np.array(self.hist_f25d, dtype=np.float32)
        p9d_arr = np.array(self.hist_p9d, dtype=np.float32)
        g9d_arr = np.array(self.hist_g9d, dtype=np.float32)

        # Compute proxies over full history (causal — each step's proxy uses only ≤t data)
        proxies_arr = _compute_proxies(f25d_arr, p9d_arr, g9d_arr, T)

        # Build 51D input [1, T, 51]
        base = np.concatenate([f25d_arr, p9d_arr, g9d_arr, proxies_arr], axis=-1)
        base_t = torch.tensor(base, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Normalize
        norm_25d = (base_t[:, :, :25] - self.n25d_m) / self.n25d_s
        norm_p9d = (base_t[:, :, 25:34] - self.np9d_m) / self.np9d_s
        norm_g9d = (base_t[:, :, 34:43] - self.ng9d_m) / self.ng9d_s
        proxies_t = base_t[:, :, 43:]
        x = torch.cat([norm_25d, norm_p9d, norm_g9d, proxies_t], dim=-1)

        with torch.no_grad():
            raw_all = self.head(self.encoder(x)).squeeze().cpu().numpy()

        raw_logit = float(np.atleast_1d(raw_all)[-1])
        cal_prob = float(_calibrated_prob(raw_logit))

        # Scheduler
        emitted_this_step = False
        if not self.latch and candidate_close and cal_prob >= TAU:
            self.persistence_counter += 1
        else:
            self.persistence_counter = 0

        if self.persistence_counter >= D_PERSIST and not self.latch:
            self.latch = True
            self.emit_step = self._t
            emitted_this_step = True

        result = {
            'step': self._t,
            'raw_logit': raw_logit,
            'calibrated_prob': cal_prob,
            'candidate_close': bool(candidate_close),
            'persistence_counter': self.persistence_counter,
            'latch': self.latch,
            'emitted_this_step': emitted_this_step,
            'emit_step': self.emit_step,
        }

        self.trajectory_logits.append(raw_logit)
        self.trajectory_cal_probs.append(cal_prob)
        self.trajectory_cc.append(bool(candidate_close))
        self._t += 1

        return result

    def get_trajectory(self):
        return {
            'raw_logits': np.array(self.trajectory_logits),
            'cal_probs': np.array(self.trajectory_cal_probs),
            'candidate_close': np.array(self.trajectory_cc),
            'emit_step': self.emit_step,
            'emitted': self.latch,
        }


_feature_adapter = None
_history_qpos = []
_history_cmd = []

def _reset_history():
    global _feature_adapter, _history_qpos, _history_cmd
    _feature_adapter = None  # Set by build_n4_inputs after import
    _history_qpos = []
    _history_cmd = []

def build_n4_inputs(obs=None, observation=None, clean_raw_action=None, raw_action=None,
                    clean_model_output=None, clean_action_raw_7d=None, policy_step=None,
                    suite=None, model=None, processor=None, **kwargs):
    """Extract 51D N4 features. Returns dict with f25d(25), p9d(9), g9d(9), candidate_close(bool)."""
    global _feature_adapter, _history_qpos, _history_cmd

    raw = clean_raw_action if clean_raw_action is not None else raw_action
    if raw is None and clean_action_raw_7d is not None:
        raw = clean_action_raw_7d
    if raw is None:
        raw = np.zeros(7, dtype=np.float32)
    raw = np.asarray(raw, dtype=np.float32)
    
    obs_dict = obs if obs is not None else observation
    if obs_dict is None:
        obs_dict = {}
    
    # Initialize history on first call (policy_step=0)
    if policy_step is not None and int(policy_step) == 0:
        _reset_history()
    if _feature_adapter is None:
        from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
        _feature_adapter = SC5StreamingFeatureAdapterV2()
    
    gripper_cmd = float(raw[6])
    gripper_qpos_arr = np.asarray(obs_dict.get('robot0_gripper_qpos', [0.0, 0.0]))
    gripper_qpos = float(gripper_qpos_arr.mean()) if gripper_qpos_arr.size > 0 else 0.0
    eef_pos = np.asarray(obs_dict.get('robot0_eef_pos', np.zeros(3)), dtype=np.float64)
    robot0_joint_vel = np.asarray(obs_dict.get('robot0_joint_vel', np.zeros(7)), dtype=np.float64)
    
    # Use SC5StreamingFeatureAdapterV2 for canonical 25D feature extraction
    # Action deltas: scaled raw action as proxy for action deltas
    action_dx = float(raw[0]) * 0.1 if len(raw) > 0 else 0.0
    action_dy = float(raw[1]) * 0.1 if len(raw) > 1 else 0.0
    action_dz = float(raw[2]) * 0.1 if len(raw) > 2 else 0.0
    action_gripper = gripper_cmd
    
    eef_vx = float(robot0_joint_vel[0]) if robot0_joint_vel.size > 0 else 0.0
    eef_vy = float(robot0_joint_vel[1]) if robot0_joint_vel.size > 1 else 0.0
    eef_vz = float(robot0_joint_vel[2]) if robot0_joint_vel.size > 2 else 0.0
    
    gripper_opening = max(0.0, 1.0 - abs(gripper_qpos))
    
    # Compute 25D via canonical adapter
    try:
        feat_dict = _feature_adapter.step(
            raw_gripper=gripper_cmd,
            gripper_qpos=gripper_qpos,
            gripper_opening_proxy=gripper_opening,
            eef_x=float(eef_pos[0]), eef_y=float(eef_pos[1]), eef_z=float(eef_pos[2]),
            eef_vx=eef_vx, eef_vy=eef_vy, eef_vz=eef_vz,
            action_dx=action_dx, action_dy=action_dy, action_dz=action_dz,
            action_gripper=action_gripper,
        )
        f25d = np.array([float(feat_dict[name]) for name in feat_dict], dtype=np.float32)
    except Exception:
        # Fallback: build basic 25D if adapter fails
        f25d = np.zeros(25, dtype=np.float32)
        f25d[0] = gripper_cmd
        f25d[1] = gripper_qpos
        f25d[2] = gripper_opening
        f25d[3] = float(eef_pos[0]); f25d[4] = float(eef_pos[1]); f25d[5] = float(eef_pos[2])
        f25d[6] = eef_vx; f25d[7] = eef_vy; f25d[8] = eef_vz
        f25d[9] = action_dx; f25d[10] = action_dy; f25d[11] = action_dz
        f25d[12] = action_gripper
    
    # Track history for response proxies
    _history_qpos.append(gripper_qpos)
    _history_cmd.append(gripper_cmd)
    if len(_history_qpos) > 32:
        _history_qpos.pop(0); _history_cmd.pop(0)
    
    # --- 9D policy intent from model.generate() scores ---
    p9d = np.zeros(9, dtype=np.float32)
    if clean_model_output is not None:
        try:
            gen = clean_model_output
            if hasattr(gen, 'scores') and gen.scores:
                last_scores = gen.scores[-1]
                if last_scores.dim() >= 2:
                    last_scores = last_scores[0] if last_scores.dim() == 2 else last_scores[0, -1]
                if last_scores.dim() >= 1:
                    log_probs = torch.log_softmax(last_scores.float(), dim=-1)
                    probs = log_probs.exp()
                    vocab_size = last_scores.shape[-1]
                    n_bins = min(256, vocab_size // 8)
                    open_ids = list(range(0, max(1, n_bins // 4)))
                    close_ids = list(range(vocab_size - n_bins // 4, vocab_size))
                    open_t = torch.tensor(open_ids, device=last_scores.device, dtype=torch.long)
                    close_t = torch.tensor(close_ids, device=last_scores.device, dtype=torch.long)
                    open_log_mass = torch.logsumexp(log_probs.index_select(-1, open_t), dim=-1)
                    close_log_mass = torch.logsumexp(log_probs.index_select(-1, close_t), dim=-1)
                    entropy = -(probs * log_probs).sum(dim=-1) / max(1.0, np.log(vocab_size))
                    top1_prob, top1_token = probs.max(dim=-1)
                    p9d[0] = float(open_log_mass.exp().cpu())
                    p9d[1] = float(close_log_mass.exp().cpu())
                    p9d[2] = float((open_log_mass - close_log_mass).cpu())
                    p9d[3] = float(entropy.cpu())
                    p9d[4] = float(top1_prob.cpu())
                    p9d[5] = 1.0 if int(top1_token) in open_ids else 0.0
                    p9d[6] = 1.0 if int(top1_token) in close_ids else 0.0
                    p9d[7] = 0.0; p9d[8] = 0.0
        except Exception:
            pass
    
    g9d = p9d.copy()
    candidate_close = bool(gripper_cmd < 0.5)
    
    return {'f25d': f25d.astype(np.float32), 'p9d': p9d.astype(np.float32),
            'g9d': g9d.astype(np.float32), 'candidate_close': candidate_close}

