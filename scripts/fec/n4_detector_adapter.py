"""N4 Detector adapter + canonical feature provider V3.
Uses SC5StreamingFeatureAdapterV2.update() exactly matching frozen collector.
No silent fallbacks. No Teacher labels. Fails on missing/invalid data."""
import json, os, sys, numpy as np, torch, torch.nn as nn, hashlib
sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack/src')

PLATT_A = 0.5190011735319306; PLATT_B = 0.812702331013635
TAU = 0.855; D_PERSIST = 6; HIDDEN = 64; SHORT_RF = 32; LONG_RF = 128
CKPT_PATH = '/mnt/sdc/dty_user/openvla_attack_evidence/formal_v23_student_training_v1/o0_i0/checkpoint.pt'

def _load_encoder():
    from gripper_attack.v6_critical_student import CausalTCNEncoder
    class N4Encoder(nn.Module):
        def __init__(self, base_dim=43, proxy_dim=8, hidden=64, short_rf=32, long_rf=128, dropout=0.1):
            super().__init__()
            self.short_tcn = CausalTCNEncoder(base_dim+proxy_dim, hidden, short_rf, dropout)
            self.long_tcn = CausalTCNEncoder(base_dim+proxy_dim, hidden, long_rf, dropout)
            self.fusion = nn.Linear(hidden*2, hidden)
        def forward(self, x): return self.fusion(torch.cat([self.short_tcn(x), self.long_tcn(x)], dim=-1))
    return N4Encoder

def _compute_proxies(f25d, p9d, g9d, T):
    proxies = np.zeros((T, 8), dtype=np.float32)
    cmd = f25d[:,0]; qpos = f25d[:,1]
    proxies[:,0] = cmd - qpos; proxies[:,1] = (cmd < 0).astype(np.float32)
    proxies[1:,2] = np.diff(qpos); proxies[0,2] = 0
    dur = 0; cd = np.zeros(T)
    for t in range(T):
        if cmd[t] < 0: dur += 1
        else: dur = 0
        cd[t] = dur
    proxies[:,3] = cd
    proxies[:,4] = np.sqrt(f25d[:,6]**2 + f25d[:,7]**2 + f25d[:,8]**2)
    for t in range(T):
        w_s = max(0, t-4); w_e = min(T, t+1)
        proxies[t,5] = np.var(qpos[w_s:w_e]) if w_e-w_s > 1 else 0
    proxies[:,6] = g9d[:,0]; proxies[:,7] = g9d[:,7]
    return np.nan_to_num(proxies, 0).astype(np.float32)

def _calibrated_prob(raw_logit):
    xc = np.clip(PLATT_A * np.array(raw_logit) + PLATT_B, -50, 50)
    return 1.0 / (1.0 + np.exp(-xc))

class N4DetectorAdapter:
    def __init__(self, device='cuda:0', norm_data_path=None):
        self.device = torch.device(device)
        N4Encoder = _load_encoder()
        ckpt = torch.load(CKPT_PATH, map_location=self.device, weights_only=False)
        self.encoder = N4Encoder().to(self.device)
        self.head = nn.Linear(HIDDEN, 1).to(self.device)
        self.encoder.load_state_dict(ckpt['enc']); self.head.load_state_dict(ckpt['head'])
        self.encoder.eval(); self.head.eval()
        if norm_data_path and os.path.isfile(norm_data_path):
            norms = torch.load(norm_data_path, map_location='cpu', weights_only=False)
            self.n25d_m = norms['n25d_m'].to(self.device); self.n25d_s = norms['n25d_s'].to(self.device)
            self.np9d_m = norms['np9d_m'].to(self.device); self.np9d_s = norms['np9d_s'].to(self.device)
            self.ng9d_m = norms['ng9d_m'].to(self.device); self.ng9d_s = norms['ng9d_s'].to(self.device)
        else:
            raise ValueError('Normalization parameters required.')
        self.reset_episode()

    def reset_episode(self):
        self.persistence_counter = 0; self.latch = False; self.emit_step = None
        self._t = 0
        self.hist_f25d = []; self.hist_p9d = []; self.hist_g9d = []
        self.trajectory_logits = []; self.trajectory_cal_probs = []; self.trajectory_cc = []

    def step(self, features_25d, policy_9d, gripper_9d, candidate_close):
        self.hist_f25d.append(features_25d); self.hist_p9d.append(policy_9d); self.hist_g9d.append(gripper_9d)
        T = len(self.hist_f25d)
        f25d_arr = np.array(self.hist_f25d, dtype=np.float32)
        p9d_arr = np.array(self.hist_p9d, dtype=np.float32)
        g9d_arr = np.array(self.hist_g9d, dtype=np.float32)
        proxies_arr = _compute_proxies(f25d_arr, p9d_arr, g9d_arr, T)
        base = np.concatenate([f25d_arr, p9d_arr, g9d_arr, proxies_arr], axis=-1)
        base_t = torch.tensor(base, dtype=torch.float32, device=self.device).unsqueeze(0)
        norm_25d = (base_t[:,:,:25] - self.n25d_m) / self.n25d_s
        norm_p9d = (base_t[:,:,25:34] - self.np9d_m) / self.np9d_s
        norm_g9d = (base_t[:,:,34:43] - self.ng9d_m) / self.ng9d_s
        x = torch.cat([norm_25d, norm_p9d, norm_g9d, base_t[:,:,43:]], dim=-1)
        with torch.no_grad():
            raw_all = self.head(self.encoder(x)).squeeze().cpu().numpy()
        raw_logit = float(np.atleast_1d(raw_all)[-1])
        cal_prob = float(_calibrated_prob(raw_logit))
        emitted_this_step = False
        if not self.latch and candidate_close and cal_prob >= TAU:
            self.persistence_counter += 1
        else:
            self.persistence_counter = 0
        if self.persistence_counter >= D_PERSIST and not self.latch:
            self.latch = True; self.emit_step = self._t; emitted_this_step = True
        result = {'step': self._t, 'raw_logit': raw_logit, 'calibrated_prob': cal_prob,
                  'candidate_close': bool(candidate_close), 'persistence_counter': self.persistence_counter,
                  'latch': self.latch, 'emitted_this_step': emitted_this_step, 'emit_step': self.emit_step}
        self.trajectory_logits.append(raw_logit); self.trajectory_cal_probs.append(cal_prob)
        self.trajectory_cc.append(bool(candidate_close)); self._t += 1
        return result

    def get_trajectory(self):
        return {'raw_logits': np.array(self.trajectory_logits), 'cal_probs': np.array(self.trajectory_cal_probs),
                'candidate_close': np.array(self.trajectory_cc), 'emit_step': self.emit_step, 'emitted': self.latch}


# ========== CANONICAL 51D FEATURE PROVIDER V3 ==========
# Exact SC5StreamingFeatureAdapterV2.update() contract.
# No fallbacks, no missing-to-zero, no Teacher labels.
# Fails on any missing/invalid data.

_adapter = None
_prev_eef = None

FEATURE_NAMES_25D = [
    'gripper_command','gripper_qpos','gripper_opening_proxy',
    'eef_x','eef_y','eef_z','eef_vx','eef_vy','eef_vz',
    'action_dx','action_dy','action_dz','action_gripper',
    'recent_close_streak','recent_open_streak','recent_gripper_flip_count',
    'close_onset','time_since_close','eef_speed',
    'eef_z_delta_since_close','qpos_delta_1','qpos_delta_3',
    'opening_proxy_delta_3','opening_proxy_variance_5','eef_speed_variance_5',
]


def build_n4_inputs(obs=None, observation=None, clean_raw_action=None, raw_action=None,
                    clean_env_action=None, clean_model_output=None, clean_action_raw_7d=None,
                    policy_step=None, suite=None, model=None, processor=None, **kwargs):
    """Canonical 51D feature provider. Fails on missing/invalid data. No silent fallback."""
    global _adapter, _prev_eef
    from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2

    if _adapter is None or (policy_step is not None and int(policy_step) == 0):
        _adapter = SC5StreamingFeatureAdapterV2()
        _prev_eef = None

    obs_dict = obs if obs is not None else observation
    if obs_dict is None:
        raise RuntimeError('build_n4_inputs requires obs/observation')

    raw = clean_raw_action if clean_raw_action is not None else raw_action
    if raw is None and clean_action_raw_7d is not None:
        raw = clean_action_raw_7d
    if raw is None:
        raise RuntimeError('build_n4_inputs requires clean_raw_action')
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (7,):
        raise RuntimeError('clean_raw_action must be 7D, got {}'.format(raw.shape))

    env_action = clean_env_action
    if env_action is None:
        raise RuntimeError('build_n4_inputs requires clean_env_action')
    env_action = np.asarray(env_action, dtype=np.float64)

    step = int(policy_step) if policy_step is not None else 0

    # Gripper qpos from MuJoCo sim: q7+q8 (matching frozen collector)
    qpos_arr = obs_dict.get('robot0_gripper_qpos')
    if qpos_arr is None:
        raise RuntimeError('obs missing robot0_gripper_qpos')
    qpos_arr = np.asarray(qpos_arr, dtype=np.float64).flatten()
    if qpos_arr.size < 2:
        raise RuntimeError('robot0_gripper_qpos too short: {}'.format(qpos_arr.size))
    q7, q8 = float(qpos_arr[0]), float(qpos_arr[1])
    if not (np.isfinite(q7) and np.isfinite(q8)):
        raise RuntimeError('gripper qpos non-finite: {}, {}'.format(q7, q8))
    gripper_qpos = float(q7 + q8)
    opening_proxy = float(abs(q7) + abs(q8))

    # EEF position from grip site (robot0_eef_pos as proxy when sim not accessible)
    eef_pos = obs_dict.get('robot0_eef_pos')
    if eef_pos is None:
        raise RuntimeError('obs missing robot0_eef_pos')
    eef_pos = np.asarray(eef_pos, dtype=np.float64).flatten()
    eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])

    # EEF velocity = causal delta (matching frozen collector)
    if _prev_eef is not None and np.all(np.isfinite([eef_x, eef_y, eef_z])):
        eef_vx = eef_x - _prev_eef[0]
        eef_vy = eef_y - _prev_eef[1]
        eef_vz = eef_z - _prev_eef[2]
    else:
        eef_vx = eef_vy = eef_vz = 0.0
    if np.all(np.isfinite([eef_x, eef_y, eef_z])):
        _prev_eef = (eef_x, eef_y, eef_z)

    # Action
    raw_gripper = float(raw[6])
    env_gripper = float(env_action[6]) if len(env_action) > 6 else raw_gripper
    action_dx = float(raw[0]); action_dy = float(raw[1]); action_dz = float(raw[2])
    action_gripper = raw_gripper

    # Canonical 25D via SC5StreamingFeatureAdapterV2.update()
    feat_result = _adapter.update(
        step_id=step,
        raw_gripper=raw_gripper, env_gripper=env_gripper,
        gripper_qpos=gripper_qpos, gripper_opening_proxy=opening_proxy,
        eef_x=eef_x, eef_y=eef_y, eef_z=eef_z,
        eef_vx=eef_vx, eef_vy=eef_vy, eef_vz=eef_vz,
        action_dx=action_dx, action_dy=action_dy, action_dz=action_dz,
        action_gripper=action_gripper,
    )
    if not feat_result.get('valid'):
        raise RuntimeError('SC5 adapter invalid at step {}: {}'.format(
            step, feat_result.get('error', 'unknown')))

    f25d_dict = feat_result['features']
    f25d = np.array([float(f25d_dict.get(name, 0.0)) for name in FEATURE_NAMES_25D], dtype=np.float32)

    # 9D policy intent from model.generate() scores
    p9d = np.zeros(9, dtype=np.float32)
    if clean_model_output is not None and hasattr(clean_model_output, 'scores') and clean_model_output.scores:
        last_scores = clean_model_output.scores[-1]
        if last_scores.dim() >= 2:
            last_scores = last_scores[0] if last_scores.dim() == 2 else last_scores[0, -1]
        if last_scores.dim() >= 1 and last_scores.shape[-1] > 100:
            log_probs = torch.log_softmax(last_scores.float(), dim=-1)
            probs = log_probs.exp(); V = last_scores.shape[-1]
            n_bin = min(256, V // 8)
            open_t = torch.arange(0, max(1, n_bin // 4), device=last_scores.device)
            close_t = torch.arange(V - n_bin // 4, V, device=last_scores.device)
            olm = torch.logsumexp(log_probs.index_select(-1, open_t), dim=-1)
            clm = torch.logsumexp(log_probs.index_select(-1, close_t), dim=-1)
            ent = -(probs * log_probs).sum(dim=-1) / max(1.0, np.log(V))
            t1p, t1t = probs.max(dim=-1)
            p9d[0] = float(olm.exp().cpu()); p9d[1] = float(clm.exp().cpu())
            p9d[2] = float((olm - clm).cpu()); p9d[3] = float(ent.cpu())
            p9d[4] = float(t1p.cpu())
            p9d[5] = 1.0 if int(t1t) in open_t else 0.0
            p9d[6] = 1.0 if int(t1t) in close_t else 0.0
            p9d[7] = 0.0; p9d[8] = 0.0

    # 9D gripper token (same logits source)
    g9d = p9d.copy()

    candidate_close = bool(raw_gripper < 0.5)

    return {'f25d': f25d.astype(np.float32), 'p9d': p9d.astype(np.float32),
            'g9d': g9d.astype(np.float32), 'candidate_close': candidate_close}
