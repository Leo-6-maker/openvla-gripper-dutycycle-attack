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
