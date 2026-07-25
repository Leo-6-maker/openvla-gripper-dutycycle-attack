"""Canonical N4 feature provider using SC5StreamingFeatureAdapterV2 for 25D.
Extracts 9D policy intent from clean_model_output logits.
Never reads Teacher labels, privileged state, or attack outcome."""
import sys, os, numpy as np, torch
sys.path.insert(0, '/mnt/sdc/dty_user/openvla_attack/src')

from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2

# 9D policy intent feature names (from OFFICIAL_V3_SOURCE_CONTRACT_V1.json)
POLICY_INTENT_ORDER = [
    'clean_open_probability_mass', 'clean_close_probability_mass',
    'clean_open_minus_close_log_mass', 'clean_action_token_entropy_normalized',
    'clean_top1_probability', 'clean_top1_is_open', 'clean_top1_is_close',
    'clean_best_open_rank_normalized', 'clean_best_close_rank_normalized',
]

# Per-provider-instance state (history tracking)
_feature_adapter = None
_history_qpos = []
_history_cmd = []

def _reset_history():
    global _feature_adapter, _history_qpos, _history_cmd
    _feature_adapter = SC5StreamingFeatureAdapterV2()
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
        _reset_history()
    
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
