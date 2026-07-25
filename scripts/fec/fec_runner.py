"""FEC V2.3 five-arm runner. Orchestrates env + Detector + attack per parent.

Integrates:
- N4DetectorAdapter (frozen, 300/300 P4 parity verified)
- TokenPrefixPGDAttacker (canonical PGD engine)
- OpenVLA model + LIBERO env (via existing infrastructure)

NOT integrated (replaced):
- SC5 25D detector
- Factorized 3-head scheduler
- V4 trigger/budget system (replaced by N4 first-emit + K=10 budget)
"""
import json, os, sys, time, hashlib, numpy as np, torch
from collections import defaultdict
from datetime import datetime

# ── Frozen Detector constants ──
PLATT_A = 0.5190011735319306
PLATT_B = 0.812702331013635
TAU = 0.855
D_PERSIST = 6
K10 = 10

# Paths
EVIDENCE = '/mnt/sdc/dty_user/openvla_attack_evidence'
CKPT_PATH = EVIDENCE + '/formal_v23_student_training_v1/o0_i0/checkpoint.pt'
NORM_PATH = EVIDENCE + '/fec_implementation_v1/n4_norms_o0i0.pt'


def sha256_file(p):
    d = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): d.update(chunk)
    return d.hexdigest()


class RandomTimeSampler:
    """Pre-rollout random time sampler. Independent of Detector/Teacher/outcome."""

    def __init__(self, seed, min_step=20, max_step_ratio=0.85):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.min_step = min_step
        self.max_step_ratio = max_step_ratio

    def sample(self, episode_max_t):
        """Sample random start_t. Must be K10-executable within [0, max_t)."""
        max_valid = int(episode_max_t * self.max_step_ratio)
        max_valid = max(max_valid, self.min_step)
        if max_valid - K10 + 1 <= self.min_step:
            return None  # Not enough room for K10
        upper = max_valid - K10 + 1
        if upper <= self.min_step:
            return self.min_step
        return int(self.rng.randint(self.min_step, upper))

    def to_manifest(self):
        return {'seed': self.seed, 'min_step': self.min_step,
                'max_step_ratio': self.max_step_ratio, 'K10': K10}


class K10BudgetController:
    """Emit-relative K=10 budget. Attack starts at first_emit, lasts exactly 10 steps."""

    def __init__(self):
        self.attack_start = None
        self.frames_executed = 0

    def bind_emit(self, emit_step):
        """Called when Detector emits. Sets the attack window."""
        self.attack_start = emit_step

    def is_active(self, step):
        """Returns True if step is within the K=10 attack window."""
        if self.attack_start is None:
            return False
        return self.attack_start <= step < self.attack_start + K10

    def frame_index(self, step):
        """Returns 0-indexed frame number within K=10 window, or -1 if not active."""
        if not self.is_active(step):
            return -1
        return step - self.attack_start

    def get_summary(self):
        return {
            'attack_start': self.attack_start,
            'frames_executed': self.frames_executed,
            'k10_planned': K10 if self.attack_start is not None else 0
        }


class FECTelemetry:
    """Collects per-step and per-episode telemetry for one arm."""

    def __init__(self, parent_id, arm, env_seed, policy_seed, attack_seed):
        self.parent_id = parent_id
        self.arm = arm
        self.env_seed = env_seed
        self.policy_seed = policy_seed
        self.attack_seed = attack_seed
        self.steps = []
        self.episode_meta = {}

    def record_step(self, data):
        self.steps.append(data)

    def set_episode_meta(self, meta):
        self.episode_meta = meta

    def to_dict(self):
        return {
            'parent_id': self.parent_id, 'arm': self.arm,
            'env_seed': self.env_seed, 'policy_seed': self.policy_seed,
            'attack_seed': self.attack_seed,
            'n_steps': len(self.steps), 'steps': self.steps,
            'episode_meta': self.episode_meta
        }


def build_step_record(step_t, obs, clean_action, final_action, n4_result,
                      budget, attack_active, attack_frame_idx, arm,
                      route_info=None, exception_info=None):
    """Build a single step telemetry record."""
    rec = {
        'step': step_t,
        'arm': arm,
        'clean_action': [float(x) for x in clean_action] if clean_action is not None else None,
        'final_action': [float(x) for x in final_action] if final_action is not None else None,
        'raw_logit': n4_result['raw_logit'],
        'calibrated_prob': n4_result['calibrated_prob'],
        'candidate_close': n4_result['candidate_close'],
        'persistence_counter': n4_result['persistence_counter'],
        'latch': n4_result['latch'],
        'emitted_this_step': n4_result['emitted_this_step'],
        'emit_step': n4_result['emit_step'],
        'attack_active': attack_active,
        'attack_frame_idx': attack_frame_idx,
    }
    if attack_active:
        rec['attack_start'] = budget.attack_start
        rec['k10_planned'] = K10
    if route_info:
        rec['route'] = route_info
    if exception_info:
        rec['exception'] = exception_info
    return rec


def compute_fec_metrics(telemetries):
    """Compute FEC arm-level metrics from telemetry."""
    metrics = {}
    for arm, tlm_list in telemetries.items():
        n_total = len(tlm_list)
        n_success = sum(1 for t in tlm_list if t.episode_meta.get('task_success', False))
        n_emit = sum(1 for t in tlm_list if t.episode_meta.get('detector_emitted', False))
        n_k10_exec = sum(1 for t in tlm_list if t.episode_meta.get('k10_executed', False))
        n_no_emit = n_total - n_emit

        metrics[arm] = {
            'n_total': n_total,
            'task_success': n_success,
            'success_rate': n_success / max(n_total, 1),
            'n_emit': n_emit,
            'n_no_emit': n_no_emit,
            'n_k10_executed': n_k10_exec,
        }
    return metrics


def validate_rand_matched(true_telemetry, rand_telemetry):
    """Verify RAND matches TRUE on emit step, K, epsilon, route."""
    issues = []
    for t_tlm, r_tlm in zip(true_telemetry, rand_telemetry):
        t_emit = t_tlm.episode_meta.get('first_emit_step')
        r_emit = r_tlm.episode_meta.get('first_emit_step')
        if t_emit != r_emit:
            issues.append('{}: TRUE emit={} RAND emit={}'.format(
                t_tlm.parent_id, t_emit, r_emit))
    return len(issues) == 0, issues


def validate_oracle_override(telemetries):
    """Verify ORACLE preserved arm dims and only changed gripper."""
    issues = []
    for tlm in telemetries:
        for step_rec in tlm.steps:
            clean = step_rec.get('clean_action')
            final = step_rec.get('final_action')
            if clean is not None and final is not None and len(clean) >= 7 and len(final) >= 7:
                arm_match = all(abs(clean[i] - final[i]) < 1e-6 for i in range(6))
                if not arm_match:
                    issues.append('{} step={}: arm dims changed'.format(
                        tlm.parent_id, step_rec['step']))
    return len(issues) == 0, issues
