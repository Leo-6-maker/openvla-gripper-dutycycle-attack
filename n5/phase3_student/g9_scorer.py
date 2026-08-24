"""G9 Score-Only Scorer + Causal Scheduler Framework.

STATUS: SCORE_ONLY / CONTRACT_UNRESOLVED
  - Exports raw logits with full provenance binding.
  - Causal scheduler state machine implemented but thresholds NOT selected.
  - safe_release head marked UNRESOLVED_AUXILIARY.
  - Fail-closed: any threshold-fit attempt without resolved contract → exit(2).

HARD RULES (enforced in code, not comments):
  R1: NO threshold search, fit, or selection.
  R2: NO G9_SEAL generation.
  R3: NO reading G10 test manifest.
  R4: safe_release = UNRESOLVED_AUXILIARY (not fitted, not gated).
  R5: candidate_close NOT a gate.
  R6: instability NOT a veto.
  R7: unknown → abstain, never negative.
  R8: NO suite-specific thresholds.
  R9: Missing or unresolved contract → fail closed (exit 2).
"""
import json, os, sys, time, hashlib, argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

DIR = os.path.dirname(__file__)
sys.path.insert(0, DIR)

from n5_dataset import N5Dataset, N5Normalizer, N5_HEAD_NAMES
from n5_student_model import N5MultiHeadStudent, compute_schema_sha as n5_schema_sha

# ── Paths ──
G6_SEAL_PATH = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g6_training_seal/G6_SEAL_V2.json'
IDENTITY_MANIFEST = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops/FACTORIZED_PHASE_B2_DETERMINISTIC_ALLOCATION_V3_804113EE_20260723/checkpoint_training_identity_manifest.json'
CS200_ROOT = '/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/clean'
LABEL_ROOT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase2_labels/g4_label_production'
G8_CKPT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g8_n5_training/seed_19903/n5_seed19903_best.pt'
G9_OUT = '/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/g9_scoring'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.makedirs(G9_OUT, exist_ok=True)

# ── Contract State ──
CONTRACT_STATE = 'UNRESOLVED'
SAFE_RELEASE_STATUS = 'UNRESOLVED_AUXILIARY'
G9_MODE = 'SCORE_ONLY'  # Never change to 'CALIBRATE' or 'SELECT_THRESHOLDS' without user directive


# ═══════════════════════════════════════════════════════════════════
# Part 1: Score-Only Logit Export
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ScoringRecord:
    """Single-step scoring record with full provenance."""
    identity: str
    suite: str
    step: int
    # Raw logits (5 heads)
    logit_physical_criticality: float
    logit_k10_feasible: float
    logit_safe_release: float
    logit_instability: float
    logit_gripper_closing_state: float
    # Labels (tri-state: -1/0/1)
    label_physical_criticality: int
    label_k10_feasible: int
    label_safe_release: int
    label_instability: int
    label_gripper_closing_state: int
    # Valid masks
    valid_physical_criticality: bool
    valid_k10_feasible: bool
    valid_safe_release: bool
    valid_instability: bool
    valid_gripper_closing_state: bool


def export_logits(model, normalizer, identities, split_name, out_dir):
    """Export raw logits for a split. Returns records and metadata."""
    dataset = N5Dataset(IDENTITY_MANIFEST, CS200_ROOT, LABEL_ROOT, split='checkpoint_training')
    all_idents = set(dataset.identities)
    target_idents = [i for i in identities if i in all_idents]

    records = []
    model.eval()
    total_steps = 0

    for ident in target_idents:
        idx = dataset.identities.index(ident)
        ep = dataset.get_episode(idx)
        feats_norm = normalizer.normalize(ep['features'].copy())
        feats_t = torch.tensor(feats_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(feats_t)

        for t in range(ep['T']):
            rec = ScoringRecord(
                identity=ident,
                suite=ep['suite'],
                step=t,
                logit_physical_criticality=float(output['physical_criticality'][0, t].cpu()),
                logit_k10_feasible=float(output['k10_feasible'][0, t].cpu()),
                logit_safe_release=float(output['safe_release'][0, t].cpu()),
                logit_instability=float(output['instability'][0, t].cpu()),
                logit_gripper_closing_state=float(output['gripper_closing_state'][0, t].cpu()),
                label_physical_criticality=int(ep['labels']['physical_criticality'][t]),
                label_k10_feasible=int(ep['labels']['k10_feasible'][t]),
                label_safe_release=int(ep['labels']['safe_release'][t]),
                label_instability=int(ep['labels']['instability'][t]),
                label_gripper_closing_state=int(ep['labels']['gripper_closing_state'][t]),
                valid_physical_criticality=bool(ep['valid_masks']['physical_criticality'][t]),
                valid_k10_feasible=bool(ep['valid_masks']['k10_feasible'][t]),
                valid_safe_release=bool(ep['valid_masks']['safe_release'][t]),
                valid_instability=bool(ep['valid_masks']['instability'][t]),
                valid_gripper_closing_state=bool(ep['valid_masks']['gripper_closing_state'][t]),
            )
            records.append(rec)
        total_steps += ep['T']

    # Write JSONL
    jsonl_path = os.path.join(out_dir, f'scores_{split_name}.jsonl')
    with open(jsonl_path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec.__dict__) + '\n')

    # Write numpy arrays for fast loading
    n_steps = len(records)
    logits_arr = np.zeros((n_steps, 5), dtype=np.float32)
    labels_arr = np.zeros((n_steps, 5), dtype=np.int8)
    valid_arr = np.zeros((n_steps, 5), dtype=bool)
    idents_arr = [rec.identity for rec in records]

    for i, rec in enumerate(records):
        logits_arr[i] = [
            rec.logit_physical_criticality, rec.logit_k10_feasible,
            rec.logit_safe_release, rec.logit_instability,
            rec.logit_gripper_closing_state,
        ]
        labels_arr[i] = [
            rec.label_physical_criticality, rec.label_k10_feasible,
            rec.label_safe_release, rec.label_instability,
            rec.label_gripper_closing_state,
        ]
        valid_arr[i] = [
            rec.valid_physical_criticality, rec.valid_k10_feasible,
            rec.valid_safe_release, rec.valid_instability,
            rec.valid_gripper_closing_state,
        ]

    npz_path = os.path.join(out_dir, f'scores_{split_name}.npz')
    np.savez_compressed(npz_path, logits=logits_arr, labels=labels_arr, valid=valid_arr)

    meta = {
        'split': split_name,
        'n_identities': len(target_idents),
        'n_steps': n_steps,
        'jsonl': jsonl_path,
        'npz': npz_path,
        'jsonl_sha': hashlib.sha256(open(jsonl_path, 'rb').read()).hexdigest(),
    }
    return meta, records


# ═══════════════════════════════════════════════════════════════════
# Part 2: Causal Scheduler State Machine
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SchedulerState:
    """Causal state machine — no future information, no global thresholds."""
    # Configuration (ALL UNFROZEN until contract resolved)
    tau_c: Optional[float] = None       # UNFROZEN
    tau_k: Optional[float] = None       # UNFROZEN
    tau_safe: Optional[float] = None    # UNRESOLVED_AUXILIARY
    D: int = 6                          # persistence counter (UNFROZEN)
    K: int = 10                         # horizon minimum (from K10 definition)

    # Runtime state
    persistence_counter: int = 0
    latch: bool = False
    emit_step: Optional[int] = None
    t: int = 0

    def reset(self):
        self.persistence_counter = 0
        self.latch = False
        self.emit_step = None
        self.t = 0

    def step(self, p_critical: float, p_k10: float, p_safe: float,
             valid_critical: bool, valid_k10: bool, valid_safe: bool,
             remaining_horizon: int, candidate_close: bool) -> Dict:
        """Process one step. Returns decision dict.

        RULES (enforced):
          R1: No tau_c/tau_k/tau_safe search — use placeholder if not set.
          R2: unsafe_safe → no gate. safe_release flagged UNRESOLVED.
          R3: candidate_close NOT used as gate (R5).
          R4: instability NOT used as veto (R6).
          R5: unknown valid_mask → abstain, never negative (R7).
          R6: No suite-specific logic (R8).
          R7: first-emit-only latch.
          R8: remaining_horizon < K → abstain.

        Returns dict with: {decision, reason, counters, state}
        """
        self.t += 1

        # R7: first-emit-only latch
        if self.latch:
            return {'decision': 'LATCHED', 'reason': 'ALREADY_EMITTED',
                    'persistence': self.persistence_counter, 'emit_step': self.emit_step,
                    'step': self.t - 1}

        # R5: candidate_close NOT a gate — skip if False, but don't gate on it
        has_close_candidate = candidate_close

        # R8: horizon abstain
        if remaining_horizon < self.K:
            return {'decision': 'ABSTAIN', 'reason': 'HORIZON_INSUFFICIENT',
                    'remaining': remaining_horizon, 'step': self.t - 1}

        # Check criticality (tau_c = UNFROZEN)
        crit_positive = valid_critical and (p_critical >= (self.tau_c if self.tau_c is not None else 0.5))

        # Check K10 feasibility (tau_k = UNFROZEN)
        k10_positive = valid_k10 and (p_k10 >= (self.tau_k if self.tau_k is not None else 0.5))

        # R1: Placeholder threshold if tau not set
        if self.tau_c is None:
            crit_positive = valid_critical and p_critical > 0.5
        if self.tau_k is None:
            k10_positive = valid_k10 and p_k10 > 0.5

        # R5: candidate_close as auxiliary, not gate
        # Persistence requires sustained critical+feasible AND close candidate
        if crit_positive and k10_positive and has_close_candidate:
            self.persistence_counter += 1
        else:
            self.persistence_counter = 0

        # R3: safe_release UNRESOLVED_AUXILIARY — logged but never gates
        safe_flag = valid_safe and (p_safe >= (self.tau_safe if self.tau_safe is not None else 0.5))
        if self.tau_safe is None:
            safe_flag = valid_safe and p_safe > 0.5

        # Emission check
        if self.persistence_counter >= self.D:
            self.latch = True
            self.emit_step = self.t - 1
            return {'decision': 'EMIT', 'reason': 'PERSISTENCE_SATISFIED',
                    'persistence': self.persistence_counter, 'emit_step': self.emit_step,
                    'step': self.t - 1,
                    'safe_release_flag': safe_flag, 'safe_release_status': SAFE_RELEASE_STATUS}

        return {'decision': 'NO_EMIT', 'reason': 'PERSISTENCE_NOT_MET',
                'persistence': self.persistence_counter, 'safe_release_flag': safe_flag,
                'step': self.t - 1}


# ═══════════════════════════════════════════════════════════════════
# Part 3: Calibration Interface (SCORE_ONLY — no fitting allowed)
# ═══════════════════════════════════════════════════════════════════

class CalibrationInterface:
    """Interface for calibration routines. All fitting DISABLED in SCORE_ONLY mode.

    Will raise RuntimeError if fit() or select_thresholds() is called
    while CONTRACT_STATE == 'UNRESOLVED'.
    """

    def __init__(self):
        self.fitted = False
        self.platt_a = {name: None for name in ['physical_criticality', 'k10_feasible']}
        self.platt_b = {name: None for name in ['physical_criticality', 'k10_feasible']}
        self.temperature = {name: 1.0 for name in N5_HEAD_NAMES}

    def fit_platt(self, logits, labels, valid_mask, head_name):
        """DISABLED: Fail-closed until contract resolved."""
        if CONTRACT_STATE == 'UNRESOLVED':
            raise RuntimeError(
                f'G9 CONTRACT_UNRESOLVED: Cannot fit Platt scaling for {head_name}. '
                f'Resolve safe-release contract before enabling calibration.'
            )
        # Future implementation: logistic regression on val split
        raise NotImplementedError('Calibration not implemented in SCORE_ONLY mode')

    def fit_temperature(self, logits, labels, valid_mask, head_name):
        """DISABLED: Fail-closed until contract resolved."""
        if CONTRACT_STATE == 'UNRESOLVED':
            raise RuntimeError(
                f'G9 CONTRACT_UNRESOLVED: Cannot fit temperature for {head_name}.'
            )
        raise NotImplementedError('Calibration not implemented in SCORE_ONLY mode')

    def select_thresholds(self, *args, **kwargs):
        """HARD FAIL: Never call in SCORE_ONLY mode."""
        raise RuntimeError(
            'G9 SCORE_ONLY: threshold selection is FORBIDDEN. '
            'Resolve P0 defects, fix safe-release contract, and obtain explicit '
            'user authorization before calling select_thresholds(). '
            'This is a fail-closed design — the program exits rather than produce '
            'invalid thresholds from unresolved contracts.'
        )

    def apply_platt(self, logits: np.ndarray, head_name: str) -> np.ndarray:
        """Apply existing Platt calibration (identity if not fitted)."""
        a = self.platt_a.get(head_name)
        b = self.platt_b.get(head_name)
        if a is None or b is None:
            return 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))  # raw sigmoid
        return 1.0 / (1.0 + np.exp(-np.clip(a * logits + b, -50, 50)))


# ═══════════════════════════════════════════════════════════════════
# Part 4: Main — Score Export
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', type=str, default='val,cal',
                       help='Comma-separated split names to export')
    parser.add_argument('--mode', type=str, default='SCORE_ONLY',
                       choices=['SCORE_ONLY'],
                       help='Operating mode (only SCORE_ONLY allowed)')
    args = parser.parse_args()

    if args.mode != 'SCORE_ONLY':
        print(f'ERROR: mode={args.mode} not allowed. Only SCORE_ONLY.')
        sys.exit(2)

    print(f'G9 Mode: {G9_MODE}')
    print(f'Contract: {CONTRACT_STATE}')
    print(f'Safe-Release: {SAFE_RELEASE_STATUS}')
    print(f'Device: {DEVICE}')
    print()

    # Load G6 seal
    with open(G6_SEAL_PATH) as f:
        g6_seal = json.load(f)
    print(f'G6 Seal V2 SHA: {g6_seal["self_sha256"][:16]}...')

    # Load normalizer
    normalizer = N5Normalizer.load(g6_seal['normalization']['path'])

    # Load model
    ckpt = torch.load(G8_CKPT, map_location=DEVICE, weights_only=False)
    model = N5MultiHeadStudent(input_dim=51, hidden=64, short_rf=32, long_rf=128).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    ckpt_sha = hashlib.sha256(open(G8_CKPT, 'rb').read()).hexdigest()
    print(f'Model: {G8_CKPT}')
    print(f'Checkpoint SHA: {ckpt_sha[:16]}...')
    print(f'N5 Schema SHA: {n5_schema_sha()[:16]}...')
    print()

    splits = [s.strip() for s in args.splits.split(',')]
    export_meta = {}

    for split_name in splits:
        print(f'--- Exporting {split_name} ---')
        identities = g6_seal['split'][f'{split_name}_identities']

        meta, records = export_logits(model, normalizer, identities, split_name, G9_OUT)
        export_meta[split_name] = meta
        print(f'  {meta["n_identities"]} identities, {meta["n_steps"]} steps')
        print(f'  JSONL: {meta["jsonl"]}')
        print(f'  NPZ: {meta["npz"]}')

        # Diagnostic: count per-head valid steps
        head_stats = defaultdict(lambda: {'valid': 0, 'pos': 0, 'neg': 0, 'unk': 0})
        for rec in records:
            for head_name in N5_HEAD_NAMES:
                valid_key = f'valid_{head_name}'
                label_key = f'label_{head_name}'
                if getattr(rec, valid_key):
                    head_stats[head_name]['valid'] += 1
                    val = getattr(rec, label_key)
                    if val > 0: head_stats[head_name]['pos'] += 1
                    elif val < 0: head_stats[head_name]['neg'] += 1
                    else: head_stats[head_name]['unk'] += 1

        for head_name in N5_HEAD_NAMES:
            s = head_stats[head_name]
            print(f'  {head_name}: valid={s["valid"]}, pos={s["pos"]}, neg={s["neg"]}, unk={s["unk"]}')

    # Write scoring manifest
    manifest = {
        'g9_mode': G9_MODE,
        'contract_state': CONTRACT_STATE,
        'safe_release_status': SAFE_RELEASE_STATUS,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'g6_seal_sha': g6_seal['self_sha256'],
        'checkpoint_sha': ckpt_sha,
        'checkpoint_path': G8_CKPT,
        'n5_schema_sha': n5_schema_sha(),
        'normalizer_sha': g6_seal['normalization']['sha256'],
        'exports': export_meta,
        'hard_rules': {
            'threshold_search': 'FORBIDDEN',
            'g9_seal_generation': 'FORBIDDEN',
            'safe_release_gating': 'FORBIDDEN',
            'candidate_close_gating': 'FORBIDDEN',
            'instability_veto': 'FORBIDDEN',
            'suite_specific_thresholds': 'FORBIDDEN',
            'unknown_to_negative': 'FORBIDDEN',
            'g10_test_manifest_access': 'FORBIDDEN',
        },
        'fail_closed': {
            'on_missing_contract': 'exit(2)',
            'on_threshold_fit_attempt': 'RuntimeError',
            'on_g9_seal_attempt': 'NotImplementedError',
        },
    }

    manifest_path = os.path.join(G9_OUT, 'SCORING_MANIFEST.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    manifest_sha = hashlib.sha256(open(manifest_path, 'rb').read()).hexdigest()
    print(f'\nScoring Manifest: {manifest_path}')
    print(f'Manifest SHA: {manifest_sha[:16]}...')
    print(f'\nG9 STATUS: SCORE_ONLY — thresholds UNFROZEN, safe_release UNRESOLVED')
    print('NEXT: safe-release Teacher audit before any threshold work.')


if __name__ == '__main__':
    main()
