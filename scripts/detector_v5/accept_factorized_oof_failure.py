#!/usr/bin/env python3
"""Phase R0: Formal acceptance of CATASTROPHIC_OOF_FAILURE.

Creates sealed artifact recording the failure status, binding the evaluation
root and all 24 checkpoints. No re-labeling, no seed selection, no threshold tuning.
"""
import hashlib, json, os, sys, uuid
from pathlib import Path
from datetime import datetime, timezone

OPS = Path('/mnt/sdc/dty_user/openvla_attack_evidence/c2g/c2g_cs200_official_v3_20260716/ops')
EVAL_ROOT = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_OOF_EVALUATION_V1_20260721'
TRAINING_ROOT = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_OOF_335048c_20260721'
RECON_ROOT = OPS / 'OFFICIAL_V3_FACTORIZED_STUDENT_OOF_PROVENANCE_RECONCILIATION_V1_20260721'
OUT = OPS / 'DETECTOR_V5_FACTORIZED_OOF_FAILURE_ACCEPTANCE_V1_20260721'


def sha256_file(p):
    d = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): d.update(b)
    return d.hexdigest()


def _atomic_text(p, v):
    t = p.with_name(f'.{p.name}.{uuid.uuid4().hex}.tmp')
    with t.open('x') as f: f.write(v); f.flush(); os.fsync(f.fileno())
    os.replace(t, p)


def write_seal(root):
    excl = {'SHA256SUMS', 'SHA256SUMS.sha256'}
    fs = sorted((p for p in root.rglob('*') if p.is_file() and p.name not in excl),
                key=lambda p: p.relative_to(root).as_posix())
    c = ''.join(f'{sha256_file(p)}  {p.relative_to(root).as_posix()}\n' for p in fs)
    _atomic_text(root / 'SHA256SUMS', c)
    _atomic_text(root / 'SHA256SUMS.sha256', f'{sha256_file(root / "SHA256SUMS")}  SHA256SUMS\n')


if OUT.exists():
    raise SystemExit(f'OUTPUT EXISTS: {OUT}')

# Load evaluation results
gate = json.loads((EVAL_ROOT / 'oof_gate_decision.json').read_text())
head_metrics = json.loads((EVAL_ROOT / 'per_head_metrics.json').read_text())
route_metrics = json.loads((EVAL_ROOT / 'per_route_metrics.json').read_text())
later_metrics = json.loads((EVAL_ROOT / 'later_event_metrics.json').read_text())
safety_metrics = json.loads((EVAL_ROOT / 'safety_emit_metrics.json').read_text())

# Build checkpoint inventory
checkpoints = []
for mt in ['25D9D', '25D']:
    for fold in [0, 1, 2, 3]:
        for seed in [42, 123, 456]:
            d = TRAINING_ROOT / mt / f'fold{fold}_seed{seed}'
            checkpoints.append({
                'model_type': mt, 'fold_id': fold, 'seed': seed,
                'dir': str(d), 'seal': sha256_file(d / 'SHA256SUMS'),
            })

acceptance = {
    'schema': 'DETECTOR_V5_FACTORIZED_OOF_FAILURE_ACCEPTANCE_V1',
    'status': 'CATASTROPHIC_OOF_FAILURE_ACCEPTED',
    'timestamp': datetime.now(timezone.utc).isoformat(),

    'evaluation_root': str(EVAL_ROOT),
    'evaluation_seal': sha256_file(EVAL_ROOT / 'SHA256SUMS'),
    'training_root': str(TRAINING_ROOT),
    'reconciliation_root': str(RECON_ROOT),

    'failure_summary': {
        'gate_outcome': 'CATASTROPHIC_OOF_FAILURE',
        'catastrophic_conditions': [
            {
                'condition': 'single_object_pick_place_min_recall',
                'value_25d9d': 0.4656, 'value_25d': 0.4051,
                'catastrophic_threshold': 0.50,
                'failing_head': 'release',
            },
            {
                'condition': 'release_later_event_recall',
                'value_25d9d': 0.4246, 'value_25d': 0.3684,
                'catastrophic_threshold': 0.50,
            },
        ],
        'failed_non_catastrophic': [
            {'condition': 'macro_grasp_recall', 'value_25d9d': 0.8544, 'threshold': 0.90},
            {'condition': 'multi_object_transfer_min_recall', 'value_25d9d': 0.5518, 'threshold': 0.75},
        ],
        'passing': [
            {'condition': 'macro_manipulation_recall', 'value_25d9d': 0.9225},
            {'condition': 'release_overlap_emit_rate', 'value_25d9d': 0.0024},
            {'condition': 'unsupported_route_emit', 'value': 0},
            {'condition': 'background_false_emit_rate', 'value_25d9d': 0.0511},
        ],
    },

    'hard_constraints_upheld': [
        'no_threshold_tuning_on_oof',
        'no_seed_selection_on_oof',
        'no_checkpoint_selection_on_oof',
        'no_metric_redefinition_after_prediction',
        'val_loss_not_a_selection_criterion',
        'all_24_checkpoints_evaluated',
    ],

    'checkpoint_inventory': {
        'count': len(checkpoints),
        'all_retained': True,
        'disposition': 'FROZEN_AS_TRAINED',
        'checkpoints': checkpoints,
    },

    'next_phase_authorization': {
        'full_fit': 'HOLD',
        'cal': 'HOLD',
        'check': 'HOLD',
        'vis_integration': 'NOT_STARTED',
        'passive_canary': 'NOT_STARTED',
        'attack_canary': 'NOT_STARTED',
        'full_attack_matrix': 'PREPARED_NOT_LAUNCHED',
        'release_failure_forensics': 'AUTHORIZED',
    },

    'diagnostic_hypotheses': [
        'release_label_observability_or_short_event_modeling',
        'fixed_threshold_probability_conservatism',
        'multi_event_state_reset_generalization',
        'ordinary_training_failure',
        'evaluator_implementation_error',
    ],
}

staging = OUT.with_name(f'.{OUT.name}.{uuid.uuid4().hex}.staging')
staging.mkdir(parents=True)
_atomic_text(staging / 'failure_acceptance.json', json.dumps(acceptance, indent=2, sort_keys=True) + '\n')
write_seal(staging)
os.replace(staging, OUT)

print(json.dumps({
    'status': 'FAILURE_ACCEPTANCE_SEALED',
    'root': str(OUT),
    'seal': sha256_file(OUT / 'SHA256SUMS'),
}, indent=2))
