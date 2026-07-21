"""Test lexicographic selection: safety precedes AUPRC, LR gate, tie-breaking."""
import json, tempfile
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts/detector_v5"))
from select_factorized_v2_candidate import select_candidate, check_lr_gate, check_safety_elimination


def make_candidate(name, release_auprc=0.8, short_auprc=0.7, gap=0.1, params=50000,
                   bg_emit=0.05, unsup=0.0, overlap=0.01,
                   per_split_scores=None):
    m = {
        'release_auprc': release_auprc, 'release_short_auprc': short_auprc,
        'first_later_recall_gap': gap, 'parameter_count': params,
        'background_false_emit_rate': bg_emit,
        'unsupported_route_emit_rate': unsup,
        'release_overlap_emit_rate': overlap,
        'release_auroc': release_auprc - 0.05,
    }
    if per_split_scores:
        m['release_auprc_per_split'] = per_split_scores
    return m


def test_safety_eliminates_high_bg():
    candidates = {
        'safe': make_candidate('safe', bg_emit=0.05),
        'unsafe': make_candidate('unsafe', bg_emit=0.15),  # >0.10
    }
    surviving, eliminated = check_safety_elimination(candidates)
    assert 'safe' in surviving
    assert 'unsafe' in eliminated
    assert 'background_false_emit_rate' in str(eliminated['unsafe']['failures'])


def test_safety_eliminates_unsupported_emit():
    candidates = {
        'safe': make_candidate('safe', unsup=0.0),
        'unsafe': make_candidate('unsafe', unsup=0.001),  # any >0
    }
    surviving, eliminated = check_safety_elimination(candidates)
    assert 'safe' in surviving
    assert 'unsafe' in eliminated


def test_higher_auprc_selected_when_safety_equal():
    candidates = {
        'low': make_candidate('low', release_auprc=0.75),
        'high': make_candidate('high', release_auprc=0.85),
    }
    selected, trace, eliminated = select_candidate(candidates)
    assert selected == 'high', f'got {selected}'


def test_lr_gate_rejects_when_candidate_lower():
    candidate = {'good': make_candidate('good', release_auprc=0.80, release_auroc=0.75)}
    lr = {'release_auprc': 0.85, 'release_auroc': 0.80}
    passed, msg, checks = check_lr_gate('good', candidate, lr)
    assert not passed, f'should reject: {msg}'


def test_lr_gate_passes_when_candidate_higher():
    candidate = {'good': make_candidate('good', release_auprc=0.85, release_auroc=0.80)}
    lr = {'release_auprc': 0.80, 'release_auroc': 0.75}
    passed, msg, checks = check_lr_gate('good', candidate, lr)
    assert passed, f'should pass: {msg}'


def test_parameter_count_tiebreak():
    candidates = {
        'big': make_candidate('big', params=60000, release_auprc=0.80),
        'small': make_candidate('small', params=30000, release_auprc=0.80),
    }
    selected, trace, eliminated = select_candidate(candidates)
    # Both survive through all metrics (identical AUPRC) → smaller params wins
    assert selected == 'small', f'got {selected}'
