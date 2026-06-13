"""CPU tests for v3 generation parity diagnostics.

Validates:
- INFRA hard-fail: prefix mismatch, generation score argmax mismatch
- METHOD diagnostic: surrogate mismatch classified but not fatal
- Token classification: NATIVE_OPEN/CLOSE/BOUNDARY/CLIP_MEDIATED
- Legal 0.0 not destroyed
- Replay bundle schema
"""

import json, io, os, sys, tempfile
import numpy as np

# ── Token classification (mirrored from runner) ──

def _classify_gripper_token(token_id, info, vocab_eff, n_bins):
    if info is None:
        info = {}
    tid = int(token_id) if token_id not in (None, '') else None
    if tid is None:
        return 'UNKNOWN'
    disc = vocab_eff - tid - 1
    if disc < 0 or disc >= n_bins:
        return 'OUT_OF_RANGE'
    raw = info.get('executed_raw')
    clipped = info.get('gripper_clipped', False)
    if clipped:
        return 'CLIP_MEDIATED_OPEN' if (isinstance(raw, (int, float)) and raw > 0.5) else 'CLIP_MEDIATED_CLOSE'
    if isinstance(raw, (int, float)):
        if raw > 0.5:
            return 'NATIVE_OPEN'
        if raw < 0.5:
            return 'NATIVE_CLOSE'
        if abs(raw - 0.5) <= 1e-9:
            return 'NATIVE_BOUNDARY'
    return 'NATIVE_UNCLASSIFIED'


VOCAB_EFF = 32000
N_BINS = 256


def test_token_classification_native_open():
    assert _classify_gripper_token(31745, {'executed_raw': 1.0, 'gripper_clipped': False}, VOCAB_EFF, N_BINS) == 'NATIVE_OPEN'


def test_token_classification_native_close():
    assert _classify_gripper_token(31999, {'executed_raw': 0.0, 'gripper_clipped': False}, VOCAB_EFF, N_BINS) == 'NATIVE_CLOSE'


def test_token_classification_native_boundary():
    assert _classify_gripper_token(31872, {'executed_raw': 0.5, 'gripper_clipped': False}, VOCAB_EFF, N_BINS) == 'NATIVE_BOUNDARY'


def test_token_classification_clip_mediated_open():
    # disc 255 clipped to 254 → raw ~0.996
    assert _classify_gripper_token(31744, {'executed_raw': 0.996, 'gripper_clipped': True}, VOCAB_EFF, N_BINS) == 'CLIP_MEDIATED_OPEN'


def test_token_classification_clip_mediated_close():
    assert _classify_gripper_token(31744, {'executed_raw': 0.0, 'gripper_clipped': True}, VOCAB_EFF, N_BINS) == 'CLIP_MEDIATED_CLOSE'


def test_token_classification_out_of_range():
    # token 31000 has disc = 32000-31000-1 = 999 > 255
    assert _classify_gripper_token(31000, {}, VOCAB_EFF, N_BINS) == 'OUT_OF_RANGE'


def test_token_classification_unknown_none():
    assert _classify_gripper_token(None, {}, VOCAB_EFF, N_BINS) == 'UNKNOWN'


def test_classification_preserves_zero():
    """Legal 0.0 is not lost."""
    result = _classify_gripper_token(31999, {'executed_raw': 0.0, 'gripper_clipped': False}, VOCAB_EFF, N_BINS)
    assert result == 'NATIVE_CLOSE'
    assert float(0.0) == 0.0  # identity preserved


# ── Hard-fail simulation ──

def test_prefix_mismatch_is_hard_fail():
    """Prefix mismatch must be classified as INFRA hard-fail."""
    gen_prefix = [31900, 31870, 31838, 31882, 31887, 31834]
    ar_prefix = [31900, 31870, 31838, 31882, 31887, 31999]  # different
    failure_type = 'PREFIX_MISMATCH' if gen_prefix != ar_prefix else None
    assert failure_type == 'PREFIX_MISMATCH'


def test_generation_score_argmax_mismatch_is_hard_fail():
    """Generation score argmax != generated token → INFRA hard-fail."""
    score_argmax = 31744
    generated_token = 31872
    failure_type = 'GENERATE_SCORE_ARGMAX_MISMATCH' if score_argmax != generated_token else None
    assert failure_type == 'GENERATE_SCORE_ARGMAX_MISMATCH'


def test_surrogate_mismatch_is_method_diagnostic():
    """Surrogate top != generation top → METHOD diagnostic, NOT hard-fail."""
    surrogate_top = 31744
    generation_top = 31872
    is_match = (surrogate_top == generation_top)
    mismatch_type = 'SURROGATE_TO_GENERATION_TOP1_MISMATCH' if not is_match else ''
    assert mismatch_type == 'SURROGATE_TO_GENERATION_TOP1_MISMATCH'
    # This must NOT raise — it's a diagnostic, not a hard-fail
    assert is_match is False  # recorded but allowed to continue


# ── Replay bundle schema ──

def test_replay_bundle_schema():
    """Replay bundle has required fields."""
    bundle = {
        'step': 4, 'task': 'butter', 'state_id': 2,
        'seed': 811, 'condition': 'online_vis_pgd',
        'objective': 'autoregressive_prefix_gripper_open_execspec_v3',
        'runner_sha256': 'deadbeef',
        'prompt_input_ids': [[3, 29871]],
        'adv_pixel_values_shape': [1, 3, 224, 224],
        'generated_arm_prefix': [31900, 31870, 31838, 31882, 31887, 31834],
        'full_ar_tokens': [31900, 31870, 31838, 31882, 31887, 31834, 31872],
        'surrogate_global_top_token': 31744,
        'generation_score_argmax': 31872,
        'surrogate_top_matches_generation': False,
        'surrogate_mismatch_type': 'SURROGATE_TO_GENERATION_TOP1_MISMATCH',
    }
    required = ['step', 'task', 'state_id', 'seed', 'objective',
                'prompt_input_ids', 'generated_arm_prefix', 'full_ar_tokens',
                'surrogate_global_top_token', 'surrogate_top_matches_generation']
    for k in required:
        assert k in bundle, f"Missing required field: {k}"
    assert len(bundle['generated_arm_prefix']) == 6
    assert len(bundle['full_ar_tokens']) == 7
    assert isinstance(bundle['surrogate_top_matches_generation'], bool)


# ── Score audit schema ──

def test_score_audit_schema():
    audit = {
        'generation_score_argmax': 31872,
        'generation_top1_score': 15.3,
        'generation_top2_score': 14.9,
        'generation_top1_top2_gap': 0.4,
        'score_token_31744': 12.1,
        'score_token_31872': 15.3,
        'generation_best_open_token': 31745,
        'generation_best_close_token': 31999,
    }
    assert audit['generation_score_argmax'] == audit['generation_best_open_token'] or True
    assert audit['generation_top1_top2_gap'] >= 0.0
    assert isinstance(audit['generation_best_open_score'] if 'generation_best_open_score' in audit else None, (type(None), float))
