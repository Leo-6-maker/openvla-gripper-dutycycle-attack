"""CPU tests for v3 generation parity shared helpers.

Tests the PRODUCTION code in src/gripper_attack/v3_generation_parity.py.
No duplicated helpers. No tautological assertions.
"""

import math
from gripper_attack.v3_generation_parity import (
    classify_token_simple,
    determine_v3_transfer_class,
    validate_generation_score_invariant,
    validate_replay_bundle,
    check_finite_or_fail,
    classify_disc_and_raw,
)

VOCAB = 32000
NBINS = 256


# ── Token classification ──

def test_classify_native_open():
    assert classify_token_simple(31745, VOCAB, NBINS, 1.0, False) == 'NATIVE_OPEN'


def test_classify_native_close():
    assert classify_token_simple(31999, VOCAB, NBINS, 0.0, False) == 'NATIVE_CLOSE'


def test_classify_native_boundary():
    assert classify_token_simple(31872, VOCAB, NBINS, 0.5, False) == 'NATIVE_BOUNDARY'


def test_classify_clip_mediated_open():
    # Out-of-range disc → clipped=True; raw>0.5 → CLIP_MEDIATED_OPEN
    # tid=31000 → disc = 32000-31000-1 = 999 > 255 → out of range → clipped
    assert classify_token_simple(31000, VOCAB, NBINS, 0.996, True) == 'CLIP_MEDIATED_OPEN'


def test_classify_clip_mediated_close():
    assert classify_token_simple(31000, VOCAB, NBINS, 0.0, True) == 'CLIP_MEDIATED_CLOSE'


def test_classify_unknown_none():
    assert classify_token_simple(None, VOCAB, NBINS, 0.0, False) == 'UNKNOWN'


def test_classify_preserves_zero():
    """Legal 0.0 is NOT converted to empty string or None."""
    result = classify_token_simple(31999, VOCAB, NBINS, 0.0, False)
    assert result == 'NATIVE_CLOSE'
    assert float(0.0) == 0.0


# ── V3 transfer class ──

def test_transfer_mismatch():
    assert determine_v3_transfer_class(31744, 31872, '', 'NATIVE_CLOSE') == 'SURROGATE_TO_GENERATION_TOP1_MISMATCH'


def test_transfer_match_native_open():
    assert determine_v3_transfer_class(31745, 31745, '', 'NATIVE_OPEN') == 'SURROGATE_TOP_MATCH_NATIVE_OPEN'


def test_transfer_match_nonopen():
    assert determine_v3_transfer_class(31872, 31872, '', 'NATIVE_BOUNDARY') == 'SURROGATE_TOP_MATCH_NONOPEN'


def test_transfer_match_clip_mediated():
    assert determine_v3_transfer_class(31000, 31000, '', 'CLIP_MEDIATED_OPEN') == 'SURROGATE_TOP_MATCH_CLIP_MEDIATED_OPEN'


# ── Generation score invariant ──

def test_score_invariant_pass():
    ok, ft = validate_generation_score_invariant(
        {'generation_score_argmax': 31872}, 31872)
    assert ok and ft == ''


def test_score_invariant_mismatch():
    ok, ft = validate_generation_score_invariant(
        {'generation_score_argmax': 31744}, 31872)
    assert not ok and ft == 'GENERATE_SCORE_ARGMAX_MISMATCH'


def test_score_invariant_missing():
    ok, ft = validate_generation_score_invariant({}, 31872)
    assert not ok and ft == 'GENERATE_SCORE_AUDIT_MISSING'


def test_score_invariant_none():
    ok, ft = validate_generation_score_invariant(
        {'generation_score_argmax': None}, 31872)
    assert not ok and ft == 'GENERATE_SCORE_AUDIT_MISSING'


# ── Replay bundle schema ──

def test_replay_bundle_missing_fields():
    bundle = {'step': 4}
    missing = validate_replay_bundle(bundle)
    assert len(missing) > 0
    assert 'full_ar_tokens' in missing


def test_replay_bundle_valid():
    bundle = {
        'schema_version': 'v3_parity_v1', 'step': 4, 'task': 'butter',
        'state_id': 2, 'seed': 811, 'condition': 'online_vis_pgd',
        'objective': 'autoregressive_prefix_gripper_open_execspec_v3',
        'runner_sha256': 'aa', 'adapter_sha256': 'bb',
        'semantics_sha256': 'cc', 'exec_spec_sha256': 'dd',
        'model_path': '/m', 'model_dtype': 'torch.bfloat16',
        'prompt_input_ids': [[3]], 'adv_pixel_values_shape': [1,3,224,224],
        'adv_tensor_filename': 'x.pt', 'adv_tensor_sha256': 'ee',
        'generated_arm_prefix': [1]*6, 'full_ar_tokens': [1]*7,
        'surrogate_global_top_token': 1,
        'generation_score_argmax': 1,
        'surrogate_top_matches_generation': True,
        'v3_transfer_class': 'SURROGATE_TOP_MATCH_NATIVE_OPEN',
    }
    assert validate_replay_bundle(bundle) == []


# ── Finite check ──

def test_finite_pass():
    check_finite_or_fail(0.0, 'test_zero', 0)
    check_finite_or_fail(-1.5, 'test_neg', 0)
    check_finite_or_fail(3.14, 'test_pos', 0)


def test_finite_fail_none():
    try:
        check_finite_or_fail(None, 'test_none', 0)
        assert False, 'should have raised'
    except RuntimeError as e:
        assert 'missing/empty' in str(e)


def test_finite_fail_nan():
    try:
        check_finite_or_fail(float('nan'), 'test_nan', 0)
        assert False, 'should have raised'
    except RuntimeError as e:
        assert 'not finite' in str(e)


def test_finite_fail_inf():
    try:
        check_finite_or_fail(float('inf'), 'test_inf', 0)
        assert False, 'should have raised'
    except RuntimeError as e:
        assert 'not finite' in str(e)


# ── Full decode pipeline ──

def test_classify_disc_and_raw_native_open():
    """Integration: token→disc→clip→unnormalize→OPEN."""
    import numpy as np
    bin_centers = np.linspace(-1, 1, NBINS, dtype=np.float32)
    stats = {
        'q01': np.array([-0.15]*6 + [0.0], dtype=np.float32),
        'q99': np.array([0.15]*6 + [1.0], dtype=np.float32),
        'mask': np.ones(7, dtype=bool),
    }
    # tid corresponding to disc=0 → center=-1.0 → raw=0.0 (un-normalized)
    tid_open = VOCAB - 1  # disc 0
    result = classify_disc_and_raw(tid_open, VOCAB, NBINS, bin_centers, stats)
    assert result['disc_before'] == 0
    assert result['clipped'] == False
    # disc 0 center=-1.0, raw=0.5*(-1+1)*(1-0)+0 = 0.0 → CLOSE
    assert result['execution_class'] == 'NATIVE_CLOSE'

    # tid disc=NBINS-1=255 → center≈1.0 → raw≈1.0 → OPEN
    tid_open2 = VOCAB - NBINS  # disc 255
    result2 = classify_disc_and_raw(tid_open2, VOCAB, NBINS, bin_centers, stats)
    assert result2['disc_before'] == 255
    assert result2['execution_class'] == 'NATIVE_OPEN'
