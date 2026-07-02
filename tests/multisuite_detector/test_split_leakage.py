#!/usr/bin/env python3
"""Test: same episode/parent in multiple splits must be rejected."""
import json, sys, tempfile, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from tools.multisuite_detector.validate_detector_splits import validate_split_file

def test_no_overlap():
    """Clean split with no overlap should pass."""
    split = {
        'split_type': 'episode_grouped', 'seed': 42,
        'splits': {'train': ['ep1','ep2'], 'val': ['ep3'], 'test': ['ep4']},
        'counts': {'train': 2, 'val': 1, 'test': 1},
        'validation_passed': True,
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(split, f); path = f.name
    result = validate_split_file(path)
    os.unlink(path)
    assert result['valid'], f'Should pass: {result["errors"]}'
    print('PASS: test_no_overlap')

def test_train_val_overlap_rejected():
    """Overlap between train and val should fail."""
    split = {
        'split_type': 'episode_grouped', 'seed': 42,
        'splits': {'train': ['ep1','ep2'], 'val': ['ep2','ep3'], 'test': ['ep4']},
        'counts': {'train': 2, 'val': 2, 'test': 1},
        'validation_passed': True,
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(split, f); path = f.name
    result = validate_split_file(path)
    os.unlink(path)
    assert not result['valid'], f'Should fail due to overlap'
    print('PASS: test_train_val_overlap_rejected')

def test_loso_missing_test_suite():
    """LOSO split without test_suite should fail."""
    split = {
        'split_type': 'loso', 'seed': 42,
        'splits': {'train': ['ep1'], 'val': ['ep2'], 'test': ['ep3']},
        'counts': {'train': 1, 'val': 1, 'test': 1},
        'validation_passed': True,
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(split, f); path = f.name
    result = validate_split_file(path)
    os.unlink(path)
    assert not result['valid'], f'Should fail: missing test_suite for LOSO'
    print('PASS: test_loso_missing_test_suite')

if __name__ == '__main__':
    test_no_overlap()
    test_train_val_overlap_rejected()
    test_loso_missing_test_suite()
    print('All split leakage tests passed.')
