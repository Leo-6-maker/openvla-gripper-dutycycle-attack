"""Test inner-CV split direction: train≈2/3, val≈1/3, no overlap, no contamination."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from gripper_attack.v5_factorized_v2_splits import (
    resolve_inner_train_val_ids, get_outer_val_ids, validate_inner_split)


def test_train_two_val_one():
    """Train ≈2/3 of outer_train, val ≈1/3."""
    # Synthetic split bundle
    ids = [f'id_{i}' for i in range(600)]
    bundle = {
        'splits': {
            'fold_0': {
                'outer_val_identities': [f'val_{i}' for i in range(200)],
                'inner_folds': [
                    {'identities': ids[0:200]},
                    {'identities': ids[200:400]},
                    {'identities': ids[400:600]},
                ]
            }
        }
    }
    train, val = resolve_inner_train_val_ids(bundle, 0, 0)
    assert len(train) == 400, f'train={len(train)}'
    assert len(val) == 200, f'val={len(val)}'
    assert len(train & val) == 0, 'overlap'


def test_no_outer_val_contamination():
    """Inner train and val must not contain outer val identities."""
    ids = [f'id_{i}' for i in range(600)]
    outer_vals = [f'val_{i}' for i in range(200)]
    bundle = {
        'splits': {
            'fold_0': {
                'outer_val_identities': outer_vals,
                'inner_folds': [
                    {'identities': ids[0:200]},
                    {'identities': ids[200:400]},
                    {'identities': ids[400:600]},
                ]
            }
        }
    }
    train, val = resolve_inner_train_val_ids(bundle, 0, 0)
    outer_val_set = set(outer_vals)
    assert len(train & outer_val_set) == 0, 'train contains outer val'
    assert len(val & outer_val_set) == 0, 'val contains outer val'


def test_validate_inner_split_passes():
    ids = [f'id_{i}' for i in range(600)]
    bundle = {
        'splits': {
            'fold_0': {
                'outer_val_identities': [f'val_{i}' for i in range(200)],
                'inner_folds': [
                    {'identities': ids[0:200]},
                    {'identities': ids[200:400]},
                    {'identities': ids[400:600]},
                ]
            }
        }
    }
    result = validate_inner_split(bundle, 0, 0)
    assert result['valid'], f'Issues: {result["issues"]}'
    assert 0.60 <= result['train_fraction'] <= 0.75, f'train fraction={result["train_fraction"]}'


def test_validate_inner_split_detects_overlap():
    ids = [f'id_{i}' for i in range(600)]
    bundle = {
        'splits': {
            'fold_0': {
                'outer_val_identities': [],
                'inner_folds': [
                    {'identities': ids[0:210]},  # overlap with fold 1
                    {'identities': ids[200:400]},
                    {'identities': ids[400:600]},
                ]
            }
        }
    }
    result = validate_inner_split(bundle, 0, 0)
    assert not result['valid'], 'should detect overlap'
    assert len(result['issues']) > 0
