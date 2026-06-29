#!/usr/bin/env python3
"""Test: feature contract validation — forbidden features, 25D check."""
import json, sys, tempfile, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from tools.multisuite_detector.validate_detector_splits import validate_feature_contract
from tools.multisuite_detector.extract_frozen_feature_contract import extract, validate_feature_list
from pathlib import Path

def test_25d_contract():
    """Extracted contract must have exactly 25 features."""
    repo = Path(__file__).resolve().parents[2]
    contract = extract(repo)
    assert contract['contract_valid'], f'Contract invalid: {contract.get("warnings", [])}'
    assert len(contract['features']['names']) == 25
    assert len(contract['phases']['names']) == 9
    print('PASS: test_25d_contract')

def test_forbidden_in_list():
    """Forbidden features in feature list must be caught."""
    bad_list = list(contract['features']['names'])
    bad_list.append('task_success')
    result = validate_feature_list(bad_list)
    assert not result['valid']
    print('PASS: test_forbidden_in_list')

def test_contract_validation():
    """Validate feature contract JSON."""
    contract_path = Path(tempfile.gettempdir()) / 'test_contract.json'
    repo = Path(__file__).resolve().parents[2]
    contract = extract(repo)
    with open(contract_path, 'w') as f:
        json.dump(contract, f)
    result = validate_feature_contract(str(contract_path))
    os.unlink(contract_path)
    assert result['valid'], f'Contract validation failed: {result["errors"]}'
    print('PASS: test_contract_validation')

if __name__ == '__main__':
    contract = extract(Path(__file__).resolve().parents[2])
    test_25d_contract()
    test_forbidden_in_list()
    test_contract_validation()
    print('All feature contract tests passed.')
