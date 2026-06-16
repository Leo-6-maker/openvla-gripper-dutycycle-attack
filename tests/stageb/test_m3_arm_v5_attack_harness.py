from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gripper_attack.m3_v5_attack_harness import (
    V5_2_CANDIDATE_COUNT,
    V5_2_FROZEN_SEED,
    V5_2_CONDITIONS,
    audit_frame_group,
    arm_match_count,
    candidate_path,
    require_candidate_index,
    require_v5_2_seed,
    select_best_feasible,
    write_candidate_artifact,
)


def _payload(*, margin=1.0, token=31744, arm=6, linf=0.0):
    return {
        "official_gripper_token": token,
        "official_exact_7_tokens": [1, 2, 3, 4, 5, 6, token],
        "arm_match_count": arm,
        "official_target_margin": margin,
        "linf": linf,
        "score_invariant_status": "PASS",
        "route_status": "PASS",
    }


def _write_group(root: Path, frame_id: str = "frame_a"):
    for condition in V5_2_CONDITIONS:
        for idx in range(V5_2_CANDIDATE_COUNT):
            base = 10.0 if condition == "TRUE_PGD21_SELECTIVE" else (1.0 if condition == "RAND21_SELECTIVE" else 0.5)
            write_candidate_artifact(root, frame_id=frame_id, condition=condition, candidate_index=idx, payload=_payload(margin=base + idx / 100.0))


def test_seed_contract_rejects_legacy_and_nonfrozen_seed():
    require_v5_2_seed(V5_2_FROZEN_SEED)
    with pytest.raises(ValueError, match="legacy seed"):
        require_v5_2_seed(85)
    with pytest.raises(ValueError, match="frozen seed"):
        require_v5_2_seed(123)


def test_candidate_index_contract_is_exact_0_to_20():
    require_candidate_index(0)
    require_candidate_index(20)
    with pytest.raises(ValueError):
        require_candidate_index(21)


def test_arm_match_count_requires_actual_six_token_prefix():
    assert arm_match_count([1, 2, 3, 4, 5, 6], [1, 2, 0, 4, 0, 6]) == 4
    with pytest.raises(ValueError, match="six tokens"):
        arm_match_count([1, 2], [1, 2])


def test_select_best_feasible_rejects_arm_or_budget_failures(tmp_path):
    root = tmp_path / "root"
    for idx in range(V5_2_CANDIDATE_COUNT):
        if idx == 0:
            payload = _payload(margin=99, arm=2)
        elif idx == 1:
            payload = _payload(margin=1, arm=6)
        else:
            payload = _payload(margin=50, linf=1.0)
        write_candidate_artifact(root, frame_id="f", condition="TRUE_PGD21_SELECTIVE", candidate_index=idx, payload=payload)
    from gripper_attack.m3_v5_attack_harness import load_condition_candidates

    selected = select_best_feasible(load_condition_candidates(root, frame_id="f", condition="TRUE_PGD21_SELECTIVE"))
    assert selected is not None
    assert selected.candidate_index == 1


def test_frame_group_auditor_requires_all_three_isolated_conditions(tmp_path):
    root = tmp_path / "root"
    _write_group(root)
    result = audit_frame_group(root, frame_ids=["frame_a"], seed=V5_2_FROZEN_SEED)
    assert result["audit_status"] == "PASS"
    assert result["frame_full_selective_pass_count"] == 1

    # Cross-condition contamination: a RAND file claiming TRUE condition must fail.
    bad_path = candidate_path(root, frame_id="frame_a", condition="RAND21_SELECTIVE", candidate_index=0)
    bad = json.loads(bad_path.read_text(encoding="utf-8"))
    bad["condition"] = "TRUE_PGD21_SELECTIVE"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="cross-contamination"):
        audit_frame_group(root, frame_ids=["frame_a"], seed=V5_2_FROZEN_SEED)


def test_frame_group_auditor_rejects_missing_candidate(tmp_path):
    root = tmp_path / "root"
    _write_group(root)
    candidate_path(root, frame_id="frame_a", condition="SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE", candidate_index=20).unlink()
    with pytest.raises(ValueError, match="0..20"):
        audit_frame_group(root, frame_ids=["frame_a"], seed=V5_2_FROZEN_SEED)


def test_mock_frame_group_script_roundtrip(tmp_path):
    out = tmp_path / "mock"
    cmd = [
        sys.executable,
        "scripts/stageb/run_m3_arm_v5_frame_group.py",
        "--mode",
        "mock_zero_perturbation",
        "--output_dir",
        str(out),
        "--frame_ids",
        "f0,f1",
        "--seed",
        str(V5_2_FROZEN_SEED),
    ]
    subprocess.check_call(cmd)
    audit = json.loads((out / "m3_arm_v5_frame_group_mock_summary.json").read_text(encoding="utf-8"))
    assert audit["audit_status"] == "PASS"
    assert audit["frame_count"] == 2
