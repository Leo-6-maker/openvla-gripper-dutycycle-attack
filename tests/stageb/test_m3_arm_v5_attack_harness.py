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
from scripts.stageb.audit_m3_arm_v5_frame_group_independent import audit_frame_group as independent_audit_frame_group


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
    frames = ",".join(f"f{i}" for i in range(8))
    cmd = [
        sys.executable,
        "scripts/stageb/run_m3_arm_v5_frame_group.py",
        "--mode",
        "mock_zero_perturbation",
        "--output_dir",
        str(out),
        "--frame_ids",
        frames,
        "--seed",
        str(V5_2_FROZEN_SEED),
    ]
    subprocess.check_call(cmd)
    audit = json.loads((out / "m3_arm_v5_frame_group_mock_summary.json").read_text(encoding="utf-8"))
    assert audit["artifact_audit_status"] == "PASS"
    assert audit["scientific_gate_status"] == "PASS"
    assert audit["frame_count"] == 8


def test_frame_group_external_script_uses_independent_auditor():
    text = Path("scripts/stageb/audit_m3_arm_v5_frame_group.py").read_text(encoding="utf-8")
    assert "audit_m3_arm_v5_frame_group_independent" in text
    assert "from gripper_attack.m3_v5_attack_harness import audit_frame_group" not in text


def test_independent_auditor_separates_artifact_pass_from_scientific_fail(tmp_path):
    out = tmp_path / "mock"
    frames = ",".join(f"f{i}" for i in range(8))
    subprocess.check_call(
        [
            sys.executable,
            "scripts/stageb/run_m3_arm_v5_frame_group.py",
            "--mode",
            "mock_zero_perturbation",
            "--output_dir",
            str(out),
            "--frame_ids",
            frames,
            "--seed",
            str(V5_2_FROZEN_SEED),
        ]
    )
    for path in (out / "frames").glob("*/TRUE_PGD21_SELECTIVE/candidate_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["target_token_score"] = 10.0
        payload["best_competitor_score"] = 10.0
        payload["official_target_margin"] = 0.0
        path.write_text(json.dumps(payload), encoding="utf-8")
    audit = independent_audit_frame_group(out, frame_ids=[f"f{i}" for i in range(8)], seed=V5_2_FROZEN_SEED)
    assert audit["artifact_audit_status"] == "PASS"
    assert audit["scientific_gate_status"] == "FAIL"
    assert audit["frame_full_selective_pass_count"] == 0


def test_independent_auditor_recomputes_arm_linf_and_margin(tmp_path):
    out = tmp_path / "mock"
    frames = ",".join(f"f{i}" for i in range(8))
    subprocess.check_call(
        [
            sys.executable,
            "scripts/stageb/run_m3_arm_v5_frame_group.py",
            "--mode",
            "mock_zero_perturbation",
            "--output_dir",
            str(out),
            "--frame_ids",
            frames,
            "--seed",
            str(V5_2_FROZEN_SEED),
        ]
    )
    path = out / "frames" / "f0" / "TRUE_PGD21_SELECTIVE" / "candidate_20.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["arm_match_count"] = 6
    payload["attacked_arm_prefix"] = [9, 9, 9, 9, 9, 9]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="attacked arm prefix does not match attacked exact generation"):
        independent_audit_frame_group(out, frame_ids=[f"f{i}" for i in range(8)], seed=V5_2_FROZEN_SEED)

    payload["attacked_arm_prefix"] = [1, 2, 3, 4, 5, 6]
    payload["official_target_margin"] = 123.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="official target margin"):
        independent_audit_frame_group(out, frame_ids=[f"f{i}" for i in range(8)], seed=V5_2_FROZEN_SEED)
