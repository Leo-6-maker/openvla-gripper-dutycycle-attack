from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stageb.audit_m3_v3_tomato_reaudit import audit_lambda_dir, run_reaudit  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _selected(condition: str, margin: float, *, arm: int = 6, token: int = 31744, selected_id: int = 3):
    return {
        "condition": condition,
        "stage_result": "FULL_SELECTIVE_V4_SEED_PASS",
        "condition_result": "SELECTED",
        "selected_candidate_id": selected_id,
        "official_gripper_token": token,
        "arm_prefix_match_count": arm,
        "arm_prefix_match_denominator": 6,
        "official_target31744_margin": margin,
        "processor_linf": 0.01,
        "score_invariant_status": "PASS",
    }


def _candidate_rows(condition: str, selected_id: int = 3, *, arm: int = 6, token: int = 31744, margin: float = 10.0):
    rows = []
    for idx in range(21):
        rows.append(
            {
                "condition": condition,
                "candidate_id": idx,
                "attack_seed": 81,
                "commit": "abc",
                "official_tokens": "[1,2,3,4,5,6,31744]",
                "processor_linf": 0.01,
                "selected": 1 if idx == selected_id else 0,
                "score_invariant_status": "PASS",
                "feasible": 1 if idx == selected_id else 0,
                "official_gripper_token": token if idx == selected_id else 31872,
                "arm_prefix_match_count": arm if idx == selected_id else 0,
                "official_target31744_margin": margin if idx == selected_id else -1.0,
            }
        )
    return rows


def _route_rows():
    base = {
        "attack_seed": 81,
        "commit": "abc",
        "strict_route": "True",
        "allow_fallback": "False",
        "fallback_used": "False",
        "resolved_adapter_class": "TokenPrefixPGDAttacker",
        "requested_objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        "resolved_objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        "target_token_id": 31744,
        "num_backwards": 20,
        "num_loss_forwards": 21,
        "num_generation_forwards": 21,
        "trajectory_candidate_count": 21,
    }
    return [
        {"condition": "TRUE_PGD_TRAJECTORY21_SELECTIVE", **base},
        {"condition": "SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE", **base},
    ]


def _lambda_dir(tmp_path: Path, *, true_arm: int = 6, fallback: bool = False) -> Path:
    d = tmp_path / "lambda_0.5" / "canary"
    _write_csv(
        d / "m3_v4_selected_results.csv",
        [
            _selected("TRUE_PGD_TRAJECTORY21_SELECTIVE", 20.0, arm=true_arm),
            _selected("RAND21_SELECTIVE", 0.5),
            _selected("SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE", 0.0),
        ],
    )
    rows = []
    rows.extend(_candidate_rows("TRUE_PGD_TRAJECTORY21_SELECTIVE", arm=true_arm, margin=20.0))
    rows.extend(_candidate_rows("RAND21_SELECTIVE", margin=0.5))
    rows.extend(_candidate_rows("SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE", margin=0.0))
    _write_csv(d / "m3_v4_candidate_audit.csv", rows)
    route = _route_rows()
    if fallback:
        route[0]["fallback_used"] = "True"
    _write_csv(d / "m3_v4_route_audit.csv", route)
    (d / "m3_artifact_hash_manifest.csv").write_text("path,sha256\n", encoding="utf-8")
    (d / "m3_v4_debug.json").write_text("{}", encoding="utf-8")
    return d


def test_reaudit_lambda_passes_with_canonical_arm_field(tmp_path):
    d = _lambda_dir(tmp_path)
    row = audit_lambda_dir("0.5", d, expected_seed=81, expected_commit="abc", epsilon=6 / 255)
    assert row["reaudit_status"] == "PASS"
    assert row["true_arm_match"] == 6
    assert row["true_minus_rand"] == pytest.approx(19.5)


def test_reaudit_lambda_rejects_arm_mismatch_and_route_fallback(tmp_path):
    d = _lambda_dir(tmp_path / "arm", true_arm=4)
    row = audit_lambda_dir("0.5", d, expected_seed=81, expected_commit="abc", epsilon=6 / 255)
    assert row["reaudit_status"] == "INFRA_INVALID"
    assert "arm_below_gate" in row["reason"]

    d = _lambda_dir(tmp_path / "route", fallback=True)
    row = audit_lambda_dir("0.5", d, expected_seed=81, expected_commit="abc", epsilon=6 / 255)
    assert row["reaudit_status"] == "INFRA_INVALID"
    assert "route_fallback_used" in row["reason"]


def test_full_reaudit_does_not_modify_original_gate(tmp_path):
    root = tmp_path / "r3"
    s3 = root / "S3_TOMATO_SCREEN"
    (s3 / "gate_result.json").parent.mkdir(parents=True)
    (s3 / "gate_result.json").write_text(json.dumps({"status": "FAIL", "failure_class": "TOMATO_NO_LAMBDA_PASS"}), encoding="utf-8")
    d = _lambda_dir(s3)
    out = tmp_path / "audit"

    class Args:
        output_root = str(root)
        audit_output_dir = str(out)
        expected_seed = 81
        expected_commit = "abc"
        epsilon = 6 / 255

    run_reaudit(Args())
    assert json.loads((s3 / "gate_result.json").read_text(encoding="utf-8"))["status"] == "FAIL"
    summary = json.loads((out / "S3_independent_reaudit_summary.json").read_text(encoding="utf-8"))
    assert summary["S3_ORIGINAL_GATE"] == "FAIL"
    assert summary["S3_INDEPENDENT_REAUDIT"] == "PASS"
