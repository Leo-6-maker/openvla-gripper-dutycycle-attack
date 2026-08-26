#!/usr/bin/env python3
"""Build the deterministic AA2R2 static semantics reconciliation report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.openvla_libero_exec_spec import raw_gripper_to_env_gripper  # noqa: E402
from stage_aa.action_semantics_v2 import MODEL_M0, MODEL_M1, MODEL_M2, validate_action_pair  # noqa: E402
from stage_z_preparation.action_semantics import validate_action_pair as validate_action_pair_v1  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action(gripper: float) -> list[float]:
    return [0.0, 0.1, -0.1, 0.2, -0.2, 0.3, gripper]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", default="UNBOUND_PRE_COMMIT")
    args = parser.parse_args()

    below = float(np.nextafter(np.float64(0.5), -np.inf))
    above = float(np.nextafter(np.float64(0.5), np.inf))
    probes = [0.4999999, 0.5, 0.5000001, below, above]
    boundary_rows = []
    for raw in probes:
        expected = float(raw_gripper_to_env_gripper(raw))
        result = validate_action_pair(MODEL_M0, action(raw), action(expected))
        assert result["accepted"] is True, result
        boundary_rows.append({"raw": raw, "expected_env": expected, "semantic_state": result["semantic_state"], "accepted": True})

    wrong_final_rows = []
    for final in (1.0, -1.0):
        result = validate_action_pair(MODEL_M0, action(0.5), action(final))
        assert result["accepted"] is False, result
        wrong_final_rows.append({"raw": 0.5, "final": final, "accepted": False, "reason": result["reason"]})

    malformed_rows = []
    for family in (MODEL_M0, MODEL_M1, MODEL_M2):
        cases = [
            ("raw_nan", validate_action_pair(family, action(float("nan")), action(0.0))),
            ("final_inf", validate_action_pair(family, action(0.0), action(float("inf")))),
            ("raw_wrong_dim", validate_action_pair(family, action(0.0)[:-1], action(0.0))),
        ]
        for case, result in cases:
            assert result["accepted"] is False, (family, case, result)
            malformed_rows.append({"family": family, "case": case, "accepted": False, "reason": result["reason"]})

    m2_raw = [2.0, -2.0, 0.2, 0.0, 0.0, 0.0, -0.9986837]
    m2_final = [1.0, -1.0, 0.2, 0.0, 0.0, 0.0, -0.9986837]
    m2_result = validate_action_pair(MODEL_M2, m2_raw, m2_final)
    assert m2_result["accepted"] is True, m2_result

    compatibility = []
    tested = (-1.0, 0.0, 0.499, 0.501, 1.0, 2.0)
    for family in (MODEL_M0, MODEL_M1):
        for raw in tested:
            expected = float(raw_gripper_to_env_gripper(raw))
            old = validate_action_pair_v1(family, action(raw), action(expected))
            new = validate_action_pair(family, action(raw), action(expected))
            compatibility.append({
                "family": family,
                "raw": raw,
                "old_accepted": bool(old["accepted"]),
                "v2_accepted": bool(new["accepted"]),
                "same_semantic_state": (not old["accepted"]) or new["semantic_state"] == ("OPEN" if raw > 0.5 else "CLOSE"),
            })
    accepted = [row for row in compatibility if row["old_accepted"]]
    assert accepted and all(row["v2_accepted"] and row["same_semantic_state"] for row in accepted)

    files = {
        "official_executable_spec": ROOT / "src/gripper_attack/openvla_libero_exec_spec.py",
        "historical_validator": ROOT / "src/stage_z_preparation/action_semantics.py",
        "v2_validator": ROOT / "src/stage_aa/action_semantics_v2.py",
        "v2_runner": ROOT / "scripts/stage_aa/run_stage_aa2r2_engineering_canary.py",
    }
    report = {
        "schema": "STAGE_AA_AA2R2_STATIC_SEMANTICS_RECONCILIATION_V1",
        "status": "STAGE_AA_AA2R2_STATIC_SEMANTICS_RECONCILIATION_PASS",
        "source_commit": args.source_commit,
        "source_files": {
            key: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for key, path in files.items()
        },
        "official_rule": "raw < 0.5 -> env +1; raw == 0.5 -> env 0; raw > 0.5 -> env -1",
        "boundary_probes": boundary_rows,
        "wrong_final_at_exact_threshold": wrong_final_rows,
        "malformed_nonfinite_probes": malformed_rows,
        "m2_clip_probe": {"accepted": True, "result": m2_result},
        "compatibility": {
            "tested_families": [MODEL_M0, MODEL_M1],
            "tested_raw_values": list(tested),
            "old_accepted_count": len(accepted),
            "old_pass_implies_v2_pass": True,
            "same_open_close_meaning": True,
            "rows": compatibility,
        },
        "scientific_firewall": {
            "scientific_parent_exposure": 0,
            "open_intervention_steps": 0,
            "attacked_env_steps": 0,
            "pgd_calls": 0,
            "aa_v_phys_reads": 0,
            "task_success_reads": 0,
            "protected_eval160_reads": 0,
            "bridge_f1_reads": 0,
            "paper_promotion": False,
        },
        "next_legal_action": "PHASE_A_9_CELL_RUNTIME_QUALIFICATION",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
