import json
from pathlib import Path
import shutil
import subprocess

from scripts.detector_v5.audit_stage_v_m4_preintervention_closure import audit


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _branch(probe, arm, *, rows=None, actions=None, receipts=None):
    return {
        "schema": "STAGE_V_M4_PHYSICAL_EXECUTION_V1",
        "probe_id": probe,
        "arm": arm,
        "branch_id": f"{probe}-{arm}",
        "protected_counters": {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0},
        "branch": {
            "status": "FAIL",
            "arm": arm,
            "dose_steps": 3 if arm == "T3" else 0,
            "error": "CausalSnapshotError:EXACT_BINDING_MISMATCH:runtime_state",
            "protected_counters": {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0},
            "rows": rows or [],
            "actions": actions or [],
            "treatment_receipts": receipts or [],
            "treatment_compliant": False,
            "state_restore_exact": False,
            "runtime_state_exact": False,
            "causal_input_binding_pass": False,
        },
    }


def _make_root(tmp_path, branches, labels=None, observations=None):
    root = tmp_path / "parent"
    root.mkdir()
    _write_jsonl(root / "M4_COUNTERFACTUAL_BRANCHES_V1.jsonl", branches)
    _write_jsonl(root / "M4_V_PHYS_LABELS_V1.jsonl", labels or [])
    _write_jsonl(root / "M4_TREATMENT_OBSERVATIONS_V1.jsonl", observations or [])
    return root


def _gate_b(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "run_stage_v_m3_5_v1_4_gate_b.py"
    source = Path(__file__).resolve().parents[2]
    historical = subprocess.check_output(
        ["git", "show", "c5d81130854cc06f581409f35ffa76e500f0a214:scripts/detector_v5/run_stage_v_m3_5_v1_4_gate_b.py"],
        cwd=source,
        text=True,
    )
    path.write_text(historical, encoding="utf-8")
    return path


def test_closure_passes_only_when_all_96_failed_before_action(tmp_path):
    arms = ("CONTROL", "T3", "T5", "T10")
    branches = [_branch(f"Q{i:02d}", arm) for i in range(24) for arm in arms]
    labels = [{"binary_label_consumable": False, "protected_counters": {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}} for _ in range(72)]
    observations = [{"protected_counters": {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}} for _ in range(72)]
    report = audit(_make_root(tmp_path, branches, labels, observations), tmp_path / "audit", "libero_10/task_01/state_42", _gate_b(tmp_path))
    assert report["status"] == "PASS_PREINTERVENTION_STRUCTURAL_INVALIDATION"
    assert report["rows_total"] == report["actions_total"] == report["treatment_receipts_total"] == 0
    assert report["binary_label_consumable_count"] == 0
    assert report["valid_v_phys_count"] == 0
    assert report["physical_intervention_executed"] is False
    assert report["pre_primary_restore_failure_count"] == 96
    assert report["post_snapshot_primary_window_steps_total"] == 0
    assert report["clean_prefix_replay_steps_counted"] is False


def test_any_treatment_receipt_blocks_preintervention_reexecution_claim(tmp_path):
    arms = ("CONTROL", "T3", "T5", "T10")
    branches = [_branch(f"Q{i:02d}", arm) for i in range(24) for arm in arms]
    branches[1]["branch"]["treatment_receipts"] = [{"relative_step": 0}]
    report = audit(_make_root(tmp_path, branches, [{} for _ in range(72)], [{} for _ in range(72)]), tmp_path / "audit", "libero_10/task_01/state_42", _gate_b(tmp_path))
    assert report["status"] == "HOLD_PHYSICAL_ACTION_EVIDENCE_NONZERO"
    assert report["physical_intervention_executed"] is True


def test_consumable_label_or_treatment_compliance_is_hold(tmp_path):
    arms = ("CONTROL", "T3", "T5", "T10")
    branches = [_branch(f"Q{i:02d}", arm) for i in range(24) for arm in arms]
    branches[0]["branch"]["treatment_compliant"] = True
    labels = [{"binary_label_consumable": False, "protected_counters": {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}} for _ in range(72)]
    labels[0]["binary_label_consumable"] = True
    labels[1]["label_class"] = "V_PHYS"
    report = audit(_make_root(tmp_path, branches, labels, [{} for _ in range(72)]), tmp_path / "audit", "libero_10/task_01/state_42", _gate_b(tmp_path))
    assert report["status"] == "HOLD_CONSUMABLE_OUTCOME_EVIDENCE"
    assert report["binary_label_consumable_count"] == 1
    assert report["valid_v_phys_count"] == 1


def test_any_row_or_action_is_hold(tmp_path):
    arms = ("CONTROL", "T3", "T5", "T10")
    for index, evidence in enumerate(("rows", "actions")):
        case_root = tmp_path / str(index)
        case_root.mkdir()
        branches = [_branch(f"Q{i:02d}", arm) for i in range(24) for arm in arms]
        branches[0]["branch"][evidence] = [{"relative_step": 0}]
        report = audit(_make_root(case_root, branches, [{} for _ in range(72)], [{} for _ in range(72)]), tmp_path / "audit" / str(index), "libero_10/task_01/state_42", _gate_b(case_root))
        assert report["status"] == "HOLD_PHYSICAL_ACTION_EVIDENCE_NONZERO"


def test_gate_b_hash_and_branch_identity_are_bound(tmp_path):
    arms = ("CONTROL", "T3", "T5", "T10")
    branches = [_branch(f"Q{i:02d}", arm) for i in range(24) for arm in arms]
    branches[1]["probe_id"] = branches[0]["probe_id"]
    branches[1]["arm"] = branches[0]["arm"]
    branches[1]["branch_id"] = branches[0]["branch_id"]
    gate_b = _gate_b(tmp_path)
    gate_b.write_text(gate_b.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8")
    report = audit(_make_root(tmp_path, branches, [{} for _ in range(72)], [{} for _ in range(72)]), tmp_path / "audit", "libero_10/task_01/state_42", gate_b)
    assert report["status"].startswith("HOLD_")
    assert "GATE_B_SHA_MISMATCH" in report["errors"]
    assert any(error.startswith("DUPLICATE_BRANCH_IDENTITY:") for error in report["errors"])
    assert any(error.startswith("MISSING_BRANCH_IDENTITY:") for error in report["errors"])
