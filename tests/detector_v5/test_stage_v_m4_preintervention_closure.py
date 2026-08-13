import json

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


def test_closure_passes_only_when_all_96_failed_before_action(tmp_path):
    arms = ("CONTROL", "T3", "T5", "T10")
    branches = [_branch(f"Q{i:02d}", arm) for i in range(24) for arm in arms]
    labels = [{"binary_label_consumable": False, "protected_counters": {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}} for _ in range(72)]
    observations = [{"protected_counters": {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}} for _ in range(72)]
    report = audit(_make_root(tmp_path, branches, labels, observations), tmp_path / "audit", "libero_10/task_01/state_42")
    assert report["status"] == "PASS_PREINTERVENTION_STRUCTURAL_INVALIDATION"
    assert report["rows_total"] == report["actions_total"] == report["treatment_receipts_total"] == 0
    assert report["binary_label_consumable_count"] == 0
    assert report["physical_intervention_executed"] is False


def test_any_treatment_receipt_blocks_preintervention_reexecution_claim(tmp_path):
    arms = ("CONTROL", "T3", "T5", "T10")
    branches = [_branch(f"Q{i:02d}", arm) for i in range(24) for arm in arms]
    branches[1]["branch"]["treatment_receipts"] = [{"relative_step": 0}]
    report = audit(_make_root(tmp_path, branches, [{} for _ in range(72)], [{} for _ in range(72)]), tmp_path / "audit", "libero_10/task_01/state_42")
    assert report["status"] == "HOLD_PHYSICAL_ACTION_EVIDENCE_NONZERO"
    assert report["physical_intervention_executed"] is True
