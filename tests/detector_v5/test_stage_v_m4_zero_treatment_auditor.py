import ast
from pathlib import Path

from scripts.detector_v5 import run_stage_v_m4_zero_treatment_auditor as auditor

TARGET = Path(__file__).resolve().parents[2] / "scripts/detector_v5/run_stage_v_m4_zero_treatment_auditor.py"


def test_zero_treatment_auditor_has_no_intervention_or_label_import_boundary():
    tree = ast.parse(TARGET.read_text(encoding="utf-8"))
    forbidden_modules = {
        "run_stage_v_m3_5_intervention_parent",
        "run_stage_v_m4_matched_parent",
        "stage_v_m3_5_physical_taxonomy",
    }
    forbidden_names = {"matched_action", "_pair_label", "v_phys_label", "treatment_compliance"}
    imported_modules = {
        node.module.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }
    attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    assert not imported_modules & forbidden_modules
    assert not (names | attributes) & forbidden_names
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_q00_authority"
        for node in ast.walk(tree)
    )


def test_zero_treatment_auditor_holds_before_snapshot_replay_without_q00_authority(tmp_path):
    class Env:
        closed = False

        def close(self):
            self.closed = True

    env = Env()
    receipt = auditor.audit_probe(tmp_path, env, [])
    assert receipt["status"] == "HOLD_ZERO_TREATMENT_COMPATIBILITY"
    assert any(error.startswith("AUDIT_ERROR:Q00AuthorityError:Q00_AUTHORITY_SCHEMA") for error in receipt["errors"])
    assert env.closed is True
