from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.detector_v5.audit_stage_v2_teacher_enrichment import audit
from scripts.detector_v5.stage_v2_teacher_enrichment import (
    StageV2PreconditionError,
    compute_report,
    load_json,
    run_analysis,
)


COMMIT = "b300e79bb0e6e754a9d384f8ea1b75034bd1d4b4"
TREE = "96881b4d53f901870dd53ede39d051c0a4c83e34"
CONFIG = Path(__file__).resolve().parents[2] / "configs" / "stage_v2_teacher_enrichment_v1.json"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def make_canary(tmp_path: Path) -> Path:
    root = tmp_path / "stage-v-canary"
    root.mkdir(parents=True)
    (root / "DIAGNOSTIC_CANARY_ONLY").write_text("NOT_FORMAL\n", encoding="utf-8")
    write_json(root / "RUN_MANIFEST.json", {"eval160_reads": 0, "protected_eval_reads": 0, "attack_rollouts": 0})
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        parent = root / suite / "task_00" / "state_48"
        parent.mkdir(parents=True)
        write_json(parent / "PARENT_RESULT.json", {"canonical_parent_key": f"{suite}/task_00/state_48", "suite": suite, "task_index": 0, "state_index": 48})
        rows = []
        for i in range(4):
            rows.append(
                {
                    "arm": ("OPEN_T3", "OPEN_T5", "OPEN_T10")[i % 3],
                    "probe_step": i,
                    "teacher_corridor_membership": True,
                    "comparison": {"local_vulnerability": i != 3, "task_vulnerability": i == 0},
                }
            )
        for i in range(4):
            rows.append(
                {
                    "arm": ("OPEN_T3", "OPEN_T5", "OPEN_T10")[i % 3],
                    "probe_step": i + 10,
                    "background_random_membership": True,
                    "comparison": {"local_vulnerability": i == 0, "task_vulnerability": False},
                }
            )
        for i in range(2):
            rows.append(
                {
                    "arm": ("OPEN_T3", "OPEN_T5")[i],
                    "probe_step": i + 20,
                    "safe_release_support_membership": True,
                    "comparison": {"local_vulnerability": False, "task_vulnerability": False},
                }
            )
        (parent / "COUNTERFACTUAL_BRANCHES.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
    return root


def test_formal_precondition_is_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(StageV2PreconditionError, match="STAGE_V_CLOSURE_RECEIPT_NOT_PASS"):
        run_analysis(
            tmp_path / "missing-root",
            tmp_path / "out",
            CONFIG,
            expected_source_commit=COMMIT,
            expected_source_tree=TREE,
        )


def test_diagnostic_canary_and_independent_audit_pass(tmp_path: Path) -> None:
    root = make_canary(tmp_path)
    output = tmp_path / "v2"
    report = run_analysis(root, output, CONFIG, expected_source_commit=COMMIT, expected_source_tree=TREE, diagnostic_canary=True)
    assert report["status"] == "STAGE_V2_TEACHER_PROPOSAL_PASS"
    audited = audit(root, output, CONFIG, expected_source_commit=COMMIT, expected_source_tree=TREE, diagnostic_canary=True)
    assert audited["verdict"] == "PASS"
    assert (output / "SHA256SUMS").is_file()
    assert load_json(output / "STAGE_V2_INDEPENDENT_AUDIT.json")["for_gate"] is False


def test_zero_denominator_is_explicit() -> None:
    report = compute_report(
        [
            {"group": "teacher_corridor", "local_vulnerability": True, "task_vulnerability": False, "arm": "OPEN_T3"}
        ],
        config=load_json(CONFIG),
        execution_class="DIAGNOSTIC_CANARY_ONLY",
        binding={},
        input_summary={"invalid_branch_rows": 0},
    )
    assert report["local_vulnerability"]["enrichment"]["status"] == "UNAVAILABLE_ZERO_DENOMINATOR"
    assert report["local_vulnerability"]["enrichment"]["ratio"] is None


def test_independent_audit_rejects_tampered_report(tmp_path: Path) -> None:
    root = make_canary(tmp_path)
    output = tmp_path / "v2"
    run_analysis(root, output, CONFIG, expected_source_commit=COMMIT, expected_source_tree=TREE, diagnostic_canary=True)
    assert audit(root, output, CONFIG, expected_source_commit=COMMIT, expected_source_tree=TREE, diagnostic_canary=True)["verdict"] == "PASS"
    report = load_json(output / "STAGE_V2_TEACHER_ENRICHMENT_REPORT.json")
    report["local_vulnerability_enrichment"] = 999.0
    write_json(output / "STAGE_V2_TEACHER_ENRICHMENT_REPORT.json", report)
    audited = audit(root, output, CONFIG, expected_source_commit=COMMIT, expected_source_tree=TREE, diagnostic_canary=True)
    assert audited["verdict"] == "FAIL"
    assert "independent_recompute_disagrees" in audited["errors"] or "parent_artifact_hash_mismatch:STAGE_V2_TEACHER_ENRICHMENT_REPORT.json" in audited["errors"]
