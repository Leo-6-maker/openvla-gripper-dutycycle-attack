"""Seal the clean Teacher/causal Student package before formal M4."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from audit_r3_contact_input import sha256_file, verify_seal


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
ACTIVE_HEADS = ["physical_criticality", "k10_feasibility", "instability", "gripper_closing_state"]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _root(path: Path) -> Path:
    value = path.resolve(strict=True)
    if not value.is_dir() or value.is_symlink():
        raise ValueError(f"invalid evidence root: {value}")
    return value


def _sealed(root: Path, name: str) -> tuple[dict[str, Any], str]:
    root = _root(root)
    seal = verify_seal(root)["sha256sums_sha256"]
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing sealed file: {path}")
    return _json(path), seal


def _sealed_exact_plan(root: Path, name: str) -> tuple[dict[str, Any], str]:
    """Verify the exact-plan producer's ROOT_SEAL format."""
    root = _root(root)
    sums = root / "SHA256SUMS"
    pointer = root / "ROOT_SEAL.sha256"
    if not sums.is_file() or not pointer.is_file():
        raise ValueError(f"exact-plan seal missing: {root}")
    digest = sha256_file(sums)
    if pointer.read_text(encoding="utf-8").strip() != f"{digest}  SHA256SUMS":
        raise ValueError(f"exact-plan seal pointer mismatch: {root}")
    listed: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        item, separator, relative_name = line.partition("  ")
        if not separator or len(item) != 64 or relative_name in listed or relative_name in {"SHA256SUMS", "ROOT_SEAL.sha256"}:
            raise ValueError(f"invalid exact-plan seal row: {line!r}")
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts or not (root / relative).is_file() or sha256_file(root / relative) != item:
            raise ValueError(f"invalid exact-plan sealed file: {relative_name}")
        listed.add(relative_name)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "ROOT_SEAL.sha256"}}
    if actual != listed:
        raise ValueError(f"exact-plan seal closure mismatch: missing={sorted(actual - listed)} extra={sorted(listed - actual)}")
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing sealed file: {path}")
    return _json(path), digest


def _file(path: Path) -> Path:
    value = path.resolve(strict=True)
    if not value.is_file() or value.is_symlink():
        raise ValueError(f"missing regular file: {value}")
    return value


def _require_boundary(value: Mapping[str, Any], label: str) -> None:
    if value.get("protected_counters") != COUNTERS:
        raise ValueError(f"{label} protected counters are not zero")
    for key in ("outcomes_read", "intervention_executed", "v_phys_generated"):
        if key in value and value[key] is not False:
            raise ValueError(f"{label} read boundary is open: {key}")


def _seal(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files), encoding="utf-8")
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def seal(*, firewall_root: Path, final_manifest: Path, final_split: Path, exact_plan_root: Path, teacher_root: Path, coverage_root: Path, t4_root: Path, g0_root: Path, g1_root: Path, g2_root: Path, student_root: Path, g7_root: Path, output_root: Path) -> dict[str, Any]:
    firewall, firewall_seal = _sealed(firewall_root, "STAGE_V_PRIMARY_DATA_FIREWALL_OVERLAP_AUDIT_V3.json")
    if firewall.get("status") != "PASS_PRIMARY_DATA_FIREWALL_EXACT55" or firewall.get("primary_identity_firewall", {}).get("zero_overlap") is not True or firewall.get("primary_identity_firewall", {}).get("attempted_overlap_count") != 0 or firewall.get("primary_identity_firewall", {}).get("final40_overlap_count") != 0:
        raise ValueError("primary firewall is not a sealed zero-overlap PASS")
    _require_boundary(firewall, "primary firewall")

    final_manifest = _file(final_manifest); final_split = _file(final_split)
    final = _json(final_manifest); split = _json(final_split)
    if final.get("schema") != "STAGE_V_M4_ELIGIBLE_FORMAL_PARENT_MANIFEST_V2" or final.get("status") != "FROZEN_COMPOSITE_40_CORRIDOR_ELIGIBLE" or final.get("parent_count") != 40 or final.get("formal_m4_authorized") is not False or final.get("outcomes_read") is not False:
        raise ValueError("final40 manifest is not frozen and outcome-blind")
    if split.get("schema") != "STAGE_V_M4_FINAL_PARENT_SPLIT_V2" or split.get("status") != "FROZEN" or split.get("final_manifest_sha256") != _sha(final_manifest) or split.get("counts") != {"TRAIN": 24, "VAL": 8, "TEST": 8}:
        raise ValueError("final split is not bound to final40")
    _require_boundary(final, "final40")
    _require_boundary(split, "final split")

    exact_root = _root(exact_plan_root)
    plan, plan_seal = _sealed_exact_plan(exact_root, "PLAN_RESULT.json")
    exact_manifest, _ = _sealed_exact_plan(exact_root, "EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json")
    if plan.get("status") != "PASS" or plan.get("manifest_status") != "PASS_EXACT_40X24_PLAN_ONLY" or plan.get("parent_count") != 40 or plan.get("probe_count_total") != 960 or plan.get("planned_branch_authority_count") != 3840 or plan.get("outcomes_read") is not False or plan.get("intervention_executed") is not False or plan.get("protected_counters") != COUNTERS:
        raise ValueError("exact 40x24 plan is not a PASS")
    if exact_manifest.get("status") != "PASS_EXACT_40X24_PLAN_ONLY" or exact_manifest.get("final40_manifest_sha256") != _sha(final_manifest) or exact_manifest.get("final_split_sha256") != _sha(final_split):
        raise ValueError("exact plan is not bound to frozen final40/split")

    teacher, teacher_seal = _sealed(teacher_root, "teacher_manifest.json")
    coverage, coverage_seal = _sealed(coverage_root, "coverage_report.json")
    t4, t4_seal = _sealed(t4_root, "TEACHER_STUDENT_TRANSITION.json")
    g0, g0_seal = _sealed(g0_root, "G0_LABEL_BASELINE_AUDIT.json")
    g1_seal = verify_seal(_root(g1_root))["sha256sums_sha256"]
    g2, g2_seal = _sealed(g2_root, "TEACHER_TO_STUDENT_GENERALIZATION_TRANSITION_V2.json")
    student, student_seal = _sealed(student_root, "heldout_report.json")
    g7, g7_seal = _sealed(g7_root, "G7_TEST_EVALUATION.json")
    if teacher.get("status") != "DEVELOPMENT_NONCONSUMABLE" or teacher.get("identity_count") != 670 or teacher.get("step_count") != 196483 or teacher.get("protected_reads") != 0 or teacher.get("future_fields_used") is not False or teacher.get("attack_authorized") is not False:
        raise ValueError("Teacher package is not clean-only and closed")
    if coverage.get("status") != "HOLD_COVERAGE" or coverage.get("identity_count") != 670 or coverage.get("step_count") != 196483 or coverage.get("protected_reads") != 0:
        raise ValueError("Teacher coverage package is not the expected closed HOLD")
    eligible = [head for head in coverage.get("coverage", {}) if coverage["coverage"][head].get("pass") is True]
    if eligible != ACTIVE_HEADS or coverage.get("coverage", {}).get("safe_release", {}).get("pass") is not False:
        raise ValueError("Teacher coverage head set is not the frozen four-head boundary")
    if t4.get("status") != "PASS_DEVELOPMENT_ELIGIBLE_HEADS" or t4.get("eligible_heads") != ACTIVE_HEADS or t4.get("formal_training_authorized") is not False or t4.get("attack_authorized") is not False:
        raise ValueError("T4 is not the four-head development transition")
    if g0.get("status") != "PASS_LABEL_AND_BASELINE_AUDIT" or g0.get("consumable") is not False or g0.get("identity_count") != 670 or g0.get("step_count") != 196483:
        raise ValueError("G0 is not the sealed diagnostic baseline")
    if g2.get("status") != "PASS_G2_DEVELOPMENT_TRANSITION" or g2.get("formal_training_authorized") is not False or g2.get("formal_inference_authorized") is not False or g2.get("teacher_privileged_fields_in_student") is not False or g2.get("consumable_for_scientific_promotion") is not False:
        raise ValueError("G2 boundary is not development-only")
    if student.get("status") != "ENGINEERING_DEVELOPMENT_NONCONSUMABLE" or student.get("random_initialization") is not True or student.get("all_670_checkpoint_loaded") is not False or student.get("test_payload_read") is not False or student.get("test_evaluation_performed") is not False or student.get("threshold_selection_split") != "validation_only" or student.get("teacher_privileged_fields_in_student") is not False or student.get("normalization_drift", {}).get("status") != "PASS_RECOMPUTED_CONSISTENT" or student.get("active_heads") != ACTIVE_HEADS:
        raise ValueError("Student validation package is not closed")
    student_permissions = student.get("permissions", {})
    if student_permissions.get("protected_reads") != 0 or student_permissions.get("formal_training") is not False or student_permissions.get("attack") is not False:
        raise ValueError("Student permission boundary is not closed")
    thresholds = _file(student_root / "thresholds.json")
    checkpoint = _file(student_root / "checkpoint.pt")
    threshold_data = _json(thresholds)
    if any(threshold_data.get(head, {}).get("status") != "SELECTED_VALIDATION_ONLY" or threshold_data[head].get("threshold") is None for head in ACTIVE_HEADS):
        raise ValueError("validation thresholds are not frozen for all active heads")
    if g7.get("status") != "PASS_R3_G7_TEST_EVALUATION" or g7.get("test_read_count") != 1 or g7.get("thresholds_frozen_before_test") is not True or g7.get("model_selection_after_test") is not False or g7.get("outcomes_read") is not False or g7.get("intervention_executed") is not False or g7.get("v_phys_generated") is not False or g7.get("formal_m4_authorized") is not False or g7.get("protected_counters") != COUNTERS or g7.get("checkpoint_sha256") != _sha(checkpoint):
        raise ValueError("G7 test read is not a one-time frozen-checkpoint evaluation")

    report = {
        "schema": "STAGE_V_PRIMARY_TEACHER_STUDENT_FREEZE_V1",
        "status": "PASS_PRIMARY_TEACHER_STUDENT_FREEZE",
        "purpose": "Clean Teacher and causal Student freeze completed before any formal M4 outcome read.",
        "teacher": {"root": str(_root(teacher_root)), "seal_sha256sums_sha256": teacher_seal, "manifest_sha256": _sha(_root(teacher_root) / "teacher_manifest.json"), "identity_count": 670, "step_count": 196483, "status": teacher["status"]},
        "coverage": {"root": str(_root(coverage_root)), "seal_sha256sums_sha256": coverage_seal, "report_sha256": _sha(_root(coverage_root) / "coverage_report.json"), "eligible_heads": ACTIVE_HEADS, "held_heads": ["safe_release"]},
        "t4": {"root": str(_root(t4_root)), "seal_sha256sums_sha256": t4_seal, "manifest_sha256": _sha(_root(t4_root) / "TEACHER_STUDENT_TRANSITION.json")},
        "g0": {"root": str(_root(g0_root)), "seal_sha256sums_sha256": g0_seal, "report_sha256": _sha(_root(g0_root) / "G0_LABEL_BASELINE_AUDIT.json")},
        "g1": {"root": str(_root(g1_root)), "seal_sha256sums_sha256": g1_seal},
        "g2": {"root": str(_root(g2_root)), "seal_sha256sums_sha256": g2_seal, "transition_sha256": _sha(_root(g2_root) / "TEACHER_TO_STUDENT_GENERALIZATION_TRANSITION_V2.json")},
        "student": {"root": str(_root(student_root)), "seal_sha256sums_sha256": student_seal, "report_sha256": _sha(_root(student_root) / "heldout_report.json"), "checkpoint_sha256": _sha(checkpoint), "checkpoint_path": str(checkpoint), "thresholds_sha256": _sha(thresholds), "active_heads": ACTIVE_HEADS, "formal_training_authorized": False, "formal_inference_authorized": False},
        "g7": {"root": str(_root(g7_root)), "seal_sha256sums_sha256": g7_seal, "report_sha256": _sha(_root(g7_root) / "G7_TEST_EVALUATION.json"), "test_read_count": 1},
        "feature_schema_sha256": g2.get("feature_binding", {}).get("sha256"),
        "feature_order_sha256": g2.get("feature_binding", {}).get("feature_order_sha256"),
        "primary_data_firewall": {"root": str(_root(firewall_root)), "seal_sha256sums_sha256": firewall_seal, "report_sha256": _sha(_root(firewall_root) / "STAGE_V_PRIMARY_DATA_FIREWALL_OVERLAP_AUDIT_V3.json"), "status": firewall["status"]},
        "final40": {"path": str(final_manifest), "sha256": _sha(final_manifest), "split_path": str(final_split), "split_sha256": _sha(final_split)},
        "exact_plan": {"root": str(exact_root), "seal_sha256sums_sha256": plan_seal, "manifest_sha256": _sha(exact_root / "EXACT_PROBE_AND_SNAPSHOT_MANIFEST.json"), "plan_result_sha256": _sha(exact_root / "PLAN_RESULT.json"), "parent_count": 40, "probe_count": 960, "branch_count": 3840},
        "architecture_order": ["CLEAN_ROLLOUT", "PRIVILEGED_CLEAN_TEACHER_C_t", "CLEAN_TEACHER_SUPERVISED_CAUSAL_STUDENT_C_HAT_t", "HELD_OUT_MATCHED_COUNTERFACTUAL_VALIDATION_V_t_d"],
        "m4_outcomes_read": False,
        "v_phys_generated": False,
        "formal_m4_authorized": False,
        "formal_training_authorized": False,
        "frozen_student_inference_only_after_m4": True,
        "protected_counters": dict(COUNTERS),
    }
    output_root = output_root.resolve()
    if output_root.parent != _root(g7_root).parent or output_root.exists():
        raise ValueError("freeze output must be a new sibling of G7")
    staging = output_root.with_name(f".{output_root.name}.staging")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    (staging / "PRIMARY_TEACHER_STUDENT_FREEZE.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = _seal(staging)
    staging.rename(output_root)
    report["sha256sums_sha256"] = digest
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("firewall-root", "final-manifest", "final-split", "exact-plan-root", "teacher-root", "coverage-root", "t4-root", "g0-root", "g1-root", "g2-root", "student-root", "g7-root", "output-root"):
        parser.add_argument(f"--{name}", dest=name.replace("-", "_"), type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seal(firewall_root=args.firewall_root, final_manifest=args.final_manifest, final_split=args.final_split, exact_plan_root=args.exact_plan_root, teacher_root=args.teacher_root, coverage_root=args.coverage_root, t4_root=args.t4_root, g0_root=args.g0_root, g1_root=args.g1_root, g2_root=args.g2_root, student_root=args.student_root, g7_root=args.g7_root, output_root=args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
