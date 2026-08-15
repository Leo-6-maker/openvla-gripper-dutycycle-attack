#!/usr/bin/env python3
"""Run the Stage VI root-cause diagnostic with clean-only M4 R3 coverage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.detector_v5 import run_stage_vi_root_cause_diagnostic as base  # noqa: E402


def load_reconstructed_teacher(root: Path) -> tuple[dict[tuple[str, int], dict[str, int]], dict[str, int]]:
    rows = base.read_jsonl(root / "M4_R3_TEACHER_LABELS_DIAGNOSTIC.jsonl")
    teacher: dict[tuple[str, int], dict[str, int]] = {}
    lengths: dict[str, int] = {}
    for row in rows:
        identity, step = str(row["canonical_parent_key"]), int(row["step"])
        key = (identity, step)
        if key in teacher:
            raise ValueError(f"RECONSTRUCTED_TEACHER_DUPLICATE:{identity}:{step}")
        labels = row.get("labels")
        if not isinstance(labels, dict):
            raise ValueError(f"RECONSTRUCTED_TEACHER_LABELS_INVALID:{identity}:{step}")
        teacher[key] = {
            head: int(str(labels[head]["value"]).upper() == "TRUE")
            for head in base.TEACHER_HEADS
        }
        lengths[identity] = max(lengths.get(identity, 0), step + 1)
    if not teacher or len({identity for identity, _ in teacher}) != 40:
        raise ValueError("RECONSTRUCTED_TEACHER_COVERAGE_NOT_40")
    return teacher, lengths


def verify_aggregate(root: Path, source_commit: str, source_tree: str) -> dict[str, Any]:
    seal = base.read_json(root / "ROOT_SEAL.json")
    if seal.get("status") != "PASS_CLEAN_ONLY_R3_RECONSTRUCTION_AGGREGATE" or seal.get("protected_counters") != base.COUNTERS or seal.get("eval160_status") != "UNREAD":
        raise ValueError("RECONSTRUCTED_AGGREGATE_BOUNDARY_INVALID")
    sums_digest = base.sha256_file(root / "SHA256SUMS")
    if seal.get("sha256sums_sha256") != sums_digest:
        raise ValueError("RECONSTRUCTED_AGGREGATE_SEAL_INVALID")
    provenance = base.read_json(root / "PROVENANCE.json")
    if provenance.get("source_commit") != source_commit or provenance.get("source_tree") != source_tree:
        raise ValueError("RECONSTRUCTED_AGGREGATE_SOURCE_MISMATCH")
    audit = base.read_json(root / "AGGREGATE_AUDIT.json")
    if audit.get("status") != "PASS_CLEAN_ONLY_R3_RECONSTRUCTION_AGGREGATE" or audit.get("parent_count") != 40 or audit.get("protected_counters") != base.COUNTERS or audit.get("eval160_status") != "UNREAD":
        raise ValueError("RECONSTRUCTED_AGGREGATE_AUDIT_INVALID")
    index = base.read_json(root / "PARENT_INDEX.json")
    if index.get("status") != "PASS_EXACT_PLAN_40_UNIQUE_FULL_CLEAN_ROOTS" or len(index.get("parents", [])) != 40:
        raise ValueError("RECONSTRUCTED_AGGREGATE_INDEX_INVALID")
    return {"root": str(root), "seal_sha256": sums_digest, "audit": audit, "provenance": provenance, "index": index}


def teacher_head_probe(teacher: dict[tuple[str, int], dict[str, int]], m4_labels: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in m4_labels if row["dose"] == "T5"]
    x, y, groups, suites, phases = [], [], [], [], []
    for row in rows:
        identity, step = str(row["canonical_parent_key"]), int(row["probe_step"])
        target = teacher.get((identity, step))
        if target is None:
            raise ValueError(f"TEACHER_HEAD_PROBE_JOIN:{identity}:{step}")
        x.append([target[head] for head in base.TEACHER_HEADS])
        y.append(int(row["y"]))
        groups.append(identity)
        suites.append(row["suite"])
        phases.append(row["split"])
    return {
        "status": "PASS_RECONSTRUCTED_CLEAN_ONLY_TEACHER_HEADS",
        "feature_schema": list(base.TEACHER_HEADS),
        "source": "R3_LABELS_DERIVED_FROM_CLEAN_ONLY_RECONSTRUCTION",
        "probe": base.grouped_probe("B_TEACHER_HEADS", np.asarray(x), np.asarray(y), np.asarray(groups), np.asarray(suites), np.asarray(phases)),
    }


def seal_output(root: Path) -> str:
    files = sorted(path for path in root.iterdir() if path.is_file() and path.name not in {"SHA256SUMS", "ROOT_SEAL.json"})
    (root / "SHA256SUMS").write_text("".join(f"{base.sha256_file(path)}  {path.name}\n" for path in files), encoding="utf-8")
    digest = base.sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    base_json = {"schema": "STAGE_VI_ROOT_CAUSE_DIAGNOSTIC_M4_R3_RECONSTRUCTED_ROOT_SEAL_V1", "status": "PASS_READ_ONLY_RECONSTRUCTED_M4_DIAGNOSTIC", "sha256sums_sha256": digest, "protected_counters": dict(base.COUNTERS), "eval160_status": "UNREAD"}
    (root / "ROOT_SEAL.json").write_text(json.dumps(base_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return digest


def run(args: argparse.Namespace) -> int:
    source_commit = base.subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=REPO, text=True).strip()
    source_tree = base.subprocess.check_output(("git", "rev-parse", "HEAD^{tree}"), cwd=REPO, text=True).strip()
    if source_commit != args.source_commit or source_tree != args.source_tree:
        raise ValueError("SOURCE_COMMIT_OR_TREE_MISMATCH")
    aggregate = verify_aggregate(args.aggregate_root.resolve(), source_commit, source_tree)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    stage_v_seal = base.read_json(base.STAGE_V_ROOT / "ROOT_SEAL.json")
    stage_v_bundle = base.read_json(base.STAGE_V_ROOT / "FINAL_EVIDENCE_BUNDLE.json")
    if stage_v_seal.get("status") != "PASS" or stage_v_bundle.get("status") != "READY_FOR_PROTECTED_EVAL" or stage_v_bundle.get("eval160_status") != "UNREAD" or stage_v_bundle.get("protected_counters") != base.COUNTERS:
        raise ValueError("STAGE_V_IMMUTABLE_BOUNDARY_INVALID")

    frozen_teacher, frozen_lengths = base.load_teacher()
    reconstructed_teacher, reconstructed_lengths = load_reconstructed_teacher(args.aggregate_root.resolve())
    overlap = sorted(set(frozen_teacher) & set(reconstructed_teacher))
    if overlap:
        raise ValueError(f"FROZEN_RECONSTRUCTED_TEACHER_OVERLAP:{overlap[0]}")
    teacher = {**frozen_teacher, **reconstructed_teacher}
    lengths = {**frozen_lengths, **reconstructed_lengths}
    parent_splits = base.load_parent_splits()
    m4_labels = base.load_m4_labels(parent_splits)
    m4_student = base.load_m4_student(parent_splits, m4_labels)
    stream = base.load_student_stream()
    features = base.load_features()
    privileged_features, privileged_provenance = base.load_privileged_features(m4_labels)

    teacher_result = base.teacher_vphys(teacher, m4_labels)
    coverage = base.teacher_m4_coverage(teacher, m4_labels)
    coverage.update({
        "status": "PASS_M4_TEACHER_COVERAGE_RECONSTRUCTED_CLEAN_ONLY",
        "reconstruction_aggregate_root": str(args.aggregate_root.resolve()),
        "reconstructed_identity_count": len({identity for identity, _ in reconstructed_teacher}),
        "reconstructed_row_count": len(reconstructed_teacher),
        "fresh_formal_m4_executed": False,
    })
    teacher_result["coverage"] = coverage
    for head in base.TEACHER_HEADS:
        for dose in base.DOSES:
            if teacher_result[head][dose]["status"] != "PASS":
                raise ValueError(f"TEACHER_VPHYS_JOIN_INVALID:{head}:{dose}")
    student_result = base.student_vphys(m4_student)
    fidelity = base.student_teacher_fidelity(teacher, lengths, m4_student)
    temporal = base.temporal_alignment(teacher, stream, m4_labels)
    probes = base.information_probes(features, privileged_features, teacher, m4_labels)
    probes["probe_B_privileged_teacher_heads"] = teacher_head_probe(teacher, m4_labels)
    matrix = base.matrix_diagnostic()
    classification = base.classify(teacher_result, student_result, probes)

    diagnostic = {
        "schema": "STAGE_VI_ROOT_CAUSE_DIAGNOSTIC_M4_R3_RECONSTRUCTED_V1",
        "status": "PASS_READ_ONLY_RECONSTRUCTED_M4_DIAGNOSTIC",
        "stage_v": {
            "bundle_root": str(base.STAGE_V_ROOT),
            "bundle_root_seal_sha256": stage_v_seal["sha256sums_sha256"],
            "conclusion_preserved": stage_v_bundle["scientific_conclusion"],
        },
        "inputs": {
            "frozen_teacher_root": str(base.TEACHER_ROOT),
            "reconstruction_aggregate_root": str(args.aggregate_root.resolve()),
            "student_root": str(base.STUDENT_ROOT),
            "m4_aggregate": str(base.M4_AGGREGATE),
            "m4_student_root": str(base.M4_STUDENT_ROOT),
            "clean_replay_root": str(base.CLEAN_ROOT),
            "m4_formal_root": str(base.M4_FORMAL_ROOT),
            "matrix_aggregate": str(base.MATRIX_AGGREGATE),
            "matrix_execution": str(base.MATRIX_EXECUTION),
            "feature_schema": list(base.FEATURE_NAMES),
            "privileged_feature_schema": list(base.PRIVILEGED_FEATURE_NAMES),
        },
        "teacher_to_v_phys": teacher_result,
        "student_to_teacher": fidelity,
        "student_to_v_phys": student_result,
        "temporal_alignment_offsets_minus16_plus16": temporal,
        "information_content_probes": probes,
        "privileged_feature_provenance": privileged_provenance,
        "matrix_forensics": matrix,
        "classification": classification,
        "diagnostic_only": True,
        "stage_v_development_outcomes_reused": True,
        "fresh_m4_execution": False,
        "fresh_intervention_outcomes_read": False,
        "outcome_informed_model_selection": False,
        "student_training_performed": False,
        "teacher_student_frozen": True,
        "protected_counters": dict(base.COUNTERS),
        "eval160_status": "UNREAD",
        "source_commit": source_commit,
        "source_tree": source_tree,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    (output / "STAGE_VI_ROOT_CAUSE_DIAGNOSTIC.json").write_text(json.dumps(diagnostic, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    provenance = {
        "schema": "STAGE_VI_ROOT_CAUSE_DIAGNOSTIC_M4_R3_RECONSTRUCTED_PROVENANCE_V1",
        "source_commit": source_commit,
        "source_tree": source_tree,
        "source_status": base.subprocess.check_output(("git", "status", "--porcelain"), cwd=REPO, text=True),
        "diagnostic_code_sha256": base.sha256_file(Path(__file__)),
        "base_diagnostic_code_sha256": base.sha256_file(REPO / "scripts/detector_v5/run_stage_vi_root_cause_diagnostic.py"),
        "stage_v_bundle_root": str(base.STAGE_V_ROOT),
        "stage_v_bundle_seal_sha256": stage_v_seal["sha256sums_sha256"],
        "reconstruction_aggregate_root": str(args.aggregate_root.resolve()),
        "reconstruction_aggregate_seal_sha256": aggregate["seal_sha256"],
        "teacher_records_sha256": base.sha256_file(base.TEACHER_ROOT / "teacher_records.jsonl"),
        "student_predictions_sha256": base.sha256_file(base.STUDENT_ROOT / "predictions.jsonl"),
        "m4_labels_sha256": base.sha256_file(base.M4_AGGREGATE / "M4_ALL_LABELS_V1.jsonl"),
        "m4_student_scores_sha256": base.sha256_file(base.M4_STUDENT_ROOT / "STUDENT_M4_PROBE_SCORES_V1.jsonl"),
        "protected_counters": dict(base.COUNTERS),
        "eval160_status": "UNREAD",
        "fresh_m4_execution": False,
        "student_training_performed": False,
    }
    (output / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = seal_output(output)
    print(json.dumps({"status": diagnostic["status"], "classification": classification, "root": str(output), "root_seal": digest}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "reason": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
