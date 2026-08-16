#!/usr/bin/env python3
"""Audit Stage VI-B2 suite, prevalence, and domain signals before training."""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.detector_v5 import run_stage_vi_root_cause_diagnostic as base  # noqa: E402

COUNTERS = dict(base.COUNTERS)
SPLITS = ("TRAIN", "VAL", "TEST")
HEAD = "physical_criticality"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def window(features: dict[tuple[str, int], np.ndarray], identity: str, step: int) -> np.ndarray:
    return np.concatenate([features.get((identity, past), np.zeros(25, dtype=np.float64)) for past in range(step - 15, step + 1)])


def summary(scores: list[float]) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("SCORE_SUMMARY_INVALID")
    return {"count": int(len(values)), "mean": float(values.mean()), "quantiles": {str(q): float(np.quantile(values, q)) for q in (0.0, 0.1, 0.5, 0.9, 1.0)}, "emission_rate_at_0.5": float(np.mean(values >= 0.5))}


def rows_for(rows: list[dict[str, Any]], suite: str | None = None, split: str | None = None) -> list[dict[str, Any]]:
    return [row for row in rows if (suite is None or str(row["suite"]) == suite) and (split is None or str(row["split"]) == split)]


def grouped_row_summary(rows: list[dict[str, Any]], score_map: dict[tuple[str, int], float], label_map: dict[tuple[str, int], int] | None = None) -> dict[str, Any]:
    suites = sorted({str(row["suite"]) for row in rows})
    output = {}
    for suite in suites:
        selected = rows_for(rows, suite=suite)
        scores = [float(score_map[(str(row["canonical_parent_key"]), int(row["probe_step"]))]) for row in selected]
        values = {"row_count": len(selected), "parent_count": len({str(row["canonical_parent_key"]) for row in selected}), "prevalence": float(np.mean([int(row["y"]) for row in selected])), "score": summary(scores)}
        if label_map is not None:
            labels = [label_map[(str(row["canonical_parent_key"]), int(row["probe_step"]))] for row in selected]
            values["positive_rate"] = float(np.mean(labels))
        output[suite] = values
    return output


def phase_distribution(m4_labels: list[dict[str, Any]], provenance: dict[str, Any]) -> dict[str, Any]:
    target = {(str(row["canonical_parent_key"]), int(row["probe_step"])) for row in m4_labels if str(row["dose"]) == "T5"}
    counts: Counter[str] = Counter()
    joined = 0
    paths = provenance.get("source_paths", {})
    for identity, raw_path in paths.items():
        path = Path(str(raw_path))
        if not path.is_file():
            raise ValueError(f"PHASE_SOURCE_MISSING:{identity}")
        data = read_json(path)
        for row in data.get("rows", []):
            key = (str(identity), int(row["step"]))
            if key not in target:
                continue
            phase = str(row.get("clean_only_phase_label") or "UNKNOWN")
            counts[phase] += 1
            joined += 1
    if joined != len(target):
        raise ValueError(f"PHASE_JOIN:{joined}:{len(target)}")
    return {"target_rows": len(target), "joined_rows": joined, "counts": dict(sorted(counts.items())), "outcomes_read": False}


def seal(root: Path) -> str:
    files = sorted(path for path in root.iterdir() if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json"})
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in files), encoding="utf-8")
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    (root / "ROOT_SEAL.json").write_text(json.dumps({"schema": "STAGE_VI_B2_SUITE_FORENSIC_ROOT_SEAL_V1", "status": "PASS_STAGE_VI_B2_SUITE_FORENSIC", "sha256sums_sha256": digest, "protected_counters": COUNTERS, "eval160_status": "UNREAD"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return digest


def run(args: argparse.Namespace) -> int:
    source_commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=REPO, text=True).strip()
    source_tree = subprocess.check_output(("git", "rev-parse", "HEAD^{tree}"), cwd=REPO, text=True).strip()
    if source_commit != args.source_commit or source_tree != args.source_tree:
        raise ValueError("SOURCE_COMMIT_OR_TREE_MISMATCH")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    diagnostic = read_json(args.diagnostic_root.resolve() / "STAGE_VI_ROOT_CAUSE_DIAGNOSTIC.json")
    diagnostic_seal = read_json(args.diagnostic_root.resolve() / "ROOT_SEAL.json")
    if diagnostic.get("classification", {}).get("case") != "B" or diagnostic_seal.get("protected_counters") != COUNTERS or diagnostic_seal.get("eval160_status") != "UNREAD":
        raise ValueError("STAGE_VI_DIAGNOSTIC_BOUNDARY_INVALID")
    parent_splits = base.load_parent_splits()
    m4_labels = base.load_m4_labels(parent_splits)
    m4_student = base.load_m4_student(parent_splits, m4_labels)
    features = base.load_features()
    privileged, privileged_provenance = base.load_privileged_features(m4_labels)
    t5 = [row for row in m4_labels if str(row["dose"]) == "T5"]
    if len(t5) != 858 or len({str(row["canonical_parent_key"]) for row in t5}) != 40:
        raise ValueError("T5_GRAIN_INVALID")
    keys = [(str(row["canonical_parent_key"]), int(row["probe_step"])) for row in t5]
    if len(keys) != len(set(keys)):
        raise ValueError("T5_DUPLICATE_PARENT_STEP")
    split_sets = {split: {str(row["canonical_parent_key"]) for row in t5 if str(row["split"]) == split} for split in SPLITS}
    if any(not split_sets[split] for split in SPLITS) or any(split_sets[left] & split_sets[right] for index, left in enumerate(SPLITS) for right in SPLITS[index + 1:]):
        raise ValueError("PARENT_SPLIT_OVERLAP")
    for key in keys:
        if key not in features or key not in privileged:
            raise ValueError(f"FEATURE_JOIN_MISSING:{key}")
    y = np.asarray([int(row["y"]) for row in t5], dtype=np.int64)
    x25 = np.asarray([features[key] for key in keys], dtype=np.float64)
    xwindow = np.asarray([window(features, identity, step) for identity, step in keys], dtype=np.float64)
    xprivileged = np.asarray([privileged[key] for key in keys], dtype=np.float64)
    groups = np.asarray([str(row["canonical_parent_key"]) for row in t5])
    suites = np.asarray([str(row["suite"]) for row in t5])
    phases = np.asarray([str(row["split"]) for row in t5])
    probes = {
        "direct_25d": base.grouped_probe("B2_DIRECT_25D", x25, y, groups, suites, phases),
        "causal_window_16x25d": base.grouped_probe("B2_CAUSAL_WINDOW", xwindow, y, groups, suites, phases),
        "privileged_clean_state": base.grouped_probe("B2_PRIVILEGED", xprivileged, y, groups, suites, phases),
    }
    aggregate = read_json(args.reconstruction_aggregate.resolve() / "AGGREGATE_AUDIT.json")
    aggregate_labels = base.read_jsonl(args.reconstruction_aggregate.resolve() / "M4_R3_TEACHER_LABELS_DIAGNOSTIC.jsonl")
    original_teacher: dict[tuple[str, int], int] = {}
    original_teacher_unknown: dict[tuple[str, int], int] = {}
    for row in aggregate_labels:
        if int(row["step"]) < 0:
            raise ValueError("TEACHER_STEP_INVALID")
        key = (str(row["canonical_parent_key"]), int(row["step"]))
        value = str(row["labels"][HEAD]["value"])
        original_teacher[key] = int(value == "TRUE")
        original_teacher_unknown[key] = int(value == "UNKNOWN")
    teacher_scores = {key: float(original_teacher[key]) for key in keys if key in original_teacher}
    teacher_unknown = {key: original_teacher_unknown[key] for key in keys if key in original_teacher_unknown}
    if len(teacher_scores) != len(keys):
        raise ValueError("ORIGINAL_TEACHER_JOIN_MISSING")
    old_root = args.s6c_root.resolve()
    with (old_root / "T_V_T5_PRIVILEGED_LOGISTIC.pkl").open("rb") as handle:
        old_teacher = pickle.load(handle)
    with (old_root / "S6_C_STUDENT.pkl").open("rb") as handle:
        old_student = pickle.load(handle)
    old_teacher_scores = old_teacher.predict_proba(xprivileged)[:, 1]
    old_student_scores = old_student.predict_proba(xwindow)[:, 1]
    old_teacher_map = dict(zip(keys, old_teacher_scores.tolist()))
    old_student_map = dict(zip(keys, old_student_scores.tolist()))
    frozen_student = {(str(row["canonical_parent_key"]), int(row["probe_step"])): float(row["scores"][HEAD]) for row in m4_student}
    if len(frozen_student) != len(keys):
        raise ValueError("FROZEN_STUDENT_JOIN_MISSING")
    suite_rows = {suite: rows_for(t5, suite=suite) for suite in sorted(suites.tolist())}
    quality = {
        "t5_row_count": len(t5),
        "t5_parent_count": len({str(row["canonical_parent_key"]) for row in t5}),
        "t5_duplicate_parent_step_count": 0,
        "feature_join_count_25d": len(x25),
        "feature_join_count_causal_window": len(xwindow),
        "feature_join_count_privileged": len(xprivileged),
        "frozen_student_join_count": len(frozen_student),
        "original_teacher_join_count": len(teacher_scores),
        "s6c_teacher_join_count": len(old_teacher_map),
        "s6c_student_join_count": len(old_student_map),
        "parent_split_overlap": False,
        "post_outcome_features_used": False,
        "future_features_used": False,
        "protected_counters": COUNTERS,
        "eval160_status": "UNREAD",
    }
    suites_out = {}
    for suite in sorted(suite_rows):
        rows = suite_rows[suite]
        suite_keys = [(str(row["canonical_parent_key"]), int(row["probe_step"])) for row in rows]
        suites_out[suite] = {
            "row_count": len(rows),
            "parent_count": len({key[0] for key in suite_keys}),
            "split_counts": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
            "v_phys_prevalence": float(np.mean([int(row["y"]) for row in rows])),
            "original_teacher_positive_rate": float(np.mean([teacher_scores[key] for key in suite_keys])),
            "original_teacher_unknown_rate": float(np.mean([teacher_unknown[key] for key in suite_keys])),
            "s6c_teacher_probability": summary([old_teacher_map[key] for key in suite_keys]),
            "s6c_student_probability": summary([old_student_map[key] for key in suite_keys]),
            "frozen_stage_v_student_probability": summary([frozen_student[key] for key in suite_keys]),
            "probes": {name: probe["per_suite"][suite] for name, probe in probes.items()},
        }
    result = {
        "schema": "STAGE_VI_B2_SUITE_FORENSIC_V1",
        "status": "PASS_STAGE_VI_B2_SUITE_FORENSIC",
        "grain": "one consumable Stage-V M4 V_phys label at one parent/probe_step/dose; forensic primary dose T5",
        "inputs": {
            "diagnostic_root": str(args.diagnostic_root.resolve()),
            "diagnostic_root_seal_sha256": diagnostic_seal["sha256sums_sha256"],
            "reconstruction_aggregate": str(args.reconstruction_aggregate.resolve()),
            "reconstruction_aggregate_audit": aggregate,
            "s6c_root": str(old_root),
            "m4_labels_sha256": sha256_file(base.M4_AGGREGATE / "M4_ALL_LABELS_V1.jsonl"),
            "m4_student_sha256": sha256_file(base.M4_STUDENT_ROOT / "STUDENT_M4_PROBE_SCORES_V1.jsonl"),
        },
        "quality_checks": quality,
        "parent_split_counts": {split: len(values) for split, values in split_sets.items()},
        "suite_forensics": suites_out,
        "phase_distribution": phase_distribution(m4_labels, privileged_provenance),
        "global": {
            "v_phys_prevalence": float(y.mean()),
            "original_teacher_positive_rate": float(np.mean(list(teacher_scores.values()))),
            "original_teacher_unknown_rate": float(np.mean(list(teacher_unknown.values()))),
            "s6c_teacher_probability": summary(list(old_teacher_map.values())),
            "s6c_student_probability": summary(list(old_student_map.values())),
            "frozen_stage_v_student_probability": summary(list(frozen_student.values())),
        },
        "grouped_probes": probes,
        "diagnostic_only": True,
        "stage_v_development_outcomes_reused": True,
        "fresh_m4_execution": False,
        "protected_counters": COUNTERS,
        "eval160_status": "UNREAD",
        "source_commit": source_commit,
        "source_tree": source_tree,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    (output / "STAGE_VI_B2_SUITE_FORENSIC.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (output / "PROVENANCE.json").write_text(json.dumps({"schema": "STAGE_VI_B2_SUITE_FORENSIC_PROVENANCE_V1", "source_commit": source_commit, "source_tree": source_tree, "code_sha256": sha256_file(Path(__file__)), "diagnostic_root": str(args.diagnostic_root.resolve()), "reconstruction_aggregate": str(args.reconstruction_aggregate.resolve()), "s6c_root": str(old_root), "privileged_feature_provenance": privileged_provenance, "protected_counters": COUNTERS, "eval160_status": "UNREAD"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = seal(output)
    print(json.dumps({"status": result["status"], "root": str(output), "root_seal": digest, "libero_10_direct_25d_auroc": suites_out["libero_10"]["probes"]["direct_25d"]["auroc"]}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--reconstruction-aggregate", type=Path, required=True)
    parser.add_argument("--s6c-root", type=Path, required=True)
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
