"""Build R9P OGS-1500 preview plan from R8Z derived suite roots.

Discovers 900 DETECTOR_TRAIN episodes (300 per suite), assigns nested
preview splits (FIT/CAL/CHECK), freezes the model contract, and binds
all source provenance. Only TRAIN-cohort episodes are enumerated; VAL,
TEST, and ATTACK_EVAL cohorts remain sealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from tools.multisuite_detector.c2g_official_suite_horizons import OFFICIAL_MAX_POLICY_STEPS
from tools.multisuite_detector.c2g_r8r_common import (
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)

SCHEMA = "c2g.r9p.preview_plan.2026-07-12.v1"

TARGET_SUITES = ("libero_spatial", "libero_object", "libero_goal")
SUITE_SLUGS = {"spatial": "libero_spatial", "object": "libero_object", "goal": "libero_goal"}

EXPECTED_TRAIN_PER_SUITE = 300
EXPECTED_TOTAL_TRAIN = 900

PREVIEW_SPLIT_SALT = "C2G_R9P_OGS1500_PREVIEW_V1"
FIT_BUCKETS = frozenset(range(8))
CAL_BUCKET = 8
CHECK_BUCKET = 9

R9P_HEAD_NAMES = (
    "window_start",
    "burst_feasible",
    "critical_window",
    "release_safe",
    "contact_grasp",
    "grounding_confidence",
)

R9P_MODEL_TARGET_MAP = {
    "window_start": "y_attack_start_b",
    "burst_feasible": "y_burst_feasible",
    "critical_window": "y_gripper_critical_window",
    "release_safe": "y_release_safe",
    "contact_grasp": "y_contact_or_grasp_stable",
    "grounding_confidence": "grounding_confidence",
}

LOSS_WEIGHTS = {
    "start": 1.0,
    "burst": 0.5,
    "critical": 0.5,
    "release": 0.2,
    "contact": 0.2,
    "grounding": 0.2,
    "early_emit": 0.25,
    "episode_miss": 0.50,
    "negative_any_emit": 0.50,
    "release_safe_emit": 0.50,
}

FORBIDDEN_STUDENT_FIELDS = (
    "object_pose", "target_pose", "object_target_distance",
    "contact_pairs", "teacher_phase", "teacher_reason_code",
    "resolved_target_objects", "attack_outcome", "post_intervention_state",
    "clean_final_success", "late_success_in_extended_source",
)

GATE_PASS = "PASS_C2G_R9P_PLAN"
GATE_HOLD_COUNT = "HOLD_C2G_R9P_PLAN_EPISODE_COUNT"
GATE_HOLD_SPLIT = "HOLD_C2G_R9P_PLAN_SPLIT_DISTRIBUTION"
GATE_HOLD_PROVENANCE = "HOLD_C2G_R9P_PLAN_PROVENANCE"


def _bucket(key: str, salt: str, modulus: int = 10) -> int:
    return hashlib.sha256(f"{salt}|{key}".encode()).digest()[0] % modulus


def assign_preview_split(parent_key: str) -> str:
    bucket = _bucket(parent_key, PREVIEW_SPLIT_SALT, 10)
    if bucket in FIT_BUCKETS:
        return "FIT"
    if bucket == CAL_BUCKET:
        return "CAL"
    return "CHECK"


def assign_splits_deterministic(rows: list[dict]) -> None:
    """Assign exact FIT/CAL/CHECK splits stratified per suite*task.

    Within each (suite, task_index) group, episodes are sorted by their
    full SHA256 integer rank. Per task: 24 FIT, 3 CAL, 3 CHECK (30 episodes).
    This guarantees exact 720/90/90 across all 30 suite×task groups.
    Uses the full 32-byte SHA256 digest as integer (not just first byte) to
    avoid collisions.
    """
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[(r["suite"], r["task_index"])].append(r)

    for (suite, task_idx), group_rows in groups.items():
        if len(group_rows) != 30:
            raise ValueError(
                f"Expected 30 train episodes per task, got {len(group_rows)} "
                f"for suite={suite} task={task_idx}"
            )
        # Full SHA256 integer rank (32 bytes) — no first-byte collision
        ranked = sorted(
            group_rows,
            key=lambda r: int.from_bytes(
                hashlib.sha256(f"{PREVIEW_SPLIT_SALT}|{r['parent_key']}".encode()).digest(),
                "big",
            ),
        )
        for i, r in enumerate(ranked):
            if i < 24:
                r["preview_split"] = "FIT"
            elif i < 27:
                r["preview_split"] = "CAL"
            else:
                r["preview_split"] = "CHECK"


def discover_episodes(suite_root: Path, suite: str) -> list[dict[str, Any]]:
    episodes_dir = suite_root / "episodes" / suite
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"episodes directory not found: {episodes_dir}")
    rows = []
    for meta_path in sorted(episodes_dir.rglob("derived_episode_metadata.json")):
        meta = read_json(meta_path)
        if meta.get("cohort") != "DETECTOR_TRAIN":
            continue
        parent_key = meta.get("parent_key", "")
        if not parent_key:
            raise ValueError(f"missing parent_key in {meta_path}")
        rows.append({
            "parent_key": parent_key,
            "suite": meta.get("suite", suite),
            "task_index": int(meta.get("task_index", -1)),
            "state_id": int(meta.get("state_id", -1)),
            "cohort": "DETECTOR_TRAIN",
            "split": meta.get("split", "train"),
            "task_language": meta.get("task_language", ""),
            "metadata_path": str(meta_path.relative_to(suite_root)),
        })
    return rows


def _validate_episode_counts(rows: list[dict], suite: str) -> None:
    count = len(rows)
    if count != EXPECTED_TRAIN_PER_SUITE:
        raise ValueError(
            f"{suite}: expected {EXPECTED_TRAIN_PER_SUITE} TRAIN episodes, found {count}"
        )


def _validate_split_distribution(rows: list[dict]) -> dict[str, int]:
    counts = Counter(row["preview_split"] for row in rows)
    if counts.get("FIT", 0) != 720:
        raise ValueError(f"FIT expected 720, got {counts.get('FIT', 0)}")
    if counts.get("CAL", 0) != 90:
        raise ValueError(f"CAL expected 90, got {counts.get('CAL', 0)}")
    if counts.get("CHECK", 0) != 90:
        raise ValueError(f"CHECK expected 90, got {counts.get('CHECK', 0)}")
    return dict(counts)


def _per_suite_split_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for suite in TARGET_SUITES:
        suite_rows = [r for r in rows if r["suite"] == suite]
        result[suite] = dict(Counter(r["preview_split"] for r in suite_rows))
    return result


def _verify_git_state(expected_commit: str) -> dict[str, Any]:
    import subprocess
    result = {"clean": True, "issues": []}
    try:
        actual = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
        if actual != expected_commit:
            result["clean"] = False
            result["issues"].append(f"HEAD mismatch: expected {expected_commit}, actual {actual}")
    except Exception as exc:
        result["clean"] = False
        result["issues"].append(f"git rev-parse failed: {exc}")

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True,
        ).stdout.strip()
        # Allow untracked files but no modifications to tracked files
        modified = [line for line in status.splitlines() if not line.startswith("??")]
        if modified:
            result["clean"] = False
            result["issues"].append(f"dirty worktree: {len(modified)} modified tracked files")
    except Exception as exc:
        result["clean"] = False
        result["issues"].append(f"git status failed: {exc}")
    return result


def build_plan(
    *,
    spatial_root: Path,
    object_root: Path,
    goal_root: Path,
    output_root: Path,
    git_commit: str,
    expected_spatial_report_sha: str = "",
    expected_object_report_sha: str = "",
    expected_goal_report_sha: str = "",
    expected_r8z1_audit_sha: str = "",
) -> dict[str, Any]:
    suite_roots = {
        "libero_spatial": spatial_root,
        "libero_object": object_root,
        "libero_goal": goal_root,
    }

    # --- Provenance verification ---
    git_state = _verify_git_state(git_commit)
    if not git_state["clean"]:
        return {
            "schema": SCHEMA,
            "status": GATE_HOLD_PROVENANCE,
            "git_issues": git_state["issues"],
        }

    # Verify R8Z suite report SHAs if provided
    provenance_issues = []
    suite_report_shas = {
        "libero_spatial": expected_spatial_report_sha,
        "libero_object": expected_object_report_sha,
        "libero_goal": expected_goal_report_sha,
    }
    for suite, expected_sha in suite_report_shas.items():
        if expected_sha:
            report_path = suite_roots[suite] / "suite_report.json"
            if report_path.exists():
                actual = sha256_file(report_path)
                if actual != expected_sha:
                    provenance_issues.append(
                        f"{suite} report SHA mismatch: expected {expected_sha}, actual {actual}"
                    )
            else:
                provenance_issues.append(f"{suite} report not found: {report_path}")

    if expected_r8z1_audit_sha:
        # R8Z1 audit SHA is verified against the audit report at a known path
        provenance_issues.append(
            "R8Z1 audit SHA binding requires external verification — "
            "ensure the audit report at the canonical R8Z1 output root matches"
        )

    if provenance_issues:
        return {
            "schema": SCHEMA,
            "status": GATE_HOLD_PROVENANCE,
            "provenance_issues": provenance_issues,
        }

    # Output root must not already exist
    if output_root.exists():
        return {
            "schema": SCHEMA,
            "status": GATE_HOLD_PROVENANCE,
            "error": f"output root already exists: {output_root}",
        }
    output_root.mkdir(parents=True)
    for suite, root in suite_roots.items():
        try:
            rows = discover_episodes(root, suite)
            _validate_episode_counts(rows, suite)
            all_rows.extend(rows)
        except Exception as exc:
            errors.append({"suite": suite, "error": str(exc)})

    if errors:
        return {
            "schema": SCHEMA,
            "status": GATE_HOLD_COUNT,
            "errors": errors,
        }

    if len(all_rows) != EXPECTED_TOTAL_TRAIN:
        return {
            "schema": SCHEMA,
            "status": GATE_HOLD_COUNT,
            "total_episodes": len(all_rows),
            "expected": EXPECTED_TOTAL_TRAIN,
        }

    assign_splits_deterministic(all_rows)
    split_counts = _validate_split_distribution(all_rows)

    per_suite = _per_suite_split_counts(all_rows)
    unique_parents = set(r["parent_key"] for r in all_rows)
    if len(unique_parents) != len(all_rows):
        return {
            "schema": SCHEMA,
            "status": GATE_HOLD_COUNT,
            "error": f"duplicate parent_keys: {len(all_rows)} rows, {len(unique_parents)} unique",
        }

    tasks_by_suite: dict[str, set] = defaultdict(set)
    for r in all_rows:
        tasks_by_suite[r["suite"]].add(r["task_index"])
    per_suite_tasks = {s: len(t) for s, t in tasks_by_suite.items()}

    manifest_path = output_root / "r9p_preview_episode_manifest.jsonl"
    manifest_rows = [
        {
            "parent_key": r["parent_key"],
            "suite": r["suite"],
            "task_index": r["task_index"],
            "state_id": r["state_id"],
            "cohort": r["cohort"],
            "task_language": r["task_language"],
            "preview_split": r["preview_split"],
        }
        for r in sorted(all_rows, key=lambda x: (x["suite"], x["parent_key"]))
    ]
    write_jsonl(manifest_path, manifest_rows)

    split_manifest_path = output_root / "r9p_preview_split_manifest.jsonl"
    split_rows = []
    for suite in TARGET_SUITES:
        split_rows.append({
            "suite": suite,
            "FIT": per_suite[suite].get("FIT", 0),
            "CAL": per_suite[suite].get("CAL", 0),
            "CHECK": per_suite[suite].get("CHECK", 0),
        })
    split_rows.append({
        "suite": "TOTAL",
        "FIT": split_counts.get("FIT", 0),
        "CAL": split_counts.get("CAL", 0),
        "CHECK": split_counts.get("CHECK", 0),
    })
    write_jsonl(split_manifest_path, split_rows)

    feature_schema = {
        "schema": SCHEMA,
        "proprio_dim": 25,
        "policy_intent_dim": 9,
        "proprio_source": "features_25d",
        "policy_intent_source": "clean_policy_intent_9d",
        "feature_dtype": "float32",
    }
    write_json(output_root / "r9p_feature_schema.json", feature_schema)

    label_schema = {
        "schema": SCHEMA,
        "head_names": list(R9P_HEAD_NAMES),
        "teacher_label_map": dict(R9P_MODEL_TARGET_MAP),
        "burst_length": 10,
        "contact_persistence_steps": 2,
        "label_known_key": "label_known_mask",
        "forbidden_student_fields": list(FORBIDDEN_STUDENT_FIELDS),
    }
    write_json(output_root / "r9p_label_schema.json", label_schema)

    model_spec = {
        "schema": SCHEMA,
        "head_names": list(R9P_HEAD_NAMES),
        "teacher_label_map": dict(R9P_MODEL_TARGET_MAP),
        "proprio_dim": 25,
        "policy_intent_dim": 9,
        "language_dim": 128,
        "visual_dim": 1152,
        "hidden": 128,
        "dropout": 0.1,
        "burst_length": 10,
        "loss_weights": dict(LOSS_WEIGHTS),
        "model_a": {
            "use_policy_intent": False,
            "use_visual": False,
            "use_language_conditioning": True,
        },
        "model_b": {
            "use_policy_intent": True,
            "use_visual": False,
            "use_language_conditioning": True,
        },
        "training": {
            "max_epochs": 30,
            "early_stop_patience": 5,
            "seeds": [42, 123, 456],
            "optimizer": "AdamW",
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "grad_clip": 5.0,
        },
    }
    write_json(output_root / "r9p_model_spec.json", model_spec)

    execution_boundary = {
        "schema": SCHEMA,
        "canonical_val_reads": 0,
        "canonical_test_reads": 0,
        "attack_eval_reads": 0,
        "nontrain_cohorts_exposed": False,
        "sealed_cohorts": ["DETECTOR_VAL", "DETECTOR_TEST_WITHIN_TASK", "ATTACK_EVAL_PREREGISTERED"],
        "preview_checkpoint_classification": "EXPLORATORY_OGS1500_PREVIEW_ONLY",
    }
    write_json(output_root / "r9p_execution_boundary.json", execution_boundary)

    artifacts = [
        "r9p_preview_episode_manifest.jsonl",
        "r9p_preview_split_manifest.jsonl",
        "r9p_feature_schema.json",
        "r9p_label_schema.json",
        "r9p_model_spec.json",
        "r9p_execution_boundary.json",
    ]
    sums_lines = []
    for name in artifacts:
        path = output_root / name
        sums_lines.append(f"{sha256_file(path)}  {name}\n")
    sums_path = output_root / "SHA256SUMS"
    sums_path.write_text("".join(sums_lines), encoding="utf-8")
    sums_sha = sha256_file(sums_path)
    (output_root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")

    plan = {
        "schema": SCHEMA,
        "status": GATE_PASS,
        "mode": "run",
        "plan_kind": "R9P_OGS1500_PREVIEW_PIPELINE",
        "target_suites": list(TARGET_SUITES),
        "git_commit": git_commit,
        "total_train_episodes": len(all_rows),
        "split_counts": split_counts,
        "per_suite_split_counts": per_suite,
        "per_suite_train_episodes": {s: EXPECTED_TRAIN_PER_SUITE for s in TARGET_SUITES},
        "per_suite_tasks": per_suite_tasks,
        "preview_split_salt": PREVIEW_SPLIT_SALT,
        "official_horizons": {s: OFFICIAL_MAX_POLICY_STEPS[s] for s in TARGET_SUITES},
        "outputs": {
            name: {
                "path": name,
                "sha256": sha256_file(output_root / name),
            }
            for name in artifacts
        },
        "sha256sums_sha256": sums_sha,
        "boundaries": {
            "canonical_val_reads": 0,
            "canonical_test_reads": 0,
            "attack_eval_reads": 0,
            "nontrain_metrics_exposed": False,
            "visual_materialization": "deferred",
            "attack_eval_accessed": False,
        },
    }
    plan_path = output_root / "r9p_preview_plan.json"
    write_json(plan_path, plan)
    plan_sha = sha256_file(plan_path)
    (output_root / "r9p_preview_plan.json.sha256").write_text(
        f"{plan_sha}  r9p_preview_plan.json\n", encoding="utf-8"
    )
    return plan


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build R9P OGS-1500 preview plan")
    parser.add_argument("--spatial-root", required=True, type=Path, help="R8Z spatial suite root")
    parser.add_argument("--object-root", required=True, type=Path, help="R8Z object suite root")
    parser.add_argument("--goal-root", required=True, type=Path, help="R8Z goal suite root")
    parser.add_argument("--output-root", required=True, type=Path, help="R9P plan output root")
    parser.add_argument("--git-commit", required=True, help="Current git commit SHA")
    parser.add_argument("--expected-spatial-report-sha", default="")
    parser.add_argument("--expected-object-report-sha", default="")
    parser.add_argument("--expected-goal-report-sha", default="")
    parser.add_argument("--expected-r8z1-audit-sha", default="")
    parser.add_argument("--mode", default="preview", choices=["preview", "run"],
                        help="preview: dry-run validation only; run: materialize plan")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "preview":
        for suite, root in [("libero_spatial", args.spatial_root),
                            ("libero_object", args.object_root),
                            ("libero_goal", args.goal_root)]:
            rows = discover_episodes(root, suite)
            _validate_episode_counts(rows, suite)
            for row in rows:
                row["preview_split"] = assign_preview_split(row["parent_key"])
            print(f"{suite}: {len(rows)} episodes, "
                  f"FIT={sum(1 for r in rows if r['preview_split']=='FIT')} "
                  f"CAL={sum(1 for r in rows if r['preview_split']=='CAL')} "
                  f"CHECK={sum(1 for r in rows if r['preview_split']=='CHECK')}")
        print("Preview OK — use --mode run to materialize")
        return 0

    plan = build_plan(
        spatial_root=args.spatial_root,
        object_root=args.object_root,
        goal_root=args.goal_root,
        output_root=args.output_root,
        git_commit=args.git_commit,
        expected_spatial_report_sha=args.expected_spatial_report_sha,
        expected_object_report_sha=args.expected_object_report_sha,
        expected_goal_report_sha=args.expected_goal_report_sha,
        expected_r8z1_audit_sha=args.expected_r8z1_audit_sha,
    )
    status = plan.get("status", "UNKNOWN")
    print(f"Plan: {status}")
    if status == GATE_PASS:
        print(f"  Episodes: {plan['total_train_episodes']}")
        print(f"  Splits: FIT={plan['split_counts']['FIT']} "
              f"CAL={plan['split_counts']['CAL']} "
              f"CHECK={plan['split_counts']['CHECK']}")
        return 0
    print(f"  Errors: {json.dumps(plan.get('errors', plan.get('error', 'unknown')), indent=2)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
