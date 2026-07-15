#!/usr/bin/env python3
"""Read-only compatibility or FIT-only materialization audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from audit_b3_retention_materialization import audit  # noqa: E402
from materialize_b3_retention_episode import (  # noqa: E402
    materialize,
    validate_materialization_inputs,
    verify_source_artifact,
)


SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
EXPECTED_TASKS = {(suite, task) for suite in SUITES for task in range(10)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_paths(source_root: Path) -> list[Path]:
    return sorted(path.parent for path in source_root.rglob("episode_metadata.json"))


def _selection_paths(selection: Path, source_root: Path) -> list[Path]:
    with selection.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    paths: list[Path] = []
    for row in rows:
        if row.get("runtime_valid", "").lower() != "true":
            continue
        raw = row.get("artifact")
        if not raw:
            raise ValueError("selection row is missing artifact")
        artifact = Path(raw)
        if not artifact.is_absolute():
            artifact = source_root / artifact
        paths.append(artifact.resolve())
    return paths


def _metadata(artifact: Path) -> dict[str, object]:
    path = artifact / "episode_metadata.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"metadata is not an object: {path}")
    return value


def _canonical(artifact: Path) -> str:
    key = _metadata(artifact).get("canonical_parent_key")
    if not isinstance(key, str) or not key:
        raise ValueError(f"metadata has no canonical_parent_key: {artifact}")
    return key


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _git_provenance(repo: Path, expected_head: str) -> tuple[str, bool, bool, str]:
    actual = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    script_path = Path(__file__).resolve()
    try:
        script_relative = script_path.relative_to(repo.resolve())
    except ValueError as exc:
        raise ValueError(f"runner script is outside runner repo: {script_path}") from exc
    subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"HEAD:{script_relative.as_posix()}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return actual, not dirty, actual == expected_head and not dirty, script_relative.as_posix()


def _fit_preflight(selected: list[Path], source_root: Path, config: Path) -> None:
    keys: list[str] = []
    for artifact in selected:
        if not _is_relative_to(artifact, source_root):
            raise ValueError(f"selection artifact is outside source root: {artifact}")
        meta = _metadata(artifact)
        suite = meta.get("suite")
        task = meta.get("task_idx")
        state = meta.get("state_id")
        if suite not in SUITES or not isinstance(task, int) or not 0 <= task < 10:
            raise ValueError(f"invalid FIT identity: {artifact}")
        if not isinstance(state, int) or not 0 <= state <= 19 or meta.get("split") != "FIT":
            raise ValueError(f"FIT preflight rejects non-FIT artifact: {artifact}")
        key = meta.get("canonical_parent_key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"missing FIT canonical identity: {artifact}")
        keys.append(key)
        validate_materialization_inputs(artifact, config, mode="fit-label-materialization")
    if len(keys) != len(set(keys)):
        raise ValueError("FIT preflight rejects duplicate canonical identity")


def run(
    source_root: Path,
    output_root: Path,
    config: Path,
    selection: Path | None,
    mode: str,
    runner_git_head: str,
    runner_repo: Path,
    require_all_suite_tasks: bool,
) -> dict[str, object]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    config = config.resolve()
    selection = selection.resolve() if selection else None
    if not source_root.is_dir():
        raise ValueError(f"missing source root: {source_root}")
    if not config.is_file():
        raise ValueError(f"missing protocol config: {config}")
    if selection and not selection.is_file():
        raise ValueError(f"missing selection manifest: {selection}")
    runner_repo = runner_repo.resolve()
    actual_git_head, actual_worktree_clean, runner_provenance_pass, runner_script_relative = _git_provenance(
        runner_repo, runner_git_head
    )
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"S0 output root is non-empty: {output_root}")
    staging_root = output_root.with_name(output_root.name + ".staging")
    if mode == "fit-label-materialization" and (output_root.exists() or staging_root.exists()):
        raise ValueError(f"FIT output or staging root already exists: {output_root}")

    started_at = _now()
    selected = _selection_paths(selection, source_root) if selection else _metadata_paths(source_root)
    selected = sorted(selected, key=_canonical)
    if not selected:
        raise ValueError("no source artifacts selected")
    outside_root = sorted(str(artifact) for artifact in selected if not _is_relative_to(artifact, source_root))
    if outside_root:
        raise ValueError(f"selection artifacts outside source root: {outside_root[:3]}")
    fit_transactional_preflight = mode != "fit-label-materialization"
    if mode == "fit-label-materialization":
        _fit_preflight(selected, source_root, config)
        fit_transactional_preflight = True
    keys = [_canonical(artifact) for artifact in selected]
    key_counts = Counter(keys)
    duplicate_keys = sorted(key for key, count in key_counts.items() if count > 1)
    suite_task_selected = Counter()
    state_counts = Counter()
    for artifact in selected:
        meta = _metadata(artifact)
        suite_task_selected[(str(meta.get("suite")), int(meta.get("task_idx", -1)))] += 1
        state_counts[(str(meta.get("suite")), int(meta.get("state_id", -1)))] += 1
    missing_suite_tasks = sorted(EXPECTED_TASKS - set(suite_task_selected))

    work_root = staging_root if mode == "fit-label-materialization" else output_root
    work_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    suite_pass = Counter()
    suite_task_pass = Counter()
    pass_count = 0
    for index, artifact in enumerate(selected):
        key = _canonical(artifact)
        meta = _metadata(artifact)
        output = work_root / "episodes" / f"{index:04d}_{key.replace('/', '__')}"
        record: dict[str, object] = {
            "canonical_parent_key": key,
            "suite": meta.get("suite"),
            "task_idx": meta.get("task_idx"),
            "state_id": meta.get("state_id"),
            "split": meta.get("split"),
            "source_artifact": str(artifact),
            "source_artifact_relative": str(artifact.relative_to(source_root))
            if _is_relative_to(artifact, source_root)
            else None,
            "source_artifact_manifest_sha256": None,
            "status": "FAIL",
        }
        before_sha = after_sha = None
        try:
            record["source_artifact_manifest_sha256"] = _sha256(artifact / "artifact_sha256.json")
            before_sha = verify_source_artifact(artifact)
            record["source_recursive_sha256_before"] = before_sha
            manifest = materialize(artifact, output, config, mode=mode)
            audit_result = audit(output)
            after_sha = verify_source_artifact(artifact)
            record["source_recursive_sha256_after"] = after_sha
            record["source_unchanged"] = before_sha == after_sha
            if before_sha != after_sha:
                raise ValueError("SOURCE_CHANGED_DURING_AUDIT")
            child_manifest = output / "materialization_manifest.json"
            record.update(
                {
                    "status": "PASS",
                    "source_artifact_sha256": manifest["source_artifact_sha256"],
                    "materialization_manifest_sha256": _sha256(child_manifest),
                    "step_count": manifest["step_count"],
                    "audit_schema": audit_result["schema"],
                    "robot_evidence_error_summary": manifest["robot_evidence_error_summary"],
                }
            )
            if mode == "fit-label-materialization":
                record["label_statistics"] = manifest["label_statistics"]
            suite_pass[str(meta["suite"])] += 1
            suite_task_pass[(str(meta["suite"]), int(meta["task_idx"]))] += 1
            pass_count += 1
        except Exception as exc:  # noqa: BLE001 - report every artifact and continue
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["source_recursive_sha256_after"] = after_sha
            record["source_unchanged"] = before_sha is not None and after_sha == before_sha
            if output.exists():
                shutil.rmtree(output)
        records.append(record)

    coverage_pass = not missing_suite_tasks and all(value == 1 for value in suite_task_selected.values())
    artifact_status = "PASS" if pass_count == len(selected) and not duplicate_keys else "HOLD"
    status = "PASS" if artifact_status == "PASS" and runner_provenance_pass and (coverage_pass or not require_all_suite_tasks) else "HOLD"
    error_keys = (
        "max_action_abs_error",
        "max_qpos_abs_error",
        "max_opening_abs_error",
        "max_eef_abs_error",
    )
    robot_evidence_max_errors = {
        key: max(
            (float(row["robot_evidence_error_summary"][key]) for row in records if "robot_evidence_error_summary" in row),
            default=0.0,
        )
        for key in error_keys
    }
    fit_output_promoted = False
    if mode == "fit-label-materialization":
        if status == "PASS":
            staging_root.replace(output_root)
            fit_output_promoted = True
        else:
            shutil.rmtree(staging_root, ignore_errors=True)
            output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "B3_RETENTION_REAL_ARTIFACT_COMPATIBILITY_AUDIT_V2",
        "mode": mode,
        "source_root": str(source_root),
        "selection_manifest": str(selection) if selection else None,
        "selection_manifest_sha256": _sha256(selection) if selection else None,
        "selected_count": len(selected),
        "unique_canonical_key_count": len(key_counts),
        "duplicate_canonical_keys": duplicate_keys,
        "unique_suite_task_count": len(suite_task_selected),
        "selected_state_counts": {f"{suite}/state_{state:02d}": count for (suite, state), count in sorted(state_counts.items())},
        "suite_task_selected_counts": {f"{suite}/task_{task:02d}": count for (suite, task), count in sorted(suite_task_selected.items())},
        "suite_task_pass_counts": {f"{suite}/task_{task:02d}": count for (suite, task), count in sorted(suite_task_pass.items())},
        "missing_suite_tasks": [f"{suite}/task_{task:02d}" for suite, task in missing_suite_tasks],
        "coverage_status": "FULL_40_TASKS" if coverage_pass else "SAMPLE_OR_INCOMPLETE",
        "require_all_suite_tasks": require_all_suite_tasks,
        "pass_count": pass_count,
        "fail_count": len(selected) - pass_count,
        "suite_pass_counts": dict(sorted(suite_pass.items())),
        "status": status,
        "artifact_compatibility_status": artifact_status,
        "classification": (
            "S0_SCHEMA_JOIN_ROBOT_EVIDENCE_PARITY_PASS"
            if artifact_status == "PASS" and mode == "compatibility-only"
            else "FIT_LABEL_MATERIALIZATION_PASS"
            if artifact_status == "PASS" and mode == "fit-label-materialization"
            else "HOLD"
        ),
        "read_only_source": True,
        "source_unchanged_all": all(row.get("source_unchanged") is True for row in records),
        "expected_runner_git_head": runner_git_head,
        "actual_runner_git_head": actual_git_head,
        "runner_git_head": actual_git_head,
        "runner_worktree_clean": actual_worktree_clean,
        "runner_provenance_pass": runner_provenance_pass,
        "runner_script_relative_path": runner_script_relative,
        "protocol_config_sha256": _sha256(config),
        "real_audit_script_sha256": _sha256(Path(__file__).resolve()),
        "robot_evidence_max_errors": robot_evidence_max_errors,
        "fit_transactional_preflight": fit_transactional_preflight,
        "fit_output_promoted": fit_output_promoted,
        "started_at": started_at,
        "finished_at": _now(),
        "formal_training_ready": False,
        "formal_attack_ready": False,
        "records": records,
    }
    report_path = output_root / "REAL_ARTIFACT_COMPATIBILITY_AUDIT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.with_name(report_path.name + ".sha256").write_text(
        f"{_sha256(report_path)}  {report_path.name}\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--mode", choices=("compatibility-only", "fit-label-materialization"), default="compatibility-only")
    parser.add_argument("--runner-git-head", required=True)
    parser.add_argument("--runner-repo", type=Path, required=True)
    parser.add_argument("--require-all-suite-tasks", action="store_true")
    args = parser.parse_args()
    report = run(
        args.source_root,
        args.output_root,
        args.config,
        args.selection_manifest,
        args.mode,
        args.runner_git_head,
        args.runner_repo,
        args.require_all_suite_tasks,
    )
    print(json.dumps({key: report[key] for key in ("status", "mode", "selected_count", "pass_count", "fail_count", "coverage_status")}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
