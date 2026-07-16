#!/usr/bin/env python3
"""Transactional materialization of the exact 800-identity FIT census."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
sys.path.insert(0, str(SCRIPT_PATH.parent))

from audit_b3_causal_25d_materialization import audit  # noqa: E402
from materialize_b3_causal_25d_episode import (  # noqa: E402
    materialize,
    sha256_file,
    validate_materialization_inputs,
)
from materialize_b3_retention_episode import verify_source_artifact  # noqa: E402


SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
EXPECTED_COUNT = 800
PASS_STATUS = "RUNTIME_VALID_MATERIALIZATION_DRYRUN_PASS"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_provenance(expected_head: str, runner_repo: Path) -> dict[str, Any]:
    actual = subprocess.run(
        ["git", "-C", str(runner_repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(runner_repo), "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    relative = SCRIPT_PATH.relative_to(runner_repo.resolve()).as_posix()
    subprocess.run(
        ["git", "-C", str(runner_repo), "cat-file", "-e", f"HEAD:{relative}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "expected_head": expected_head,
        "actual_head": actual,
        "worktree_clean": not dirty,
        "provenance_pass": actual == expected_head and not dirty,
        "script_relative": relative,
    }


def _canonical_key(row: dict[str, str]) -> str:
    return row.get("canonical_parent_key", "")


def _load_census(census: Path, source_root: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    with census.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keys = [_canonical_key(row) for row in rows]
    errors: list[str] = []
    if len(rows) != EXPECTED_COUNT:
        errors.append(f"CENSUS_ROW_COUNT:{len(rows)}")
    if len(set(keys)) != len(keys) or any(not key for key in keys):
        errors.append("CENSUS_IDENTITY_SET_INVALID")
    if any(row.get("status") != PASS_STATUS for row in rows):
        counts = Counter(row.get("status", "") for row in rows)
        errors.append("CENSUS_NOT_ALL_DRYRUN_PASS:" + json.dumps(dict(sorted(counts.items())), sort_keys=True))
    for row in rows:
        artifact = Path(row.get("artifact_root", "")).resolve()
        try:
            artifact.relative_to(source_root.resolve())
        except ValueError:
            errors.append(f"ARTIFACT_OUTSIDE_SOURCE_ROOT:{row.get('canonical_parent_key')}")
    return rows, {
        "census_sha256": sha256_file(census),
        "row_count": len(rows),
        "unique_identity_count": len(set(keys)),
        "errors": errors,
    }


def _write_hold_report(path: Path, report: dict[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"report root already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)
    output = path / "B3_CAUSAL_25D_FIT_MATERIALIZATION_REPORT_V1.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_name(output.name + ".sha256").write_text(
        f"{sha256_file(output)}  {output.name}\n", encoding="utf-8"
    )


def _seal_tree_checksums(root: Path) -> None:
    sums = root / "SHA256SUMS"
    sidecar = root / "SHA256SUMS.sha256"
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path not in {sums, sidecar}
    )
    lines = "".join(
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in paths
    )
    sums.write_text(lines, encoding="utf-8")
    sidecar.write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def run(
    source_root: Path,
    census: Path,
    output_root: Path,
    source_protocol: Path,
    feature_config: Path,
    materialization_config: Path,
    expected_head: str,
    runner_repo: Path,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    census = census.resolve()
    output_root = output_root.resolve()
    source_protocol = source_protocol.resolve()
    feature_config = feature_config.resolve()
    materialization_config = materialization_config.resolve()
    runner_repo = runner_repo.resolve()
    if output_root.exists():
        raise ValueError(f"output root already exists: {output_root}")

    provenance = _git_provenance(expected_head, runner_repo)
    rows, census_info = _load_census(census, source_root)
    report: dict[str, Any] = {
        "schema": "B3_CAUSAL_25D_FIT_MATERIALIZATION_AUDIT_V1",
        "status": "HOLD",
        "source_root": str(source_root),
        "census": str(census),
        "census_sha256": census_info["census_sha256"],
        "source_protocol_sha256": sha256_file(source_protocol),
        "feature_config_sha256": sha256_file(feature_config),
        "materialization_config_sha256": sha256_file(materialization_config),
        "expected_identity_count": EXPECTED_COUNT,
        "census_row_count": census_info["row_count"],
        "census_unique_identity_count": census_info["unique_identity_count"],
        "census_errors": census_info["errors"],
        "runner_provenance": provenance,
        "student_policy_intent_read": False,
        "formal_training_ready": False,
        "formal_attack_ready": False,
        "started_at": _now(),
        "records": [],
    }
    if not provenance["provenance_pass"]:
        report["hold_reasons"] = ["RUNNER_PROVENANCE_NOT_CLEAN"]
        _write_hold_report(output_root.with_name(output_root.name + "_HOLD"), report)
        return report
    if census_info["errors"]:
        report["hold_reasons"] = ["FIT_CENSUS_NOT_COMPLETE"]
        _write_hold_report(output_root.with_name(output_root.name + "_HOLD"), report)
        return report

    staging = output_root.with_name(output_root.name + ".staging")
    if staging.exists():
        raise ValueError(f"staging root already exists: {staging}")
    staging.mkdir(parents=True, exist_ok=False)
    passed = 0
    try:
        for index, row in enumerate(sorted(rows, key=_canonical_key)):
            key = _canonical_key(row)
            artifact = Path(row["artifact_root"]).resolve()
            episode_output = staging / "episodes" / f"{index:04d}_{key.replace('/', '__')}"
            item: dict[str, Any] = {"canonical_parent_key": key, "status": "HOLD", "artifact_root": str(artifact)}
            try:
                preflight = validate_materialization_inputs(
                    artifact, source_protocol, feature_config, materialization_config
                )
                if preflight["source_artifact_sha256"] != row.get("source_artifact_sha256"):
                    raise ValueError("CENSUS_SOURCE_SHA_MISMATCH")
                before = verify_source_artifact(artifact)
                manifest = materialize(
                    artifact,
                    episode_output,
                    source_protocol,
                    feature_config,
                    materialization_config,
                )
                result = audit(episode_output)
                after = verify_source_artifact(artifact)
                if before != after:
                    raise ValueError("SOURCE_CHANGED_DURING_MATERIALIZATION")
                item.update({
                    "status": "PASS",
                    "source_artifact_sha256": manifest["source_artifact_sha256"],
                    "step_count": manifest["step_count"],
                    "causal_event_count": manifest["causal_event_count"],
                    "materialization_manifest_sha256": sha256_file(episode_output / "materialization_manifest.json"),
                    "student_projection_pass": result["student_projection_pass"],
                    "teacher_invariant_pass": result["teacher_invariant_pass"],
                })
                passed += 1
            except Exception as exc:  # retain exact failure, then fail closed
                item["error"] = f"{type(exc).__name__}: {exc}"
                if episode_output.exists():
                    shutil.rmtree(episode_output)
            report["records"].append(item)

        report["passed_count"] = passed
        report["failed_count"] = EXPECTED_COUNT - passed
        report["status"] = "PASS" if passed == EXPECTED_COUNT else "HOLD"
        report["hold_reasons"] = [] if report["status"] == "PASS" else ["EPISODE_MATERIALIZATION_OR_AUDIT_HOLD"]
        report["finished_at"] = _now()
        if report["status"] != "PASS":
            shutil.rmtree(staging, ignore_errors=True)
            _write_hold_report(output_root.with_name(output_root.name + "_HOLD"), report)
            return report

        # Seal the aggregate report inside staging before the atomic promote.
        aggregate = staging / "B3_CAUSAL_25D_FIT_MATERIALIZATION_AUDIT_V1.json"
        report["output_promoted"] = True
        aggregate.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        aggregate.with_name(aggregate.name + ".sha256").write_text(
            f"{sha256_file(aggregate)}  {aggregate.name}\n", encoding="utf-8"
        )
        staging.replace(output_root)
        _seal_tree_checksums(output_root)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--feature-config", type=Path, required=True)
    parser.add_argument("--materialization-config", type=Path, required=True)
    parser.add_argument("--expected-runner-head", required=True)
    parser.add_argument("--runner-repo", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.source_root,
        args.census,
        args.output_root,
        args.source_protocol,
        args.feature_config,
        args.materialization_config,
        args.expected_runner_head,
        args.runner_repo,
    )
    print(json.dumps({key: report.get(key) for key in ("status", "census_row_count", "passed_count", "failed_count", "hold_reasons")}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
