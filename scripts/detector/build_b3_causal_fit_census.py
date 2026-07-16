#!/usr/bin/env python3
"""Build the exact 800-identity causal-25D FIT census without labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from materialize_b3_causal_25d_episode import (  # noqa: E402
    validate_materialization_inputs,
)
from materialize_b3_retention_episode import sha256_file  # noqa: E402


SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
EXPECTED_COUNT = 800
PASS_STATUS = "RUNTIME_VALID_MATERIALIZATION_DRYRUN_PASS"


def canonical_key(suite: str, task_idx: int, state_id: int) -> str:
    return f"{suite}/task_{task_idx:02d}/state_{state_id:02d}"


def expected_identities() -> list[dict[str, Any]]:
    return [
        {
            "suite": suite,
            "task_idx": task,
            "state_id": state,
            "canonical_parent_key": canonical_key(suite, task, state),
            "split": "FIT",
        }
        for suite in SUITES
        for task in range(10)
        for state in range(20)
    ]


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def _metadata_paths(source_root: Path) -> dict[str, list[Path]]:
    paths: dict[str, list[Path]] = defaultdict(list)
    for metadata_path in sorted(source_root.rglob("episode_metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            key = metadata.get("canonical_parent_key")
        except (OSError, ValueError, TypeError):
            continue
        if isinstance(key, str) and key:
            paths[key].append(metadata_path.parent)
    return paths


def _runtime_valid(artifact: Path, metadata: dict[str, Any]) -> bool:
    if metadata.get("runtime_valid") is not True:
        return False
    runtime = json.loads((artifact / "runtime_audit.json").read_text(encoding="utf-8"))
    return runtime.get("runtime_valid") is True


def _identity_matches(metadata: dict[str, Any], identity: dict[str, Any]) -> bool:
    try:
        return (
            metadata.get("suite") == identity["suite"]
            and metadata.get("canonical_parent_key") == identity["canonical_parent_key"]
            and int(metadata.get("task_idx")) == identity["task_idx"]
            and int(metadata.get("state_id")) == identity["state_id"]
        )
    except (TypeError, ValueError):
        return False


def build_census(
    source_root: Path,
    source_protocol: Path,
    feature_config: Path,
    materialization_config: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_root = source_root.resolve()
    artifacts = _metadata_paths(source_root)
    source_protocol = source_protocol.resolve()
    feature_config = feature_config.resolve()
    materialization_config = materialization_config.resolve()
    rows: list[dict[str, Any]] = []
    config_sha = _sha256_file(materialization_config)
    for identity in expected_identities():
        key = identity["canonical_parent_key"]
        candidates = artifacts.get(key, [])
        row: dict[str, Any] = {
            **identity,
            "status": "MISSING",
            "artifact_root": "",
            "reason": "NO_CANONICAL_ARTIFACT",
            "source_artifact_sha256": "",
            "materialization_config_sha256": config_sha,
            "source_protocol_config_sha256": _sha256_file(source_protocol),
            "feature_config_sha256": _sha256_file(feature_config),
            "dryrun_step_count": "",
            "causal_event_count": "",
        }
        if len(candidates) > 1:
            row.update(status="PROTOCOL_HOLD", reason="DUPLICATE_CANONICAL_ARTIFACT")
        elif candidates:
            artifact = candidates[0].resolve()
            row["artifact_root"] = str(artifact)
            try:
                metadata = json.loads((artifact / "episode_metadata.json").read_text(encoding="utf-8"))
                if not _identity_matches(metadata, identity):
                    row.update(status="PROTOCOL_HOLD", reason="IDENTITY_MISMATCH")
                elif metadata.get("condition") != "CLEAN":
                    row.update(status="PROTOCOL_HOLD", reason="NON_CLEAN_SOURCE")
                elif metadata.get("split") != "FIT":
                    row.update(status="PROTOCOL_HOLD", reason="NON_FIT_SOURCE")
                elif not _runtime_valid(artifact, metadata):
                    row.update(status="RUNTIME_INVALID", reason="RUNTIME_VALID_FALSE")
                else:
                    try:
                        dryrun = validate_materialization_inputs(
                            artifact,
                            source_protocol,
                            feature_config,
                            materialization_config,
                        )
                        row.update(
                            status=PASS_STATUS,
                            reason="CAUSAL_25D_STUDENT_DRYRUN_PASS",
                            source_artifact_sha256=dryrun["source_artifact_sha256"],
                            dryrun_step_count=dryrun["step_count"],
                            causal_event_count=dryrun["causal_event_count"],
                            teacher_labels_materialized=dryrun["teacher_labels_materialized"],
                        )
                    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                        row.update(
                            status="MATERIALIZATION_DRYRUN_HOLD",
                            reason=f"MATERIALIZATION_DRYRUN_HOLD:{type(exc).__name__}:{exc}",
                        )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                row.update(status="PROTOCOL_HOLD", reason=f"METADATA_PARSE_ERROR:{type(exc).__name__}")
        rows.append(row)

    counts = Counter(row["status"] for row in rows)
    summary = {
        "schema": "B3_CAUSAL_FIT_CENSUS_V1",
        "identity_accounting_status": "COMPLETE"
        if len(rows) == EXPECTED_COUNT and len({row["canonical_parent_key"] for row in rows}) == EXPECTED_COUNT
        else "HOLD",
        "training_input_status": "PASS"
        if len(rows) == EXPECTED_COUNT and all(row["status"] == PASS_STATUS for row in rows)
        else "HOLD",
        "source_root": str(source_root),
        "source_protocol_config_sha256": _sha256_file(source_protocol),
        "feature_config_sha256": _sha256_file(feature_config),
        "materialization_config_sha256": config_sha,
        "identity_count": len(rows),
        "unique_identity_count": len({row["canonical_parent_key"] for row in rows}),
        "materializable_count": counts.get(PASS_STATUS, 0),
        "status_counts": dict(sorted(counts.items())),
        "by_suite": {
            suite: dict(sorted(Counter(row["status"] for row in rows if row["suite"] == suite).items()))
            for suite in SUITES
        },
        "teacher_labels_read": False,
        "teacher_labels_materialized": False,
        "student_policy_intent_read": False,
        "formal_training_ready": False,
        "formal_attack_ready": False,
    }
    return rows, summary


def write_census(rows: list[dict[str, Any]], summary: dict[str, Any], output_root: Path) -> None:
    output_root = output_root.resolve()
    if output_root.exists():
        raise ValueError(f"output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    fields = [
        "suite", "task_idx", "state_id", "canonical_parent_key", "split", "status", "reason",
        "artifact_root", "source_artifact_sha256", "materialization_config_sha256",
        "source_protocol_config_sha256", "feature_config_sha256", "dryrun_step_count",
        "causal_event_count", "teacher_labels_materialized",
    ]
    csv_path = output_root / "B3_CAUSAL_FIT_CENSUS_V1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    summary["census_sha256"] = _sha256_file(csv_path)
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary_path = output_root / "B3_CAUSAL_FIT_CENSUS_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    names = sorted(path.name for path in output_root.iterdir() if path.is_file())
    sums = output_root / "SHA256SUMS"
    sums.write_text("".join(f"{_sha256_file(output_root / name)}  {name}\n" for name in names), encoding="utf-8")
    (output_root / "SHA256SUMS.sha256").write_text(
        f"{_sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--feature-config", type=Path, required=True)
    parser.add_argument("--materialization-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = build_census(
        args.source_root,
        args.source_protocol,
        args.feature_config,
        args.materialization_config,
    )
    write_census(rows, summary, args.output_root)
    print(json.dumps({key: summary[key] for key in ("identity_accounting_status", "training_input_status", "identity_count", "status_counts")}, sort_keys=True))
    return 0 if summary["identity_accounting_status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
