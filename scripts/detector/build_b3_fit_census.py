#!/usr/bin/env python3
"""Build the 800-identity FIT census without reading Teacher labels."""

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

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from materialize_b3_retention_episode import validate_materialization_inputs  # noqa: E402

SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
REQUIRED_FILES = {
    "episode_metadata.json",
    "episode_summary.json",
    "runtime_audit.json",
    "condition_config.json",
    "attack_config.json",
    "step_records.jsonl",
    "policy_intent_records.jsonl",
    "privileged_teacher_sidecar.jsonl",
    "artifact_sha256.json",
}
DEFAULT_MATERIALIZER_CONFIG = REPO_ROOT / "configs" / "B3_RETENTION_PROTOCOL_V1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_manifest(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return expected_identities()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        try:
            state = int(row["state_id"])
            task = int(row["task_idx"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("manifest has an invalid task_idx/state_id") from exc
        suite = row.get("suite", "")
        if suite not in SUITES or not 0 <= task < 10 or not 0 <= state < 20:
            continue
        key = row.get("canonical_parent_key") or canonical_key(suite, task, state)
        selected.append({
            "suite": suite,
            "task_idx": task,
            "state_id": state,
            "canonical_parent_key": key,
            "split": row.get("split", "FIT"),
        })
    selected.sort(key=lambda row: (row["suite"], row["task_idx"], row["state_id"]))
    expected = sorted(expected_identities(), key=lambda row: (row["suite"], row["task_idx"], row["state_id"]))
    expected_keys = [row["canonical_parent_key"] for row in expected]
    actual_keys = [row["canonical_parent_key"] for row in selected]
    if actual_keys != expected_keys or any(row["split"] != "FIT" for row in selected):
        raise ValueError("FIT census manifest is not exactly the frozen 800-identity set")
    return selected


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
    if metadata.get("suite") != identity["suite"]:
        return False
    if metadata.get("canonical_parent_key") != identity["canonical_parent_key"]:
        return False
    try:
        return (
            int(metadata.get("task_idx")) == identity["task_idx"]
            and int(metadata.get("state_id")) == identity["state_id"]
        )
    except (TypeError, ValueError):
        return False


def build_census(
    manifest_path: Path | None,
    source_root: Path,
    materializer_config_path: Path = DEFAULT_MATERIALIZER_CONFIG,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    identities = load_manifest(manifest_path)
    artifacts = _metadata_paths(source_root)
    materializer_config_path = materializer_config_path.resolve()
    materializer_config_sha = sha256_file(materializer_config_path)
    rows: list[dict[str, Any]] = []
    for identity in identities:
        key = identity["canonical_parent_key"]
        candidates = artifacts.get(key, [])
        row = {
            **identity,
            "status": "MISSING",
            "artifact_root": "",
            "reason": "NO_CANONICAL_ARTIFACT",
            "source_artifact_sha256": "",
            "materializer_config_sha256": materializer_config_sha,
            "dryrun_step_count": "",
        }
        if len(candidates) > 1:
            row.update(status="PROTOCOL_HOLD", reason="DUPLICATE_CANONICAL_ARTIFACT")
        elif candidates:
            artifact = candidates[0]
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
                    missing = sorted(name for name in REQUIRED_FILES if not (artifact / name).is_file())
                    if missing:
                        row.update(status="RUNTIME_VALID_SOURCE_PRESENT", reason="MISSING_SOURCE_FILE:" + ",".join(missing))
                    else:
                        try:
                            dryrun = validate_materialization_inputs(
                                artifact,
                                materializer_config_path,
                                mode="fit-label-materialization",
                            )
                            row.update(
                                status="RUNTIME_VALID_MATERIALIZATION_DRYRUN_PASS",
                                reason="MATERIALIZATION_DRYRUN_PASS",
                                source_artifact_sha256=dryrun["source_artifact_sha256"],
                                dryrun_step_count=dryrun["step_count"],
                            )
                        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                            row.update(
                                status="MATERIALIZATION_DRYRUN_HOLD",
                                reason=f"MATERIALIZATION_DRYRUN_HOLD:{type(exc).__name__}:{exc}",
                            )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                row.update(status="PROTOCOL_HOLD", reason=f"METADATA_OR_RUNTIME_PARSE_ERROR:{type(exc).__name__}")
        rows.append(row)

    counts = Counter(row["status"] for row in rows)
    materializable = counts.get("RUNTIME_VALID_MATERIALIZATION_DRYRUN_PASS", 0)
    source_present = counts.get("RUNTIME_VALID_SOURCE_PRESENT", 0)
    dryrun_holds = counts.get("MATERIALIZATION_DRYRUN_HOLD", 0)
    summary = {
        "schema": "B3_FIT_CENSUS_V1",
        "status": "IDENTITY_ACCOUNTING_COMPLETE" if len(rows) == 800 and len({row["canonical_parent_key"] for row in rows}) == 800 else "HOLD",
        "identity_accounting_status": "COMPLETE" if len(rows) == 800 and len({row["canonical_parent_key"] for row in rows}) == 800 else "HOLD",
        "training_input_status": "PASS" if materializable > 0 and source_present == 0 and dryrun_holds == 0 else "HOLD",
        "source_root": str(source_root.resolve()),
        "manifest": str(manifest_path.resolve()) if manifest_path else None,
        "materializer_config": str(materializer_config_path),
        "materializer_config_sha256": materializer_config_sha,
        "identity_count": len(rows),
        "unique_identity_count": len({row["canonical_parent_key"] for row in rows}),
        "materializable_count": materializable,
        "status_counts": dict(sorted(counts.items())),
        "by_suite": {
            suite: dict(sorted(Counter(row["status"] for row in rows if row["suite"] == suite).items()))
            for suite in SUITES
        },
        "teacher_labels_read": False,
        "teacher_source_sidecar_read": materializable > 0,
        "teacher_files_opened": False,
        "formal_training_ready": False,
        "formal_attack_ready": False,
    }
    return rows, summary


def write_census(rows: list[dict[str, Any]], summary: dict[str, Any], output_root: Path) -> None:
    if output_root.exists():
        raise ValueError(f"output root already exists; use a new census root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    fields = [
        "suite", "task_idx", "state_id", "canonical_parent_key", "split", "status", "reason",
        "artifact_root", "source_artifact_sha256", "materializer_config_sha256", "dryrun_step_count",
    ]
    with (output_root / "B3_FIT_CENSUS_V1.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    summary["census_sha256"] = sha256_file(output_root / "B3_FIT_CENSUS_V1.csv")
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary_path = output_root / "B3_FIT_CENSUS_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in sorted(output_root.iterdir()) if path.is_file()),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--materializer-config", type=Path, default=DEFAULT_MATERIALIZER_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = build_census(args.manifest, args.source_root.resolve(), args.materializer_config)
    write_census(rows, summary, args.output_root.resolve())
    print(json.dumps({key: summary[key] for key in ("status", "training_input_status", "identity_count", "status_counts")}, sort_keys=True))
    return 0 if summary["identity_accounting_status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
