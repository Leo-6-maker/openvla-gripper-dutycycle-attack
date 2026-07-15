#!/usr/bin/env python3
"""Read-only S0 compatibility audit for real Official CLEAN artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from audit_b3_retention_materialization import audit  # noqa: E402
from materialize_b3_retention_episode import materialize  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_paths(source_root: Path) -> list[Path]:
    return sorted(source_root.rglob("episode_metadata.json"))


def _selection_paths(selection: Path, source_root: Path) -> list[Path]:
    with selection.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    paths: list[Path] = []
    for row in rows:
        raw = row.get("artifact")
        if not raw:
            raise ValueError("selection row is missing artifact")
        if row.get("runtime_valid", "").lower() != "true":
            continue
        artifact = Path(raw)
        if not artifact.is_absolute():
            artifact = source_root / artifact
        artifact = artifact.resolve()
        if artifact not in paths:
            paths.append(artifact)
    return paths


def _canonical(meta_path: Path) -> str:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    key = payload.get("canonical_parent_key")
    if not isinstance(key, str) or not key:
        raise ValueError(f"metadata has no canonical_parent_key: {meta_path}")
    return key


def run(source_root: Path, output_root: Path, config: Path, selection: Path | None) -> dict[str, object]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if not source_root.is_dir():
        raise ValueError(f"missing source root: {source_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"S0 output root is non-empty: {output_root}")
    selected = _selection_paths(selection, source_root) if selection else _metadata_paths(source_root)
    selected = sorted(selected, key=_canonical)
    if not selected:
        raise ValueError("no runtime-valid source artifacts selected")
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate source artifact selection")

    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    suite_counts: Counter[str] = Counter()
    pass_count = 0
    for index, artifact in enumerate(selected):
        meta_path = artifact / "episode_metadata.json"
        key = _canonical(meta_path)
        output = output_root / "episodes" / f"{index:04d}_{key.replace('/', '__')}"
        record: dict[str, object] = {
            "canonical_parent_key": key,
            "source_artifact": str(artifact),
            "source_artifact_relative": str(artifact.relative_to(source_root))
            if artifact.is_relative_to(source_root)
            else None,
            "status": "FAIL",
        }
        try:
            manifest = materialize(artifact, output, config)
            result = audit(output)
            record.update(
                {
                    "status": "PASS",
                    "suite": manifest["source_identity"]["suite"],
                    "task_idx": manifest["source_identity"]["task_idx"],
                    "state_id": manifest["source_identity"]["state_id"],
                    "source_artifact_sha256": manifest["source_artifact_sha256"],
                    "step_count": manifest["step_count"],
                    "label_statistics": manifest["label_statistics"],
                    "audit_schema": result["schema"],
                }
            )
            suite_counts[str(record["suite"])] += 1
            pass_count += 1
        except Exception as exc:  # noqa: BLE001 - report every artifact and continue
            record["error"] = f"{type(exc).__name__}: {exc}"
            if output.exists():
                shutil.rmtree(output)
        records.append(record)

    report = {
        "schema": "B3_RETENTION_REAL_ARTIFACT_COMPATIBILITY_AUDIT_V1",
        "source_root": str(source_root),
        "selection_manifest": str(selection.resolve()) if selection else None,
        "selected_count": len(selected),
        "pass_count": pass_count,
        "fail_count": len(selected) - pass_count,
        "suite_pass_counts": dict(sorted(suite_counts.items())),
        "status": "PASS" if pass_count == len(selected) else "HOLD",
        "read_only_source": True,
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
    args = parser.parse_args()
    report = run(args.source_root, args.output_root, args.config, args.selection_manifest)
    print(json.dumps({key: report[key] for key in ("status", "selected_count", "pass_count", "fail_count")}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
