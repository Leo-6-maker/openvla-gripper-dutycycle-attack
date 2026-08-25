#!/usr/bin/env python3
"""Materialize one frozen OFT checkpoint without changing its authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()

    manifest_sha256 = sha256_file(args.manifest)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS_SEALED_FOUR_OFT_BYTE_MANIFESTS":
        raise RuntimeError("M1_MANIFEST_NOT_SEALED")
    suite = manifest.get("suites", {}).get(args.suite)
    if not isinstance(suite, dict) or not isinstance(suite.get("rows"), list):
        raise RuntimeError(f"M1_MANIFEST_SUITE_MISSING:{args.suite}")
    if args.target.exists():
        raise RuntimeError("Z3R1_TARGET_ALREADY_EXISTS")
    if args.staging.resolve() == args.target.resolve():
        raise RuntimeError("Z3R1_STAGING_MUST_BE_SEPARATE")

    rows = []
    for row in suite["rows"]:
        relative = str(row["path"]).replace("\\", "/")
        if relative.startswith("/") or ".." in Path(relative).parts:
            raise RuntimeError(f"M1_MANIFEST_PATH_ESCAPE:{relative}")
        rows.append({"path": relative, "sha256": str(row["sha256"]), "size": int(row["size"])})
    expected_paths = {row["path"] for row in rows}
    repo_id = str(suite["repo_id"])
    revision = str(suite["revision"])
    source_base = f"{args.endpoint.rstrip('/')}/{quote(repo_id, safe='/')}/resolve/{revision}"
    expected_suite_manifest = str(suite.get("server_manifest_sha256") or suite.get("local_manifest_sha256") or "")
    receipt = {
        "schema": "STAGE_Z_Z3R1_M1_MATERIALIZATION_RECOVERY_RECEIPT_V1",
        "status": "RUNNING",
        "started_utc": now(),
        "suite": args.suite,
        "repo_id": repo_id,
        "revision": revision,
        "endpoint": args.endpoint.rstrip("/"),
        "source_base": source_base,
        "manifest_path": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "suite_manifest_sha256": expected_suite_manifest,
        "expected_files": len(rows),
        "expected_bytes": sum(row["size"] for row in rows),
        "target": str(args.target),
        "staging": str(args.staging),
        "retention": "RETAIN_UNTIL_Z3_SENTINEL_AND_DECISION",
        "scientific_exposure": 0,
        "runtime_counters": {"model_inference_calls": 0, "env_step_calls": 0, "scientific_parent_exposure": 0},
    }
    write_json(args.receipt, receipt)
    args.staging.mkdir(parents=True, exist_ok=True)

    try:
        for index, row in enumerate(rows, start=1):
            output = args.staging / row["path"]
            output.parent.mkdir(parents=True, exist_ok=True)
            url = f"{source_base}/{quote(row['path'], safe='/')}"
            partial = output.with_name(output.name + ".partial")
            if output.exists() and (output.stat().st_size != row["size"] or sha256_file(output) != row["sha256"]):
                output.unlink()
            if not output.exists():
                if partial.exists() and partial.stat().st_size > row["size"]:
                    partial.unlink()
                subprocess.run(
                    [
                        "curl", "--fail", "--location", "--retry", "8", "--retry-all-errors",
                        "--connect-timeout", "30", "--continue-at", "-", "--output", str(partial), url,
                    ],
                    check=True,
                )
                if partial.stat().st_size != row["size"] or sha256_file(partial) != row["sha256"]:
                    raise RuntimeError(f"Z3R1_DOWNLOAD_MANIFEST_MISMATCH:{row['path']}")
                partial.replace(output)
            receipt["completed_files"] = index
            receipt["completed_bytes"] = sum(item["size"] for item in rows[:index])
            receipt["last_completed_path"] = row["path"]
            write_json(args.receipt, receipt)

        actual = {
            path.relative_to(args.staging).as_posix(): path
            for path in args.staging.rglob("*")
            if path.is_file()
        }
        if set(actual) != expected_paths:
            raise RuntimeError("Z3R1_CHECKPOINT_FILE_SET_MISMATCH")
        for row in rows:
            path = actual[row["path"]]
            if path.stat().st_size != row["size"] or sha256_file(path) != row["sha256"]:
                raise RuntimeError(f"Z3R1_CHECKPOINT_MANIFEST_MISMATCH:{row['path']}")
        args.staging.replace(args.target)
        receipt.update(
            {
                "status": "PASS_EXACT_RETAINED",
                "finished_utc": now(),
                "completed_files": len(rows),
                "completed_bytes": sum(row["size"] for row in rows),
                "materialized_path": str(args.target),
            }
        )
        receipt.pop("staging", None)
        write_json(args.receipt, receipt)
    except Exception as exc:
        receipt.update({"status": "ENGINEERING_INVALID_MATERIALIZATION_FAILURE", "finished_utc": now(), "error": repr(exc)})
        write_json(args.receipt, receipt)
        raise


if __name__ == "__main__":
    main()
