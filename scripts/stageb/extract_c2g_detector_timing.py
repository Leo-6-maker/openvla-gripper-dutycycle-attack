#!/usr/bin/env python3
"""Extract preregistered detector starts from CLEAN detector-only rollout logs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-trigger", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    root = args.clean_output_root.resolve()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for metadata_path in sorted(root.rglob("CLEAN/episode_metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not bool(metadata.get("runtime_valid")):
                raise ValueError("runtime_valid is false")
            step_path = metadata_path.with_name("step_records.jsonl")
            steps = read_jsonl(step_path)
            starts = [int(row["step"]) for row in steps if bool(row.get("trigger_started"))]
            if len(starts) > 1:
                raise ValueError(f"multiple detector starts: {starts}")
            if not starts and args.require_trigger:
                raise ValueError("detector did not trigger")
            rows.append(
                {
                    "parent_key": str(metadata["parent_key"]),
                    "detector_start_step": starts[0] if starts else -1,
                    "suite": str(metadata["suite"]),
                    "task_index": int(metadata["task_index"]),
                    "state_id": int(metadata["state_id"]),
                    "clean_metadata_sha256": sha256_file(metadata_path),
                    "clean_steps_sha256": sha256_file(step_path),
                }
            )
        except Exception as exc:
            errors.append({"path": str(metadata_path), "error": f"{type(exc).__name__}: {exc}"})
    if errors or not rows:
        print(json.dumps({"status": "HOLD_C2G_DETECTOR_TIMING", "rows": len(rows), "errors": errors}, indent=2))
        return 2
    if len({row["parent_key"] for row in rows}) != len(rows):
        raise ValueError("duplicate parent_key in CLEAN timing outputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    report = {
        "status": "PASS_C2G_DETECTOR_TIMING_EXTRACTED",
        "parent_count": len(rows),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output.resolve()),
    }
    args.output.with_suffix(args.output.suffix + ".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
