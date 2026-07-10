#!/usr/bin/env python3
"""Bind evaluation parents to CLEAN detector runs and official init-state hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROTOCOL_NAME = "C2G_CLEAN_WINDOW_VIS_PGD"
PROTOCOL_VERSION = "2026-07-10.v1"


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value.get("parents", value.get("episodes", value)) if isinstance(value, dict) else value
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("manifest must contain a list of objects")
    return [dict(row) for row in rows]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_file_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        content_sha = sha256_file(path)
        digest.update(f"{path.name}|{path.stat().st_size}|{content_sha}\n".encode("utf-8"))
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(b"|")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(b"|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--clean-output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    args = parser.parse_args(argv)

    from libero.libero import benchmark

    rows = read_rows(args.parents.resolve())
    bound: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    suite_cache: dict[str, Any] = {}
    for source in rows:
        try:
            suite = str(source["suite"])
            task_index = int(source["task_index"])
            state_id = int(source["state_id"])
            parent_key = str(source.get("parent_key") or f"{suite}/task_{task_index}/state_{state_id}")
            metadata_path = args.clean_output_root.resolve() / parent_key / "CLEAN" / "episode_metadata.json"
            steps_path = metadata_path.with_name("step_records.jsonl")
            if not metadata_path.is_file() or not steps_path.is_file() or steps_path.stat().st_size == 0:
                raise FileNotFoundError("CLEAN detector output is incomplete")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not bool(metadata.get("runtime_valid")):
                raise ValueError("CLEAN detector output runtime_valid=false")
            if metadata.get("protocol_name") != PROTOCOL_NAME or metadata.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError("CLEAN detector output protocol mismatch")
            if metadata.get("git_commit") != args.expected_git_commit:
                raise ValueError("CLEAN detector output commit mismatch")
            for field, expected in (
                ("parent_key", parent_key), ("condition", "CLEAN"), ("suite", suite),
                ("task_index", task_index), ("state_id", state_id),
            ):
                if metadata.get(field) != expected:
                    raise ValueError(f"CLEAN detector identity mismatch for {field}")
            if suite not in suite_cache:
                suite_cache[suite] = benchmark.get_benchmark_dict()[suite]()
            init_states = suite_cache[suite].get_task_init_states(task_index)
            if state_id < 0 or state_id >= len(init_states):
                raise IndexError("state_id outside official init-state range")
            initial_state_sha = array_sha256(init_states[state_id])
            clean_parent_sha = combined_file_sha256((metadata_path, steps_path))
            bound.append(
                {
                    "parent_key": parent_key,
                    "suite": suite,
                    "task_index": task_index,
                    "state_id": state_id,
                    "eval_seed": int(source.get("eval_seed", source.get("seed", 42))),
                    "max_steps": int(source.get("max_steps", metadata.get("total_steps", 300) or 300)),
                    "clean_parent_sha256": clean_parent_sha,
                    "initial_state_sha256": initial_state_sha,
                    "clean_metadata_sha256": sha256_file(metadata_path),
                    "clean_steps_sha256": sha256_file(steps_path),
                    "clean_success": metadata.get("success"),
                    "detector_start_step": next(
                        (
                            int(row["step"])
                            for row in (
                                json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines() if line.strip()
                            )
                            if bool(row.get("trigger_started"))
                        ),
                        None,
                    ),
                }
            )
        except Exception as exc:
            errors.append({"source": json.dumps(source, sort_keys=True), "error": f"{type(exc).__name__}: {exc}"})
    if errors or not bound:
        report = {"status": "HOLD_C2G_EVAL_PARENTS", "parent_count": len(bound), "errors": errors}
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    if len({row["parent_key"] for row in bound}) != len(bound):
        raise ValueError("duplicate parent_key")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in bound), encoding="utf-8")
    report = {
        "status": "PASS_C2G_EVAL_PARENTS_BOUND",
        "parent_count": len(bound),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output.resolve()),
        "expected_git_commit": args.expected_git_commit,
    }
    args.output.with_suffix(args.output.suffix + ".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
