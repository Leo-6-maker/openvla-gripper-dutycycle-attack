#!/usr/bin/env python3
"""Build the frozen 2,000-row official LIBERO task/state manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
from collections import Counter
from pathlib import Path


SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
STATES_PER_TASK = 50
TASKS_PER_SUITE = 10
SPLITS = {
    "FIT": range(0, 24),
    "CAL": range(24, 27),
    "CHECK": range(27, 30),
    "FINAL_EVAL_CANDIDATE": range(30, 50),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def state_sha256(state: object) -> str:
    return sha256_bytes(pickle.dumps(state, protocol=4))


def split_for_state(state_id: int) -> str:
    for name, values in SPLITS.items():
        if state_id in values:
            return name
    raise ValueError(f"state outside official split: {state_id}")


def csv_bytes(rows: list[dict[str, object]], fields: list[str]) -> bytes:
    from io import StringIO

    buf = StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def write_hashed_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> str:
    data = csv_bytes(rows, fields)
    path.write_bytes(data)
    digest = sha256_bytes(data)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--upstream-root", required=True, type=Path)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import sys

    sys.path.insert(0, str(args.upstream_root))
    from libero.libero import benchmark

    horizons = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
    }
    rows: list[dict[str, object]] = []
    for suite in SUITES:
        task_suite = benchmark.get_benchmark_dict()[suite]()
        if int(task_suite.n_tasks) != TASKS_PER_SUITE:
            raise SystemExit(f"OFFICIAL_TASK_COUNT_FAIL {suite}: {task_suite.n_tasks}")
        for task_idx in range(TASKS_PER_SUITE):
            task = task_suite.get_task(task_idx)
            states = task_suite.get_task_init_states(task_idx)
            if len(states) < STATES_PER_TASK:
                raise SystemExit(f"OFFICIAL_STATE_COUNT_FAIL {suite} task={task_idx}: {len(states)}")
            for state_id in range(STATES_PER_TASK):
                canonical = f"{suite}/task_{task_idx:02d}/state_{state_id:02d}"
                rows.append(
                    {
                        "suite": suite,
                        "task_idx": task_idx,
                        "task_name": str(task.name),
                        "task_language": str(task.language),
                        "state_id": state_id,
                        "canonical_parent_key": canonical,
                        "split": split_for_state(state_id),
                        "official_horizon": horizons[suite],
                        "initial_state_sha256": state_sha256(states[state_id]),
                    }
                )

    fields = list(rows[0])
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "OFFICIAL_CLEAN_2000_MANIFEST.csv"
    manifest_sha = write_hashed_csv(manifest_path, rows, fields)
    split_path = args.output_root / "OFFICIAL_IDENTITY_SPLIT_V1.csv"
    split_sha = write_hashed_csv(split_path, rows, fields)

    keys = [(r["suite"], r["task_idx"], r["state_id"]) for r in rows]
    if len(rows) != 2000 or len(set(keys)) != 2000:
        raise SystemExit("OFFICIAL_MANIFEST_IDENTITY_FAIL")
    suite_counts = Counter(str(r["suite"]) for r in rows)
    task_counts = Counter((str(r["suite"]), int(r["task_idx"])) for r in rows)
    if any(suite_counts[s] != 500 for s in SUITES) or any(v != 50 for v in task_counts.values()):
        raise SystemExit("OFFICIAL_MANIFEST_BALANCE_FAIL")

    summary = {
        "status": "OFFICIAL_CLEAN_MANIFEST_FROZEN",
        "rows": len(rows),
        "suites": dict(suite_counts),
        "tasks_per_suite": TASKS_PER_SUITE,
        "states_per_task": STATES_PER_TASK,
        "splits": {k: len(list(v)) * 40 for k, v in SPLITS.items()},
        "manifest_sha256": manifest_sha,
        "identity_split_sha256": split_sha,
        "horizons": horizons,
    }
    (args.output_root / "OFFICIAL_CLEAN_MANIFEST_AUDIT.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
