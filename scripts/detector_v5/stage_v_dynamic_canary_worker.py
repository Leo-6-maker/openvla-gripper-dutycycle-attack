"""Synthetic artifact producer used only for the control-plane canary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

try:
    from .stage_v_dynamic_common import atomic_write_json, utc_now
except ImportError:
    from stage_v_dynamic_common import atomic_write_json, utc_now


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-key", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--sleep-seconds", type=float, default=0)
    args = parser.parse_args(argv)
    if args.sleep_seconds > 0:
        time.sleep(args.sleep_seconds)
    suite, task, state = args.parent_key.split("/")
    task_index = int(task.removeprefix("task_"))
    state_index = int(state.removeprefix("state_"))
    out = args.output_dir / suite / task / state
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for probe_step in range(24):
        for arm in ("CLEAN", "OPEN_T3", "OPEN_T5"):
            rows.append({
                "canonical_parent_key": args.parent_key, "probe_step": probe_step, "k": 0,
                "arm": arm, "status": "PASS", "task_success": True,
                "eval160_reads": 0, "protected_eval_reads": 0, "attack_rollouts": 0,
            })
    (out / "COUNTERFACTUAL_BRANCHES.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (out / "CLEAN_TRACE.jsonl").write_text(json.dumps({"step": 0, "status": "PASS"}) + "\n", encoding="utf-8")
    atomic_write_json(out / "PARENT_RESULT.json", {
        "schema": "D8_STAGE_V_COUNTERFACTUAL_PARENT_RESULT_V1",
        "status": "PASS", "canonical_parent_key": args.parent_key,
        "suite": suite, "task_index": task_index, "state_index": state_index,
        "clean_success": True, "branch_count": 72, "probe_count": 24,
        "branch_arms": ["CLEAN", "OPEN_T3", "OPEN_T5"],
        "current_source_commit": args.source_commit, "current_source_tree": args.source_tree,
        "exact_snapshot_replay": True, "label_status": "VALID",
        "eval160_reads": 0, "protected_eval_reads": 0, "attack_rollouts": 0,
        "generated_utc": utc_now(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
