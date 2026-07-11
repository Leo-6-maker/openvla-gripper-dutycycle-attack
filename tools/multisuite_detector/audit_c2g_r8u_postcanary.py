#!/usr/bin/env python3
"""R8U post-canary L10 Teacher-v2 unknown reason audit — read-only, CPU only.

Reads the 24 existing R8T canary step_records.jsonl and produces per-step
unknown-reason decomposition without modifying Teacher-v2 label semantics.
"""
from __future__ import annotations

import argparse, json, csv, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="R8U Teacher-v2 unknown reason audit")
    ap.add_argument("--run-root", required=True, help="R8T GPU collection root")
    ap.add_argument("--output-dir", required=True, help="Output directory (must not exist)")
    args = ap.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(str(output_dir))
    output_dir.mkdir(parents=True)

    episode_dirs = sorted(run_root.glob("shards/*/clean_collection/episodes/**/step_records.jsonl"))
    episode_dirs = [p.parent for p in episode_dirs]

    ledger: List[dict] = []
    reason_counts: Dict[str, int] = defaultdict(int)
    l10_decomp: Dict[str, int] = defaultdict(int)
    per_suite_task_state: Dict[str, dict] = defaultdict(
        lambda: {"total": 0, "unknown": 0, "before_first_contact": 0, "after_first_contact": 0,
                 "single_binding": 0, "multi_binding": 0}
    )

    for ep_dir in episode_dirs:
        meta = read_json(ep_dir / "episode_metadata.json")
        steps = read_jsonl(ep_dir / "step_records.jsonl")
        suite = meta["suite"]
        task_index = meta["task_index"]
        state_id = meta["state_id"]
        parent_key = meta["parent_key"]
        bindings = meta.get("goal_event_bindings", [])
        binding_count = len(bindings)

        total_steps = len(steps)
        unknown_steps = 0
        first_resolved_step: int | None = None

        for i, row in enumerate(steps):
            known = row.get("label_known_mask", row.get("teacher_known", True))
            reason = row.get("teacher_reason_code", "UNKNOWN")
            phase = row.get("teacher_phase", "UNKNOWN")
            active_reason = row.get("active_target_reason", "")
            active_known = row.get("active_target_known", True)
            subgoal_idx = row.get("active_subgoal_index", -1)
            contacted = row.get("contacted_goal_targets", 0)
            bilateral = row.get("bilateral_goal_targets", 0)

            if not known:
                unknown_steps += 1
                reason_counts[reason] += 1

                # L10 decomposition
                if suite == "libero_10":
                    if first_resolved_step is None and active_known:
                        first_resolved_step = i
                    before = first_resolved_step is None or i < first_resolved_step
                    if before:
                        l10_decomp["unknown_before_first_resolved_target"] += 1
                    else:
                        l10_decomp["unknown_after_first_resolved_target"] += 1
                    if binding_count > 1:
                        l10_decomp["unknown_multi_binding"] += 1
                    else:
                        l10_decomp["unknown_single_binding"] += 1
                    if contacted == 0:
                        l10_decomp["unknown_no_goal_target_contact"] += 1
                    elif contacted > 1:
                        l10_decomp["unknown_multiple_contacted_targets"] += 1
                    reason_lower = reason.lower()
                    if "contact" in reason_lower:
                        l10_decomp["unknown_contact_semantics"] += 1
                    if "progress" in reason_lower:
                        l10_decomp["unknown_progress_semantics"] += 1
                    if "release" in reason_lower:
                        l10_decomp["unknown_release_semantics"] += 1
                    if "target" in reason_lower and "unresolved" in reason_lower:
                        l10_decomp["unknown_target_resolution"] += 1
                    if "mechanism" in reason_lower or "unsupported" in reason_lower:
                        l10_decomp["unknown_unsupported_mechanism"] += 1

            ledger.append({
                "suite": suite, "task_index": task_index, "state_id": state_id,
                "parent_key": parent_key, "step": i,
                "teacher_reason_code": reason,
                "teacher_phase": phase,
                "label_known_mask": known,
                "active_target_reason": active_reason,
                "active_target_known": active_known,
                "active_subgoal_index": subgoal_idx,
                "goal_binding_count": binding_count,
                "contacted_goal_targets": contacted,
                "bilateral_goal_targets": bilateral,
            })

        suite_key = f"{suite}/task_{task_index}/state_{state_id}"
        st = per_suite_task_state[suite_key]
        st["total"] = total_steps
        st["unknown"] = unknown_steps

    # Write outputs
    ledger_fields = ["suite", "task_index", "state_id", "parent_key", "step",
                     "teacher_reason_code", "teacher_phase", "label_known_mask",
                     "active_target_reason", "active_target_known", "active_subgoal_index",
                     "goal_binding_count", "contacted_goal_targets", "bilateral_goal_targets"]
    write_csv(output_dir / "r8u_teacher_unknown_reason_ledger.csv", ledger, ledger_fields)

    with open(output_dir / "r8u_teacher_unknown_reason_counts.json", "w") as f:
        json.dump({
            "reason_counts": dict(sorted(reason_counts.items())),
            "l10_decomposition": dict(sorted(l10_decomp.items())),
            "per_suite_task_state": {k: dict(v) for k, v in sorted(per_suite_task_state.items())},
            "total_steps": len(ledger),
        }, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "status": "PASS_C2G_R8U_TEACHER_V2_UNKNOWN_AUDIT",
        "total_steps": len(ledger),
        "unknown_steps": sum(1 for r in ledger if not r["label_known_mask"]),
        "l10_decomposition": dict(sorted(l10_decomp.items())),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
