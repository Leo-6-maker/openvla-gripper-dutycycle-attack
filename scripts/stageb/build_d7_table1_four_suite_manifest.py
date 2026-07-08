#!/usr/bin/env python3
"""D7 Table1 four-suite manifest builder.

Selects balanced parent episodes from CLEAN2000 across four LIBERO suites.
Outputs preregistered queue manifest for CLEAN/TRUE_T10/RAND_T10/COMMAND_OPEN_ORACLE.

CPU-only. No OpenVLA. No LIBERO. No env.step.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, random, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ============ Constants ============
SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
CONDITIONS = ["CLEAN", "TRUE_T10", "RAND_T10", "COMMAND_OPEN_ORACLE"]
DEFAULT_N_PER_SUITE = 50
RANDOM_SEED = 42

# Task index constraints per suite (from LIBERO benchmark)
SUITE_TASK_RANGES = {
    "libero_10": range(0, 10),
    "libero_goal": range(0, 10),
    "libero_object": range(0, 10),
    "libero_spatial": range(0, 10),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def is_attack_eligible(row: Dict[str, str]) -> bool:
    """Episode is eligible if it has a primary/attackable event (runtime_emit_eligible=1)
    and is NOT suppressed as no-primary (no_primary_suppress=0)."""
    emit_eligible = row.get("runtime_emit_eligible", "0")
    no_suppress = row.get("no_primary_suppress", "1")
    return emit_eligible == "1" and no_suppress == "0"


def select_parents(
    context_rows: List[Dict[str, str]],
    n_per_suite: int,
    seed: int,
) -> Dict[str, List[Dict[str, str]]]:
    """Select balanced parent episodes per suite."""
    random.seed(seed)
    parents: Dict[str, List[Dict[str, str]]] = {s: [] for s in SUITES}

    # Filter eligible rows
    eligible = [r for r in context_rows if r.get("suite", "") in SUITES
                and is_attack_eligible(r) and is_attack_eligible(r)]

    # Group by suite
    suite_groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in eligible:
        suite_groups[r["suite"]].append(r)

    for suite in SUITES:
        group = suite_groups[suite]
        print(f"  {suite}: {len(group)} eligible parents")

        # Stratify by task if possible
        task_groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for r in group:
            task_idx = r.get("task_index", "0")
            task_groups[task_idx].append(r)

        # Select evenly from tasks
        n_tasks = len(task_groups)
        per_task = max(1, n_per_suite // max(1, n_tasks))

        selected: List[Dict[str, str]] = []
        for task_idx in sorted(task_groups.keys()):
            candidates = task_groups[task_idx]
            random.shuffle(candidates)
            selected.extend(candidates[:per_task])

        # Trim to exact count
        if len(selected) > n_per_suite:
            random.shuffle(selected)
            selected = selected[:n_per_suite]

        parents[suite] = selected
        print(f"  → selected {len(selected)} parents from {n_tasks} tasks")

    return parents


def build_manifest(
    parents: Dict[str, List[Dict[str, str]]],
    conditions: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build queue manifest and preregistered parent keys."""
    queue_rows: List[Dict[str, Any]] = []
    parent_keys: List[Dict[str, Any]] = []

    for suite in SUITES:
        for parent in parents[suite]:
            parent_key = parent.get("group_key", parent.get("record_id", ""))
            parent_keys.append({
                "suite": suite,
                "parent_key": parent_key,
                "task_index": parent.get("task_index", ""),
                "state_id": parent.get("state_id", ""),
            })

            for condition in conditions:
                queue_rows.append({
                    "suite": suite,
                    "condition": condition,
                    "parent_key": parent_key,
                    "group_key": parent.get("group_key", ""),
                    "task_index": parent.get("task_index", ""),
                    "state_id": parent.get("state_id", ""),
                    "record_id": parent.get("record_id", ""),
                    "temporal_path": parent.get("temporal_path", ""),
                    "label_status": parent.get("teacher_label_status", ""),
                    "split": parent.get("split", ""),
                    "is_attack_eligible": is_attack_eligible(parent),
                    "is_attack_eligible": is_attack_eligible(parent),
                    # C2e3 detector trigger info (placeholder — filled at bridge time)
                    "detector_trigger_step": "",
                    "detector_emit_p": "",
                    "detector_suppress_p": "",
                })

    return queue_rows, parent_keys


def main():
    ap = argparse.ArgumentParser(description="D7 Table1 four-suite manifest builder")
    ap.add_argument("--context-dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-per-suite", type=int, default=DEFAULT_N_PER_SUITE)
    ap.add_argument("--conditions", default="CLEAN,TRUE_T10,RAND_T10,COMMAND_OPEN_ORACLE")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--git-commit", required=True)
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    conditions = [c.strip() for c in args.conditions.split(",")]
    print(f"D7 Manifest: n={args.n_per_suite} suites={SUITES} conditions={conditions}")

    # Load context dataset
    context_rows = read_csv(args.context_dataset)
    print(f"  Context rows: {len(context_rows)}")

    # Select parents
    parents = select_parents(context_rows, args.n_per_suite, args.seed)

    # Build manifest
    queue_rows, parent_keys = build_manifest(parents, conditions)

    total_episodes = len(queue_rows)
    total_parents = len(parent_keys)
    print(f"\n  Total: {total_parents} parents → {total_episodes} episodes ({len(conditions)} conditions each)")

    # Summary by suite
    suite_counts = defaultdict(lambda: defaultdict(int))
    for r in queue_rows:
        suite_counts[r["suite"]][r["condition"]] += 1
    for suite in SUITES:
        parts = ", ".join(f"{c}={suite_counts[suite][c]}" for c in conditions)
        print(f"  {suite}: {parts}")

    # Write outputs
    queue_fields = ["suite", "condition", "parent_key", "group_key", "task_index",
                    "state_id", "record_id", "temporal_path", "label_status",
                    "split", "is_attack_eligible", "is_attack_eligible",
                    "detector_trigger_step", "detector_emit_p", "detector_suppress_p"]
    write_csv(out / "d7_table1_queue_manifest.csv", queue_rows, queue_fields)

    parent_fields = ["suite", "parent_key", "task_index", "state_id"]
    write_csv(out / "d7_table1_preregistered_parent_keys.csv", parent_keys, parent_fields)

    # Manifest report
    report = {
        "gate": "D7_TABLE1_MANIFEST",
        "status": "PASS_D7_MANIFEST_BUILT",
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - t0,
        "git_commit": args.git_commit,
        "n_per_suite": args.n_per_suite,
        "conditions": conditions,
        "total_parents": total_parents,
        "total_episodes": total_episodes,
        "suite_parent_counts": {s: len(parents.get(s, [])) for s in SUITES},
        "suite_condition_counts": {s: dict(suite_counts[s]) for s in SUITES},
        "seed": args.seed,
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
            "OpenVLA_model": "NOT_LOADED",
            "LIBERO_runtime": "NOT_PERFORMED",
            "env_step": "NOT_PERFORMED",
            "attack": "NOT_PERFORMED",
            "rollout": "NOT_PERFORMED",
        },
    }
    write_json(out / "d7_table1_manifest_report.json", report)

    # Checksums
    csums = {}
    for fn in sorted(out.glob("*")):
        if fn.is_file() and fn.name != "checksum_report.json":
            csums[fn.name] = sha256_file(fn)
    write_json(out / "checksum_report.json", csums)
    with open(out / "SHA256SUMS", "w") as f:
        for fn, sha in sorted(csums.items()):
            if fn not in ("SHA256SUMS", "SHA256SUMS.sha256"):
                f.write(f"{sha}  {fn}\n")
    (out / "SHA256SUMS.sha256").write_text(
        f"{sha256_file(out / 'SHA256SUMS')}  SHA256SUMS\n"
    )

    print(f"\nD7 Manifest built: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
