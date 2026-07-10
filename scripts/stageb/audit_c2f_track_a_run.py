#!/usr/bin/env python3
"""CPU-only C2F Track A completion audit."""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


CONDITIONS = ["CLEAN", "TRUE_CMDOPEN_T10_C2F", "RAND_ACTION_NOISE_T10_C2F"]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_runtime_valid_metadata(path: Path) -> bool:
    try:
        return bool(load_json(path).get("runtime_valid") is True)
    except Exception:
        return False


def archive_invalid_attempt(output_root: Path, parent_key: str, condition: str, archive_root: Path) -> bool:
    ep_dir = output_root / parent_key / condition
    meta = ep_dir / "episode_metadata.json"
    if not meta.exists() or is_runtime_valid_metadata(meta):
        return False
    dest = archive_root / parent_key / condition
    i = 1
    while (dest / f"attempt_{i:03d}").exists():
        i += 1
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(str(ep_dir), str(dest / f"attempt_{i:03d}"))
    return True


def audit_run(run_root: Path, output_root: Path, parent_manifest: Path, jobs_file: Path | None = None) -> Dict[str, Any]:
    if jobs_file:
        expected_jobs = [
            {"parent_key": p, "condition": c}
            for p, c, *_ in (line.split("|") for line in jobs_file.read_text(encoding="utf-8").splitlines() if line.strip())
        ]
        expected_parents = sorted({j["parent_key"] for j in expected_jobs})
    else:
        expected_parents = [load_json_line["parent_key"] for load_json_line in (
            json.loads(line) for line in parent_manifest.read_text(encoding="utf-8").splitlines() if line.strip()
        )]
        expected_jobs = [{"parent_key": p, "condition": c} for p in expected_parents for c in CONDITIONS]
    metas: List[Dict[str, Any]] = []
    invalid: List[Dict[str, str]] = []
    for path in sorted(output_root.glob("**/episode_metadata.json")):
        try:
            meta = load_json(path)
        except Exception as exc:
            invalid.append({"path": str(path), "error": str(exc)})
            continue
        meta["_path"] = str(path)
        metas.append(meta)
        if meta.get("runtime_valid") is not True:
            invalid.append({
                "path": str(path),
                "parent_key": str(meta.get("parent_key", "")),
                "condition": str(meta.get("condition", "")),
                "error_type": str(meta.get("error_type", "")),
                "error_message": str(meta.get("error_message", "")),
            })

    valid = [m for m in metas if m.get("runtime_valid") is True]
    parents_done = defaultdict(set)
    for m in valid:
        parents_done[m.get("parent_key")].add(m.get("condition"))

    missing = []
    for job in expected_jobs:
        if job["condition"] not in parents_done[job["parent_key"]]:
            missing.append(job)

    delivery = [int(m.get("delivery_count", 0)) for m in valid if m.get("condition") in CONDITIONS[1:]]
    no_emit = [
        {"parent_key": m.get("parent_key"), "condition": m.get("condition")}
        for m in valid
        if m.get("condition") in CONDITIONS[1:] and int(m.get("attack_window_start", -1)) < 0
    ]
    audit = {
        "expected_parents": len(expected_parents),
        "expected_episodes": len(expected_jobs),
        "metadata_count": len(metas),
        "valid_episode_count": len(valid),
        "invalid_episode_count": len(invalid),
        "by_condition_valid": dict(Counter(m.get("condition") for m in valid)),
        "by_suite_valid": dict(Counter(m.get("suite") for m in valid)),
        "missing": missing,
        "invalid": invalid,
        "complete": len(missing) == 0 and len(invalid) == 0,
        "no_emit": no_emit,
        "delivery_count_min": min(delivery) if delivery else None,
        "delivery_count_max": max(delivery) if delivery else None,
        "delivery_count_mean": sum(delivery) / len(delivery) if delivery else None,
    }
    (run_root / "postrun_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = ["parent_key,suite,task_index,state_id,condition,success,runtime_valid,total_steps,attack_window_start,attack_window_end,delivery_count"]
    for m in valid:
        rows.append(",".join(str(m.get(k, "")) for k in [
            "parent_key", "suite", "task_index", "state_id", "condition", "success", "runtime_valid",
            "total_steps", "attack_window_start", "attack_window_end", "delivery_count",
        ]))
    (run_root / "summary_table.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-complete")
    ap.add_argument("--archive-invalid-output-root")
    ap.add_argument("--parent-key")
    ap.add_argument("--condition")
    ap.add_argument("--invalid-archive-root")
    ap.add_argument("--run-root")
    ap.add_argument("--output-root")
    ap.add_argument("--parent-manifest")
    ap.add_argument("--jobs-file")
    args = ap.parse_args()

    if args.metadata_complete:
        return 0 if is_runtime_valid_metadata(Path(args.metadata_complete)) else 1
    if args.archive_invalid_output_root:
        moved = archive_invalid_attempt(
            Path(args.archive_invalid_output_root),
            args.parent_key or "",
            args.condition or "",
            Path(args.invalid_archive_root or "invalid_attempts"),
        )
        print(json.dumps({"archived": moved}, sort_keys=True))
        return 0
    audit = audit_run(
        Path(args.run_root),
        Path(args.output_root),
        Path(args.parent_manifest),
        Path(args.jobs_file) if args.jobs_file else None,
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
