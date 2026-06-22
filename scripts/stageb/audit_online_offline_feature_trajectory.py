#!/usr/bin/env python3
"""Compare frozen offline SC5 features with online restore-runner telemetry.

CPU-only. Does not load OpenVLA, LIBERO, GPU, or attack code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from gripper_attack.sc5_detector_runtime import SC5_FEATURES


RAW_FIELDS = [
    "gripper_qpos",
    "gripper_opening_proxy",
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_vx",
    "eef_vy",
    "eef_vz",
    "action_dx",
    "action_dy",
    "action_dz",
    "action_gripper",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def finite_abs_diff(a: float, b: float) -> float:
    if math.isfinite(a) and math.isfinite(b):
        return abs(a - b)
    if math.isnan(a) and math.isnan(b):
        return 0.0
    return float("inf")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    dataset = Path(args.dataset)
    online = Path(args.online_telemetry)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    dataset_rows = [
        row
        for row in read_csv(dataset)
        if row.get("episode_key") == args.episode_key
    ]
    online_rows = read_csv(online)
    if not dataset_rows:
        raise ValueError(f"episode_key not found in dataset: {args.episode_key}")
    by_offline = {int(float(row["step"])): row for row in dataset_rows}
    by_online = {int(float(row["step"])): row for row in online_rows}
    shared_steps = sorted(set(by_offline) & set(by_online))
    if not shared_steps:
        raise ValueError("no shared steps")

    diff_rows: list[dict[str, Any]] = []
    feature_summary: dict[str, dict[str, Any]] = {}
    fields = [(name, name, "raw") for name in RAW_FIELDS]
    fields += [(name, "f_" + name, "feature25d") for name in SC5_FEATURES]
    for offline_key, online_key, group in fields:
        max_abs = -1.0
        first_diff_step = ""
        count_gt_tol = 0
        for step in shared_steps:
            off = as_float(by_offline[step], offline_key)
            on = as_float(by_online[step], online_key)
            diff = finite_abs_diff(off, on)
            if diff > max_abs:
                max_abs = diff
            if diff > args.tolerance:
                count_gt_tol += 1
                if first_diff_step == "":
                    first_diff_step = step
            diff_rows.append(
                {
                    "step": step,
                    "group": group,
                    "field": offline_key,
                    "offline": off,
                    "online": on,
                    "abs_diff": diff,
                    "gt_tolerance": diff > args.tolerance,
                }
            )
        feature_summary[offline_key] = {
            "group": group,
            "max_abs_diff": max_abs,
            "count_gt_tolerance": count_gt_tol,
            "first_diff_step": first_diff_step,
        }

    offline_emit = ""
    online_emit = ""
    for step in sorted(by_offline):
        # Offline dataset rows do not store detector outputs; leave explicit.
        pass
    for step, row in sorted(by_online.items()):
        if row.get("detector_state_after") == "EMITTED" or row.get("detector_emitted_after") == "True":
            online_emit = step
            break

    raw_bad = sum(1 for f in RAW_FIELDS if feature_summary[f]["count_gt_tolerance"] > 0)
    feature_bad = sum(1 for f in SC5_FEATURES if feature_summary[f]["count_gt_tolerance"] > 0)
    summary = {
        "episode_key": args.episode_key,
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "online_telemetry": str(online),
        "online_telemetry_sha256": sha256_file(online),
        "offline_row_count": len(dataset_rows),
        "online_row_count": len(online_rows),
        "shared_step_count": len(shared_steps),
        "first_shared_step": shared_steps[0],
        "last_shared_step": shared_steps[-1],
        "tolerance": args.tolerance,
        "raw_fields_with_diff": raw_bad,
        "feature25d_fields_with_diff": feature_bad,
        "online_emit_step": online_emit,
        "classification": "FEATURE_TRAJECTORY_MATCH" if raw_bad == 0 and feature_bad == 0 else "FEATURE_TRAJECTORY_DRIFT",
        "field_summary": feature_summary,
    }
    write_csv(out / "online_offline_feature_step_diffs.csv", diff_rows)
    write_json(out / "online_offline_feature_summary.json", summary)
    report = [
        "# Online/Offline SC5 Feature Trajectory Audit",
        "",
        f"- Episode: `{args.episode_key}`",
        f"- Shared steps: {len(shared_steps)} ({shared_steps[0]}..{shared_steps[-1]})",
        f"- Tolerance: `{args.tolerance}`",
        f"- Raw fields with diff: {raw_bad}/{len(RAW_FIELDS)}",
        f"- 25D fields with diff: {feature_bad}/{len(SC5_FEATURES)}",
        f"- Online emit step: `{online_emit}`",
        f"- Classification: `{summary['classification']}`",
        "",
        "Largest field diffs:",
    ]
    top = sorted(feature_summary.items(), key=lambda kv: float(kv[1]["max_abs_diff"]), reverse=True)[:12]
    for name, stats in top:
        report.append(
            f"- `{name}` ({stats['group']}): max_abs_diff={stats['max_abs_diff']} "
            f"count_gt_tol={stats['count_gt_tolerance']} first={stats['first_diff_step']}"
        )
    report.append("")
    report.append("No GPU, LIBERO, VIS, RAND, shuffled, oracle, or attack code was executed.")
    (out / "ONLINE_OFFLINE_FEATURE_TRAJECTORY_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--online-telemetry", required=True)
    p.add_argument("--episode-key", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tolerance", type=float, default=1e-6)
    return p.parse_args()


def main() -> None:
    print(json.dumps(audit(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
