"""Deterministic FIT-only window baselines.

The selectors operate on the same already-built candidate-window records as
V5.  They do not inspect Teacher fields except for offline scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import uuid
from pathlib import Path
from typing import Any, Sequence


def choose_window(windows: Sequence[dict[str, Any]], method: str, *, seed: int = 20260717) -> int | None:
    if not windows:
        return None
    if method == "B0_RANDOM":
        return random.Random(seed).randrange(len(windows))
    if method == "B1_EARLIEST":
        return min(range(len(windows)), key=lambda i: (int(windows[i]["start_step"]), str(windows[i].get("window_id", ""))))
    if method == "B2_LATEST":
        return max(range(len(windows)), key=lambda i: (int(windows[i]["start_step"]), str(windows[i].get("window_id", ""))))
    if method == "B3_LONGEST":
        return max(range(len(windows)), key=lambda i: (int(windows[i]["step_count"]), -int(windows[i]["start_step"])))
    if method == "B4_MAX_TIME_SINCE_CLOSE":
        if not any("time_since_close" in row for row in windows):
            return None
        return max(range(len(windows)), key=lambda i: float(windows[i].get("time_since_close", 0.0)))
    if method == "B5_MAX_CLOSE_STREAK":
        if not any("recent_close_streak" in row for row in windows):
            return None
        return max(range(len(windows)), key=lambda i: float(windows[i].get("recent_close_streak", 0.0)))
    if method in {"B6_V4_C0", "B7_V4_C2"}:
        field = "v4_c0_score" if method == "B6_V4_C0" else "v4_c2_score"
        if not any(field in row for row in windows):
            return None
        return max(range(len(windows)), key=lambda i: float(windows[i].get(field, float("-inf"))))
    raise ValueError(f"unknown baseline: {method}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-root", type=Path)
    parser.add_argument("--windows-json", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    if args.geometry_root is not None:
        if args.output_root is None:
            raise ValueError("--output-root is required with --geometry-root")
        window_rows = list(csv.DictReader((args.geometry_root / "DETECTOR_V5_WINDOW_GEOMETRY.csv").open(newline="", encoding="utf-8")))
        episode_rows = {row["canonical_parent_key"]: row for row in csv.DictReader((args.geometry_root / "DETECTOR_V5_EPISODE_GEOMETRY.csv").open(newline="", encoding="utf-8"))}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in window_rows:
            row["start_step"] = int(row["start_step"])
            row["step_count"] = int(row["step_count"])
            row["utility_tier"] = int(row["utility_tier"])
            grouped.setdefault(row["canonical_parent_key"], []).append(row)
        results: dict[str, Any] = {}
        for method in ("B0_RANDOM", "B1_EARLIEST", "B2_LATEST", "B3_LONGEST", "B4_MAX_TIME_SINCE_CLOSE", "B5_MAX_CLOSE_STREAK", "B6_V4_C0", "B7_V4_C2"):
            rows_out = []
            for identity, windows in sorted(grouped.items()):
                selected = choose_window(windows, method, seed=args.seed)
                if selected is None:
                    continue
                selected_tier = int(windows[selected]["utility_tier"])
                max_tier = max(int(window["utility_tier"]) for window in windows)
                category = episode_rows[identity]["category"]
                rows_out.append({"canonical_parent_key": identity, "category": category, "selected_tier": selected_tier, "top1_hit": selected_tier == max_tier})
            mixed = [row for row in rows_out if row["category"] == "TRUE_MIXED"]
            pure = [row for row in rows_out if row["category"] == "PURE_NEGATIVE"]
            results[method] = {
                "selected_episode_count": len(rows_out),
                "true_mixed_count": len(mixed),
                "true_mixed_top1_hit_rate": (sum(row["top1_hit"] for row in mixed) / len(mixed)) if mixed else None,
                "pure_negative_abstention_rate": 0.0 if pure else None,
                "identity_rows": rows_out,
            }
        output = args.output_root.resolve()
        if output.exists():
            raise FileExistsError(output)
        staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
        try:
            staging.mkdir(parents=True)
            (staging / "baseline_results.json").write_text(json.dumps({"schema": "DETECTOR_V5_WINDOW_BASELINES_V2", "results": results}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            from gripper_attack.b3_training_protocol import seal_directory
            seal_directory(staging)
            os.replace(staging, output)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        print(json.dumps(results, indent=2, sort_keys=True))
        return 0
    if args.windows_json is None or args.output_json is None:
        raise ValueError("provide --geometry-root/--output-root or --windows-json/--output-json")
    payload = json.loads(args.windows_json.read_text(encoding="utf-8"))
    results = {}
    for method in ("B0_RANDOM", "B1_EARLIEST", "B2_LATEST", "B3_LONGEST", "B4_MAX_TIME_SINCE_CLOSE", "B5_MAX_CLOSE_STREAK", "B6_V4_C0", "B7_V4_C2"):
        results[method] = [choose_window(windows, method, seed=args.seed) for windows in payload]
    args.output_json.write_text(json.dumps({"schema": "DETECTOR_V5_WINDOW_BASELINES_V2", "results": results}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
