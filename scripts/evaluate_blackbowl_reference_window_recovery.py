#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import yaml


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_references(path: Path) -> dict[str, dict[str, tuple[int, int]]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    refs: dict[str, dict[str, tuple[int, int]]] = {}
    for task_id, states in (raw.get("references") or {}).items():
        refs[str(task_id)] = {}
        for state, window in states.items():
            refs[str(task_id)][str(state)] = (int(window[0]), int(window[1]))
    return refs


def interval_iou(start: str, end: str, fixed_start: int, fixed_end: int) -> tuple[str, str, str]:
    if start in ("", None) or end in ("", None):
        return "", "", ""
    start_i = int(start)
    end_i = int(end)
    intersection = max(0, min(end_i, fixed_end) - max(start_i, fixed_start) + 1)
    union = max(end_i, fixed_end) - min(start_i, fixed_start) + 1
    return str(intersection), str(union), str(intersection / union if union else 0.0)


def compare(candidates: list[dict[str, str]], refs: dict[str, dict[str, tuple[int, int]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in candidates:
        task_id = str(row.get("task_id", ""))
        state = str(row.get("state", ""))
        if task_id not in refs or state not in refs[task_id]:
            continue
        fixed_start, fixed_end = refs[task_id][state]
        auto_start = row.get("auto_window_start", "")
        auto_end = row.get("auto_window_end", "")
        intersection, union, iou = interval_iou(auto_start, auto_end, fixed_start, fixed_end)
        if auto_start in ("", None) or auto_end in ("", None):
            center_offset = ""
            strict = False
            loose = False
        else:
            auto_center = (int(auto_start) + int(auto_end)) / 2
            fixed_center = (fixed_start + fixed_end) / 2
            center_offset = abs(auto_center - fixed_center)
            confidence = row.get("confidence", "")
            strict = center_offset <= 5 and float(iou) >= 0.4 and confidence in {"high", "medium"}
            loose = center_offset <= 10 and float(iou) > 0 and confidence != "low"
        rows.append(
            {
                "run_id": row.get("run_id", ""),
                "task_id": task_id,
                "state": state,
                "seed": row.get("seed", ""),
                "fixed_start": fixed_start,
                "fixed_end": fixed_end,
                "auto_start": auto_start,
                "auto_end": auto_end,
                "center_offset": center_offset,
                "intersection_len": intersection,
                "union_len": union,
                "iou": iou,
                "confidence": row.get("confidence", ""),
                "detector_mode": row.get("detector_mode", ""),
                "pass_strict": strict,
                "pass_loose": loose,
                "notes": row.get("failure_reason", ""),
            }
        )
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]]) -> str:
    strict = {str(row["state"]) for row in rows if bool(row.get("pass_strict"))}
    loose = {str(row["state"]) for row in rows if bool(row.get("pass_loose"))}
    if rows and ({"5", "7"}.issubset(strict) or (strict and {"5", "7"}.issubset(strict | loose))):
        decision = "A. generic_detector_ready_for_blackbowl_autowindow_attack_sanity"
    elif rows:
        decision = "B. generic_detector_audit_passed_but_window_quality_needs_fix"
    else:
        decision = "D. artifact_or_signal_insufficient"
    lines = [
        "# Generic Auto-Window Black Bowl Reference Evaluation",
        "",
        f"Decision: {decision}",
        "",
        "Reference windows are evaluation-only and are not read by the generic detector.",
        "",
        "## Comparisons",
    ]
    for row in rows:
        lines.append(
            f"- State{row.get('state')} seed{row.get('seed')}: "
            f"fixed={row.get('fixed_start')}-{row.get('fixed_end')}, "
            f"auto={row.get('auto_start')}-{row.get('auto_end')}, "
            f"offset={row.get('center_offset')}, IoU={row.get('iou')}, "
            f"strict={row.get('pass_strict')}, loose={row.get('pass_loose')}, "
            f"mode={row.get('detector_mode')}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-hoc evaluation against Black Bowl reference windows.")
    parser.add_argument("--candidates_csv", required=True)
    parser.add_argument("--reference_config", default="configs/reference_windows_blackbowl_eval_only.yaml")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_md", required=True)
    args = parser.parse_args()

    candidates = read_csv(Path(args.candidates_csv))
    refs = load_references(Path(args.reference_config))
    rows = compare(candidates, refs)
    write_csv(Path(args.output_csv), rows)
    write_summary(Path(args.summary_md), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
