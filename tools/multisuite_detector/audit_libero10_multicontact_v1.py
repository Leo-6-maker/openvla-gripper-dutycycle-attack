#!/usr/bin/env python3
"""Audit LIBERO-10 long-horizon multi-contact support.

This tool is read-only and diagnostic-only. It inspects frozen detector dataset,
Label V2 rows, split assignments, and 25D clean features to explain why the
LIBERO-10 suite may have no positive single-window support under Label V2.

It never trains a detector, mutates artifacts, runs OpenVLA/LIBERO, performs
rollouts, interventions, or attacks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.multisuite_detector.detector_dataset_closure_v1 import (  # noqa: E402
    SC5_FEATURES,
    load_dataset_manifest,
    sha256_file,
)

TARGET_SUITE_DEFAULT = "libero_10"
SPLIT_COLUMNS = ["split_type", "fold_id", "group_id", "episode_key", "split"]


class Libero10AuditError(ValueError):
    pass


def fail(message: str) -> None:
    raise Libero10AuditError(message)


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            fail(f"{path.name}: empty CSV header")
        rows = []
        for line_no, row in enumerate(reader, start=2):
            if None in row:
                fail(f"{path.name}:{line_no}: extra cells")
            if any(v is None for v in row.values()):
                fail(f"{path.name}:{line_no}: missing cells")
            rows.append(row)
        return rows


def write_csv(path: str | Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_int(value: str, field: str, episode: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        fail(f"{episode}: {field} must be int")
    if str(parsed) != str(value):
        fail(f"{episode}: {field} must be canonical int")
    return parsed


def parse_float(value: str, field: str, episode: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        fail(f"{episode}: {field} must be finite float")
    if not math.isfinite(parsed):
        fail(f"{episode}: {field} must be finite float")
    return parsed


def read_labels(label_csv: str | Path) -> dict[str, dict[str, Any]]:
    rows = read_csv_rows(label_csv)
    required = {"episode_key", "suite", "task_id", "event_present", "window_valid", "window_start", "window_end"}
    if not rows or not required <= set(rows[0]):
        fail("label CSV missing required Label V2 support columns")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        ep = row["episode_key"]
        if ep in out:
            fail(f"duplicate label episode: {ep}")
        start = parse_int(row["window_start"], "window_start", ep)
        end = parse_int(row["window_end"], "window_end", ep)
        event_present = row["event_present"] == "true"
        window_valid = row["window_valid"] == "true"
        out[ep] = {
            "suite": row["suite"],
            "task_id": row["task_id"],
            "event_present": event_present,
            "window_valid": window_valid,
            "window_start": start,
            "window_end": end,
            "positive": event_present and window_valid and end > start,
        }
    return out


def read_splits(split_csv: str | Path | None, fold_id: str | None) -> dict[str, str]:
    if not split_csv:
        return {}
    rows = read_csv_rows(split_csv)
    if list(rows[0]) != SPLIT_COLUMNS:
        fail("split CSV header mismatch")
    if fold_id:
        rows = [row for row in rows if row["fold_id"] == fold_id]
    elif len({row["fold_id"] for row in rows}) > 1:
        fail("split CSV has multiple folds; pass --fold-id")
    out: dict[str, str] = {}
    for row in rows:
        ep = row["episode_key"]
        if ep in out:
            fail(f"duplicate split assignment: {ep}")
        out[ep] = row["split"]
    return out


def read_features(feature_csv: str | Path) -> dict[str, list[dict[str, float]]]:
    rows = read_csv_rows(feature_csv)
    required = {"episode_key", "step", *SC5_FEATURES}
    if not rows or not required <= set(rows[0]):
        missing = sorted(required - set(rows[0] if rows else []))
        fail(f"feature CSV missing columns: {missing}")
    by_ep: dict[str, list[dict[str, float]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for row in rows:
        ep = row["episode_key"]
        step = parse_int(row["step"], "step", ep)
        key = (ep, step)
        if key in seen:
            fail(f"duplicate feature row: {ep}:{step}")
        seen.add(key)
        feat = {name: parse_float(row[name], name, ep) for name in SC5_FEATURES}
        feat["step"] = float(step)
        by_ep[ep].append(feat)
    for ep in by_ep:
        by_ep[ep].sort(key=lambda item: int(item["step"]))
    return dict(by_ep)


def bool_series(rows: list[dict[str, float]], kind: str) -> list[bool]:
    out = []
    for row in rows:
        command = row.get("gripper_command", 0.0)
        action = row.get("action_gripper", 0.0)
        close_streak = row.get("recent_close_streak", 0.0)
        open_streak = row.get("recent_open_streak", 0.0)
        close_onset = row.get("close_onset", 0.0)
        if kind == "close_like":
            out.append(command < -0.05 or action < -0.05 or close_streak > 0.0 or close_onset > 0.0)
        elif kind == "open_like":
            out.append(command > 0.05 or action > 0.05 or open_streak > 0.0)
        else:
            fail(f"unknown series kind: {kind}")
    return out


def contiguous_segments(flags: list[bool], min_len: int = 1) -> list[tuple[int, int]]:
    segs = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        if (not flag or i == len(flags) - 1) and start is not None:
            end = i + 1 if flag and i == len(flags) - 1 else i
            if end - start >= min_len:
                segs.append((start, end))
            start = None
    return segs


def segment_stats(rows: list[dict[str, float]], start: int, end: int) -> dict[str, float]:
    chunk = rows[start:end]
    def max_abs(name: str) -> float:
        return max((abs(row.get(name, 0.0)) for row in chunk), default=0.0)
    def max_val(name: str) -> float:
        return max((row.get(name, 0.0) for row in chunk), default=0.0)
    return {
        "max_recent_close_streak": max_val("recent_close_streak"),
        "max_recent_open_streak": max_val("recent_open_streak"),
        "max_abs_qpos_delta_1": max_abs("qpos_delta_1"),
        "max_abs_qpos_delta_3": max_abs("qpos_delta_3"),
        "max_abs_opening_proxy_delta_3": max_abs("opening_proxy_delta_3"),
        "max_close_onset": max_val("close_onset"),
    }


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def classify_episode(label_positive: bool, candidate_count: int) -> str:
    if label_positive:
        return "LABEL_V2_POSITIVE"
    if candidate_count <= 0:
        return "TRUE_NO_CONTACT_OR_FEATURE_UNDETECTED"
    if candidate_count >= 2:
        return "MULTI_CONTACT_LONG_HORIZON"
    return "LABEL_V2_SINGLE_WINDOW_MISMATCH"


def write_sha256sums(root: Path) -> tuple[str, str]:
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}):
        rel = path.relative_to(root).as_posix()
        lines.append(f"{sha256_file(path)}  {rel}")
    sums = root / "SHA256SUMS"
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    side = root / "SHA256SUMS.sha256"
    side.write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")
    return sha256_file(sums), sha256_file(side)


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_dataset_manifest(args.dataset_csv)
    labels = read_labels(args.label_csv)
    splits = read_splits(args.split_csv, args.fold_id)
    features = read_features(args.feature_csv)
    target_rows = [row for row in dataset if row["suite"] == args.target_suite]
    if not target_rows:
        fail(f"target suite not present in dataset: {args.target_suite}")
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    episode_rows = []
    segment_rows = []
    alignment_rows = []
    candidate_rows = []
    classification_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    label_positive_count = 0
    candidate_episode_count = 0

    for ep_row in sorted(target_rows, key=lambda row: row["episode_key"]):
        ep = ep_row["episode_key"]
        label = labels.get(ep)
        if label is None:
            fail(f"missing label for {ep}")
        if ep not in features:
            fail(f"missing feature rows for {ep}")
        rows = features[ep]
        close_segments = contiguous_segments(bool_series(rows, "close_like"), min_len=args.min_segment_len)
        open_segments = contiguous_segments(bool_series(rows, "open_like"), min_len=args.min_segment_len)
        candidate_segments = []
        for start, end in close_segments:
            stats = segment_stats(rows, start, end)
            has_response = (
                stats["max_abs_qpos_delta_1"] >= args.min_response_delta
                or stats["max_abs_qpos_delta_3"] >= args.min_response_delta
                or stats["max_abs_opening_proxy_delta_3"] >= args.min_response_delta
                or stats["max_close_onset"] > 0.0
            )
            if has_response:
                candidate_segments.append((start, end, stats))
        label_start = int(label["window_start"])
        label_end = int(label["window_end"])
        label_positive = bool(label["positive"])
        if label_positive:
            label_positive_count += 1
        if candidate_segments:
            candidate_episode_count += 1
        classification = classify_episode(label_positive, len(candidate_segments))
        classification_counts[classification] += 1
        split_name = splits.get(ep, "UNASSIGNED")
        split_counts[split_name] += 1
        close_onset_count = sum(1 for row in rows if row.get("close_onset", 0.0) > 0.0)
        episode_rows.append({
            "episode_key": ep,
            "suite": ep_row["suite"],
            "task_id": ep_row["task_id"],
            "split": split_name,
            "trace_length": ep_row["trace_length"],
            "label_event_present": str(label["event_present"]).lower(),
            "label_window_valid": str(label["window_valid"]).lower(),
            "label_window_start": label_start,
            "label_window_end": label_end,
            "label_positive": str(label_positive).lower(),
            "close_onset_count": close_onset_count,
            "sustained_close_segment_count": len(close_segments),
            "open_segment_count": len(open_segments),
            "candidate_contact_segment_count": len(candidate_segments),
            "classification": classification,
        })
        seg_id = 0
        for kind, segs in [("close_like", close_segments), ("open_like", open_segments)]:
            for start, end in segs:
                stats = segment_stats(rows, start, end)
                segment_rows.append({
                    "episode_key": ep,
                    "segment_id": f"{kind}_{seg_id}",
                    "segment_type": kind,
                    "start": start,
                    "end": end,
                    "length": end - start,
                    "overlaps_label_window": str(label_positive and overlaps(start, end, label_start, label_end)).lower(),
                    **stats,
                })
                seg_id += 1
        for idx, (start, end, stats) in enumerate(candidate_segments):
            overlap = label_positive and overlaps(start, end, label_start, label_end)
            row = {
                "episode_key": ep,
                "candidate_id": f"candidate_{idx}",
                "start": start,
                "end": end,
                "length": end - start,
                "overlaps_label_window": str(overlap).lower(),
                "label_positive": str(label_positive).lower(),
                **stats,
            }
            candidate_rows.append(row)
            alignment_rows.append(row)

    write_csv(root / "libero10_episode_support_summary.csv", [
        "episode_key", "suite", "task_id", "split", "trace_length",
        "label_event_present", "label_window_valid", "label_window_start", "label_window_end", "label_positive",
        "close_onset_count", "sustained_close_segment_count", "open_segment_count", "candidate_contact_segment_count", "classification",
    ], episode_rows)
    write_csv(root / "libero10_gripper_event_segments.csv", [
        "episode_key", "segment_id", "segment_type", "start", "end", "length", "overlaps_label_window",
        "max_recent_close_streak", "max_recent_open_streak", "max_abs_qpos_delta_1", "max_abs_qpos_delta_3", "max_abs_opening_proxy_delta_3", "max_close_onset",
    ], segment_rows)
    write_csv(root / "libero10_label_v2_alignment.csv", [
        "episode_key", "candidate_id", "start", "end", "length", "overlaps_label_window", "label_positive",
        "max_recent_close_streak", "max_recent_open_streak", "max_abs_qpos_delta_1", "max_abs_qpos_delta_3", "max_abs_opening_proxy_delta_3", "max_close_onset",
    ], alignment_rows)
    write_csv(root / "libero10_multicontact_candidate_windows.csv", [
        "episode_key", "candidate_id", "start", "end", "length", "overlaps_label_window", "label_positive",
        "max_recent_close_streak", "max_recent_open_streak", "max_abs_qpos_delta_1", "max_abs_qpos_delta_3", "max_abs_opening_proxy_delta_3", "max_close_onset",
    ], candidate_rows)

    report = {
        "status": "PASS",
        "schema_version": "libero10_multicontact_audit_v1",
        "target_suite": args.target_suite,
        "episode_count": len(target_rows),
        "label_positive_episode_count": label_positive_count,
        "candidate_contact_episode_count": candidate_episode_count,
        "classification_counts": dict(classification_counts),
        "split_counts": dict(split_counts),
        "dataset_csv_sha256": sha256_file(Path(args.dataset_csv)),
        "feature_csv_sha256": sha256_file(Path(args.feature_csv)),
        "label_csv_sha256": sha256_file(Path(args.label_csv)),
        "split_csv_sha256": sha256_file(Path(args.split_csv)) if args.split_csv else "NOT_PROVIDED",
        "new_training": "NOT_PERFORMED",
        "OpenVLA": "NOT_PERFORMED",
        "LIBERO": "NOT_PERFORMED",
        "rollout": "NOT_PERFORMED",
        "attack": "NOT_PERFORMED",
        "intervention": "NOT_PERFORMED",
        "label_mutation": "NOT_PERFORMED",
        "feature_mutation": "NOT_PERFORMED",
        "dataset_mutation": "NOT_PERFORMED",
    }
    write_json(root / "libero10_no_positive_reason_report.json", report)
    write_json(root / "libero10_detector_score_overlay.json", {
        "status": "NOT_AVAILABLE",
        "reason": "detector score artifact not supplied; this audit is feature/label support only",
        "target_suite": args.target_suite,
    })
    sums_sha, side_sha = write_sha256sums(root)
    report["SHA256SUMS"] = sums_sha
    report["SHA256SUMS.sha256"] = side_sha
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--feature-csv", required=True)
    parser.add_argument("--label-csv", required=True)
    parser.add_argument("--split-csv")
    parser.add_argument("--fold-id")
    parser.add_argument("--target-suite", default=TARGET_SUITE_DEFAULT)
    parser.add_argument("--min-segment-len", type=int, default=2)
    parser.add_argument("--min-response-delta", type=float, default=1e-6)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    try:
        report = run_audit(args)
    except (OSError, json.JSONDecodeError, csv.Error, Libero10AuditError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
