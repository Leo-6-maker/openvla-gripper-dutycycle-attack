#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

GATE = "D1A_CLEAN2000_LIBERO10_SEGMENT_CANDIDATE_RESOLVER"
PASS = "PASS_LIBERO10_SEGMENT_CANDIDATES_BUILT"
OUT_FILES = ["libero10_segment_resolver_report.json", "libero10_segment_candidates.csv", "libero10_segment_resolver_failures.csv", "checksum_report.json"]
STEP_FIELDS = ["step", "timestep", "frame", "frame_idx", "step_idx", "index"]
COMMAND_FIELDS = ["action_gripper", "gripper_command", "gripper_action", "a_gripper"]
OPENING_FIELDS = ["gripper_opening_proxy", "gripper_qpos", "gripper_width", "robot0_gripper_qpos", "obs_gripper_qpos"]
TEMPORAL_PATH_FIELDS = ["path_step_telemetry_csv", "path_step_records_jsonl", "path_phase_cues_csv", "path_episode_manifest_json"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"{path}:{line_no} is not a JSON object")
            rows.append(dict(obj))
    return rows


def stable_float(value: Any, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


def first_existing_source(row: Dict[str, Any]) -> Tuple[str, Path | None]:
    for field in TEMPORAL_PATH_FIELDS:
        raw = str(row.get(field, "") or "").strip()
        if raw and Path(raw).exists():
            return field, Path(raw)
    artifact_dir = str(row.get("artifact_dir", "") or "").strip()
    if artifact_dir and Path(artifact_dir).exists():
        for name in ["step_telemetry.csv", "step_records.jsonl", "phase_cues.csv", "episode_manifest.json"]:
            p = Path(artifact_dir) / name
            if p.exists():
                return name, p
    return "", None


def load_rows(kind: str, path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return read_csv(path)
    if path.suffix.lower() in {".jsonl", ".jl"}:
        return read_jsonl(path)
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            for key in ["steps", "records", "trajectory", "telemetry", "phase_cues"]:
                if isinstance(obj.get(key), list):
                    return [dict(x) for x in obj[key] if isinstance(x, dict)]
        raise ValueError(f"no list-like temporal field in {path}")
    raise ValueError(f"unsupported temporal source: {path}")


def get_series(rows: List[Dict[str, Any]], fields: List[str]) -> Tuple[str, np.ndarray]:
    for field in fields:
        vals = np.asarray([stable_float(r.get(field)) for r in rows], dtype=np.float64)
        if vals.size and np.isfinite(vals).sum() >= max(3, int(0.2 * len(vals))):
            return field, vals
    return "", np.full((len(rows),), np.nan)


def get_steps(rows: List[Dict[str, Any]]) -> np.ndarray:
    field, vals = get_series(rows, STEP_FIELDS)
    if field:
        vals = np.where(np.isfinite(vals), vals, np.arange(len(rows)))
        return vals.astype(int)
    return np.arange(len(rows), dtype=int)


def smooth(mask: np.ndarray, min_run: int) -> np.ndarray:
    if min_run <= 1:
        return mask.astype(bool)
    out = np.zeros_like(mask, dtype=bool)
    start = None
    values = mask.astype(bool).tolist() + [False]
    for i, v in enumerate(values):
        if v and start is None:
            start = i
        if not v and start is not None:
            if i - start >= min_run:
                out[start:i] = True
            start = None
    return out


def find_onsets(mask: np.ndarray, min_gap: int) -> List[int]:
    out = []
    prev = False
    last = -10**9
    for i, v in enumerate(mask.astype(bool)):
        if v and not prev and i - last >= min_gap:
            out.append(i)
            last = i
        prev = bool(v)
    return out


def infer_activity(rows: List[Dict[str, Any]], q: float, min_run: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
    field, cmd = get_series(rows, COMMAND_FIELDS)
    if field:
        finite = cmd[np.isfinite(cmd)]
        low = np.nanpercentile(finite, q * 100.0)
        high = np.nanpercentile(finite, (1.0 - q) * 100.0)
        return smooth(cmd <= low, min_run), smooth(cmd >= high, min_run), field, "command_quantile"
    field, opening = get_series(rows, OPENING_FIELDS)
    if field:
        d = np.diff(opening, prepend=opening[0])
        finite = d[np.isfinite(d)]
        if finite.size:
            low = np.nanpercentile(finite, q * 100.0)
            high = np.nanpercentile(finite, (1.0 - q) * 100.0)
            return smooth(d <= low, min_run), smooth(d >= high, min_run), field, "opening_delta_quantile"
    return np.zeros(len(rows), dtype=bool), np.zeros(len(rows), dtype=bool), "", "no_signal"


def phase_rows_to_segments(reg: Dict[str, Any], rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("segment_start_step") not in (None, "") and row.get("segment_end_step") not in (None, ""):
            start = int(float(row["segment_start_step"]))
            end = int(float(row["segment_end_step"]))
            anchor = int(float(row.get("teacher_anchor_step", row.get("event_step", (start + end) // 2))))
            out.append(segment_row(reg, len(out), start, end, anchor, "phase_cues_existing_segment", args))
    return out


def telemetry_to_segments(reg: Dict[str, Any], rows: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    steps = get_steps(rows)
    active, inverse_active, field, method = infer_activity(rows, args.activity_quantile, args.min_activity_run)
    onsets = find_onsets(active, args.min_segment_gap)
    segments = []
    for onset in onsets:
        start_step = int(max(int(steps[0]), int(steps[onset]) - args.pre_window))
        future_end = [i for i in range(onset + args.min_segment_len, len(rows)) if inverse_active[i]]
        if future_end:
            end_idx = min(future_end[0], onset + args.max_segment_len)
        else:
            end_idx = min(len(rows) - 1, onset + args.default_segment_len)
        end_step = int(min(int(steps[-1]), int(steps[end_idx]) + args.post_window))
        anchor_step = int(steps[min(len(rows) - 1, onset + args.anchor_offset)])
        segments.append(segment_row(reg, len(segments), start_step, end_step, anchor_step, f"heuristic_{method}", args))
        segments[-1]["resolver_signal_field"] = field
    return segments, {"signal_field": field, "signal_method": method, "onset_count": len(onsets), "row_count": len(rows)}


def segment_row(reg: Dict[str, Any], idx: int, start: int, end: int, anchor: int, method: str, args: argparse.Namespace) -> Dict[str, Any]:
    rid = reg.get("record_id", "")
    return {
        "record_id": rid,
        "suite": reg.get("suite", ""),
        "task_id": reg.get("task_id", ""),
        "task_name": reg.get("task_name", ""),
        "artifact_dir": reg.get("artifact_dir", ""),
        "event_id": f"{rid}::event_{idx:02d}",
        "segment_id": f"{rid}::segment_{idx:02d}",
        "segment_index": idx,
        "segment_start_step": int(start),
        "segment_end_step": int(end),
        "teacher_anchor_step": int(anchor),
        "teacher_window_start": int(anchor - args.teacher_window_pre),
        "teacher_window_end": int(anchor + args.teacher_window_post),
        "segment_role": "candidate_manipulation",
        "subtask_index": idx,
        "event_role": "auxiliary_manipulation",
        "event_status": "CANDIDATE",
        "phase_label": "stable_carry",
        "corridor_label": 1,
        "release_safe_label": 0,
        "label_confidence": args.default_label_confidence,
        "resolver_method": method,
    }


def assign_primary(segments: List[Dict[str, Any]], policy: str) -> None:
    if not segments:
        return
    primary = 0 if policy == "first_segment" else len(segments) - 1
    for i, seg in enumerate(segments):
        if i == primary:
            seg["event_role"] = "primary_attackable"
            seg["event_status"] = "VALID_PRIMARY_CANDIDATE"
            seg["segment_role"] = "primary_candidate"
        else:
            seg["event_role"] = "auxiliary_manipulation"
            seg["event_status"] = "VALID_AUXILIARY_CANDIDATE"
            seg["segment_role"] = "auxiliary_candidate"


def resolve_one(reg: Dict[str, Any], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
    kind, path = first_existing_source(reg)
    if path is None:
        return [], failure(reg, "NO_EXISTING_TEMPORAL_SOURCE", "")
    try:
        rows = load_rows(kind, path)
        if not rows:
            return [], failure(reg, "EMPTY_TEMPORAL_SOURCE", str(path))
        if "phase_cues" in str(path.name):
            segments = phase_rows_to_segments(reg, rows, args)
            debug = {"resolver_source_kind": kind, "resolver_source_path": str(path), "row_count": len(rows), "signal_method": "phase_cues"}
        else:
            segments, debug = telemetry_to_segments(reg, rows, args)
            debug.update({"resolver_source_kind": kind, "resolver_source_path": str(path)})
        if not segments:
            return [], failure(reg, "NO_SEGMENTS_INFERRED", json.dumps(debug, sort_keys=True))
        assign_primary(segments, args.primary_policy)
        for seg in segments:
            seg.update(debug)
        return segments, None
    except Exception as exc:
        return [], failure(reg, "RESOLVER_EXCEPTION", f"{type(exc).__name__}: {exc}")


def failure(reg: Dict[str, Any], reason: str, detail: str) -> Dict[str, Any]:
    return {"record_id": reg.get("record_id", ""), "suite": reg.get("suite", ""), "artifact_dir": reg.get("artifact_dir", ""), "failure_reason": reason, "detail": detail}


def write_checksums(out: Path) -> None:
    reported = {name: sha256_file(out / name) for name in OUT_FILES[:-1] if (out / name).exists()}
    write_json(out / "checksum_report.json", {"algorithm": "sha256", "reported_files": reported, "self_referential_checksum_fields": "ABSENT_BY_DESIGN"})
    present = [name for name in OUT_FILES if (out / name).exists()]
    sums = out / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(out / name)}  {name}\n" for name in present), encoding="utf-8")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    registry = read_csv(Path(args.artifact_registry))
    segments: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for reg in registry:
        segs, fail = resolve_one(reg, args)
        segments.extend(segs)
        if fail:
            failures.append(fail)
    primary_count = sum(1 for s in segments if s.get("event_role") == "primary_attackable")
    if len(registry) != args.expected_records:
        status = "HOLD_REGISTRY_RECORD_COUNT_MISMATCH"
        reason = f"expected={args.expected_records} observed={len(registry)}"
    elif failures:
        status = "HOLD_LIBERO10_SEGMENT_RESOLUTION_INCOMPLETE"
        reason = f"failures={len(failures)}"
    elif primary_count != len(registry):
        status = "HOLD_LIBERO10_PRIMARY_SEGMENT_COUNT_MISMATCH"
        reason = f"primary_count={primary_count} records={len(registry)}"
    else:
        status = PASS
        reason = ""
    fields = ["record_id", "suite", "task_id", "task_name", "artifact_dir", "event_id", "segment_id", "segment_index", "segment_start_step", "segment_end_step", "teacher_anchor_step", "teacher_window_start", "teacher_window_end", "segment_role", "subtask_index", "event_role", "event_status", "phase_label", "corridor_label", "release_safe_label", "label_confidence", "resolver_method", "resolver_signal_field", "resolver_source_kind", "resolver_source_path", "signal_field", "signal_method", "onset_count", "row_count"]
    write_csv(out / "libero10_segment_candidates.csv", segments, fields)
    write_csv(out / "libero10_segment_resolver_failures.csv", failures, ["record_id", "suite", "artifact_dir", "failure_reason", "detail"])
    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "artifact_registry": args.artifact_registry,
        "artifact_registry_sha256": sha256_file(Path(args.artifact_registry)),
        "registry_record_count": len(registry),
        "segment_count": len(segments),
        "primary_segment_count": primary_count,
        "failure_count": len(failures),
        "primary_policy": args.primary_policy,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "interpretation": "CPU-only candidate segment resolver. PASS means every registry row has one primary candidate segment; it is still not a final dataset freeze.",
        "boundaries": {"CUDA_required": "NOT_REQUIRED", "OpenVLA_model": "NOT_LOADED", "model_inference": "NOT_PERFORMED", "LIBERO_runtime": "NOT_PERFORMED", "env_step": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED", "attack_condition": "NOT_PERFORMED"},
        "git_commit": args.git_commit,
        "files_changed": args.files_changed,
        "tests": args.tests,
    }
    write_json(out / "libero10_segment_resolver_report.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if not status.startswith("HOLD_") else 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-registry", required=True)
    p.add_argument("--expected-records", type=int, default=500)
    p.add_argument("--primary-policy", choices=["last_segment", "first_segment"], default="last_segment")
    p.add_argument("--activity-quantile", type=float, default=0.10)
    p.add_argument("--min-activity-run", type=int, default=2)
    p.add_argument("--min-segment-gap", type=int, default=12)
    p.add_argument("--min-segment-len", type=int, default=5)
    p.add_argument("--default-segment-len", type=int, default=50)
    p.add_argument("--max-segment-len", type=int, default=120)
    p.add_argument("--pre-window", type=int, default=5)
    p.add_argument("--post-window", type=int, default=10)
    p.add_argument("--anchor-offset", type=int, default=8)
    p.add_argument("--teacher-window-pre", type=int, default=3)
    p.add_argument("--teacher-window-post", type=int, default=12)
    p.add_argument("--default-label-confidence", type=float, default=0.60)
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
