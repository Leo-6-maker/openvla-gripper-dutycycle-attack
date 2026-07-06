#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, hashlib, json, math, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

GATE = "D1D1_CLEAN2000_SINGLE_EVENT_ANCHOR_RESOLUTION_FROM_BINDINGS"
PASS = "PASS_CLEAN2000_SINGLE_EVENT_ANCHORS_RESOLVED_FROM_BINDINGS"
OUT_FILES = ["single_event_anchor_resolution_report.json", "single_event_teacher_label_candidates.csv", "single_event_anchor_resolution_failures.csv", "single_event_anchor_resolution_summary_by_suite.csv", "checksum_report.json"]
STEP_FIELDS = ["step", "timestep", "frame", "frame_idx", "step_idx", "index"]
COMMAND_FIELDS = ["action_gripper", "gripper_command", "gripper_action", "a_gripper"]
OPENING_FIELDS = ["gripper_opening_proxy", "gripper_qpos", "gripper_width", "robot0_gripper_qpos", "obs_gripper_qpos"]
DIRECT_ANCHOR_FIELDS = ["teacher_anchor_step", "positive_anchor_step", "anchor_step", "event_step", "selected_preplace_step", "preplace_step", "release_intent_step"]
DIRECT_WINDOW_START = ["teacher_window_start", "window_start", "positive_window_start", "event_window_start"]
DIRECT_WINDOW_END = ["teacher_window_end", "window_end", "positive_window_end", "event_window_end"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if isinstance(obj, dict): rows.append(dict(obj))
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in fields})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_float(v: Any, default: float = math.nan) -> float:
    if v in (None, ""): return default
    try: return float(v)
    except Exception: return default


def to_int(v: Any) -> int | None:
    if v in (None, ""): return None
    try: return int(float(v))
    except Exception: return None


def first_int(row: Dict[str, Any], fields: List[str]) -> int | None:
    for f in fields:
        val = to_int(row.get(f))
        if val is not None: return val
    return None


def get_series(rows: List[Dict[str, Any]], fields: List[str]) -> Tuple[str, List[float]]:
    for field in fields:
        vals = [stable_float(r.get(field)) for r in rows]
        if sum(1 for x in vals if math.isfinite(x)) >= max(3, int(0.2 * len(rows))):
            return field, vals
    return "", [math.nan for _ in rows]


def get_steps(rows: List[Dict[str, Any]]) -> List[int]:
    field, vals = get_series(rows, STEP_FIELDS)
    if field: return [int(v) if math.isfinite(v) else i for i, v in enumerate(vals)]
    return list(range(len(rows)))


def percentile(vals: List[float], q: float) -> float:
    xs = sorted([x for x in vals if math.isfinite(x)])
    if not xs: return math.nan
    if len(xs) == 1: return xs[0]
    pos = (len(xs) - 1) * q; lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def smooth(mask: List[bool], min_run: int) -> List[bool]:
    out = [False] * len(mask); start = None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None: start = i
        if not v and start is not None:
            if i - start >= min_run:
                for j in range(start, i): out[j] = True
            start = None
    return out


def onsets(mask: List[bool], gap: int) -> List[int]:
    out = []; prev = False; last = -10**9
    for i, v in enumerate(mask):
        if v and not prev and i - last >= gap:
            out.append(i); last = i
        prev = v
    return out


def load_rows(binding: Dict[str, Any]) -> Tuple[str, str, List[Dict[str, Any]]]:
    for key in ["temporal_path", "path_step_telemetry_csv", "path_step_records_jsonl", "path_phase_cues_csv"]:
        p = str(binding.get(key, "") or "")
        if p and Path(p).exists():
            if p.endswith(".csv"): return key, p, read_csv(Path(p))
            if p.endswith(".jsonl") or p.endswith(".jl"): return key, p, read_jsonl(Path(p))
    return "", "", []


def resolve_anchor(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[int | None, int | None, int | None, str, str]:
    for r in rows:
        anchor = first_int(r, DIRECT_ANCHOR_FIELDS)
        if anchor is not None:
            ws = first_int(r, DIRECT_WINDOW_START); we = first_int(r, DIRECT_WINDOW_END)
            return anchor, ws if ws is not None else anchor - args.teacher_window_pre, we if we is not None else anchor + args.teacher_window_post, "direct_temporal_anchor", ""
    steps = get_steps(rows)
    field, vals = get_series(rows, COMMAND_FIELDS)
    method = "command_quantile"
    if not field:
        field, opening = get_series(rows, OPENING_FIELDS)
        if not field: return None, None, None, "NO_GRIPPER_SIGNAL", ""
        vals = [opening[i] - (opening[i-1] if i > 0 and math.isfinite(opening[i-1]) else opening[i]) for i in range(len(opening))]
        method = "opening_delta_quantile"
    lo = percentile(vals, args.activity_quantile); hi = percentile(vals, 1 - args.activity_quantile)
    if not math.isfinite(lo) or not math.isfinite(hi): return None, None, None, "NO_FINITE_SIGNAL", field
    active = smooth([math.isfinite(v) and v <= lo for v in vals], args.min_activity_run)
    idxs = onsets(active, args.min_segment_gap)
    if not idxs: return None, None, None, "NO_ACTIVITY_ONSET", field
    onset = idxs[-1] if args.primary_policy == "last_segment" else idxs[0]
    anchor = int(steps[min(len(rows)-1, onset + args.anchor_offset)])
    return anchor, anchor - args.teacher_window_pre, anchor + args.teacher_window_post, f"heuristic_{method}", field


def label_for_no_event(b: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    rid = b.get("record_id", "")
    return {"record_id": rid, "suite": b.get("suite", ""), "task_id": "", "task_name": "", "event_id": f"{rid}::no_event", "segment_id": f"{rid}::no_segment", "segment_index": -1, "segment_start_step": -1, "segment_end_step": -1, "teacher_anchor_step": -1, "teacher_window_start": -1, "teacher_window_end": -1, "segment_role": "no_event", "subtask_index": -1, "event_role": "unsupported_or_abstain", "event_status": "NO_EVENT", "phase_label": "abstain_unsupported", "corridor_label": 0, "release_safe_label": 0, "label_confidence": args.no_event_label_confidence, "label_source": "single_event_source_no_event", "source_status": b.get("source_status", ""), "resolver_method": "no_event_passthrough", "resolver_signal_field": "", "artifact_dir": b.get("artifact_dir", ""), "online_input_allowed": "NO_LABEL_ONLY"}


def write_checksums(out: Path) -> None:
    reported = {name: sha256_file(out / name) for name in OUT_FILES[:-1] if (out / name).exists()}
    write_json(out / "checksum_report.json", {"algorithm":"sha256", "reported_files": reported, "self_referential_checksum_fields":"ABSENT_BY_DESIGN"})
    present = [name for name in OUT_FILES if (out / name).exists()]
    sums = out / "SHA256SUMS"; sums.write_text("".join(f"{sha256_file(out/name)}  {name}\n" for name in present), encoding="utf-8")
    (out/"SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)
    bindings = read_csv(Path(args.single_event_bindings))
    labels, failures = [], []
    for b in bindings:
        rid = b.get("record_id", ""); status = b.get("source_status", "")
        if status == "NO_EVENT":
            labels.append(label_for_no_event(b, args)); continue
        if status != "SOURCE_POSITIVE" and not args.allow_unknown_status_positive:
            failures.append({"record_id": rid, "suite": b.get("suite",""), "failure_reason":"UNKNOWN_SOURCE_STATUS", "detail": status}); continue
        kind, path, rows = load_rows(b)
        if not rows:
            failures.append({"record_id": rid, "suite": b.get("suite",""), "failure_reason":"EMPTY_TEMPORAL_SOURCE", "detail": b.get("temporal_path","")}); continue
        anchor, ws, we, method, field = resolve_anchor(rows, args)
        if anchor is None or ws is None or we is None:
            failures.append({"record_id": rid, "suite": b.get("suite",""), "failure_reason":"ANCHOR_RESOLUTION_FAILED", "detail": f"method={method} field={field} path={path}"}); continue
        labels.append({"record_id": rid, "suite": b.get("suite", ""), "task_id":"", "task_name":"", "event_id":f"{rid}::event_00", "segment_id":f"{rid}::segment_00", "segment_index":0, "segment_start_step":ws, "segment_end_step":we, "teacher_anchor_step":anchor, "teacher_window_start":ws, "teacher_window_end":we, "segment_role":"primary_single_event", "subtask_index":0, "event_role":"primary_attackable", "event_status":"VALID_PRIMARY_CANDIDATE", "phase_label":"stable_carry", "corridor_label":1, "release_safe_label":0, "label_confidence":args.positive_label_confidence, "label_source":"single_event_anchor_resolver_from_bindings", "source_status":status, "resolver_method":method, "resolver_signal_field":field, "artifact_dir":b.get("artifact_dir", ""), "online_input_allowed":"NO_LABEL_ONLY"})
    by_suite = defaultdict(Counter)
    for b in bindings: by_suite[b.get("suite","")][b.get("source_status","")] += 1; by_suite[b.get("suite","")]["bindings"] += 1
    for l in labels: by_suite[l.get("suite","")][l.get("event_status","")] += 1; by_suite[l.get("suite","")]["labels"] += 1
    for f in failures: by_suite[f.get("suite","")]["failures"] += 1
    summary = [{"suite":s, "bindings":c.get("bindings",0), "source_positive":c.get("SOURCE_POSITIVE",0), "source_no_event":c.get("NO_EVENT",0), "valid_primary":c.get("VALID_PRIMARY_CANDIDATE",0), "no_event_labels":c.get("NO_EVENT",0), "labels":c.get("labels",0), "failures":c.get("failures",0)} for s,c in sorted(by_suite.items())]
    pos = sum(1 for b in bindings if b.get("source_status") == "SOURCE_POSITIVE"); noev = sum(1 for b in bindings if b.get("source_status") == "NO_EVENT")
    resolved = sum(1 for l in labels if l.get("event_status") == "VALID_PRIMARY_CANDIDATE"); noev_lab = sum(1 for l in labels if l.get("event_status") == "NO_EVENT")
    status = PASS if len(bindings)==args.expected_records and not failures and resolved==pos and noev_lab==noev else "HOLD_SINGLE_EVENT_ANCHOR_RESOLUTION_INCOMPLETE"
    reason = "" if status==PASS else f"bindings={len(bindings)} expected={args.expected_records} failures={len(failures)} resolved={resolved}/{pos} no_event={noev_lab}/{noev}"
    fields = ["record_id","suite","task_id","task_name","event_id","segment_id","segment_index","segment_start_step","segment_end_step","teacher_anchor_step","teacher_window_start","teacher_window_end","segment_role","subtask_index","event_role","event_status","phase_label","corridor_label","release_safe_label","label_confidence","label_source","source_status","resolver_method","resolver_signal_field","artifact_dir","online_input_allowed"]
    write_csv(out/"single_event_teacher_label_candidates.csv", labels, fields)
    write_csv(out/"single_event_anchor_resolution_failures.csv", failures, ["record_id","suite","failure_reason","detail"])
    write_csv(out/"single_event_anchor_resolution_summary_by_suite.csv", summary, ["suite","bindings","source_positive","source_no_event","valid_primary","no_event_labels","labels","failures"])
    report = {"gate":GATE, "status":status, "reason":reason, "single_event_bindings":args.single_event_bindings, "single_event_bindings_sha256":sha256_file(Path(args.single_event_bindings)), "expected_records":args.expected_records, "binding_count":len(bindings), "label_row_count":len(labels), "positive_source_count":pos, "resolved_positive_count":resolved, "no_event_source_count":noev, "no_event_label_count":noev_lab, "failure_count":len(failures), "failures_by_reason":dict(Counter(f["failure_reason"] for f in failures)), "summary_by_suite":summary, "created_at":time.strftime("%Y-%m-%dT%H:%M:%S"), "interpretation":"CPU-only anchor resolver for Spatial/Goal/Object using audited temporal bindings; PASS is required before four-suite teacher-label-v2 materialization.", "boundaries":{"CUDA_required":"NOT_REQUIRED","OpenVLA_model":"NOT_LOADED","model_inference":"NOT_PERFORMED","LIBERO_runtime":"NOT_PERFORMED","env_step":"NOT_PERFORMED","rollout":"NOT_PERFORMED","intervention":"NOT_PERFORMED","detector_training":"NOT_PERFORMED"}, "git_commit":args.git_commit, "files_changed":args.files_changed, "tests":args.tests}
    write_json(out/"single_event_anchor_resolution_report.json", report); write_checksums(out); print(json.dumps(report, sort_keys=True))
    return 0 if not status.startswith("HOLD_") else 2


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--single-event-bindings", required=True); p.add_argument("--expected-records", type=int, default=1500); p.add_argument("--primary-policy", choices=["last_segment","first_segment"], default="last_segment"); p.add_argument("--activity-quantile", type=float, default=0.10); p.add_argument("--min-activity-run", type=int, default=2); p.add_argument("--min-segment-gap", type=int, default=12); p.add_argument("--anchor-offset", type=int, default=8); p.add_argument("--teacher-window-pre", type=int, default=3); p.add_argument("--teacher-window-post", type=int, default=12); p.add_argument("--positive-label-confidence", type=float, default=0.60); p.add_argument("--no-event-label-confidence", type=float, default=1.0); p.add_argument("--allow-unknown-status-positive", action="store_true"); p.add_argument("--output-root", required=True); p.add_argument("--git-commit", required=True); p.add_argument("--files-changed", action="append", default=[]); p.add_argument("--tests", action="append", default=[]); return run(p.parse_args())

if __name__ == "__main__":
    raise SystemExit(main())
