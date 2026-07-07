#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, hashlib, json, re, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

GATE = "D1D0_CLEAN2000_SINGLE_EVENT_SOURCE_ARTIFACT_AUDIT"
PASS = "PASS_SINGLE_EVENT_SOURCE_ARTIFACT_AUDIT_READY_FOR_RESOLVER"
OUT_FILES = [
    "single_event_source_artifact_audit_report.json",
    "single_event_source_artifact_bindings.csv",
    "single_event_source_artifact_failures.csv",
    "single_event_source_column_audit.csv",
    "single_event_artifact_key_coverage.csv",
    "checksum_report.json",
]
SUITES = ["libero_spatial", "libero_goal", "libero_object", "libero_10"]
SINGLE_EVENT_SUITES = ["libero_spatial", "libero_goal", "libero_object"]
ID_FIELDS = ["parent_id", "episode_key", "run_id", "record_id", "id"]
TEXT_FIELDS = ["suite", "suite_name", "benchmark", "libero_suite", "task_id", "task_name", "instruction", "language_instruction", "path", "output_root", "state_id"]
STATUS_FIELDS = ["teacher_label_status", "label_status", "source_label_status", "source_event_status", "event_status"]
TASK_PAT = re.compile(r"(?:^|[^a-z0-9])task[_\- /]*0*([0-9]+)(?:[^0-9]|$)", re.I)
STATE_PAT = re.compile(r"(?:^|[^a-z0-9])state[_\- /]*0*([0-9]+)(?:[^0-9]|$)", re.I)
TEMPORAL_FIELDS = ["path_step_telemetry.csv", "path_step_records.jsonl", "path_phase_cues.csv", "path_episode_manifest.json"]
SIGNAL_HINTS = ["gripper", "qpos", "width", "action", "eef", "step", "timestep", "phase", "segment", "anchor", "window"]


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
        rows = []
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            row = dict(row)
            row["__source_file"] = str(path)
            row["__source_line"] = line_no
            rows.append(row)
        return rows


def first_value(row: Dict[str, Any], keys: Iterable[str], default: str = "") -> Any:
    for key in keys:
        if row.get(key) not in (None, ""):
            return row[key]
    return default


def record_id(row: Dict[str, Any]) -> str:
    return str(first_value(row, ID_FIELDS, f"{row.get('__source_file')}:{row.get('__source_line')}"))


def infer_suite(row: Dict[str, Any]) -> str:
    direct = str(first_value(row, ["suite", "suite_name", "benchmark", "libero_suite"], "")).strip()
    if direct in SUITES:
        return direct
    text = " ".join(str(row.get(k, "") or "") for k in TEXT_FIELDS + ID_FIELDS).lower()
    if "libero_10" in text or "libero-10" in text or "libero10" in text or "moka" in text:
        return "libero_10"
    if "libero_spatial" in text or "spatial" in text or "black_bowl" in text:
        return "libero_spatial"
    if "libero_goal" in text or "goal" in text or "drawer" in text:
        return "libero_goal"
    if "libero_object" in text or "object" in text or "alphabet_soup" in text:
        return "libero_object"
    return "UNKNOWN"


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    s = str(value).strip()
    if re.fullmatch(r"0*[0-9]+", s):
        return int(s)
    try:
        f = float(s)
        if f.is_integer() and f >= 0:
            return int(f)
    except Exception:
        pass
    return None


def key_from_record(row: Dict[str, Any]) -> Tuple[str, int | None, int | None, str]:
    suite = infer_suite(row)
    task = next((parse_int(row.get(k)) for k in ["task_index", "task_idx", "task_id"] if parse_int(row.get(k)) is not None), None)
    state = next((parse_int(row.get(k)) for k in ["state_index", "state_idx", "state_id", "initial_state_id", "init_state_id", "state"] if parse_int(row.get(k)) is not None), None)
    text = " ".join(str(row.get(k, "") or "") for k in TEXT_FIELDS + ID_FIELDS)
    if task is None:
        m = TASK_PAT.search(text); task = int(m.group(1)) if m else None
    if state is None:
        m = STATE_PAT.search(text); state = int(m.group(1)) if m else None
    return suite, task, state, f"{suite}/task_{task:02d}/state_{state:03d}" if task is not None and state is not None else ""


def key_from_artifact(row: Dict[str, Any]) -> Tuple[str, int | None, int | None, str]:
    suite = str(row.get("suite_hint", "UNKNOWN") or "UNKNOWN")
    path = str(row.get("artifact_dir", "") or "")
    low = path.lower()
    if suite == "UNKNOWN":
        for s in SINGLE_EVENT_SUITES:
            if s in low: suite = s
    tm = TASK_PAT.search(path); sm = STATE_PAT.search(path)
    task = int(tm.group(1)) if tm else None
    state = int(sm.group(1)) if sm else None
    return suite, task, state, f"{suite}/task_{task:02d}/state_{state:03d}" if task is not None and state is not None else ""


def source_status(row: Dict[str, Any]) -> str:
    status = str(first_value(row, STATUS_FIELDS, "")).strip()
    if status: return status
    for key in ["source_positive_anchor_valid", "positive_anchor_valid", "has_positive_anchor", "source_event_valid"]:
        if str(row.get(key, "")).strip() not in {"", "0", "False", "false", "NO", "no"}: return "SOURCE_POSITIVE"
    for key in ["source_no_event", "no_event", "clean_failed"]:
        if str(row.get(key, "")).strip() not in {"", "0", "False", "false", "NO", "no"}: return "NO_EVENT"
    return "UNKNOWN"


def existing_temporal(row: Dict[str, Any]) -> Tuple[str, str]:
    for f in TEMPORAL_FIELDS:
        p = str(row.get(f, "") or "").strip()
        if p and Path(p).exists(): return f, p
    d = str(row.get("artifact_dir", "") or "")
    for name in ["step_telemetry.csv", "step_records.jsonl", "phase_cues.csv", "episode_manifest.json"]:
        p = Path(d) / name
        if p.exists(): return name, str(p)
    return "", ""


def sample_temporal_columns(path: str) -> str:
    try:
        p = Path(path)
        if p.suffix == ".csv":
            with p.open(newline="", encoding="utf-8") as f:
                return ";".join(csv.DictReader(f).fieldnames or [])
        if p.suffix in {".jsonl", ".jl"}:
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        return ";".join(sorted(obj.keys())) if isinstance(obj, dict) else ""
    except Exception as exc:
        return f"READ_ERROR:{type(exc).__name__}"
    return ""


def write_checksums(out: Path) -> None:
    reported = {name: sha256_file(out / name) for name in OUT_FILES[:-1] if (out / name).exists()}
    write_json(out / "checksum_report.json", {"algorithm": "sha256", "reported_files": reported, "self_referential_checksum_fields": "ABSENT_BY_DESIGN"})
    present = [name for name in OUT_FILES if (out / name).exists()]
    sums = out / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(out / name)}  {name}\n" for name in present), encoding="utf-8")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    out = Path(args.output_root); out.mkdir(parents=True, exist_ok=True)
    source = read_csv(Path(args.clean2000_records)); inv = read_csv(Path(args.artifact_inventory))
    target_suites = set(args.target_suite)
    target = [r for r in source if infer_suite(r) in target_suites]
    grouped = defaultdict(list)
    for row in inv:
        suite, task, state, key = key_from_artifact(row)
        if suite in target_suites and task is not None and state is not None:
            kind, path = existing_temporal(row)
            if kind:
                row = dict(row); row["structured_key"] = key; row["temporal_kind"] = kind; row["temporal_path"] = path
                grouped[(suite, task, state)].append(row)
    bindings, failures, coverage = [], [], []
    for row in target:
        rid = record_id(row); suite, task, state, key = key_from_record(row); status = source_status(row)
        candidates = grouped.get((suite, task, state), []) if task is not None and state is not None else []
        chosen = None; note = ""
        if len(candidates) == 1:
            chosen = candidates[0]; note = "unique_exact_key"
        elif candidates:
            current = candidates
            for pref in args.prefer_artifact_substring:
                hits = [c for c in current if pref in str(c.get("artifact_dir", ""))]
                if len(hits) == 1:
                    chosen = hits[0]; note = f"selected_by_prefer_substring:{pref}"; break
                if len(hits) > 1: current = hits
            if chosen is None and len(current) == 1:
                chosen = current[0]; note = "unique_after_preference"
        if chosen is None:
            failures.append({"record_id": rid, "suite": suite, "structured_key": key, "source_status": status, "failure_reason": "NO_UNIQUE_TEMPORAL_ARTIFACT_BINDING", "candidate_count": len(candidates)})
            continue
        cols = sample_temporal_columns(chosen["temporal_path"])
        signal_cols = [c for c in cols.split(";") if any(h in c.lower() for h in SIGNAL_HINTS)]
        bindings.append({
            "record_id": rid, "suite": suite, "structured_key": key, "source_status": status,
            "artifact_dir": chosen.get("artifact_dir", ""), "selection_note": note,
            "temporal_kind": chosen.get("temporal_kind", ""), "temporal_path": chosen.get("temporal_path", ""),
            "temporal_columns": cols, "signal_like_columns": ";".join(signal_cols),
        })
    by_suite = defaultdict(Counter)
    for row in target: by_suite[infer_suite(row)]["source_records"] += 1; by_suite[infer_suite(row)][source_status(row)] += 1
    for row in bindings: by_suite[row["suite"]]["bindings"] += 1
    for row in failures: by_suite[row["suite"]]["failures"] += 1
    for suite in sorted(target_suites):
        c = by_suite[suite]
        coverage.append({"suite": suite, "source_records": c.get("source_records",0), "source_positive": c.get("SOURCE_POSITIVE",0), "source_no_event": c.get("NO_EVENT",0), "bindings": c.get("bindings",0), "failures": c.get("failures",0)})
    col_counts = defaultdict(Counter)
    for row in source:
        suite = infer_suite(row)
        if suite not in target_suites: continue
        for col, val in row.items():
            if col.startswith("__"): continue
            if re.search(r"anchor|window|event|positive|no_event|status|task|state", col, re.I):
                col_counts[(suite,col)]["rows"] += 1; col_counts[(suite,col)]["nonempty"] += int(val not in (None,""))
    column_rows = [{"suite": s, "column": c, "nonempty_count": v["nonempty"], "row_count": v["rows"]} for (s,c),v in sorted(col_counts.items())]
    key_rows = [{"structured_key": f"{s}/task_{t:02d}/state_{st:03d}", "candidate_count": len(v), "artifact_dirs": ";".join(str(x.get("artifact_dir","")) for x in v[:5])} for (s,t,st),v in sorted(grouped.items())]
    status = PASS if len(target) == args.expected_target and not failures else "HOLD_SINGLE_EVENT_SOURCE_ARTIFACT_AUDIT_INCOMPLETE"
    reason = "" if status == PASS else f"target={len(target)} expected={args.expected_target} failures={len(failures)}"
    write_csv(out/"single_event_source_artifact_bindings.csv", bindings, ["record_id","suite","structured_key","source_status","artifact_dir","selection_note","temporal_kind","temporal_path","temporal_columns","signal_like_columns"])
    write_csv(out/"single_event_source_artifact_failures.csv", failures, ["record_id","suite","structured_key","source_status","failure_reason","candidate_count"])
    write_csv(out/"single_event_source_column_audit.csv", column_rows, ["suite","column","nonempty_count","row_count"])
    write_csv(out/"single_event_artifact_key_coverage.csv", key_rows, ["structured_key","candidate_count","artifact_dirs"])
    report = {"gate": GATE, "status": status, "reason": reason, "clean2000_records": args.clean2000_records, "clean2000_records_sha256": sha256_file(Path(args.clean2000_records)), "artifact_inventory": args.artifact_inventory, "artifact_inventory_sha256": sha256_file(Path(args.artifact_inventory)), "source_record_count": len(source), "target_record_count": len(target), "expected_target": args.expected_target, "binding_count": len(bindings), "failure_count": len(failures), "coverage_by_suite": coverage, "failures_by_reason": dict(Counter(f["failure_reason"] for f in failures)), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "interpretation": "CPU-only audit that verifies Spatial/Goal/Object source records have unique temporal artifact bindings and observable columns before anchor resolution.", "boundaries": {"CUDA_required":"NOT_REQUIRED","OpenVLA_model":"NOT_LOADED","model_inference":"NOT_PERFORMED","LIBERO_runtime":"NOT_PERFORMED","env_step":"NOT_PERFORMED","rollout":"NOT_PERFORMED","intervention":"NOT_PERFORMED","detector_training":"NOT_PERFORMED"}, "git_commit": args.git_commit, "files_changed": args.files_changed, "tests": args.tests}
    write_json(out/"single_event_source_artifact_audit_report.json", report); write_checksums(out); print(json.dumps(report, sort_keys=True))
    return 0 if not status.startswith("HOLD_") else 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--clean2000-records", required=True)
    p.add_argument("--artifact-inventory", required=True)
    p.add_argument("--target-suite", action="append", default=SINGLE_EVENT_SUITES)
    p.add_argument("--expected-target", type=int, default=1500)
    p.add_argument("--prefer-artifact-substring", action="append", default=["sc5_cross_suite_clean1500_v1"])
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())

if __name__ == "__main__":
    raise SystemExit(main())
