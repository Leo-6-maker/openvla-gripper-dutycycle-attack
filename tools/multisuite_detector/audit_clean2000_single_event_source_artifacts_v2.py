#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

GATE = "D1D0B_CLEAN2000_SINGLE_EVENT_SOURCE_ARTIFACT_AUDIT_NO_EVENT_AWARE"
PASS = "PASS_SINGLE_EVENT_SOURCE_ARTIFACT_AUDIT_NO_EVENT_AWARE_READY_FOR_RESOLVER"
PASS_PARTIAL_DEBUG = "PASS_SINGLE_EVENT_SOURCE_ARTIFACT_AUDIT_NO_EVENT_AWARE_PARTIAL_DEBUG"
OUT_FILES = [
    "single_event_source_artifact_audit_v2_report.json",
    "single_event_source_artifact_bindings_v2.csv",
    "single_event_source_artifact_failures_v2.csv",
    "single_event_source_artifact_candidates_v2.csv",
    "single_event_source_artifact_coverage_by_suite_v2.csv",
    "checksum_report.json",
]
SUITES = ["libero_spatial", "libero_goal", "libero_object", "libero_10"]
SINGLE_EVENT_SUITES = ["libero_spatial", "libero_goal", "libero_object"]
ID_FIELDS = ["parent_id", "episode_key", "run_id", "record_id", "id"]
TEXT_FIELDS = ["suite", "suite_name", "benchmark", "libero_suite", "task_id", "task_name", "instruction", "language_instruction", "path", "output_root", "state_id"]
STATUS_FIELDS = ["teacher_label_status", "label_status", "source_label_status", "source_event_status", "event_status"]
TASK_PAT = re.compile(r"(?:^|[^a-z0-9])task[_\- /]*0*([0-9]+)(?:[^0-9]|$)", re.I)
STATE_PATTERNS = [
    re.compile(r"(?:^|[^a-z0-9])state[_\- /]*0*([0-9]+)(?:[^0-9]|$)", re.I),
    re.compile(r"(?:^|[^a-z0-9])init[_\- /]*state[_\- /]*0*([0-9]+)(?:[^0-9]|$)", re.I),
    re.compile(r"(?:^|[^a-z0-9])initial[_\- /]*state[_\- /]*0*([0-9]+)(?:[^0-9]|$)", re.I),
]
TEMPORAL_FIELDS = ["path_step_telemetry.csv", "path_step_records.jsonl", "path_phase_cues.csv", "path_episode_manifest.json"]
SUITE_ALIASES = {
    "libero_spatial": ["libero_spatial", "libero-spatial", "libero spatial", "/spatial/", "black_bowl"],
    "libero_goal": ["libero_goal", "libero-goal", "libero goal", "/goal/", "drawer"],
    "libero_object": ["libero_object", "libero-object", "libero object", "liberoobject", "/object/", "alphabet_soup", "tomato_sauce", "cream_cheese", "orange_juice", "milk", "butter"],
}
SIGNAL_HINTS = ["gripper", "qpos", "width", "action", "eef", "step", "timestep", "phase", "anchor", "window"]


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


def alias_suite_from_text(text: str) -> str:
    low = text.lower()
    if "libero_10" in low or "libero-10" in low or "libero10" in low or "moka" in low:
        return "libero_10"
    for suite, aliases in SUITE_ALIASES.items():
        if any(a in low for a in aliases):
            return suite
    return "UNKNOWN"


def infer_suite(row: Dict[str, Any]) -> str:
    direct = str(first_value(row, ["suite", "suite_name", "benchmark", "libero_suite", "suite_hint"], "")).strip()
    if direct in SUITES:
        return direct
    text = " ".join(str(row.get(k, "") or "") for k in TEXT_FIELDS + ID_FIELDS + ["artifact_dir"])
    return alias_suite_from_text(text)


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


def parse_task_state_text(text: str) -> Tuple[int | None, int | None]:
    tm = TASK_PAT.search(text)
    task = int(tm.group(1)) if tm else None
    state = None
    for pat in STATE_PATTERNS:
        sm = pat.search(text)
        if sm:
            state = int(sm.group(1))
            break
    return task, state


def key_from_record(row: Dict[str, Any]) -> Tuple[str, int | None, int | None, str]:
    suite = infer_suite(row)
    task = None
    for k in ["task_index", "task_idx", "task_num", "task_id"]:
        task = parse_int(row.get(k))
        if task is not None:
            break
    state = None
    for k in ["state_index", "state_idx", "state_num", "state_id", "initial_state_id", "init_state_id", "state"]:
        state = parse_int(row.get(k))
        if state is not None:
            break
    text = " ".join(str(row.get(k, "") or "") for k in TEXT_FIELDS + ID_FIELDS)
    t2, s2 = parse_task_state_text(text)
    if task is None:
        task = t2
    if state is None:
        state = s2
    key = f"{suite}/task_{task:02d}/state_{state:03d}" if task is not None and state is not None else ""
    return suite, task, state, key


def key_from_artifact(row: Dict[str, Any]) -> Tuple[str, int | None, int | None, str]:
    path = str(row.get("artifact_dir", "") or "")
    suite = infer_suite(row)
    task, state = parse_task_state_text(path)
    key = f"{suite}/task_{task:02d}/state_{state:03d}" if task is not None and state is not None else ""
    return suite, task, state, key


def source_status(row: Dict[str, Any]) -> str:
    status = str(first_value(row, STATUS_FIELDS, "")).strip()
    if status:
        return status
    for key in ["source_positive_anchor_valid", "positive_anchor_valid", "has_positive_anchor", "source_event_valid"]:
        if key in row and str(row.get(key, "")).strip() not in {"", "0", "False", "false", "NO", "no"}:
            return "SOURCE_POSITIVE"
    for key in ["source_no_event", "no_event", "clean_failed"]:
        if key in row and str(row.get(key, "")).strip() not in {"", "0", "False", "false", "NO", "no"}:
            return "NO_EVENT"
    return "UNKNOWN"


def existing_temporal(row: Dict[str, Any]) -> Tuple[str, str]:
    for f in TEMPORAL_FIELDS:
        p = str(row.get(f, "") or "").strip()
        if p and Path(p).exists():
            return f, p
    d = str(row.get("artifact_dir", "") or "")
    for name in ["step_telemetry.csv", "step_records.jsonl", "phase_cues.csv", "episode_manifest.json"]:
        p = Path(d) / name
        if p.exists():
            return name, str(p)
    return "", ""


def sample_columns(path: str) -> Tuple[str, str]:
    cols = ""
    try:
        p = Path(path)
        if p.suffix == ".csv":
            with p.open(newline="", encoding="utf-8") as f:
                cols = ";".join(csv.DictReader(f).fieldnames or [])
        elif p.suffix in {".jsonl", ".jl"}:
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            cols = ";".join(sorted(obj.keys()))
                        break
    except Exception as exc:
        cols = f"READ_ERROR:{type(exc).__name__}"
    sig = ";".join([c for c in cols.split(";") if any(h in c.lower() for h in SIGNAL_HINTS)])
    return cols, sig


def choose_candidate(candidates: List[Dict[str, Any]], prefer: List[str]) -> Tuple[Dict[str, Any] | None, str]:
    unique = {str(c.get("artifact_dir", "")): c for c in candidates}
    current = sorted(unique.values(), key=lambda c: str(c.get("artifact_dir", "")))
    if len(current) == 1:
        return current[0], "unique_exact_key"
    for pref in prefer:
        hits = [c for c in current if pref and pref in str(c.get("artifact_dir", ""))]
        if len(hits) == 1:
            return hits[0], f"selected_by_prefer_substring:{pref}"
        if len(hits) > 1:
            current = hits
    if len(current) == 1:
        return current[0], "unique_after_preference"
    return None, "ambiguous_or_missing"


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
    source = read_csv(Path(args.clean2000_records))
    inventory = read_csv(Path(args.artifact_inventory))
    target_suites = set(args.target_suite)
    target = [r for r in source if infer_suite(r) in target_suites]
    grouped: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in inventory:
        suite, task, state, key = key_from_artifact(row)
        if suite not in target_suites or task is None or state is None:
            continue
        temporal_kind, temporal_path = existing_temporal(row)
        if not temporal_kind:
            continue
        new = dict(row)
        new["structured_key"] = key
        new["temporal_kind"] = temporal_kind
        new["temporal_path"] = temporal_path
        grouped[(suite, task, state)].append(new)
    bindings: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    candidates_out: List[Dict[str, Any]] = []
    for row in target:
        rid = record_id(row)
        suite, task, state, key = key_from_record(row)
        status = source_status(row)
        candidates = grouped.get((suite, task, state), []) if task is not None and state is not None else []
        chosen, note = choose_candidate(candidates, args.prefer_artifact_substring)
        for cand in candidates[: args.max_candidate_rows_per_record]:
            candidates_out.append({"record_id": rid, "suite": suite, "structured_key": key, "source_status": status, "artifact_dir": cand.get("artifact_dir", ""), "temporal_kind": cand.get("temporal_kind", ""), "selection_note": note})
        if chosen is None:
            if status == "NO_EVENT" and args.allow_no_event_without_artifact:
                bindings.append({"record_id": rid, "suite": suite, "structured_key": key, "source_status": status, "artifact_dir": "", "selection_note": "no_event_without_artifact", "temporal_kind": "", "temporal_path": "", "temporal_columns": "", "signal_like_columns": ""})
                continue
            reason = "NO_UNIQUE_TEMPORAL_ARTIFACT_BINDING"
            if status == "NO_EVENT":
                reason = "NO_EVENT_TEMPORAL_ARTIFACT_BINDING_MISSING"
            if status == "SOURCE_POSITIVE":
                reason = "POSITIVE_TEMPORAL_ARTIFACT_BINDING_MISSING"
            failures.append({"record_id": rid, "suite": suite, "structured_key": key, "source_status": status, "failure_reason": reason, "candidate_count": len(candidates)})
            continue
        cols, sig = sample_columns(str(chosen.get("temporal_path", "")))
        bindings.append({"record_id": rid, "suite": suite, "structured_key": key, "source_status": status, "artifact_dir": chosen.get("artifact_dir", ""), "selection_note": note, "temporal_kind": chosen.get("temporal_kind", ""), "temporal_path": chosen.get("temporal_path", ""), "temporal_columns": cols, "signal_like_columns": sig})
    by_suite: Dict[str, Counter] = defaultdict(Counter)
    for row in target:
        by_suite[infer_suite(row)]["source_records"] += 1
        by_suite[infer_suite(row)][source_status(row)] += 1
    for row in bindings:
        by_suite[row["suite"]]["bindings"] += 1
        by_suite[row["suite"]][row["selection_note"]] += 1
    for row in failures:
        by_suite[row["suite"]]["failures"] += 1
    coverage = []
    for suite in sorted(target_suites):
        c = by_suite[suite]
        coverage.append({"suite": suite, "source_records": c.get("source_records", 0), "source_positive": c.get("SOURCE_POSITIVE", 0), "source_no_event": c.get("NO_EVENT", 0), "bindings": c.get("bindings", 0), "failures": c.get("failures", 0), "no_event_without_artifact": c.get("no_event_without_artifact", 0)})
    positive_failures = [f for f in failures if f.get("source_status") == "SOURCE_POSITIVE"]
    complete = len(target) == args.expected_target and not failures
    if complete:
        status_out = PASS
        reason = ""
    elif args.allow_partial_debug and not positive_failures:
        status_out = PASS_PARTIAL_DEBUG
        reason = "only no-event artifact gaps remain; debug bindings emitted"
    else:
        status_out = "HOLD_SINGLE_EVENT_SOURCE_ARTIFACT_AUDIT_V2_INCOMPLETE"
        reason = f"target={len(target)} expected={args.expected_target} bindings={len(bindings)} failures={len(failures)} positive_failures={len(positive_failures)}"
    write_csv(out / "single_event_source_artifact_bindings_v2.csv", bindings, ["record_id", "suite", "structured_key", "source_status", "artifact_dir", "selection_note", "temporal_kind", "temporal_path", "temporal_columns", "signal_like_columns"])
    write_csv(out / "single_event_source_artifact_failures_v2.csv", failures, ["record_id", "suite", "structured_key", "source_status", "failure_reason", "candidate_count"])
    write_csv(out / "single_event_source_artifact_candidates_v2.csv", candidates_out, ["record_id", "suite", "structured_key", "source_status", "artifact_dir", "temporal_kind", "selection_note"])
    write_csv(out / "single_event_source_artifact_coverage_by_suite_v2.csv", coverage, ["suite", "source_records", "source_positive", "source_no_event", "bindings", "failures", "no_event_without_artifact"])
    report = {"gate": GATE, "status": status_out, "reason": reason, "clean2000_records": args.clean2000_records, "clean2000_records_sha256": sha256_file(Path(args.clean2000_records)), "artifact_inventory": args.artifact_inventory, "artifact_inventory_sha256": sha256_file(Path(args.artifact_inventory)), "source_record_count": len(source), "target_record_count": len(target), "expected_target": args.expected_target, "binding_count": len(bindings), "failure_count": len(failures), "positive_failure_count": len(positive_failures), "coverage_by_suite": coverage, "failures_by_reason": dict(Counter(f["failure_reason"] for f in failures)), "binding_source_status": dict(Counter(b["source_status"] for b in bindings)), "selection_note_counts": dict(Counter(b["selection_note"] for b in bindings)), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "interpretation": "CPU-only no-event-aware single-event source/artifact audit. It separates positive records that require temporal artifacts from no-event records that can be represented in the label manifest without an anchor.", "boundaries": {"CUDA_required": "NOT_REQUIRED", "OpenVLA_model": "NOT_LOADED", "model_inference": "NOT_PERFORMED", "LIBERO_runtime": "NOT_PERFORMED", "env_step": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED", "attack_condition": "NOT_PERFORMED", "detector_training": "NOT_PERFORMED"}, "git_commit": args.git_commit, "files_changed": args.files_changed, "tests": args.tests}
    write_json(out / "single_event_source_artifact_audit_v2_report.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if not status_out.startswith("HOLD_") else 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--clean2000-records", required=True)
    p.add_argument("--artifact-inventory", required=True)
    p.add_argument("--target-suite", action="append", default=SINGLE_EVENT_SUITES)
    p.add_argument("--expected-target", type=int, default=1500)
    p.add_argument("--prefer-artifact-substring", action="append", default=["sc5_cross_suite_clean1500_v1"])
    p.add_argument("--allow-no-event-without-artifact", action="store_true")
    p.add_argument("--allow-partial-debug", action="store_true")
    p.add_argument("--max-candidate-rows-per-record", type=int, default=5)
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())

if __name__ == "__main__":
    raise SystemExit(main())
