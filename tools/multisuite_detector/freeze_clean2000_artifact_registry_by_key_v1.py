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

GATE = "D0D2_CLEAN2000_ARTIFACT_REGISTRY_STRUCTURED_KEY_FREEZE"
PASS = "PASS_CLEAN2000_ARTIFACT_REGISTRY_STRUCTURED_KEY_FROZEN"
PASS_PARTIAL_DEBUG = "PASS_CLEAN2000_ARTIFACT_REGISTRY_STRUCTURED_KEY_PARTIAL_DEBUG_ONLY"
OUT_FILES = [
    "clean2000_artifact_registry_by_key_report.json",
    "clean2000_artifact_registry_by_key.csv",
    "clean2000_artifact_registry_by_key_rejections.csv",
    "clean2000_artifact_registry_by_key_ambiguities.csv",
    "clean2000_artifact_key_coverage.csv",
    "checksum_report.json",
]
SUITES = ["libero_spatial", "libero_goal", "libero_object", "libero_10"]
TARGET_SUITE_DEFAULT = "libero_10"
ID_FIELDS = ["parent_id", "episode_key", "run_id", "record_id", "id"]
TEXT_FIELDS = [
    "suite", "suite_name", "benchmark", "libero_suite", "task_id", "task_name",
    "instruction", "language_instruction", "output_root", "path", "episode_root", "run_root",
    "state_id", "initial_state_id", "init_state_id", "state", "task_index", "state_index",
]
TEMPORAL_INVENTORY_FIELDS = [
    "path_step_records.jsonl",
    "path_step_telemetry.csv",
    "path_phase_cues.csv",
    "path_episode_manifest.json",
]
TEMPORAL_OUTPUT_MAP = {
    "path_step_records.jsonl": "path_step_records_jsonl",
    "path_step_telemetry.csv": "path_step_telemetry_csv",
    "path_phase_cues.csv": "path_phase_cues_csv",
    "path_episode_manifest.json": "path_episode_manifest_json",
}
TASK_PATTERNS = [
    re.compile(r"(?:^|[^a-z0-9])task[_\- /]*0*([0-9]+)(?:[^0-9]|$)", re.I),
    re.compile(r"(?:^|[^a-z0-9])taskid[_\- /]*0*([0-9]+)(?:[^0-9]|$)", re.I),
]
STATE_PATTERNS = [
    re.compile(r"(?:^|[^a-z0-9])state[_\- /]*0*([0-9]+)(?:[^0-9]|$)", re.I),
    re.compile(r"(?:^|[^a-z0-9])stateid[_\- /]*0*([0-9]+)(?:[^0-9]|$)", re.I),
    re.compile(r"(?:^|[^a-z0-9])init[_\- /]*state[_\- /]*0*([0-9]+)(?:[^0-9]|$)", re.I),
]


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
        out = []
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            row = dict(row)
            row["__source_file"] = str(path)
            row["__source_line"] = line_no
            out.append(row)
        return out


def first_value(row: Dict[str, Any], keys: Iterable[str], default: str = "") -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def record_id(row: Dict[str, Any]) -> str:
    return str(first_value(row, ID_FIELDS, f"{row.get('__source_file')}:{row.get('__source_line')}"))


def infer_suite(row: Dict[str, Any]) -> str:
    direct = str(first_value(row, ["suite", "suite_name", "benchmark", "libero_suite"], "")).strip()
    if direct in SUITES:
        return direct
    text = " ".join(str(row.get(k, "") or "") for k in TEXT_FIELDS + ID_FIELDS)
    low = text.lower()
    if "libero_10" in low or "libero-10" in low or "libero10" in low or "moka" in low:
        return "libero_10"
    if "libero_spatial" in low or "spatial" in low or "black_bowl" in low:
        return "libero_spatial"
    if "libero_goal" in low or "goal" in low or "drawer" in low:
        return "libero_goal"
    if "libero_object" in low or "object" in low or "alphabet_soup" in low:
        return "libero_object"
    return "UNKNOWN"


def parse_explicit_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    s = str(value).strip()
    if re.fullmatch(r"0*[0-9]+", s):
        return int(s)
    return None


def parse_pattern_int(text: str, patterns: List[re.Pattern[str]]) -> int | None:
    for pat in patterns:
        m = pat.search(text)
        if m:
            return int(m.group(1))
    return None


def structured_key_from_record(row: Dict[str, Any]) -> Tuple[int | None, int | None, str]:
    # Prefer explicit task/state columns when present; fall back to text patterns.
    task = None
    for key in ["task_index", "task_idx", "task_num", "task_id"]:
        task = parse_explicit_int(row.get(key))
        if task is not None:
            break
    state = None
    for key in ["state_index", "state_idx", "state_num", "state_id", "initial_state_id", "init_state_id", "state"]:
        state = parse_explicit_int(row.get(key))
        if state is not None:
            break
    text = " ".join(str(row.get(k, "") or "") for k in TEXT_FIELDS + ID_FIELDS)
    if task is None:
        task = parse_pattern_int(text, TASK_PATTERNS)
    if state is None:
        state = parse_pattern_int(text, STATE_PATTERNS)
    key_text = f"task_{task:02d}/state_{state:03d}" if task is not None and state is not None else ""
    return task, state, key_text


def structured_key_from_artifact_dir(path_text: str) -> Tuple[int | None, int | None, str]:
    text = str(path_text or "")
    task = parse_pattern_int(text, TASK_PATTERNS)
    state = parse_pattern_int(text, STATE_PATTERNS)
    key_text = f"task_{task:02d}/state_{state:03d}" if task is not None and state is not None else ""
    return task, state, key_text


def temporal_paths(row: Dict[str, Any]) -> List[str]:
    out = []
    for field in TEMPORAL_INVENTORY_FIELDS:
        value = str(row.get(field, "") or "").strip()
        if value:
            out.append(value)
    return out


def has_existing_temporal(row: Dict[str, Any], require_files_exist: bool) -> Tuple[bool, str]:
    paths = temporal_paths(row)
    if not paths and str(row.get("present_temporal_sentinels", "") or "").strip():
        return True, ""
    if not require_files_exist:
        return bool(paths), ""
    existing = [p for p in paths if Path(p).exists()]
    missing = [p for p in paths if not Path(p).exists()]
    return bool(existing), ";".join(missing)


def inventory_by_key(inventory_rows: List[Dict[str, Any]], allowed_suite_hints: set[str], require_files_exist: bool) -> Tuple[Dict[Tuple[int, int], List[Dict[str, Any]]], List[Dict[str, Any]]]:
    by_key: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    rejects: List[Dict[str, Any]] = []
    for row in inventory_rows:
        artifact_dir = str(row.get("artifact_dir", "") or "").strip()
        suite_hint = str(row.get("suite_hint", "UNKNOWN") or "UNKNOWN")
        task, state, key_text = structured_key_from_artifact_dir(artifact_dir)
        has_temporal, missing = has_existing_temporal(row, require_files_exist)
        if suite_hint not in allowed_suite_hints:
            continue
        if task is None or state is None:
            if suite_hint == TARGET_SUITE_DEFAULT or "libero_10" in artifact_dir.lower():
                rejects.append({"artifact_dir": artifact_dir, "reject_reason": "MISSING_ARTIFACT_TASK_STATE_KEY", "detail": "", "suite_hint": suite_hint})
            continue
        if not has_temporal:
            rejects.append({"artifact_dir": artifact_dir, "reject_reason": "NO_EXISTING_TEMPORAL_SENTINEL", "detail": missing, "suite_hint": suite_hint, "structured_key": key_text})
            continue
        new = dict(row)
        new["task_index"] = task
        new["state_index"] = state
        new["structured_key"] = key_text
        new["missing_temporal_paths"] = missing
        by_key[(task, state)].append(new)
    for key in list(by_key.keys()):
        # De-duplicate exact same artifact_dir while preserving row fields.
        seen = {}
        for row in by_key[key]:
            seen[str(row.get("artifact_dir", ""))] = row
        by_key[key] = sorted(seen.values(), key=lambda r: str(r.get("artifact_dir", "")))
    return by_key, rejects


def select_candidate(candidates: List[Dict[str, Any]], prefer_substrings: List[str]) -> Tuple[Dict[str, Any] | None, List[Dict[str, Any]], str]:
    if not candidates:
        return None, [], "NO_EXACT_TASK_STATE_CANDIDATE"
    if len(candidates) == 1:
        return candidates[0], [], ""
    current = candidates
    preference_notes = []
    for pref in prefer_substrings:
        if not pref:
            continue
        pref_hits = [c for c in current if pref in str(c.get("artifact_dir", ""))]
        if len(pref_hits) == 1:
            return pref_hits[0], [c for c in candidates if c is not pref_hits[0]], f"selected_by_prefer_substring:{pref}"
        if len(pref_hits) > 1:
            current = pref_hits
            preference_notes.append(f"prefer_substring_retained_{len(pref_hits)}:{pref}")
    return None, candidates, "AMBIGUOUS_EXACT_TASK_STATE_CANDIDATES" + (";" + ";".join(preference_notes) if preference_notes else "")


def make_registry_row(record: Dict[str, Any], inv: Dict[str, Any], selection_note: str) -> Dict[str, Any]:
    task, state, key_text = structured_key_from_record(record)
    out = {
        "record_id": record_id(record),
        "suite": infer_suite(record),
        "task_index": task,
        "state_index": state,
        "structured_key": key_text,
        "task_id": str(first_value(record, ["task_id"], "")),
        "task_name": str(first_value(record, ["task_name", "instruction", "language_instruction"], "")),
        "clean_success": str(first_value(record, ["clean_success", "success", "task_success", "episode_success"], "")),
        "artifact_dir": str(inv.get("artifact_dir", "")),
        "suite_hint": str(inv.get("suite_hint", "")),
        "present_temporal_sentinels": str(inv.get("present_temporal_sentinels", "")),
        "selection_note": selection_note,
        "registry_status": "FROZEN_STRUCTURED_KEY_BINDING",
        "missing_temporal_paths": str(inv.get("missing_temporal_paths", "")),
    }
    for in_field, out_field in TEMPORAL_OUTPUT_MAP.items():
        out[out_field] = str(inv.get(in_field, ""))
    return out


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
    source_rows = read_csv(Path(args.clean2000_records))
    inventory_rows = read_csv(Path(args.artifact_inventory))
    allowed = set(args.allowed_suite_hint)
    by_key, inventory_rejects = inventory_by_key(inventory_rows, allowed, args.require_files_exist)
    target_records = [r for r in source_rows if infer_suite(r) == args.target_suite]
    registry: List[Dict[str, Any]] = []
    rejections: List[Dict[str, Any]] = []
    ambiguities: List[Dict[str, Any]] = []
    for record in target_records:
        rid = record_id(record)
        task, state, key_text = structured_key_from_record(record)
        if task is None or state is None:
            rejections.append({"record_id": rid, "suite": infer_suite(record), "structured_key": key_text, "reject_reason": "MISSING_RECORD_TASK_STATE_KEY", "detail": ""})
            continue
        candidates = by_key.get((task, state), [])
        chosen, others, note = select_candidate(candidates, args.prefer_artifact_substring)
        if chosen is None:
            rejections.append({"record_id": rid, "suite": infer_suite(record), "structured_key": key_text, "reject_reason": note, "detail": f"candidate_count={len(candidates)}"})
            for cand in others[: args.max_ambiguity_rows_per_record]:
                ambiguities.append({"record_id": rid, "suite": infer_suite(record), "structured_key": key_text, "artifact_dir": cand.get("artifact_dir", ""), "suite_hint": cand.get("suite_hint", ""), "present_temporal_sentinels": cand.get("present_temporal_sentinels", ""), "note": note})
            continue
        registry.append(make_registry_row(record, chosen, note))
        for cand in others[: args.max_ambiguity_rows_per_record]:
            ambiguities.append({"record_id": rid, "suite": infer_suite(record), "structured_key": key_text, "artifact_dir": cand.get("artifact_dir", ""), "suite_hint": cand.get("suite_hint", ""), "present_temporal_sentinels": cand.get("present_temporal_sentinels", ""), "note": "nonselected_exact_key_candidate"})
    duplicate_artifact_dirs = [d for d, n in Counter(row["artifact_dir"] for row in registry).items() if d and n > 1]
    if duplicate_artifact_dirs and args.require_unique_artifact_dir:
        dup_set = set(duplicate_artifact_dirs)
        registry = [r for r in registry if r["artifact_dir"] not in dup_set]
        for d in duplicate_artifact_dirs:
            rejections.append({"record_id": "MULTIPLE", "suite": args.target_suite, "structured_key": "", "reject_reason": "DUPLICATE_ARTIFACT_DIR_BINDING", "detail": d})
    expected_target = args.expected_target if args.expected_target > 0 else len(target_records)
    complete = len(registry) == expected_target and not rejections
    if len(source_rows) != args.expected_total:
        status = "HOLD_CLEAN2000_TOTAL_COUNT_MISMATCH"
        reason = f"expected_total={args.expected_total} observed_total={len(source_rows)}"
    elif len(target_records) != expected_target:
        status = "HOLD_TARGET_RECORD_COUNT_MISMATCH"
        reason = f"expected_target={expected_target} observed_target={len(target_records)}"
    elif complete:
        status = PASS
        reason = ""
    elif args.allow_partial_debug and registry:
        status = PASS_PARTIAL_DEBUG
        reason = "partial structured-key registry emitted; not authoritative and not training-ready"
    else:
        status = "HOLD_STRUCTURED_KEY_REGISTRY_FREEZE_INCOMPLETE"
        reason = f"accepted={len(registry)} expected={expected_target} rejections={len(rejections)} ambiguities={len(ambiguities)}"
    key_rows = []
    for key, rows in sorted(by_key.items()):
        key_rows.append({"structured_key": f"task_{key[0]:02d}/state_{key[1]:03d}", "candidate_count": len(rows), "artifact_dirs": ";".join(str(r.get("artifact_dir", "")) for r in rows[:10])})
    write_csv(out / "clean2000_artifact_registry_by_key.csv", registry, [
        "record_id", "suite", "task_index", "state_index", "structured_key", "task_id", "task_name", "clean_success",
        "artifact_dir", "suite_hint", "present_temporal_sentinels", "path_step_records_jsonl", "path_step_telemetry_csv",
        "path_phase_cues_csv", "path_episode_manifest_json", "missing_temporal_paths", "selection_note", "registry_status",
    ])
    write_csv(out / "clean2000_artifact_registry_by_key_rejections.csv", rejections, ["record_id", "suite", "structured_key", "reject_reason", "detail"])
    write_csv(out / "clean2000_artifact_registry_by_key_ambiguities.csv", ambiguities, ["record_id", "suite", "structured_key", "artifact_dir", "suite_hint", "present_temporal_sentinels", "note"])
    write_csv(out / "clean2000_artifact_key_coverage.csv", key_rows, ["structured_key", "candidate_count", "artifact_dirs"])
    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "clean2000_records": args.clean2000_records,
        "clean2000_records_sha256": sha256_file(Path(args.clean2000_records)),
        "artifact_inventory": args.artifact_inventory,
        "artifact_inventory_sha256": sha256_file(Path(args.artifact_inventory)),
        "expected_total": args.expected_total,
        "observed_total": len(source_rows),
        "target_suite": args.target_suite,
        "expected_target": expected_target,
        "observed_target": len(target_records),
        "accepted_count": len(registry),
        "rejection_count": len(rejections),
        "ambiguity_count": len(ambiguities),
        "inventory_key_count": len(by_key),
        "inventory_reject_count": len(inventory_rejects),
        "rejections_by_reason": dict(Counter(r["reject_reason"] for r in rejections)),
        "parameters": {
            "allowed_suite_hint": list(args.allowed_suite_hint),
            "prefer_artifact_substring": list(args.prefer_artifact_substring),
            "require_files_exist": args.require_files_exist,
            "require_unique_artifact_dir": args.require_unique_artifact_dir,
            "allow_partial_debug": args.allow_partial_debug,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "interpretation": "CPU-only structured-key registry freeze. PASS requires exact task/state key matching and unique temporal artifact bindings.",
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
            "OpenVLA_model": "NOT_LOADED",
            "model_inference": "NOT_PERFORMED",
            "LIBERO_runtime": "NOT_PERFORMED",
            "env_step": "NOT_PERFORMED",
            "rollout": "NOT_PERFORMED",
            "intervention": "NOT_PERFORMED",
            "attack_condition": "NOT_PERFORMED",
        },
        "git_commit": args.git_commit,
        "files_changed": args.files_changed,
        "tests": args.tests,
    }
    write_json(out / "clean2000_artifact_registry_by_key_report.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if not status.startswith("HOLD_") else 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--clean2000-records", required=True)
    p.add_argument("--artifact-inventory", required=True)
    p.add_argument("--target-suite", default="libero_10")
    p.add_argument("--expected-total", type=int, default=2000)
    p.add_argument("--expected-target", type=int, default=500)
    p.add_argument("--allowed-suite-hint", action="append", default=["libero_10"])
    p.add_argument("--prefer-artifact-substring", action="append", default=[])
    p.add_argument("--require-files-exist", action="store_true")
    p.add_argument("--require-unique-artifact-dir", action="store_true")
    p.add_argument("--allow-partial-debug", action="store_true")
    p.add_argument("--max-ambiguity-rows-per-record", type=int, default=5)
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
