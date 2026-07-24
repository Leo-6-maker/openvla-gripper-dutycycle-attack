#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

GATE = "D0C_CLEAN2000_ARTIFACT_REGISTRY_RECOVERY_DEBUG"
PASS = "PASS_CLEAN2000_ARTIFACT_REGISTRY_DEBUG_BUILT"
PASS_WITH_BINDINGS = "PASS_CLEAN2000_ARTIFACT_REGISTRY_DEBUG_BUILT_WITH_LIBERO10_BINDINGS"
OUT_FILES = [
    "clean2000_artifact_registry_recovery_report.json",
    "artifact_sentinel_directory_inventory.csv",
    "libero10_artifact_binding_candidates.csv",
    "libero10_unmatched_records.csv",
    "scan_root_summary.csv",
    "checksum_report.json",
]
SUITES = ["libero_spatial", "libero_goal", "libero_object", "libero_10"]
SENTINELS = [
    "step_records.jsonl",
    "step_telemetry.csv",
    "detector_telemetry.csv",
    "episode_manifest.json",
    "episode_summary.json",
    "results.json",
    "run_manifest.json",
    "summary.json",
    "phase_cues.csv",
    "video_manifest.json",
]
TEMPORAL_SENTINELS = ["step_records.jsonl", "step_telemetry.csv", "phase_cues.csv", "episode_manifest.json"]
ID_FIELDS = ["parent_id", "episode_key", "run_id", "record_id", "id"]
TEXT_FIELDS = ["suite", "suite_name", "benchmark", "libero_suite", "task_id", "task_name", "instruction", "language_instruction", "output_root", "path", "episode_root", "run_root"]
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


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


def read_csv_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            row = dict(row)
            row["__source_file"] = str(path)
            row["__source_line"] = line_no
            yield row


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"{path}:{line_no} is not a JSON object")
            obj = dict(obj)
            obj["__source_file"] = str(path)
            obj["__source_line"] = line_no
            yield obj


def read_records(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(str(path))
        suffix = path.suffix.lower()
        if suffix == ".csv":
            rows.extend(read_csv_rows(path))
        elif suffix in (".jsonl", ".jl"):
            rows.extend(read_jsonl(path))
        elif suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            data = obj.get("records") if isinstance(obj, dict) else obj
            if not isinstance(data, list):
                raise TypeError(f"{path} must be list or mapping with records")
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    raise TypeError(f"{path}[{i}] is not a JSON object")
                item = dict(item)
                item["__source_file"] = str(path)
                item["__source_line"] = i
                rows.append(item)
        else:
            raise ValueError(f"unsupported record suffix: {path}")
    return rows


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


def tokenize(text: str) -> List[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(str(text or ""))]
    return [t for t in toks if len(t) >= 2]


def record_tokens(row: Dict[str, Any]) -> List[str]:
    text = " ".join(str(row.get(k, "") or "") for k in TEXT_FIELDS + ID_FIELDS)
    toks = set(tokenize(text))
    # Keep suite-specific aliases useful for matching paths.
    suite = infer_suite(row)
    if suite == "libero_10":
        toks.update(["libero", "10", "libero10", "moka"])
    elif suite == "libero_object":
        toks.update(["libero", "object"])
    elif suite == "libero_goal":
        toks.update(["libero", "goal"])
    elif suite == "libero_spatial":
        toks.update(["libero", "spatial"])
    return sorted(toks)


def path_tokens(path: Path) -> List[str]:
    return tokenize(str(path))


def suite_hint_from_path(path: Path) -> str:
    low = str(path).lower()
    if "libero_10" in low or "libero-10" in low or "libero10" in low or "moka" in low:
        return "libero_10"
    if "libero_spatial" in low or "spatial" in low or "black_bowl" in low:
        return "libero_spatial"
    if "libero_goal" in low or "goal" in low or "drawer" in low:
        return "libero_goal"
    if "libero_object" in low or "object" in low or "alphabet_soup" in low:
        return "libero_object"
    return "UNKNOWN"


def scan_root(root: Path, max_dirs: int, max_sentinel_dirs: int, exclude_names: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    root = Path(os.path.expandvars(os.path.expanduser(str(root))))
    rows: List[Dict[str, Any]] = []
    dirs_seen = 0
    files_seen = 0
    skipped_dirs = 0
    if not root.exists():
        return rows, {"root": str(root), "exists": False, "dirs_seen": 0, "files_seen": 0, "sentinel_dirs": 0, "skipped_dirs": 0, "truncated": False}
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirs_seen += 1
        # Prune obvious irrelevant/heavy metadata dirs by name only.
        original_len = len(dirnames)
        dirnames[:] = [d for d in dirnames if d not in set(exclude_names)]
        skipped_dirs += original_len - len(dirnames)
        files_seen += len(filenames)
        present = [name for name in SENTINELS if name in filenames]
        if present:
            p = Path(dirpath)
            temporal = [name for name in TEMPORAL_SENTINELS if name in filenames]
            row = {
                "artifact_dir": str(p),
                "scan_root": str(root),
                "suite_hint": suite_hint_from_path(p),
                "present_sentinels": ";".join(present),
                "present_temporal_sentinels": ";".join(temporal),
                "has_temporal_artifacts": bool(temporal),
                "mtime": float(p.stat().st_mtime),
            }
            for name in SENTINELS:
                row[f"path_{name}"] = str(p / name) if name in filenames else ""
            rows.append(row)
            if len(rows) >= max_sentinel_dirs:
                truncated = True
                break
        if dirs_seen >= max_dirs:
            truncated = True
            break
    summary = {"root": str(root), "exists": True, "dirs_seen": dirs_seen, "files_seen": files_seen, "sentinel_dirs": len(rows), "skipped_dirs": skipped_dirs, "truncated": truncated}
    return rows, summary


def score_candidate(record: Dict[str, Any], inv: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    rid = record_id(record)
    suite = infer_suite(record)
    artifact_dir = inv["artifact_dir"]
    low_dir = artifact_dir.lower()
    reasons: List[str] = []
    score = 0
    if rid and rid.lower() in low_dir:
        score += 100
        reasons.append("record_id_substring")
    for field in ID_FIELDS:
        value = str(record.get(field, "") or "").strip()
        if len(value) >= 4 and value.lower() in low_dir:
            score += 80
            reasons.append(f"{field}_substring")
    suite_hint = inv.get("suite_hint", "UNKNOWN")
    if suite_hint == suite:
        score += 20
        reasons.append("suite_hint_match")
    elif suite_hint != "UNKNOWN" and suite_hint != suite:
        score -= 40
        reasons.append("suite_hint_conflict")
    rtoks = set(record_tokens(record))
    ptoks = set(path_tokens(Path(artifact_dir)))
    overlap = sorted((rtoks & ptoks) - {"libero"})
    if overlap:
        score += min(30, 5 * len(overlap))
        reasons.append("token_overlap")
    if inv.get("has_temporal_artifacts"):
        score += 10
        reasons.append("temporal_artifact_present")
    return score, ";".join(reasons), overlap[:20]


def build_bindings(lib10_records: List[Dict[str, Any]], inventory: List[Dict[str, Any]], top_k: int, min_score: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidate_rows: List[Dict[str, Any]] = []
    unmatched_rows: List[Dict[str, Any]] = []
    # Prefer lib10-like or unknown dirs for lib10 records; retain conflict dirs only if exact id match gives enough score.
    inv_rows = [r for r in inventory if r.get("has_temporal_artifacts")]
    for record in lib10_records:
        scored: List[Tuple[int, str, List[str], Dict[str, Any]]] = []
        for inv in inv_rows:
            score, reasons, overlap = score_candidate(record, inv)
            if score >= min_score:
                scored.append((score, reasons, overlap, inv))
        scored.sort(key=lambda x: (-x[0], x[3].get("artifact_dir", "")))
        if not scored:
            unmatched_rows.append({
                "record_id": record_id(record),
                "suite": infer_suite(record),
                "task_name": str(first_value(record, ["task_name", "instruction", "language_instruction"], "")),
                "reason": "NO_TEMPORAL_ARTIFACT_CANDIDATE_ABOVE_MIN_SCORE",
                "record_tokens": ";".join(record_tokens(record)),
                "source_file": record.get("__source_file", ""),
                "source_line": record.get("__source_line", ""),
            })
            continue
        for rank, (score, reasons, overlap, inv) in enumerate(scored[:top_k], start=1):
            candidate_rows.append({
                "record_id": record_id(record),
                "suite": infer_suite(record),
                "rank": rank,
                "score": score,
                "match_reasons": reasons,
                "token_overlap": ";".join(overlap),
                "artifact_dir": inv["artifact_dir"],
                "suite_hint": inv.get("suite_hint", ""),
                "present_temporal_sentinels": inv.get("present_temporal_sentinels", ""),
                "path_step_records_jsonl": inv.get("path_step_records.jsonl", ""),
                "path_step_telemetry_csv": inv.get("path_step_telemetry.csv", ""),
                "path_phase_cues_csv": inv.get("path_phase_cues.csv", ""),
                "path_episode_manifest_json": inv.get("path_episode_manifest.json", ""),
                "source_file": record.get("__source_file", ""),
                "source_line": record.get("__source_line", ""),
            })
    return candidate_rows, unmatched_rows


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
    records = read_records(args.clean2000_records)
    expected_total = int(args.expected_total or 2000)
    lib10_records = [r for r in records if infer_suite(r) == "libero_10"]
    inventory: List[Dict[str, Any]] = []
    root_summaries: List[Dict[str, Any]] = []
    for root in args.scan_root:
        rows, summary = scan_root(Path(root), args.max_dirs_per_root, args.max_sentinel_dirs_per_root, args.exclude_dir_name)
        inventory.extend(rows)
        root_summaries.append(summary)
    candidate_rows, unmatched_rows = build_bindings(lib10_records, inventory, args.top_k, args.min_score)
    temporal_dirs = [r for r in inventory if r.get("has_temporal_artifacts")]
    exact_or_candidate_bound_records = len({r["record_id"] for r in candidate_rows})
    if len(records) != expected_total:
        status = "HOLD_CLEAN2000_TOTAL_COUNT_MISMATCH"
        reason = f"expected_total={expected_total} observed_total={len(records)}"
    elif not args.scan_root:
        status = "HOLD_NO_SCAN_ROOTS_PROVIDED"
        reason = "provide one or more --scan-root values"
    elif not inventory:
        status = "HOLD_NO_SENTINEL_ARTIFACT_DIRS_FOUND"
        reason = "no directories containing known sentinel files were found under scan roots"
    elif not temporal_dirs:
        status = "HOLD_NO_TEMPORAL_ARTIFACT_DIRS_FOUND"
        reason = "sentinel dirs found, but none contain step_records/step_telemetry/phase_cues/episode_manifest"
    elif exact_or_candidate_bound_records > 0:
        status = PASS_WITH_BINDINGS
        reason = "temporal artifact dirs found and at least one LIBERO-10 record has candidate bindings; review candidates before materializing registry"
    else:
        status = PASS
        reason = "temporal artifact dirs found but no LIBERO-10 bindings exceeded threshold; use inventory to identify correct roots or lower threshold only for debug"

    inv_fields = ["artifact_dir", "scan_root", "suite_hint", "present_sentinels", "present_temporal_sentinels", "has_temporal_artifacts", "mtime"] + [f"path_{name}" for name in SENTINELS]
    write_csv(out / "artifact_sentinel_directory_inventory.csv", inventory, inv_fields)
    write_csv(out / "libero10_artifact_binding_candidates.csv", candidate_rows, [
        "record_id", "suite", "rank", "score", "match_reasons", "token_overlap", "artifact_dir", "suite_hint",
        "present_temporal_sentinels", "path_step_records_jsonl", "path_step_telemetry_csv", "path_phase_cues_csv", "path_episode_manifest_json", "source_file", "source_line",
    ])
    write_csv(out / "libero10_unmatched_records.csv", unmatched_rows, ["record_id", "suite", "task_name", "reason", "record_tokens", "source_file", "source_line"])
    write_csv(out / "scan_root_summary.csv", root_summaries, ["root", "exists", "dirs_seen", "files_seen", "sentinel_dirs", "skipped_dirs", "truncated"])
    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "input_files": [str(p) for p in args.clean2000_records],
        "input_file_sha256": {str(p): sha256_file(Path(p)) for p in args.clean2000_records},
        "expected_total": expected_total,
        "observed_total": len(records),
        "libero10_record_count": len(lib10_records),
        "scan_roots": args.scan_root,
        "root_summaries": root_summaries,
        "artifact_sentinel_dir_count": len(inventory),
        "temporal_artifact_dir_count": len(temporal_dirs),
        "libero10_candidate_binding_rows": len(candidate_rows),
        "libero10_candidate_bound_record_count": exact_or_candidate_bound_records,
        "libero10_unmatched_record_count": len(unmatched_rows),
        "min_score": args.min_score,
        "top_k": args.top_k,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "interpretation": "CPU-only artifact registry recovery debug. Candidate bindings are not authoritative until reviewed/materialized by a later registry-freeze gate.",
        "boundaries": {
            "CUDA_required": "NOT_REQUIRED",
            "OpenVLA_model": "NOT_LOADED",
            "model_inference": "NOT_PERFORMED",
            "LIBERO_runtime": "NOT_PERFORMED",
            "env_reset": "NOT_PERFORMED",
            "env_set_init_state": "NOT_PERFORMED",
            "env_step": "NOT_PERFORMED",
            "rollout": "NOT_PERFORMED",
            "intervention": "NOT_PERFORMED",
            "attack_condition": "NOT_PERFORMED",
        },
        "git_commit": args.git_commit,
        "files_changed": args.files_changed,
        "tests": args.tests,
    }
    write_json(out / "clean2000_artifact_registry_recovery_report.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if not status.startswith("HOLD_") else 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--clean2000-records", action="append", required=True)
    p.add_argument("--expected-total", type=int, default=2000)
    p.add_argument("--scan-root", action="append", default=[])
    p.add_argument("--max-dirs-per-root", type=int, default=250000)
    p.add_argument("--max-sentinel-dirs-per-root", type=int, default=10000)
    p.add_argument("--exclude-dir-name", action="append", default=[".git", "__pycache__", ".cache", "wandb", "node_modules"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--min-score", type=int, default=40)
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
