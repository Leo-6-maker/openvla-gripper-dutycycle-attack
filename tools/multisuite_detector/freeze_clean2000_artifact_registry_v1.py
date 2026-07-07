#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

GATE = "D0D_CLEAN2000_ARTIFACT_REGISTRY_FREEZE"
PASS = "PASS_CLEAN2000_ARTIFACT_REGISTRY_FROZEN"
PASS_PARTIAL_DEBUG = "PASS_CLEAN2000_ARTIFACT_REGISTRY_PARTIAL_DEBUG_ONLY"
OUT_FILES = [
    "clean2000_artifact_registry_freeze_report.json",
    "clean2000_artifact_registry.csv",
    "clean2000_artifact_registry_rejections.csv",
    "clean2000_artifact_registry_ambiguities.csv",
    "checksum_report.json",
]
SUITES = ["libero_spatial", "libero_goal", "libero_object", "libero_10"]
ID_FIELDS = ["parent_id", "episode_key", "run_id", "record_id", "id"]
TEXT_FIELDS = ["suite", "suite_name", "benchmark", "libero_suite", "task_id", "task_name", "instruction", "language_instruction"]
TEMPORAL_PATH_FIELDS = [
    "path_step_records_jsonl",
    "path_step_telemetry_csv",
    "path_phase_cues_csv",
    "path_episode_manifest_json",
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
        rows = []
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            row = dict(row)
            row["__source_file"] = str(path)
            row["__source_line"] = line_no
            rows.append(row)
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


def parse_float(row: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def parse_int(row: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return default


def has_temporal_path(row: Dict[str, Any]) -> bool:
    return any(str(row.get(k, "") or "").strip() for k in TEMPORAL_PATH_FIELDS) or bool(str(row.get("present_temporal_sentinels", "") or "").strip())


def temporal_paths_exist(row: Dict[str, Any]) -> Tuple[bool, str]:
    present = []
    missing = []
    for key in TEMPORAL_PATH_FIELDS:
        value = str(row.get(key, "") or "").strip()
        if not value:
            continue
        if Path(value).exists():
            present.append(value)
        else:
            missing.append(value)
    return bool(present), ";".join(missing)


def source_record_map(source_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for row in source_rows:
        rid = record_id(row)
        if rid in out:
            raise ValueError(f"duplicate source record_id={rid}")
        out[rid] = row
    return out


def group_candidates(candidate_rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        rid = str(row.get("record_id", "") or "").strip()
        if not rid:
            continue
        by_id[rid].append(row)
    for rid in by_id:
        by_id[rid].sort(key=lambda r: (-parse_float(r, "score"), parse_int(r, "rank", 999999), str(r.get("artifact_dir", ""))))
    return by_id


def decide_one(rid: str, source_row: Dict[str, Any], candidates: List[Dict[str, Any]], args: argparse.Namespace) -> Tuple[Dict[str, Any] | None, List[Dict[str, Any]], List[Dict[str, Any]]]:
    rejects: List[Dict[str, Any]] = []
    ambiguities: List[Dict[str, Any]] = []
    suite = infer_suite(source_row)
    if not candidates:
        rejects.append(reject_row(rid, suite, "NO_CANDIDATES", source_row, None, ""))
        return None, rejects, ambiguities
    top = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    score = parse_float(top, "score")
    second_score = parse_float(second, "score") if second else float("-inf")
    margin = score - second_score if second else float("inf")
    rank = parse_int(top, "rank", 999999)
    suite_hint = str(top.get("suite_hint", "UNKNOWN") or "UNKNOWN")
    artifact_dir = str(top.get("artifact_dir", "") or "").strip()
    reasons = []
    if args.require_rank1 and rank != 1:
        reasons.append(f"rank_not_1:{rank}")
    if score < args.min_score:
        reasons.append(f"score_below_min:{score}<{args.min_score}")
    if len(candidates) > 1 and margin < args.min_margin:
        reasons.append(f"margin_below_min:{margin}<{args.min_margin}")
    if suite_hint not in set(args.allowed_suite_hint):
        reasons.append(f"suite_hint_not_allowed:{suite_hint}")
    if not has_temporal_path(top):
        reasons.append("no_temporal_path_in_candidate")
    temporal_exists, missing_paths = temporal_paths_exist(top)
    if args.require_files_exist and not temporal_exists:
        reasons.append("no_existing_temporal_file")
    if artifact_dir and args.require_files_exist and not Path(artifact_dir).exists():
        reasons.append("artifact_dir_missing")
    if reasons:
        rejects.append(reject_row(rid, suite, "CANDIDATE_REJECTED", source_row, top, ";".join(reasons)))
        if len(candidates) > 1:
            ambiguities.append(ambiguity_row(rid, suite, top, second, margin, ";".join(reasons)))
        return None, rejects, ambiguities
    if len(candidates) > 1 and margin < args.ambiguity_warn_margin:
        ambiguities.append(ambiguity_row(rid, suite, top, second, margin, "accepted_but_low_margin_warn"))
    accepted = {
        "record_id": rid,
        "suite": suite,
        "task_id": str(first_value(source_row, ["task_id"], "")),
        "task_name": str(first_value(source_row, ["task_name", "instruction", "language_instruction"], "")),
        "clean_success": str(first_value(source_row, ["clean_success", "success", "task_success", "episode_success"], "")),
        "artifact_dir": artifact_dir,
        "suite_hint": suite_hint,
        "score": score,
        "rank": rank,
        "score_margin_to_rank2": margin if second else "INF",
        "match_reasons": str(top.get("match_reasons", "")),
        "token_overlap": str(top.get("token_overlap", "")),
        "present_temporal_sentinels": str(top.get("present_temporal_sentinels", "")),
        "path_step_records_jsonl": str(top.get("path_step_records_jsonl", "")),
        "path_step_telemetry_csv": str(top.get("path_step_telemetry_csv", "")),
        "path_phase_cues_csv": str(top.get("path_phase_cues_csv", "")),
        "path_episode_manifest_json": str(top.get("path_episode_manifest_json", "")),
        "missing_temporal_paths": missing_paths,
        "registry_status": "FROZEN_CANDIDATE_BINDING",
    }
    return accepted, rejects, ambiguities


def reject_row(rid: str, suite: str, reason: str, source_row: Dict[str, Any], cand: Dict[str, Any] | None, detail: str) -> Dict[str, Any]:
    cand = cand or {}
    return {
        "record_id": rid,
        "suite": suite,
        "reject_reason": reason,
        "detail": detail,
        "candidate_rank": cand.get("rank", ""),
        "candidate_score": cand.get("score", ""),
        "candidate_artifact_dir": cand.get("artifact_dir", ""),
        "candidate_suite_hint": cand.get("suite_hint", ""),
        "source_file": source_row.get("__source_file", ""),
        "source_line": source_row.get("__source_line", ""),
    }


def ambiguity_row(rid: str, suite: str, top: Dict[str, Any], second: Dict[str, Any] | None, margin: float, note: str) -> Dict[str, Any]:
    second = second or {}
    return {
        "record_id": rid,
        "suite": suite,
        "margin": margin,
        "note": note,
        "top_score": top.get("score", ""),
        "top_artifact_dir": top.get("artifact_dir", ""),
        "top_reasons": top.get("match_reasons", ""),
        "second_score": second.get("score", ""),
        "second_artifact_dir": second.get("artifact_dir", ""),
        "second_reasons": second.get("match_reasons", ""),
    }


def duplicate_artifact_rejections(registry_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_dir = defaultdict(list)
    for row in registry_rows:
        by_dir[str(row.get("artifact_dir", ""))].append(row)
    rejects = []
    for artifact_dir, rows in by_dir.items():
        if artifact_dir and len(rows) > 1:
            for row in rows:
                rejects.append({
                    "record_id": row["record_id"],
                    "suite": row["suite"],
                    "reject_reason": "DUPLICATE_ARTIFACT_DIR_BINDING",
                    "detail": f"artifact_dir shared by {len(rows)} records",
                    "candidate_rank": row.get("rank", ""),
                    "candidate_score": row.get("score", ""),
                    "candidate_artifact_dir": artifact_dir,
                    "candidate_suite_hint": row.get("suite_hint", ""),
                    "source_file": "",
                    "source_line": "",
                })
    return rejects


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
    candidates = read_csv(Path(args.candidate_bindings))
    src_map = source_record_map(source_rows)
    by_id = group_candidates(candidates)
    target_suite = set(args.target_suite)
    target_records = {rid: row for rid, row in src_map.items() if infer_suite(row) in target_suite}
    registry_rows: List[Dict[str, Any]] = []
    rejects: List[Dict[str, Any]] = []
    ambiguities: List[Dict[str, Any]] = []
    for rid, row in sorted(target_records.items()):
        accepted, rj, amb = decide_one(rid, row, by_id.get(rid, []), args)
        if accepted:
            registry_rows.append(accepted)
        rejects.extend(rj)
        ambiguities.extend(amb)
    dup_rejects = duplicate_artifact_rejections(registry_rows) if args.require_unique_artifact_dir else []
    if dup_rejects:
        dup_ids = {r["record_id"] for r in dup_rejects}
        registry_rows = [r for r in registry_rows if r["record_id"] not in dup_ids]
        rejects.extend(dup_rejects)
    expected_target = args.expected_target if args.expected_target > 0 else len(target_records)
    complete = len(registry_rows) == expected_target and not rejects
    if len(source_rows) != args.expected_total:
        status = "HOLD_CLEAN2000_TOTAL_COUNT_MISMATCH"
        reason = f"expected_total={args.expected_total} observed_total={len(source_rows)}"
    elif len(target_records) != expected_target:
        status = "HOLD_TARGET_RECORD_COUNT_MISMATCH"
        reason = f"expected_target={expected_target} observed_target={len(target_records)}"
    elif complete:
        status = PASS
        reason = ""
    elif args.allow_partial_debug and registry_rows:
        status = PASS_PARTIAL_DEBUG
        reason = "partial debug registry emitted; not authoritative and not training-ready"
    else:
        status = "HOLD_ARTIFACT_REGISTRY_FREEZE_INCOMPLETE"
        reason = f"accepted={len(registry_rows)} expected={expected_target} rejects={len(rejects)} ambiguities={len(ambiguities)}"
    write_csv(out / "clean2000_artifact_registry.csv", registry_rows, [
        "record_id", "suite", "task_id", "task_name", "clean_success", "artifact_dir", "suite_hint",
        "score", "rank", "score_margin_to_rank2", "match_reasons", "token_overlap", "present_temporal_sentinels",
        "path_step_records_jsonl", "path_step_telemetry_csv", "path_phase_cues_csv", "path_episode_manifest_json",
        "missing_temporal_paths", "registry_status",
    ])
    write_csv(out / "clean2000_artifact_registry_rejections.csv", rejects, [
        "record_id", "suite", "reject_reason", "detail", "candidate_rank", "candidate_score", "candidate_artifact_dir", "candidate_suite_hint", "source_file", "source_line",
    ])
    write_csv(out / "clean2000_artifact_registry_ambiguities.csv", ambiguities, [
        "record_id", "suite", "margin", "note", "top_score", "top_artifact_dir", "top_reasons", "second_score", "second_artifact_dir", "second_reasons",
    ])
    report = {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "clean2000_records": args.clean2000_records,
        "clean2000_records_sha256": sha256_file(Path(args.clean2000_records)),
        "candidate_bindings": args.candidate_bindings,
        "candidate_bindings_sha256": sha256_file(Path(args.candidate_bindings)),
        "expected_total": args.expected_total,
        "observed_total": len(source_rows),
        "target_suites": list(args.target_suite),
        "expected_target": expected_target,
        "observed_target": len(target_records),
        "accepted_count": len(registry_rows),
        "rejection_count": len(rejects),
        "ambiguity_count": len(ambiguities),
        "accepted_by_suite": dict(Counter(row["suite"] for row in registry_rows)),
        "rejections_by_reason": dict(Counter(row["reject_reason"] for row in rejects)),
        "parameters": {
            "min_score": args.min_score,
            "min_margin": args.min_margin,
            "ambiguity_warn_margin": args.ambiguity_warn_margin,
            "allowed_suite_hint": list(args.allowed_suite_hint),
            "require_rank1": args.require_rank1,
            "require_files_exist": args.require_files_exist,
            "require_unique_artifact_dir": args.require_unique_artifact_dir,
            "allow_partial_debug": args.allow_partial_debug,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "interpretation": "CPU-only registry freeze. PASS means target records have unique reviewed artifact bindings and can feed the event resolver. PARTIAL_DEBUG is not training-ready.",
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
    write_json(out / "clean2000_artifact_registry_freeze_report.json", report)
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if not status.startswith("HOLD_") else 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--clean2000-records", required=True)
    p.add_argument("--candidate-bindings", required=True)
    p.add_argument("--target-suite", action="append", default=["libero_10"])
    p.add_argument("--expected-total", type=int, default=2000)
    p.add_argument("--expected-target", type=int, default=500)
    p.add_argument("--min-score", type=float, default=40.0)
    p.add_argument("--min-margin", type=float, default=0.0)
    p.add_argument("--ambiguity-warn-margin", type=float, default=10.0)
    p.add_argument("--allowed-suite-hint", action="append", default=["libero_10", "UNKNOWN"])
    p.add_argument("--require-rank1", action="store_true")
    p.add_argument("--require-files-exist", action="store_true")
    p.add_argument("--require-unique-artifact-dir", action="store_true")
    p.add_argument("--allow-partial-debug", action="store_true")
    p.add_argument("--output-root", required=True)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
