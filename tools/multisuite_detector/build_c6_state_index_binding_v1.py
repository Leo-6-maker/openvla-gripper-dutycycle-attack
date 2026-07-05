#!/usr/bin/env python3
"""Build a CPU-only C6 state-index binding audit from C6_1G output.

This gate searches metadata and filesystem state artifacts for a concrete binding from
an audited initial_state_hash / parent identity to a loadable state_path, state_id,
or episode_idx. It does not import or run LIBERO/OpenVLA.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

GATE = "C6_1H_STATE_INDEX_BINDING_AUDIT_BUILD"
C6_1G_ALLOWED = {
    "HOLD_RESET_HASH_NOT_RESOLVABLE_TO_STATE_ARTIFACT",
    "PASS_STATIC_RESET_ADAPTER_PATCHABLE",
    "PASS_STATIC_RESET_ADAPTER_NOT_REQUIRED",
}
PASS_FILE_HASH = "PASS_STATE_HASH_FILE_SHA256_MATCH"
PASS_STATE_PATH = "PASS_PARENT_METADATA_BINDS_STATE_PATH"
PASS_STATE_INDEX = "PASS_PARENT_METADATA_BINDS_STATE_INDEX"
HOLD_NO_BINDING = "HOLD_NO_PARENT_STATE_BINDING"
HOLD_PATH_MISSING = "HOLD_BOUND_STATE_PATH_MISSING"
HOLD_AMBIGUOUS = "HOLD_AMBIGUOUS_STATE_BINDING"
STATE_PATH_FIELDS = ["state_path", "state_file", "initial_state_path", "initial_state_file", "init_state_path", "reset_state_path"]
STATE_INDEX_FIELDS = ["state_id", "state_key", "initial_state_id", "episode_idx", "episode_index", "benchmark_episode_idx", "benchmark_initial_state_index"]
IDENTITY_FIELDS = ["initial_state_hash", "parent_id", "episode_key", "suite", "task_id"]
META_SUFFIXES = {".json", ".jsonl", ".csv", ".txt", ".md"}
STATE_SUFFIXES = {".pkl", ".pickle", ".npz", ".npy", ".pt", ".pth", ".bin", ".h5", ".hdf5", ".msgpack", ".json"}
BOUNDARIES = {
    "legacy_runner_execution": "NOT_PERFORMED",
    "OpenVLA": "NOT_PERFORMED",
    "LIBERO": "NOT_PERFORMED",
    "rollout": "NOT_PERFORMED",
    "intervention": "NOT_PERFORMED",
    "attack_condition": "NOT_PERFORMED",
    "artifact_mutation": "NOT_PERFORMED",
}
OUT_FILES = [
    "state_index_binding_audit.json",
    "state_index_binding_candidates.csv",
    "artifact_file_candidates.csv",
    "adapter_recommendation.json",
    "checksum_report.json",
]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def as_text(x: Any) -> str:
    return "" if x is None else str(x)


def selected_parent(report: dict[str, Any]) -> dict[str, Any]:
    parent = dict(report.get("selected_parent") or {})
    if not parent.get("initial_state_hash"):
        parent["initial_state_hash"] = (((report.get("reset_binding") or {}).get("value")) or "")
    return parent


def task_token(task_id: Any) -> str:
    try:
        return f"task_{int(task_id):02d}"
    except Exception:
        return f"task_{task_id}"


def state_token(parent_id: str) -> str:
    return parent_id.rstrip("/").split("/")[-1] if parent_id else ""


def identity_match_reason(obj: Any, parent: dict[str, Any]) -> list[str]:
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False) if not isinstance(obj, str) else obj
    reasons = []
    for field in IDENTITY_FIELDS:
        val = as_text(parent.get(field))
        if val and val in text:
            reasons.append(field)
    st = state_token(as_text(parent.get("parent_id")))
    tt = task_token(parent.get("task_id"))
    suite = as_text(parent.get("suite"))
    if st and tt and suite and st in text and tt in text and suite in text:
        reasons.append("suite_task_state_tokens")
    return sorted(set(reasons))


def collect_fields(obj: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k)
            if key in STATE_PATH_FIELDS + STATE_INDEX_FIELDS and v not in (None, ""):
                out[key] = as_text(v)
            for ck, cv in collect_fields(v).items():
                out.setdefault(ck, cv)
    elif isinstance(obj, list):
        for item in obj:
            for ck, cv in collect_fields(item).items():
                out.setdefault(ck, cv)
    return out


def resolve_path(value: str, source_path: Path, roots: list[Path]) -> tuple[str, bool]:
    p = Path(value).expanduser()
    candidates = [p] if p.is_absolute() else [source_path.parent / p, *[root / p for root in roots]]
    for cand in candidates:
        if cand.exists():
            return str(cand.resolve()), True
    return str(candidates[0]), False


def candidate_from_obj(obj: Any, source_path: Path, line: int, source_kind: str, parent: dict[str, Any], roots: list[Path]) -> dict[str, Any] | None:
    reasons = identity_match_reason(obj, parent)
    if not reasons:
        return None
    fields = collect_fields(obj)
    path_fields = {k: v for k, v in fields.items() if k in STATE_PATH_FIELDS}
    index_fields = {k: v for k, v in fields.items() if k in STATE_INDEX_FIELDS}
    resolved_path = ""
    path_exists = False
    if path_fields:
        first = next(iter(path_fields.values()))
        resolved_path, path_exists = resolve_path(first, source_path, roots)
    return {
        "source_path": str(source_path),
        "line": line,
        "source_kind": source_kind,
        "match_reasons": ";".join(reasons),
        "path_fields": json.dumps(path_fields, sort_keys=True),
        "index_fields": json.dumps(index_fields, sort_keys=True),
        "resolved_path": resolved_path,
        "resolved_path_exists": path_exists,
        "is_concrete_binding": bool(path_fields or index_fields),
        "preview": json.dumps(obj, sort_keys=True, ensure_ascii=False)[:700] if not isinstance(obj, str) else obj[:700],
    }


def json_candidates(path: Path, parent: dict[str, Any], roots: list[Path]) -> list[dict[str, Any]]:
    try:
        obj = read_json(path)
    except Exception:
        return []
    rows: list[dict[str, Any]] = []

    def visit(node: Any, loc: str) -> None:
        row = candidate_from_obj(node, path, 0, "json_object", parent, roots)
        if row:
            row["locator"] = loc
            rows.append(row)
        if isinstance(node, dict):
            for k, v in node.items():
                visit(v, f"{loc}.{k}" if loc else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                visit(v, f"{loc}[{i}]")

    visit(obj, "")
    return rows


def jsonl_candidates(path: Path, parent: dict[str, Any], roots: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        row = candidate_from_obj(obj, path, i, "jsonl_object", parent, roots)
        if row:
            row["locator"] = str(i)
            rows.append(row)
    return rows


def csv_candidates(path: Path, parent: dict[str, Any], roots: list[Path]) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row_obj in enumerate(reader, start=2):
            row = candidate_from_obj(row_obj, path, i, "csv_row", parent, roots)
            if row:
                row["locator"] = str(i)
                rows.append(row)
    return rows


def line_candidates(path: Path, parent: dict[str, Any], roots: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        row = candidate_from_obj(line, path, i, "line_scan", parent, roots)
        if row:
            row["locator"] = str(i)
            rows.append(row)
    return rows


def metadata_candidates(roots: list[Path], parent: dict[str, Any], max_files: int, max_file_bytes: int) -> list[dict[str, Any]]:
    out, seen = [], set()
    scanned = 0
    for root in roots:
        if not root.exists():
            continue
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in META_SUFFIXES]
        for path in files:
            if scanned >= max_files:
                return out
            try:
                if path.stat().st_size > max_file_bytes:
                    continue
            except OSError:
                continue
            scanned += 1
            rows: list[dict[str, Any]] = []
            try:
                if path.suffix.lower() == ".json":
                    rows += json_candidates(path, parent, roots)
                elif path.suffix.lower() == ".jsonl":
                    rows += jsonl_candidates(path, parent, roots)
                elif path.suffix.lower() == ".csv":
                    rows += csv_candidates(path, parent, roots)
                rows += line_candidates(path, parent, roots)
            except Exception:
                continue
            for row in rows:
                key = (row["source_path"], row.get("line"), row.get("source_kind"), row.get("preview"))
                if key not in seen:
                    seen.add(key)
                    out.append(row)
    return out


def artifact_file_candidates(roots: list[Path], parent: dict[str, Any], max_files: int, max_file_bytes: int) -> list[dict[str, Any]]:
    out = []
    state_hash = as_text(parent.get("initial_state_hash"))
    tokens = [as_text(parent.get("suite")), task_token(parent.get("task_id")), state_token(as_text(parent.get("parent_id"))), state_hash[:12]]
    scanned = 0
    for root in roots:
        if not root.exists() or root.is_file():
            continue
        for path in root.rglob("*"):
            if scanned >= max_files:
                return out
            if not path.is_file() or path.suffix.lower() not in STATE_SUFFIXES:
                continue
            scanned += 1
            text_path = str(path)
            name_match = any(t and t in text_path for t in tokens)
            try:
                size = path.stat().st_size
            except OSError:
                continue
            digest = ""
            exact = False
            skipped = size > max_file_bytes
            if not skipped:
                digest = sha256_file(path)
                exact = bool(state_hash and digest == state_hash)
            if name_match or exact:
                out.append({"path": str(path), "size_bytes": size, "sha256": digest, "sha256_matches_initial_state_hash": exact, "name_matches_parent_tokens": name_match, "hash_skipped_too_large": skipped})
    return out


def unique_handles(rows: list[dict[str, Any]]) -> set[str]:
    handles = set()
    for r in rows:
        if r.get("resolved_path"):
            handles.add("path:" + str(r["resolved_path"]))
        idx = json.loads(r.get("index_fields") or "{}")
        for k, v in idx.items():
            handles.add(f"{k}:{v}")
    return handles


def decide_status(meta_rows: list[dict[str, Any]], file_rows: list[dict[str, Any]]) -> str:
    exact = [r for r in file_rows if str(r.get("sha256_matches_initial_state_hash")) == "True"]
    if len(exact) == 1:
        return PASS_FILE_HASH
    if len(exact) > 1:
        return HOLD_AMBIGUOUS
    concrete = [r for r in meta_rows if str(r.get("is_concrete_binding")) == "True"]
    existing_paths = [r for r in concrete if r.get("resolved_path") and str(r.get("resolved_path_exists")) == "True"]
    missing_paths = [r for r in concrete if r.get("resolved_path") and str(r.get("resolved_path_exists")) != "True"]
    index_only = [r for r in concrete if not r.get("resolved_path") and json.loads(r.get("index_fields") or "{}")]
    handles = unique_handles(existing_paths + index_only)
    if len(handles) > 1:
        return HOLD_AMBIGUOUS
    if len(existing_paths) == 1 or (len(existing_paths) > 1 and len(handles) == 1):
        return PASS_STATE_PATH
    if len(index_only) >= 1 and len(handles) == 1:
        return PASS_STATE_INDEX
    if missing_paths:
        return HOLD_PATH_MISSING
    return HOLD_NO_BINDING


def recommendation(status: str) -> str:
    if status == PASS_FILE_HASH:
        return "Use exact file SHA match as reset artifact candidate; next patch may bind this path in shim dry-run only."
    if status == PASS_STATE_PATH:
        return "Use parent-matched state_path as reset artifact candidate; next patch should pass state_path through shim dry-run only."
    if status == PASS_STATE_INDEX:
        return "Use parent-matched state index as reset candidate; next patch should bind index-to-benchmark initial state without rollout."
    if status == HOLD_PATH_MISSING:
        return "Metadata binds a path-like reset handle, but the file is missing from searched roots; locate or mount the state artifact first."
    if status == HOLD_AMBIGUOUS:
        return "Multiple distinct reset handles matched; narrow search roots or add deterministic parent/state selection before patching."
    return "No concrete state_path/state_id/episode_idx binding found; inspect upstream collector or build a state index artifact."


def write_checksums(out: Path) -> None:
    reported = {name: sha256_file(out / name) for name in OUT_FILES[:-1] if (out / name).exists()}
    write_json(out / "checksum_report.json", {"algorithm": "sha256", "reported_files": reported, "self_referential_checksum_fields": "ABSENT_BY_DESIGN"})
    present = [name for name in OUT_FILES if (out / name).exists()]
    sums = out / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(out / name)}  {name}\n" for name in present), encoding="utf-8")
    (out / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def main_from_args(args: argparse.Namespace) -> int:
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    observed = sha256_file(args.input_c6_1g_json)
    if observed != args.expected_c6_1g_sha256:
        status = "HOLD_C6_1G_HASH_MISMATCH"
        parent = {}
        meta_rows: list[dict[str, Any]] = []
        file_rows: list[dict[str, Any]] = []
    else:
        c6 = read_json(args.input_c6_1g_json)
        parent = selected_parent(c6)
        if c6.get("status") not in C6_1G_ALLOWED:
            status = "HOLD_C6_1G_STATUS_UNEXPECTED"
            meta_rows, file_rows = [], []
        else:
            roots = [Path(p).resolve() for p in args.search_root]
            roots.append(Path(args.input_c6_1g_json).parent)
            meta_rows = metadata_candidates(roots, parent, args.max_files, args.max_file_bytes)
            file_rows = artifact_file_candidates(roots, parent, args.max_files, args.max_file_bytes)
            status = decide_status(meta_rows, file_rows)
    concrete = [r for r in meta_rows if str(r.get("is_concrete_binding")) == "True"]
    report = {
        "gate": GATE,
        "status": status,
        "input_c6_1g_json": str(args.input_c6_1g_json),
        "input_c6_1g_json_sha256": observed,
        "expected_c6_1g_json_sha256": args.expected_c6_1g_sha256,
        "selected_parent": parent,
        "binding_summary": {
            "metadata_candidate_count": len(meta_rows),
            "concrete_metadata_binding_count": len(concrete),
            "artifact_file_candidate_count": len(file_rows),
            "exact_file_sha256_match_count": len([r for r in file_rows if str(r.get("sha256_matches_initial_state_hash")) == "True"]),
            "unique_handles": sorted(unique_handles(concrete)),
        },
        "recommended_next_patch": recommendation(status),
        "boundaries": dict(BOUNDARIES),
        "files_changed": args.files_changed,
        "git_commit": args.git_commit,
        "tests": args.tests,
    }
    write_json(out / "state_index_binding_audit.json", report)
    write_csv(out / "state_index_binding_candidates.csv", meta_rows, ["source_path", "line", "source_kind", "locator", "match_reasons", "path_fields", "index_fields", "resolved_path", "resolved_path_exists", "is_concrete_binding", "preview"])
    write_csv(out / "artifact_file_candidates.csv", file_rows, ["path", "size_bytes", "sha256", "sha256_matches_initial_state_hash", "name_matches_parent_tokens", "hash_skipped_too_large"])
    write_json(out / "adapter_recommendation.json", {"status": status, "recommended_next_patch": recommendation(status)})
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if status.startswith("PASS_") else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-c6-1g-json", required=True)
    ap.add_argument("--expected-c6-1g-sha256", required=True)
    ap.add_argument("--search-root", action="append", default=[])
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--max-files", type=int, default=100000)
    ap.add_argument("--max-file-bytes", type=int, default=64 * 1024 * 1024)
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--files-changed", action="append", default=[])
    ap.add_argument("--tests", action="append", default=[])
    return main_from_args(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
