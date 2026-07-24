#!/usr/bin/env python3
"""Static audit for binding C6 parent reset identity into the legacy runner source."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

GATE = "C6_1G_LEGACY_RUNNER_SOURCE_RESET_ADAPTER_STATIC_AUDIT"
C6_1F_PASS = "PASS_SHIM_DRY_RUN_RESET_ARGS_BOUND"
PASS_NOT_REQUIRED = "PASS_STATIC_RESET_ADAPTER_NOT_REQUIRED"
PASS_PATCHABLE = "PASS_STATIC_RESET_ADAPTER_PATCHABLE"
RESET_ARGS = {
    "--initial-state-hash",
    "--initial_state_hash",
    "--initial-state",
    "--initial_state",
    "--state-path",
    "--state_path",
    "--state-id",
    "--state_id",
    "--episode-idx",
    "--episode_idx",
}
PARENT_STATE_ARGS = RESET_ARGS | {"--parent-id", "--episode-key", "--suite", "--task-id"}
SEARCH_TERMS = [
    "initial_state",
    "init_state",
    "state_hash",
    "state_path",
    "reset",
    "set_state",
    "sim.set_state",
    "env.reset",
    "initial_states",
    "episode_idx",
    "task_id",
    "suite",
]
STATE_ARTIFACT_FIELDS = [
    "state_path",
    "state_file",
    "initial_state_path",
    "initial_state_file",
    "init_state_path",
    "init_state_file",
    "reset_state_path",
    "reset_state_file",
    "state_id",
    "state_key",
    "initial_state_id",
    "episode_idx",
    "episode_index",
    "benchmark_episode_idx",
    "benchmark_initial_state_index",
]
OUTPUT_FILES = [
    "legacy_runner_source_reset_adapter_static_audit.json",
    "source_matches.csv",
    "reset_resolution_candidates.csv",
    "adapter_recommendation.json",
    "checksum_report.json",
]
BOUNDARIES = {
    "legacy_runner_execution": "NOT_PERFORMED",
    "OpenVLA": "NOT_PERFORMED",
    "LIBERO": "NOT_PERFORMED",
    "rollout": "NOT_PERFORMED",
    "intervention": "NOT_PERFORMED",
    "attack_condition": "NOT_PERFORMED",
    "artifact_mutation": "NOT_PERFORMED",
}


class Hold(Exception):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def accepted_args_from_source(text: str) -> set[str]:
    return set(re.findall(r"add_argument\(\s*[\"'](--[^\"']+)", text))


def normalize_dest(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


def line_records(path: Path, root: Path, terms: list[str]) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = str(path)
    rows = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        matched = [term for term in terms if term in line]
        if matched:
            rows.append({"path": rel, "line": lineno, "matched_terms": ";".join(matched), "text": line.strip()[:400]})
    return rows


def safe_sources(root: Path, extra: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for p in extra:
        if p.exists() and p.is_file():
            candidates.append(p)
    for sub in ["scripts", "tools", "experiments"]:
        d = root / sub
        if d.exists():
            candidates.extend(p for p in d.rglob("*.py") if p.is_file())
    seen = set()
    out = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def reset_usage_evidence(path: Path, accepted_reset_args: list[str]) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    dests = {normalize_dest(arg) for arg in accepted_reset_args}
    evidence = []
    for idx, line in enumerate(lines, start=1):
        lower = line.lower()
        if not any(dest in line for dest in dests):
            continue
        if any(token in lower for token in ["reset", "set_state", "load_state", "initial_states"]):
            evidence.append({"path": str(path), "line": idx, "text": line.strip()[:400]})
            continue
        window = "\n".join(lines[idx - 1 : min(len(lines), idx + 8)]).lower()
        if any(token in window for token in ["reset", "set_state", "load_state", "initial_states"]):
            evidence.append({"path": str(path), "line": idx, "text": line.strip()[:400]})
    return evidence


def reset_insertion_evidence(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    evidence = []
    for idx, line in enumerate(text.splitlines(), start=1):
        lower = line.lower()
        if any(token in lower for token in ["env.reset", ".reset(", "set_state", "sim.set_state", "load_state", "initial_states"]):
            evidence.append({"path": str(path), "line": idx, "text": line.strip()[:400]})
    return evidence


def candidate_fields_from_line(line: str) -> list[str]:
    return [field for field in STATE_ARTIFACT_FIELDS if field in line]


def contains_hash(obj: Any, state_hash: str) -> bool:
    if isinstance(obj, dict):
        return any(contains_hash(k, state_hash) or contains_hash(v, state_hash) for k, v in obj.items())
    if isinstance(obj, list):
        return any(contains_hash(v, state_hash) for v in obj)
    return state_hash in str(obj)


def collect_artifact_fields(obj: Any) -> dict[str, str]:
    found: dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k)
            if key in STATE_ARTIFACT_FIELDS and v not in (None, ""):
                found[key] = str(v)
            for child_key, child_value in collect_artifact_fields(v).items():
                found.setdefault(child_key, child_value)
    elif isinstance(obj, list):
        for item in obj:
            for child_key, child_value in collect_artifact_fields(item).items():
                found.setdefault(child_key, child_value)
    return found


def preview_json(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)[:600]
    except TypeError:
        return str(obj)[:600]


def structured_json_candidates(path: Path, state_hash: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        obj = read_json(path)
    except Exception:
        return rows

    def visit(node: Any, locator: str) -> None:
        if isinstance(node, dict):
            if contains_hash(node, state_hash):
                fields = collect_artifact_fields(node)
                rows.append(
                    {
                        "path": str(path),
                        "line": 0,
                        "source_kind": "json_object",
                        "hash_found": True,
                        "candidate_fields": ";".join(sorted(fields)),
                        "candidate_values": json.dumps(fields, sort_keys=True),
                        "resolves_to_state_artifact": bool(fields),
                        "text": preview_json(node),
                    }
                )
            for key, value in node.items():
                visit(value, f"{locator}.{key}" if locator else str(key))
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                visit(value, f"{locator}[{idx}]")

    visit(obj, "")
    return rows


def structured_jsonl_candidates(path: Path, state_hash: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return rows
    for lineno, line in enumerate(lines, start=1):
        if state_hash not in line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        fields = collect_artifact_fields(obj) if contains_hash(obj, state_hash) else {}
        rows.append(
            {
                "path": str(path),
                "line": lineno,
                "source_kind": "jsonl_object",
                "hash_found": True,
                "candidate_fields": ";".join(sorted(fields)),
                "candidate_values": json.dumps(fields, sort_keys=True),
                "resolves_to_state_artifact": bool(fields),
                "text": preview_json(obj),
            }
        )
    return rows


def structured_csv_candidates(path: Path, state_hash: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            artifact_fields = [field for field in fieldnames if field in STATE_ARTIFACT_FIELDS]
            for lineno, row in enumerate(reader, start=2):
                if not any(state_hash in str(value) for value in row.values()):
                    continue
                values = {field: row.get(field, "") for field in artifact_fields if row.get(field, "") not in (None, "")}
                rows.append(
                    {
                        "path": str(path),
                        "line": lineno,
                        "source_kind": "csv_row",
                        "hash_found": True,
                        "candidate_fields": ";".join(sorted(values)),
                        "candidate_values": json.dumps(values, sort_keys=True),
                        "resolves_to_state_artifact": bool(values),
                        "text": json.dumps(row, sort_keys=True)[:600],
                    }
                )
    except OSError:
        return rows
    return rows


def line_scan_candidates(path: Path, state_hash: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return rows
    header_fields = candidate_fields_from_line(lines[0]) if path.suffix.lower() == ".csv" and lines else []
    for lineno, line in enumerate(lines, start=1):
        if state_hash not in line:
            continue
        fields = sorted(set(candidate_fields_from_line(line) + header_fields))
        rows.append(
            {
                "path": str(path),
                "line": lineno,
                "source_kind": "line_scan",
                "hash_found": True,
                "candidate_fields": ";".join(fields),
                "candidate_values": "{}",
                "resolves_to_state_artifact": bool(fields),
                "text": line.strip()[:600],
            }
        )
    return rows


def candidate_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (str(row.get("path")), str(row.get("line")), str(row.get("source_kind")), str(row.get("text")))


def scan_resolution_candidates(
    search_roots: list[Path],
    state_hash: str,
    *,
    max_files: int,
    max_file_bytes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    suffixes = {".json", ".jsonl", ".csv", ".txt", ".md"}
    scanned = 0
    for root in search_roots:
        if not root.exists():
            continue
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]
        for path in files:
            if scanned >= max_files:
                return rows
            try:
                if path.stat().st_size > max_file_bytes:
                    continue
            except OSError:
                continue
            scanned += 1
            suffix = path.suffix.lower()
            candidates: list[dict[str, Any]] = []
            if suffix == ".json":
                candidates.extend(structured_json_candidates(path, state_hash))
            elif suffix == ".jsonl":
                candidates.extend(structured_jsonl_candidates(path, state_hash))
            elif suffix == ".csv":
                candidates.extend(structured_csv_candidates(path, state_hash))
            candidates.extend(line_scan_candidates(path, state_hash))
            for row in candidates:
                key = candidate_key(row)
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
    return rows


def make_adapter_classification(
    accepted_reset_args: list[str],
    uses_reset_arg: bool,
    resolves: bool,
    insertion: bool,
    runner_exists: bool,
) -> str:
    if not runner_exists:
        return "BLOCKED_NO_RESET_ENTRYPOINT"
    if accepted_reset_args and uses_reset_arg:
        return "NO_ADAPTER_NEEDED_LEGACY_RUNNER_ACCEPTS_RESET_DIRECTLY"
    if not resolves:
        return "BLOCKED_HASH_NOT_RESOLVABLE"
    if accepted_reset_args and not uses_reset_arg:
        return "ADAPTER_NEEDED_HASH_TO_STATE_PATH"
    if insertion:
        return "ADAPTER_NEEDED_HASH_TO_STATE_PATH"
    return "BLOCKED_NO_RESET_ENTRYPOINT"


def status_from(
    accepted_reset_args: list[str],
    uses_reset_arg: bool,
    resolves: bool,
    insertion: bool,
) -> str:
    if accepted_reset_args and uses_reset_arg:
        return PASS_NOT_REQUIRED
    if accepted_reset_args and not uses_reset_arg:
        return "HOLD_LEGACY_RUNNER_ARG_PARSED_BUT_NOT_USED"
    if resolves and insertion:
        return PASS_PATCHABLE
    if not resolves:
        return "HOLD_RESET_HASH_NOT_RESOLVABLE_TO_STATE_ARTIFACT"
    return "HOLD_LEGACY_RUNNER_RESET_ARG_NOT_ACCEPTED"


def selected_parent_from_c6_1f(c6_1f: dict[str, Any]) -> dict[str, Any]:
    parent = dict(c6_1f.get("selected_parent") or {})
    if "initial_state_hash" not in parent:
        parent["initial_state_hash"] = ""
    return parent


def find_legacy_runner_path(c6_1f: dict[str, Any], shim_result: dict[str, Any], repo_root: Path, override: str | None) -> Path:
    if override:
        return (repo_root / override).resolve() if not Path(override).is_absolute() else Path(override)
    legacy = shim_result.get("legacy_runner") or "scripts/v4_run_eval_openvla.py"
    p = Path(str(legacy))
    return (repo_root / p).resolve() if not p.is_absolute() else p


def audit(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    repo_root = Path(args.repo_root).resolve()
    c6_1f_path = Path(args.input_c6_1f_json)
    observed = sha256_file(c6_1f_path)
    if observed != args.expected_c6_1f_sha256:
        raise Hold("HOLD_C6_1F_HASH_MISMATCH", "C6_1F artifact hash mismatch")
    c6_1f = read_json(c6_1f_path)
    if c6_1f.get("status") != C6_1F_PASS:
        raise Hold("HOLD_C6_1F_STATUS_NOT_PASS", f"C6_1F status is {c6_1f.get('status')!r}")
    shim_result_path = Path(args.shim_result_json) if args.shim_result_json else c6_1f_path.parent / "shim_result.json"
    shim_result = read_json(shim_result_path) if shim_result_path.exists() else {}
    parent = selected_parent_from_c6_1f(c6_1f)
    state_hash = str(parent.get("initial_state_hash", ""))
    runner_path = find_legacy_runner_path(c6_1f, shim_result, repo_root, args.legacy_runner)
    shim_path = repo_root / "scripts/c6_run_one_condition_openvla_libero.py"

    if not runner_path.exists():
        raise Hold("HOLD_LEGACY_RUNNER_SOURCE_NOT_FOUND", f"legacy runner source not found: {runner_path}")

    runner_text = runner_path.read_text(encoding="utf-8", errors="replace")
    accepted = accepted_args_from_source(runner_text)
    accepted_reset_args = sorted(accepted & RESET_ARGS)
    accepted_parent_state_args = sorted(accepted & PARENT_STATE_ARGS)
    usage = reset_usage_evidence(runner_path, accepted_reset_args)
    insertion = reset_insertion_evidence(runner_path)

    roots = [Path(p).resolve() for p in args.search_root]
    roots.extend([c6_1f_path.parent, repo_root / "reports", repo_root / "docs"])
    candidates = (
        scan_resolution_candidates(
            roots,
            state_hash,
            max_files=args.max_files,
            max_file_bytes=args.max_file_bytes,
        )
        if state_hash
        else []
    )
    resolves = any(bool(r.get("resolves_to_state_artifact")) for r in candidates)
    hash_found = bool(candidates)

    status = status_from(accepted_reset_args, bool(usage), resolves, bool(insertion))
    classification = make_adapter_classification(accepted_reset_args, bool(usage), resolves, bool(insertion), True)
    source_matches = []
    for p in safe_sources(repo_root, [runner_path, shim_path]):
        try:
            source_matches.extend(line_records(p, repo_root, SEARCH_TERMS))
        except OSError:
            continue

    recommendation = {
        "classification": classification,
        "recommended_next_patch": (
            "No adapter needed; next gate may remain static unless a no-model runner dry-run is added."
            if classification == "NO_ADAPTER_NEEDED_LEGACY_RUNNER_ACCEPTS_RESET_DIRECTLY"
            else "Patch the C6 shim/adapter to resolve initial_state_hash to the identified state artifact or deterministic index before any rollout."
            if classification.startswith("ADAPTER_NEEDED")
            else "Hold; first create a resolvable hash-to-state artifact/index binding."
        ),
    }
    report = {
        "gate": GATE,
        "status": status,
        "input_c6_1f_json": str(c6_1f_path),
        "input_c6_1f_json_sha256": observed,
        "input_c6_1f_expected_sha256": args.expected_c6_1f_sha256,
        "selected_parent": parent,
        "shim": {
            "path": str(shim_path.relative_to(repo_root)) if shim_path.exists() else str(shim_path),
            "passes_initial_state_hash": state_hash in " ".join(map(str, (c6_1f.get("executed_command") or {}).get("argv", []))),
            "dry_run_only_previous_gate": (c6_1f.get("executed_command") or {}).get("mode") == "SHIM_DRY_RUN_ONLY",
        },
        "legacy_runner": {
            "path": str(runner_path),
            "source_exists": True,
            "accepted_reset_args": accepted_reset_args,
            "accepted_parent_or_state_args": accepted_parent_state_args,
            "uses_reset_arg_for_env_reset": bool(usage),
            "reset_code_evidence": usage,
            "reset_insertion_evidence": insertion,
        },
        "reset_resolution": {
            "initial_state_hash_found": hash_found,
            "resolves_to_state_artifact": resolves,
            "candidate_fields": sorted({field for row in candidates for field in str(row.get("candidate_fields", "")).split(";") if field}),
            "candidate_artifacts": candidates,
            "structured_resolution_enabled": True,
            "max_files": args.max_files,
            "max_file_bytes": args.max_file_bytes,
        },
        "adapter_classification": classification,
        "recommended_next_patch": recommendation["recommended_next_patch"],
        "boundaries": dict(BOUNDARIES),
        "files_changed": args.files_changed,
        "git_commit": args.git_commit,
        "tests": args.tests,
    }
    return report, source_matches, candidates, recommendation


def empty_report(args: argparse.Namespace, status: str, reason: str) -> dict[str, Any]:
    return {
        "gate": GATE,
        "status": status,
        "reason": reason,
        "input_c6_1f_json": str(args.input_c6_1f_json),
        "input_c6_1f_expected_sha256": args.expected_c6_1f_sha256,
        "selected_parent": {},
        "shim": {},
        "legacy_runner": {"source_exists": False, "accepted_reset_args": [], "accepted_parent_or_state_args": [], "uses_reset_arg_for_env_reset": False, "reset_code_evidence": []},
        "reset_resolution": {"initial_state_hash_found": False, "resolves_to_state_artifact": False, "candidate_fields": [], "candidate_artifacts": [], "structured_resolution_enabled": True},
        "adapter_classification": "BLOCKED_NO_RESET_ENTRYPOINT" if status == "HOLD_LEGACY_RUNNER_SOURCE_NOT_FOUND" else "BLOCKED_HASH_NOT_RESOLVABLE",
        "recommended_next_patch": "Resolve the blocking HOLD before patching runtime execution.",
        "boundaries": dict(BOUNDARIES),
        "files_changed": args.files_changed,
        "git_commit": args.git_commit,
        "tests": args.tests,
    }


def write_checksum_artifacts(output_root: Path) -> None:
    reported = {name: sha256_file(output_root / name) for name in OUTPUT_FILES[:-1] if (output_root / name).exists()}
    write_json(
        output_root / "checksum_report.json",
        {
            "algorithm": "sha256",
            "reported_files": reported,
            "self_referential_checksum_fields": "ABSENT_BY_DESIGN",
        },
    )
    present = [name for name in OUTPUT_FILES if (output_root / name).exists()]
    sums = output_root / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(output_root / name)}  {name}\n" for name in present), encoding="utf-8")
    sidecar = output_root / "SHA256SUMS.sha256"
    sidecar.write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-c6-1f-json", required=True)
    p.add_argument("--expected-c6-1f-sha256", required=True)
    p.add_argument("--shim-result-json")
    p.add_argument("--legacy-runner")
    p.add_argument("--search-root", action="append", default=[])
    p.add_argument("--max-files", type=int, default=20000)
    p.add_argument("--max-file-bytes", type=int, default=5_000_000)
    p.add_argument("--output-root", required=True)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    args = p.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        report, matches, candidates, recommendation = audit(args)
        rc = 0 if str(report["status"]).startswith("PASS_") else 2
    except Hold as exc:
        report = empty_report(args, exc.status, exc.reason)
        matches, candidates, recommendation = [], [], {"classification": report["adapter_classification"], "recommended_next_patch": report["recommended_next_patch"]}
        rc = 2
    write_json(output_root / "legacy_runner_source_reset_adapter_static_audit.json", report)
    write_csv(output_root / "source_matches.csv", matches, ["path", "line", "matched_terms", "text"])
    write_csv(
        output_root / "reset_resolution_candidates.csv",
        candidates,
        ["path", "line", "source_kind", "hash_found", "candidate_fields", "candidate_values", "resolves_to_state_artifact", "text"],
    )
    write_json(output_root / "adapter_recommendation.json", recommendation)
    write_checksum_artifacts(output_root)
    print(json.dumps(report, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
