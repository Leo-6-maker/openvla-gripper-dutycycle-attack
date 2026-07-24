#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

GATE = "C6_1I_PARENT_SUFFIX_STATE_BINDING_SELECTOR"
PASS = "PASS_PARENT_SUFFIX_SELECTS_STATE_INDEX_CANDIDATE"
INDEX_FIELDS = {"state_id", "state_key", "initial_state_id", "episode_idx", "episode_index", "benchmark_episode_idx", "benchmark_initial_state_index"}
TARGET_FIELD_ORDER = ["state_id", "episode_idx", "episode_index", "benchmark_episode_idx", "benchmark_initial_state_index", "initial_state_id", "state_key"]
OUT_FILES = ["parent_suffix_state_binding_selector.json", "selected_candidate_rows.csv", "handle_breakdown.csv", "checksum_report.json"]
BOUNDARIES = {"legacy_runner_execution": "NOT_PERFORMED", "OpenVLA": "NOT_PERFORMED", "LIBERO": "NOT_PERFORMED", "rollout": "NOT_PERFORMED", "intervention": "NOT_PERFORMED", "attack_condition": "NOT_PERFORMED", "artifact_mutation": "NOT_PERFORMED"}


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
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def state_suffix_index(parent_id: str) -> int | None:
    m = re.search(r"(?:^|/)state_(\d+)(?:/|$)", parent_id or "")
    return int(m.group(1)) if m else None


def parse_json_obj(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def index_handles(row: dict[str, Any]) -> list[str]:
    fields = parse_json_obj(row.get("index_fields", "{}"))
    return [f"{k}:{v}" for k, v in fields.items() if k in INDEX_FIELDS and v not in (None, "")]


def target_handles(n: int) -> list[str]:
    return [f"{field}:{n}" for field in TARGET_FIELD_ORDER]


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def handle_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if str(row.get("is_concrete_binding")) != "True":
            continue
        for handle in index_handles(row):
            counts[handle] = counts.get(handle, 0) + 1
    return counts


def selected_rows(rows: list[dict[str, Any]], handle: str, limit: int) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if handle in set(index_handles(row)):
            out.append(row)
            if len(out) >= limit:
                break
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
    observed = sha256_file(args.input_c6_1h_json)
    parent: dict[str, Any] = {}
    suffix = None
    rows: list[dict[str, Any]] = []
    breakdown: dict[str, int] = {}
    matches: list[str] = []
    chosen = ""
    status = "HOLD_C6_1H_HASH_MISMATCH"
    if observed == args.expected_c6_1h_sha256:
        c6 = read_json(args.input_c6_1h_json)
        parent = dict(c6.get("selected_parent") or {})
        if c6.get("status") != "HOLD_AMBIGUOUS_STATE_BINDING":
            status = "HOLD_C6_1H_STATUS_NOT_AMBIGUOUS"
        else:
            suffix = state_suffix_index(str(parent.get("parent_id", "")))
            if suffix is None:
                status = "HOLD_PARENT_ID_STATE_SUFFIX_MISSING"
            else:
                rows = load_rows(Path(args.candidate_csv))
                breakdown = handle_breakdown(rows)
                matches = [h for h in target_handles(suffix) if h in breakdown]
                if not matches:
                    status = "HOLD_PARENT_SUFFIX_TARGET_HANDLE_NOT_FOUND"
                elif len(set(matches)) > 1:
                    status = "HOLD_PARENT_SUFFIX_SELECTS_MULTIPLE_HANDLES"
                else:
                    chosen = matches[0]
                    status = PASS
    picked = selected_rows(rows, chosen, args.max_selected_rows) if chosen else []
    report = {"gate": GATE, "status": status, "input_c6_1h_json": str(args.input_c6_1h_json), "input_c6_1h_json_sha256": observed, "expected_c6_1h_json_sha256": args.expected_c6_1h_sha256, "candidate_csv": str(args.candidate_csv), "selected_parent": parent, "parent_state_suffix_index": suffix, "target_handles": target_handles(suffix) if suffix is not None else [], "matching_target_handles": matches, "selected_handle": chosen, "handle_count": len(breakdown), "selected_row_count_written": len(picked), "boundaries": dict(BOUNDARIES), "files_changed": args.files_changed, "git_commit": args.git_commit, "tests": args.tests}
    write_json(out / "parent_suffix_state_binding_selector.json", report)
    write_csv(out / "selected_candidate_rows.csv", picked, ["source_path", "line", "source_kind", "locator", "match_reasons", "path_fields", "index_fields", "resolved_path", "resolved_path_exists", "is_concrete_binding", "preview"])
    write_csv(out / "handle_breakdown.csv", [{"handle": k, "count": v} for k, v in sorted(breakdown.items())], ["handle", "count"])
    write_checksums(out)
    print(json.dumps(report, sort_keys=True))
    return 0 if status == PASS else 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-c6-1h-json", required=True)
    p.add_argument("--expected-c6-1h-sha256", required=True)
    p.add_argument("--candidate-csv", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--max-selected-rows", type=int, default=200)
    p.add_argument("--git-commit", required=True)
    p.add_argument("--files-changed", action="append", default=[])
    p.add_argument("--tests", action="append", default=[])
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
