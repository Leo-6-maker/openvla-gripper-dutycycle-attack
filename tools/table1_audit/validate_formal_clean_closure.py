from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from tools.table1_audit.common import (
    add_path_arg,
    canonical_json,
    job_key,
    load_json,
    load_jsonl,
    output_dir,
    parent_key,
    replicate_key,
    sha256_file,
    write_json,
)


LEGAL_TERMINAL_INVALID = {
    "TERMINAL_INVALID",
    "SCIENTIFIC_INVALID",
    "QUARANTINED",
    "RETRY_EXHAUSTED",
}


def classify_summary(path: Path) -> tuple[str, dict | None, str]:
    if not path.exists():
        return "active_or_incomplete", None, "missing episode_summary.json"
    if path.stat().st_size == 0:
        return "malformed", None, "zero-byte episode_summary.json"
    try:
        data = load_json(path)
    except Exception as exc:
        return "malformed", None, str(exc)
    status = str(data.get("terminal_status") or data.get("status") or data.get("result_status") or "")
    if data.get("task_success") is not None or status in {"COMPLETE", "SUCCESS", "OK"}:
        return "complete", data, ""
    if status in LEGAL_TERMINAL_INVALID:
        return "terminal_invalid", data, status
    return "terminal_invalid", data, f"unrecognized terminal status: {status or 'missing'}"


def validate(args: argparse.Namespace) -> dict:
    manifest = args.manifest.resolve()
    rows = load_jsonl(manifest)
    manifest_dir = manifest.parent
    condition_root = args.condition_root.resolve()
    problems: list[dict] = []

    dirs = []
    for row in rows:
        try:
            dirs.append(str(output_dir(row, manifest_dir).resolve()))
        except Exception as exc:
            problems.append({"class": "manifest_error", "job_key": job_key(row), "detail": str(exc)})
            dirs.append("")
    dup_dirs = {d for d, n in Counter(dirs).items() if d and n > 1}
    for row, d in zip(rows, dirs):
        if d in dup_dirs:
            problems.append({"class": "duplicate_output", "job_key": job_key(row), "output_dir": d})

    parents: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        parents[parent_key(row)].append(row)
    for key, items in sorted(parents.items()):
        reps = Counter(replicate_key(r) for r in items)
        if len(items) != args.expected_replicates or any(v != 1 for v in reps.values()):
            problems.append({"class": "replicate_count", "parent": list(key), "count": len(items), "replicates": dict(reps)})

    row_classes = Counter()
    accepted_job_keys = []
    rows_out = []
    provenance_values: dict[str, set[str]] = defaultdict(set)
    manifest_sha = sha256_file(manifest)

    for row in rows:
        out = output_dir(row, manifest_dir).resolve()
        summary_path = out / "episode_summary.json"
        cls, summary, detail = classify_summary(summary_path)
        row_classes[cls] += 1
        rec = {
            "job_key": job_key(row),
            "parent_key": list(parent_key(row)),
            "replicate": replicate_key(row),
            "output_dir": str(out),
            "class": cls,
            "detail": detail,
            "retry_attempt": row.get("retry_attempt") or row.get("attempt") or row.get("attempt_id"),
            "retry_source": "manifest" if (row.get("retry_attempt") or row.get("attempt") or row.get("attempt_id")) is not None else "",
        }
        if cls == "terminal_invalid" and detail.startswith("unrecognized terminal status"):
            problems.append({"class": "terminal_invalid", "job_key": job_key(row), "detail": detail})
        if cls in {"complete", "terminal_invalid"}:
            accepted_job_keys.append(job_key(row))
        if summary:
            if "state_id" in summary and str(summary.get("state_id")) != str(row.get("state_id")):
                rec["class"] = "replaced_state"
                row_classes[cls] -= 1
                row_classes["replaced_state"] += 1
                problems.append({"class": "replaced_state", "job_key": job_key(row), "manifest_state_id": row.get("state_id"), "summary_state_id": summary.get("state_id")})
            for key in ["runner_sha256", "bridge_sha256", "protocol_sha256", "manifest_sha256", "metric_schema_sha256"]:
                val = summary.get(key) or row.get(key)
                if val:
                    provenance_values[key].add(str(val))
            for key in ["checkpoint_sha256", "detector_checkpoint_sha256"]:
                sval = summary.get(key)
                mval = row.get(key)
                if sval and mval and str(sval) != str(mval):
                    problems.append({"class": "provenance_mismatch", "job_key": job_key(row), "field": key, "manifest": mval, "summary": sval})
            if rec["retry_attempt"] is None:
                rec["retry_attempt"] = summary.get("retry_attempt") or summary.get("attempt") or summary.get("attempt_id")
                rec["retry_source"] = "summary" if rec["retry_attempt"] is not None else ""
        for p in out.glob("*") if out.exists() else []:
            if p.is_file() and p.stat().st_size == 0:
                problems.append({"class": "zero_byte_artifact", "job_key": job_key(row), "path": str(p)})
        rows_out.append(rec)

    for key, vals in sorted(provenance_values.items()):
        if key not in {"checkpoint_sha256", "detector_checkpoint_sha256"} and len(vals) > 1:
            problems.append({"class": "mixed_provenance", "field": key, "values": sorted(vals)})

    referenced = {Path(d).resolve() for d in dirs if d}
    orphan_summaries = []
    if condition_root.exists():
        for summary in condition_root.rglob("episode_summary.json"):
            if summary.parent.resolve() not in referenced:
                orphan_summaries.append(str(summary.parent.resolve()))
    for p in sorted(orphan_summaries):
        problems.append({"class": "orphan_artifact", "output_dir": p})

    row_count_ok = len(rows) == args.expected_rows
    parent_count_ok = len(parents) == args.expected_parents
    legal_terminal = row_classes["complete"] + row_classes["terminal_invalid"]
    closure_pass = (
        row_count_ok
        and parent_count_ok
        and legal_terminal == args.expected_rows
        and not problems
    )
    return {
        "closure_pass": closure_pass,
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "condition_root": str(condition_root),
        "row_count": len(rows),
        "expected_rows": args.expected_rows,
        "parent_count": len(parents),
        "expected_parents": args.expected_parents,
        "row_classes": dict(sorted(row_classes.items())),
        "accepted_job_keys": sorted(k for k in accepted_job_keys if k),
        "provenance": {k: sorted(v) for k, v in sorted(provenance_values.items())},
        "problems": problems,
        "rows": rows_out,
    }


def markdown_report(result: dict) -> str:
    verdict = "PASS" if result["closure_pass"] else "HOLD"
    lines = [
        "# Formal CLEAN Closure Validation",
        "",
        f"Verdict: `{verdict}`",
        f"Manifest: `{result['manifest']}`",
        f"Manifest SHA256: `{result['manifest_sha256']}`",
        f"Rows: {result['row_count']} / {result['expected_rows']}",
        f"Parents: {result['parent_count']} / {result['expected_parents']}",
        "",
        "## Row Classes",
        "",
    ]
    for key, val in result["row_classes"].items():
        lines.append(f"- `{key}`: {val}")
    lines += ["", "## Problems", ""]
    if result["problems"]:
        for p in result["problems"]:
            lines.append(f"- `{p.get('class')}`: `{canonical_json(p).strip()}`")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only Formal CLEAN closure validator.")
    add_path_arg(ap, "--manifest", required=True)
    add_path_arg(ap, "--condition-root", required=True)
    add_path_arg(ap, "--output-json")
    add_path_arg(ap, "--output-md")
    ap.add_argument("--expected-rows", type=int, default=162)
    ap.add_argument("--expected-parents", type=int, default=54)
    ap.add_argument("--expected-replicates", type=int, default=3)
    args = ap.parse_args()
    result = validate(args)
    if args.output_json:
        write_json(args.output_json, result)
    if args.output_md:
        args.output_md.write_text(markdown_report(result), encoding="utf-8")
    if not args.output_json and not args.output_md:
        print(canonical_json(result), end="")
    return 0 if result["closure_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
