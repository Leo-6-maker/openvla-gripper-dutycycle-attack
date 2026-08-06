"""Run deterministic clean A/B qualification before Stage V R2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

try:
    from .stage_v_dynamic_common import atomic_write_json, canonical_parent_key, load_rows, normalize_parent, sha256_file, sha256_text, utc_now
except ImportError:  # direct server execution
    from stage_v_dynamic_common import atomic_write_json, canonical_parent_key, load_rows, normalize_parent, sha256_file, sha256_text, utc_now


FORBIDDEN = re.compile(r"(?<![A-Za-z0-9_])(?:OPEN(?:_T[0-9]+)?|VIS|PGD|ATTACK|EVAL160|PROTECTED|TEACHER)(?![A-Za-z0-9_])", re.IGNORECASE)


def ranked(rows: list[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        normalized = normalize_parent(row)
        key = normalized["canonical_parent_key"]
        normalized["qualification_rank_sha256"] = hashlib.sha256(f"{salt}::{key}".encode()).hexdigest()
        output.append(normalized)
    return sorted(output, key=lambda item: (item["qualification_rank_sha256"], item["canonical_parent_key"]))


def _result_from_directory(directory: Path) -> Mapping[str, Any] | None:
    for name in ("CONTROL_RESULT.json", "RESULT.json", "PARENT_RESULT.json"):
        value = json.loads((directory / name).read_text(encoding="utf-8")) if (directory / name).is_file() else None
        if isinstance(value, Mapping):
            return value
    return None


def _run_once(template: str, *, candidate_path: Path, output_dir: Path, replicate: str, source_commit: str, source_tree: str) -> tuple[int, dict[str, Any]]:
    command_text = template.format(
        candidate_path=str(candidate_path), output_dir=str(output_dir), replicate=replicate,
        source_commit=source_commit, source_tree=source_tree,
    )
    if FORBIDDEN.search(command_text):
        return 2, {"status": "FAIL", "reason": "FORBIDDEN_COMMAND_TOKEN"}
    command = command_text if isinstance(command_text, str) else str(command_text)
    completed = subprocess.run(command, shell=True, check=False, capture_output=True, text=True)
    result = _result_from_directory(output_dir)
    if result is None:
        for line in reversed(completed.stdout.splitlines()):
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, Mapping):
                result = candidate
                break
    payload = dict(result) if result else {}
    payload.update({
        "replicate": replicate,
        "exit_code": completed.returncode,
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    })
    return completed.returncode, payload


def qualifies(row: Mapping[str, Any], a: Mapping[str, Any], b: Mapping[str, Any], source_commit: str, source_tree: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for name, result in (("A", a), ("B", b)):
        if result.get("exit_code") != 0 or result.get("status") not in ("PASS", "DONE", "QUALIFIED"):
            errors.append(f"{name}_NOT_COMPLETE")
        if result.get("clean_success") is not True:
            errors.append(f"{name}_CLEAN_SUCCESS_FALSE")
        if result.get("snapshot_restore_valid") is not True or result.get("runtime_valid") is not True:
            errors.append(f"{name}_SNAPSHOT_OR_RUNTIME_INVALID")
        if result.get("metrics_finite") is not True:
            errors.append(f"{name}_NONFINITE")
        if result.get("source_commit") != source_commit or result.get("source_tree") != source_tree:
            errors.append(f"{name}_PROVENANCE_MISMATCH")
        if result.get("remaining_horizon_complete") is not True:
            errors.append(f"{name}_HORIZON_INCOMPLETE")
    for field in ("terminal_outcome", "terminal_state_sha256", "key_state_identity_sha256"):
        if a.get(field) is None or b.get(field) is None or a.get(field) != b.get(field):
            errors.append(f"AB_MISMATCH:{field}")
    if a.get("canonical_parent_key") != row.get("canonical_parent_key"):
        errors.append("A_PARENT_IDENTITY_MISMATCH")
    if b.get("canonical_parent_key") != row.get("canonical_parent_key"):
        errors.append("B_PARENT_IDENTITY_MISMATCH")
    return not errors, sorted(set(errors))


def qualify(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    raw_rows = load_rows(args.candidate_manifest)
    rows = ranked(
        [row for row in raw_rows if row.get("audit_status", "PASS") == "PASS" and int(row.get("remaining_policy_steps", 1) or 0) > 0],
        args.salt,
    )
    if not rows:
        raise ValueError("candidate manifest is empty")
    suites = sorted({str(row["suite"]) for row in rows})
    if args.suites:
        suites = [suite for suite in suites if suite in set(args.suites.split(","))]
    by_suite = {suite: [row for row in rows if str(row["suite"]) == suite] for suite in suites}
    rows_out: list[dict[str, Any]] = []
    selected: dict[str, list[dict[str, Any]]] = {suite: [] for suite in suites}
    for suite in suites:
        suite_rows = by_suite[suite]
        next_index = 0
        pool_size = min(args.initial_per_suite, len(suite_rows))
        while len(selected[suite]) < args.target_per_suite and next_index < len(suite_rows):
            end = min(pool_size, len(suite_rows))
            for row in suite_rows[next_index:end]:
                key = row["canonical_parent_key"]
                base = args.output_dir / "qualification" / suite / key.replace("/", "__")
                base.mkdir(parents=True, exist_ok=False)
                candidate_path = base / "CANDIDATE.json"
                atomic_write_json(candidate_path, row)
                replicate_rows = {}
                for replicate in ("A", "B"):
                    output = base / replicate
                    output.mkdir()
                    code, result = _run_once(
                        args.runner_command,
                        candidate_path=candidate_path,
                        output_dir=output,
                        replicate=replicate,
                        source_commit=args.source_commit,
                        source_tree=args.source_tree,
                    )
                    replicate_rows[replicate] = result
                    if code != 0:
                        result.setdefault("status", "FAIL")
                ok, errors = qualifies(row, replicate_rows["A"], replicate_rows["B"], args.source_commit, args.source_tree)
                record = {
                    "schema": "STAGE_V_CONTROL_QUALIFICATION_ROW_V2",
                    "canonical_parent_key": key,
                    "suite": suite,
                    "qualification_rank_sha256": row["qualification_rank_sha256"],
                    "candidate_sha256": sha256_file(candidate_path),
                    "replicates": replicate_rows,
                    "qualified": ok,
                    "errors": errors,
                    "evaluated_utc": utc_now(),
                }
                atomic_write_json(base / "QUALIFICATION_ROW.json", record)
                rows_out.append(record)
                if ok and len(selected[suite]) < args.target_per_suite:
                    selected[suite].append(row)
            next_index = end
            pool_size = min(pool_size + args.batch_size, len(suite_rows))
            if end == len(suite_rows):
                break
    report = {
        "schema": "STAGE_V_CONTROL_QUALIFICATION_REPORT_V2",
        "status": "PASS" if all(len(selected[suite]) >= args.target_per_suite for suite in suites) else "FAIL",
        "salt": args.salt,
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "initial_per_suite": args.initial_per_suite,
        "batch_size": args.batch_size,
        "target_per_suite": args.target_per_suite,
        "suites": suites,
        "qualified_by_suite": {suite: len(selected[suite]) for suite in suites},
        "evaluated_rows": len(rows_out),
        "eval160_reads": 0,
        "protected_eval_reads": 0,
        "vis_pgd_attack_rollouts": 0,
        "generated_utc": utc_now(),
    }
    manifest_rows = [row for suite in suites for row in selected[suite][:args.target_per_suite]]
    manifest = {
        "schema": "STAGE_V_FORMAL_PARENT_MANIFEST_V2",
        "status": report["status"],
        "salt": args.salt,
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "parents": manifest_rows,
        "planned_parent_count": len(manifest_rows),
        "parents_by_suite": {suite: args.target_per_suite for suite in suites},
        "old_artifacts_reused": False,
        "generated_utc": utc_now(),
    }
    audit = {
        "schema": "STAGE_V_CONTROL_QUALIFICATION_INDEPENDENT_AUDIT_V2",
        "verdict": report["status"],
        "recomputed_qualified_by_suite": {
            suite: sum(1 for row in rows_out if row["suite"] == suite and row["qualified"])
            for suite in suites
        },
        "manifest_parent_count": len(manifest_rows),
        "duplicate_parent_keys": sorted({key for key in [row["canonical_parent_key"] for row in manifest_rows] if [item["canonical_parent_key"] for item in manifest_rows].count(key) > 1}),
        "boundaries": {"eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0},
        "audited_utc": utc_now(),
    }
    return report, rows_out, {"manifest": manifest, "audit": audit}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner-command", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--salt", default="STAGE_V_CONTROL_QUALIFICATION_V2_20260806")
    parser.add_argument("--initial-per-suite", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--target-per-suite", type=int, default=10)
    parser.add_argument("--suites", default="")
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        if any(args.output_dir.iterdir()):
            parser.error(f"qualification output must be new/empty: {args.output_dir}")
    else:
        args.output_dir.mkdir(parents=True)
    report, rows, extras = qualify(args)
    atomic_write_json(args.output_dir / "CONTROL_QUALIFICATION_REPORT.json", report)
    with (args.output_dir / "CONTROL_QUALIFICATION_ROWS.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    atomic_write_json(args.output_dir / "CONTROL_QUALIFICATION_INDEPENDENT_AUDIT.json", extras["audit"])
    manifest_path = args.output_dir / "STAGE_V_FORMAL_PARENT_MANIFEST_V2.json"
    if report["status"] == "PASS":
        atomic_write_json(manifest_path, extras["manifest"])
        (args.output_dir / "STAGE_V_FORMAL_PARENT_MANIFEST_V2.sha256").write_text(sha256_file(manifest_path) + "  STAGE_V_FORMAL_PARENT_MANIFEST_V2.json\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
