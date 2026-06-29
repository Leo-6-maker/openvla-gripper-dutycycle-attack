from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from tools.table1_audit.common import (
    add_path_arg,
    canonical_digest,
    canonical_json,
    first_non_none,
    is_valid_sha256,
    job_key,
    load_json,
    output_dir,
    parent_key,
    parse_manifest,
    recursive_inventory,
    replicate_key,
    sha256_file,
    validate_artifact_syntax,
    write_json,
)
from tools.table1_audit.contracts import RequiredArtifactSchema, RetryPolicy, RuntimeLock, load_contract


VALIDATOR_SCHEMA_VERSION = "formal_clean_closure_validator.v2"
REQUIRED_MANIFEST_FIELDS = {
    "job_key",
    "fold",
    "task_id",
    "state_id",
    "detector_seed",
    "perturbation_seed",
    "output_dir",
    "checkpoint_sha256",
    "detector_checkpoint_sha256",
    "condition_id",
}
PROVENANCE_FIELDS = [
    "runner_sha256",
    "worker_sha256",
    "bridge_sha256",
    "protocol_sha256",
    "metric_schema_sha256",
    "manifest_sha256",
    "state_selection_sha256",
    "global_freeze_sha256",
]


def _load_json_contract(path: Path, expected: str) -> tuple[dict, dict]:
    data = load_json(path)
    version = str(data.get("schema_version") or "")
    if version != expected:
        raise ValueError(f"{path}: expected schema_version {expected}, got {version!r}")
    return data, {"path": str(path.resolve()), "actual_sha256": sha256_file(path), "schema_version": version}


def _problem(problems: list[dict], cls: str, row: dict | None = None, **detail) -> None:
    item = {"class": cls, **detail}
    if row is not None:
        item.setdefault("job_key", job_key(row))
        item.setdefault("line", row.get("_line_no"))
    problems.append(item)


def _state_design(state_selection: dict) -> tuple[set[str], set[tuple[str, str]], set[str], set[str]]:
    folds = {str(f) for f in state_selection["folds"]}
    states_by_fold = state_selection["states_by_fold"]
    fold_states = {(str(f), str(s)) for f, states in states_by_fold.items() for s in states}
    det = {str(s) for s in state_selection["detector_seeds"]}
    pert = {str(s) for s in state_selection["perturbation_seeds"]}
    return folds, fold_states, det, pert


def _checkpoint_for(global_freeze: dict, row: dict, field: str) -> str | None:
    mapping = global_freeze.get(field, {})
    keys = [
        f"{row.get('fold')}|{row.get('detector_seed')}",
        f"{row.get('fold')}|{row.get('state_id')}|{row.get('detector_seed')}",
        str(row.get("fold")),
        "default",
    ]
    for key in keys:
        if key in mapping:
            return str(mapping[key])
    return None


def _read_terminal_ledger(out: Path) -> dict | None:
    p = out / "terminal_ledger.json"
    if not p.exists():
        return None
    return load_json(p)


def _classify(row: dict, out: Path, retry_policy: dict, artifact_schema: dict, problems: list[dict]) -> tuple[str, dict | None]:
    summary_path = out / "episode_summary.json"
    ledger = None
    summary = None
    if summary_path.exists():
        if summary_path.is_symlink():
            _problem(problems, "symlink_artifact", row, path=str(summary_path))
            return "malformed", None
        err = validate_artifact_syntax(summary_path)
        if err:
            _problem(problems, "malformed_artifact", row, path=str(summary_path), detail=err)
            return "malformed", None
        summary = load_json(summary_path)
    else:
        try:
            ledger = _read_terminal_ledger(out)
        except Exception as exc:
            _problem(problems, "malformed_artifact", row, path=str(out / "terminal_ledger.json"), detail=str(exc))
            return "malformed", None

    status = str(first_non_none(
        summary.get("terminal_status") if summary else None,
        summary.get("status") if summary else None,
        summary.get("result_status") if summary else None,
        ledger.get("terminal_status") if ledger else None,
        "",
    ))
    if summary and (summary.get("task_success") is not None or status in {"COMPLETE", "SUCCESS", "OK"}):
        required = artifact_schema["complete"].get("required_files", [])
        for rel in required:
            p = out / rel
            if not p.exists():
                _problem(problems, "missing_required_artifact", row, path=rel)
            elif p.is_symlink():
                _problem(problems, "symlink_artifact", row, path=str(p))
            else:
                err = validate_artifact_syntax(p)
                if err:
                    _problem(problems, "malformed_artifact", row, path=rel, detail=err)
        return "complete", summary

    legal_statuses = {str(s) for s in retry_policy["legal_terminal_invalid_statuses"]}
    if status not in legal_statuses:
        _problem(problems, "unknown_terminal_status", row, status=status or "missing")
        return "hold", summary or ledger

    ledger = ledger or summary
    if not ledger:
        _problem(problems, "terminal_invalid_without_policy_evidence", row)
        return "hold", None
    if str(ledger.get("job_key")) != job_key(row):
        _problem(problems, "terminal_ledger_job_key_mismatch", row, ledger_job_key=ledger.get("job_key"))
    attempts = ledger.get("attempt_history")
    if not isinstance(attempts, list) or not attempts:
        _problem(problems, "terminal_ledger_missing_attempt_history", row)
    if ledger.get("no_retry_remaining") is not True:
        _problem(problems, "terminal_invalid_retry_remaining", row)
    if str(ledger.get("terminal_reason") or "") not in {str(r) for r in retry_policy["terminal_reasons"]}:
        _problem(problems, "terminal_reason_not_allowed", row, terminal_reason=ledger.get("terminal_reason"))
    for rel in artifact_schema["terminal_invalid"].get("required_files", []):
        if not (out / rel).exists():
            _problem(problems, "missing_required_artifact", row, path=rel)
    return "terminal_invalid", summary or ledger


def validate(args: argparse.Namespace) -> dict:
    manifest = args.manifest.resolve()
    condition_root = args.condition_root.resolve()
    state_selection, state_meta = _load_json_contract(args.state_selection.resolve(), "state_selection.v1")
    global_freeze, freeze_meta = _load_json_contract(args.global_freeze.resolve(), "global_freeze.v1")
    runtime = load_contract(args.runtime_lock.resolve(), RuntimeLock)
    retry = load_contract(args.retry_policy.resolve(), RetryPolicy)
    artifact_schema = load_contract(args.required_artifact_schema.resolve(), RequiredArtifactSchema)
    rows, problems = parse_manifest(manifest)
    manifest_sha = sha256_file(manifest)
    folds_expected, fold_states_expected, det_expected, pert_expected = _state_design(state_selection)

    contracts = {
        "state_selection": state_meta,
        "global_freeze": freeze_meta,
        "runtime_lock": runtime.meta(),
        "retry_policy": retry.meta(),
        "required_artifact_schema": artifact_schema.meta(),
    }
    contract_shas = {k: v["actual_sha256"] for k, v in contracts.items()}
    condition_root_identity = canonical_digest({"condition_root": str(condition_root)})

    seen_jobs: Counter[str] = Counter()
    seen_dirs: Counter[str] = Counter()
    dirs_by_job: dict[str, Path] = {}
    parents: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    rows_out: list[dict] = []
    row_classes: Counter[str] = Counter()
    row_problem_counts: Counter[str] = Counter()
    provenance_values: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        key = job_key(row)
        if not key:
            _problem(problems, "empty_job_key", row)
        seen_jobs[key] += 1
        missing = sorted(f for f in REQUIRED_MANIFEST_FIELDS if row.get(f) is None or row.get(f) == "")
        if missing:
            _problem(problems, "missing_required_manifest_field", row, fields=missing)
        parents[parent_key(row)].append(row)
        if str(row.get("fold")) not in folds_expected:
            _problem(problems, "wrong_fold_set", row, fold=row.get("fold"))
        if (str(row.get("fold")), str(row.get("state_id"))) not in fold_states_expected:
            _problem(problems, "wrong_state_selection", row, fold=row.get("fold"), state_id=row.get("state_id"))
        if str(row.get("detector_seed")) not in det_expected:
            _problem(problems, "wrong_detector_seed_domain", row, detector_seed=row.get("detector_seed"))
        if str(row.get("perturbation_seed")) not in pert_expected:
            _problem(problems, "wrong_perturbation_seed_domain", row, perturbation_seed=row.get("perturbation_seed"))
        try:
            out = output_dir(row, manifest.parent, condition_root)
            seen_dirs[str(out)] += 1
            dirs_by_job[key] = out
        except Exception as exc:
            _problem(problems, "unsafe_output_dir", row, detail=str(exc))

    for key, n in seen_jobs.items():
        if key and n > 1:
            problems.append({"class": "duplicate_job_key", "job_key": key, "count": n})
    for path, n in seen_dirs.items():
        if n > 1:
            problems.append({"class": "duplicate_output", "output_dir": path, "count": n})
    for key, items in sorted(parents.items()):
        reps = Counter(replicate_key(r) for r in items)
        if len(items) != args.expected_replicates or set(reps) != pert_expected or any(v != 1 for v in reps.values()):
            problems.append({"class": "replicate_count", "parent": list(key), "count": len(items), "replicates": dict(reps)})

    baseline_problem_count = len(problems)
    for row in rows:
        key = job_key(row)
        before = len(problems)
        out = dirs_by_job.get(key)
        if out is None:
            row_classes["hold"] += 1
            continue
        cls, evidence = _classify(row, out, retry.data, artifact_schema.data, problems)
        row_classes[cls] += 1
        rec = {
            "job_key": key,
            "parent_key": list(parent_key(row)),
            "replicate": replicate_key(row),
            "output_dir": str(out),
            "class": cls,
            "retry_attempt": first_non_none(row.get("retry_attempt"), row.get("attempt"), row.get("attempt_id")),
        }
        if evidence:
            if "state_id" in evidence and str(evidence.get("state_id")) != str(row.get("state_id")):
                _problem(problems, "replaced_state", row, manifest_state_id=row.get("state_id"), artifact_state_id=evidence.get("state_id"))
            for field in PROVENANCE_FIELDS:
                val = first_non_none(evidence.get(field), row.get(field))
                if val is None:
                    _problem(problems, "missing_required_provenance_field", row, field=field)
                    continue
                val = str(val)
                provenance_values[field].add(val)
                if field.endswith("_sha256") and not is_valid_sha256(val):
                    _problem(problems, "invalid_sha256", row, field=field, value=val)
            if str(first_non_none(evidence.get("manifest_sha256"), row.get("manifest_sha256"), "")) != manifest_sha:
                _problem(problems, "manifest_sha_mismatch", row, expected=manifest_sha, actual=first_non_none(evidence.get("manifest_sha256"), row.get("manifest_sha256")))
            for field, expected_sha in runtime.data["required_sha256"].items():
                actual = first_non_none(evidence.get(field), row.get(field))
                if actual != expected_sha:
                    _problem(problems, "runtime_lock_mismatch", row, field=field, expected=expected_sha, actual=actual)
            for field, map_name in [("checkpoint_sha256", "victim_checkpoint_sha256"), ("detector_checkpoint_sha256", "detector_checkpoint_sha256")]:
                expected = _checkpoint_for(global_freeze, row, map_name)
                actual = first_non_none(evidence.get(field), row.get(field))
                if expected and actual != expected:
                    _problem(problems, "global_freeze_checkpoint_mismatch", row, field=field, expected=expected, actual=actual)
        for p in out.rglob("*") if out.exists() else []:
            if p.is_symlink():
                _problem(problems, "symlink_artifact", row, path=str(p))
            elif p.is_file() and p.stat().st_size == 0:
                _problem(problems, "zero_byte_artifact", row, path=str(p))
        if len(problems) > before:
            row_problem_counts[key] += len(problems) - before
        rows_out.append(rec)

    for field, vals in sorted(provenance_values.items()):
        allowed = set(runtime.data.get("allowed_values", {}).get(field, []))
        if allowed:
            unexpected = vals - allowed
            if unexpected:
                problems.append({"class": "runtime_lock_scope_mismatch", "field": field, "values": sorted(unexpected)})

    referenced = {p.resolve() for p in dirs_by_job.values()}
    if condition_root.exists():
        try:
            inventory = recursive_inventory(condition_root)
        except Exception as exc:
            inventory = []
            problems.append({"class": "artifact_inventory_error", "detail": str(exc)})
        for summary in condition_root.rglob("episode_summary.json"):
            if summary.parent.resolve() not in referenced:
                problems.append({"class": "orphan_artifact", "output_dir": str(summary.parent.resolve())})
    else:
        inventory = []
        problems.append({"class": "missing_condition_root", "path": str(condition_root)})

    accepted_job_keys = sorted(
        key for key in seen_jobs
        if key and seen_jobs[key] == 1 and key in dirs_by_job and row_problem_counts[key] == 0
    )
    artifact_inventory_digest = canonical_digest(inventory)
    accepted_job_keys_sha256 = canonical_digest(accepted_job_keys)
    row_count_ok = len(rows) == args.expected_rows
    parent_count_ok = len(parents) == args.expected_parents
    closure_pass = (
        row_count_ok
        and parent_count_ok
        and len(accepted_job_keys) == args.expected_rows
        and not problems
    )
    return {
        "validator_schema_version": VALIDATOR_SCHEMA_VERSION,
        "validator_source_sha256": sha256_file(Path(__file__)),
        "closure_pass": closure_pass,
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "condition_root": str(condition_root),
        "condition_root_identity": condition_root_identity,
        "contracts": contracts,
        "state_selection_sha256": contract_shas["state_selection"],
        "global_freeze_sha256": contract_shas["global_freeze"],
        "runtime_lock_sha256": contract_shas["runtime_lock"],
        "retry_policy_sha256": contract_shas["retry_policy"],
        "required_artifact_schema_sha256": contract_shas["required_artifact_schema"],
        "accepted_job_keys": accepted_job_keys,
        "accepted_count": len(accepted_job_keys),
        "accepted_job_keys_sha256": accepted_job_keys_sha256,
        "artifact_inventory_digest": artifact_inventory_digest,
        "row_count": len(rows),
        "expected_rows": args.expected_rows,
        "parent_count": len(parents),
        "expected_parents": args.expected_parents,
        "row_classes": dict(sorted(row_classes.items())),
        "provenance": {k: sorted(v) for k, v in sorted(provenance_values.items())},
        "problems": problems,
        "rows": rows_out,
        "manifest_parse_problem_count": max(0, baseline_problem_count),
    }


def markdown_report(result: dict) -> str:
    verdict = "PASS" if result["closure_pass"] else "HOLD"
    lines = [
        "# Formal CLEAN Closure Validation",
        "",
        f"Verdict: `{verdict}`",
        f"Validator schema: `{result['validator_schema_version']}`",
        f"Manifest SHA256: `{result['manifest_sha256']}`",
        f"Condition root identity: `{result['condition_root_identity']}`",
        f"Rows: {result['row_count']} / {result['expected_rows']}",
        f"Parents: {result['parent_count']} / {result['expected_parents']}",
        f"Accepted job keys: {result['accepted_count']} / {result['expected_rows']}",
        "",
        "## Contract SHAs",
        "",
    ]
    for key in ["state_selection_sha256", "global_freeze_sha256", "runtime_lock_sha256", "retry_policy_sha256", "required_artifact_schema_sha256"]:
        lines.append(f"- `{key}`: `{result[key]}`")
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
    add_path_arg(ap, "--state-selection", required=True)
    add_path_arg(ap, "--global-freeze", required=True)
    add_path_arg(ap, "--runtime-lock", required=True)
    add_path_arg(ap, "--retry-policy", required=True)
    add_path_arg(ap, "--required-artifact-schema", required=True)
    add_path_arg(ap, "--output-json", required=True)
    add_path_arg(ap, "--output-md", required=True)
    ap.add_argument("--expected-rows", type=int, default=162)
    ap.add_argument("--expected-parents", type=int, default=54)
    ap.add_argument("--expected-replicates", type=int, default=3)
    args = ap.parse_args()
    result = validate(args)
    write_json(args.output_json, result)
    args.output_md.write_text(markdown_report(result), encoding="utf-8")
    print(canonical_json({"closure_pass": result["closure_pass"], "problems": len(result["problems"])}), end="")
    return 0 if result["closure_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
