"""Audit action-stable causal execution equivalence from sealed M1 traces."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

try:
    from .stage_v_rb1_runtime_equivalence import (
        BOUNDARY_FIELDS,
        RuntimeEquivalenceError,
        validate_pair,
        validate_protocol,
        verify_artifact_files,
    )
except ImportError:  # direct execution on the server
    from stage_v_rb1_runtime_equivalence import (
        BOUNDARY_FIELDS,
        RuntimeEquivalenceError,
        validate_pair,
        validate_protocol,
        verify_artifact_files,
    )


PAIR_LABELS = (("Q1", "C1"), ("Q2", "C2"))


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeEquivalenceError(f"{path} must be a JSON object")
    return dict(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_binding(repo: Path) -> tuple[str, str]:
    def run(*args: str) -> str:
        return subprocess.check_output(("git", "-C", str(repo), *args), text=True).strip()

    return run("rev-parse", "HEAD"), run("rev-parse", "HEAD^{tree}")


def _m1_binding(m1_root: Path) -> tuple[list[int], dict[str, Any], dict[str, Any]]:
    audit = _load(m1_root / "M1_V2_INDEPENDENT_AUDIT.json")
    complete = _load(m1_root / "M1_V2_COMPLETE.json")
    if audit.get("schema") != "STAGE_V_M1_V2_2_INDEPENDENT_AUDIT_V1" or audit.get("verdict") != "PASS":
        raise RuntimeEquivalenceError("RB1_V2_M1_AUDIT_NOT_PASS")
    if complete.get("schema") != "STAGE_V_M1_V2_COMPLETE_V1" or complete.get("status") != "PASS_CLASSIFIED":
        raise RuntimeEquivalenceError("RB1_V2_M1_COMPLETION_NOT_PASS_CLASSIFIED")
    if complete.get("completion_owner") != "INDEPENDENT_AUDITOR":
        raise RuntimeEquivalenceError("RB1_V2_M1_COMPLETION_OWNER_INVALID")
    evidence = audit.get("evidence_profile")
    if not isinstance(evidence, Mapping) or evidence.get("action_stable") is not True or evidence.get("action_divergent_pairs") != []:
        raise RuntimeEquivalenceError("RB1_V2_M1_ACTION_STABILITY_NOT_PROVEN")
    cohort = audit.get("primary_clean_gpu_set")
    if not isinstance(cohort, list) or not cohort or any(not isinstance(gpu, int) for gpu in cohort):
        raise RuntimeEquivalenceError("RB1_V2_M1_PRIMARY_COHORT_INVALID")
    protected = audit.get("protected_boundaries")
    if not isinstance(protected, Mapping) or any(protected.get(field) != 0 for field in BOUNDARY_FIELDS):
        raise RuntimeEquivalenceError("RB1_V2_M1_PROTECTED_BOUNDARY")
    return cohort, audit, complete


def audit(m1_root: Path, protocol_path: Path, repo: Path, run_sets: list[str]) -> dict[str, Any]:
    m1_root = m1_root.resolve()
    protocol = _load(protocol_path.resolve())
    validate_protocol(protocol)
    cohort, m1_audit, m1_complete = _m1_binding(m1_root)
    auditor_commit, auditor_tree = _git_binding(repo.resolve())
    pairs: list[dict[str, Any]] = []
    receipt_paths: list[Path] = []
    protected_totals = {field: 0 for field in BOUNDARY_FIELDS}

    for run_set in run_sets:
        base = "runs" if run_set == "r1" else "raw_runs"
        for gpu in cohort:
            for left_label, right_label in PAIR_LABELS:
                left_dir = m1_root / base / f"gpu_{gpu:02d}" / left_label
                right_dir = m1_root / base / f"gpu_{gpu:02d}" / right_label
                left_path = left_dir / "RB1_INDEPENDENT_RECEIPT.json"
                right_path = right_dir / "RB1_INDEPENDENT_RECEIPT.json"
                left = _load(left_path)
                right = _load(right_path)
                verify_artifact_files(left, left_dir / "trace", protocol)
                verify_artifact_files(right, right_dir / "trace", protocol)
                result = validate_pair(left, right, protocol, "RB1A_CLEAN_PATH")
                for receipt in (left, right):
                    receipt_paths.append(left_path if receipt is left else right_path)
                    for field in BOUNDARY_FIELDS:
                        protected_totals[field] += int(receipt.get(field, 0))
                pairs.append({
                    "run_set": run_set,
                    "gpu": gpu,
                    "left": f"{base}/gpu_{gpu:02d}/{left_label}",
                    "right": f"{base}/gpu_{gpu:02d}/{right_label}",
                    "audit": result,
                })

    source_pairs = {str(path): _load(path) for path in receipt_paths}
    source_commits = sorted({str(receipt.get("source_commit")) for receipt in source_pairs.values()})
    source_trees = sorted({str(receipt.get("source_tree")) for receipt in source_pairs.values()})
    if len(source_commits) != 1 or len(source_trees) != 1:
        raise RuntimeEquivalenceError("RB1_V2_RECEIPT_SOURCE_BINDING_MISMATCH")
    allowed_fields = {
        "trace": sorted({
            field
            for item in pairs
            for field in item["audit"].get("allowed_trace_difference_fields", [])
        }),
        "diagnostic": sorted({
            field
            for item in pairs
            for field in item["audit"].get("allowed_diagnostic_difference_fields", [])
        }),
    }
    return {
        "schema": "STAGE_V_RB1_V2_CAUSAL_AUDIT_V1",
        "status": "PASS_CLASSIFIED",
        "verdict": "PASS",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha256(protocol_path.resolve()),
        "auditor_source_commit": auditor_commit,
        "auditor_source_tree": auditor_tree,
        "m1_root": str(m1_root),
        "m1_source_commit": source_commits[0],
        "m1_source_tree": source_trees[0],
        "m1_status": m1_complete["status"],
        "m1_completion_owner": m1_complete["completion_owner"],
        "m1_classification": m1_audit["classification"],
        "primary_clean_gpu_set": cohort,
        "primary_clean_gpu_count": len(cohort),
        "run_sets": run_sets,
        "pair_count": len(pairs),
        "pair_pass_count": len(pairs),
        "causal_execution_equivalence": "PASS",
        "visual_input_differences_allowed": True,
        "allowed_difference_fields_observed": allowed_fields,
        "protected_boundaries": protected_totals,
        "evidence_mode": "READ_ONLY_REUSE_OF_SEALED_M1_R1_R2_RECEIPTS",
        "m1_artifacts_modified": False,
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-set", choices=("r1", "r2"), action="append", dest="run_sets")
    args = parser.parse_args()
    run_sets = args.run_sets or ["r1", "r2"]
    try:
        result = audit(args.m1_root, args.protocol, args.repo, run_sets)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.output.with_name(args.output.name + ".tmp")
        tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(args.output)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, RuntimeEquivalenceError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"schema": "STAGE_V_RB1_V2_CAUSAL_AUDIT_V1", "status": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"schema": result["schema"], "status": result["status"], "pair_count": result["pair_count"], "protected_boundaries": result["protected_boundaries"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
