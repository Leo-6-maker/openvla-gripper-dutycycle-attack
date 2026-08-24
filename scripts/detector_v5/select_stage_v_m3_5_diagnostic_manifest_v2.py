"""Select exposed M3.5 parents from sealed clean-only corridor evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.detector_v5.build_stage_v_m3_5_probe_plan import ProbePlanError, select_probe_steps  # noqa: E402


SCHEMA = "STAGE_V_M3_5_DIAGNOSTIC_PARENT_SELECTION_V2"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _rank(salt: str, parent_key: str) -> str:
    return hashlib.sha256(f"{salt}::{parent_key}".encode("utf-8")).hexdigest()


def _identity(parent_key: str) -> tuple[str, int, int]:
    suite, task, state = parent_key.split("/")
    if suite not in SUITES or not task.startswith("task_") or not state.startswith("state_"):
        raise ValueError(f"PARENT_KEY_INVALID:{parent_key}")
    return suite, int(task.removeprefix("task_")), int(state.removeprefix("state_"))


def _verify_seal(root: Path, required: set[str]) -> None:
    sums = root / "SHA256SUMS"
    sums_sha = root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sums_sha.is_file():
        raise ValueError(f"CLEAN_COVERAGE_SEAL_MISSING:{root}")
    header = sums_sha.read_text(encoding="utf-8").split()
    if len(header) != 2 or header[1] != "SHA256SUMS" or header[0] != _sha256_file(sums):
        raise ValueError(f"CLEAN_COVERAGE_SEAL_HEADER_INVALID:{root}")
    sealed: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        path = Path(relative)
        if not separator or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"CLEAN_COVERAGE_SEAL_ROW_INVALID:{root}")
        target = root / path
        if not target.is_file() or _sha256_file(target) != digest:
            raise ValueError(f"CLEAN_COVERAGE_SEAL_FILE_INVALID:{target}")
        sealed.add(path.as_posix())
    if not required.issubset(sealed):
        raise ValueError(f"CLEAN_COVERAGE_SEAL_REQUIRED_FILES_MISSING:{root}")


def _coverage_rows(roots: Sequence[Path], exposure_keys: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for root in roots:
        for result_path in sorted(root.resolve().rglob("PARENT_RESULT.json")):
            result = _load(result_path)
            if result.get("schema") != "STAGE_V_M3_5_CLEAN_COVERAGE_RESULT_V1":
                continue
            if (
                result.get("status") != "PASS" or result.get("coverage_only") is not True
                or result.get("parent_atomic") is not True or result.get("protected_counters") != COUNTERS
            ):
                raise ValueError(f"CLEAN_COVERAGE_RESULT_INVALID:{result_path}")
            parent_key = str(result.get("canonical_parent_key", ""))
            if parent_key not in exposure_keys or parent_key in rows:
                raise ValueError(f"COVERAGE_PARENT_UNEXPECTED_OR_DUPLICATED:{parent_key}")
            if (result_path.parent / "COUNTERFACTUAL_BRANCHES.jsonl").exists():
                raise ValueError(f"COUNTERFACTUAL_BRANCH_FILE_FORBIDDEN:{parent_key}")
            trajectory_path = result_path.with_name("CLEAN_TRAJECTORY.json")
            if not trajectory_path.is_file():
                raise ValueError(f"CLEAN_TRAJECTORY_MISSING:{parent_key}")
            trajectory = _load(trajectory_path)
            if trajectory.get("schema") != "STAGE_V_M3_5_CLEAN_TRAJECTORY_V1" or trajectory.get("outcomes_read") is not False:
                raise ValueError(f"CLEAN_TRAJECTORY_CONTRACT_INVALID:{parent_key}")
            _verify_seal(result_path.parent, {"PARENT_RESULT.json", "CLEAN_TRAJECTORY.json"})
            clean_rows = trajectory.get("rows")
            if not isinstance(clean_rows, list):
                raise ValueError(f"CLEAN_TRAJECTORY_ROWS_INVALID:{parent_key}")
            plan = None
            plan_reason = ""
            try:
                plan = select_probe_steps(clean_rows, parent_key)
            except ProbePlanError as exc:
                plan_reason = str(exc)
            rows[parent_key] = {
                "canonical_parent_key": parent_key,
                "suite": str(result.get("suite")),
                "clean_success": result.get("clean_success") is True,
                "clean_result_path": str(result_path.resolve()),
                "clean_result_sha256": _sha256_file(result_path),
                "clean_trajectory_path": str(trajectory_path.resolve()),
                "clean_trajectory_file_sha256": _sha256_file(trajectory_path),
                "prospective_probe_plan_status": "PASS" if plan else "PROBE_PLAN_INSUFFICIENT",
                "prospective_probe_plan_reason": plan_reason,
                "prospective_probe_plan_sha256": _sha256_json(plan) if plan else None,
                "prospective_probe_steps": [int(item["step"]) for item in plan["probe_steps"]] if plan else [],
                "prospective_corridor_candidate_count": int(plan["corridor_candidate_count"]) if plan else 0,
                "protected_counters": dict(COUNTERS),
            }
    return rows


def select(
    exposure_manifest: Path,
    taxonomy_audit: Path,
    coverage_roots: Sequence[Path],
    output: Path,
    *,
    per_suite: int = 2,
    salt: str = "STAGE_V_M3_5_DIAGNOSTIC_SELECTION_V2",
    source_commit: str | None = None,
    source_tree: str | None = None,
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"REFUSE_OVERWRITE:{output}")
    exposure_path = exposure_manifest.resolve()
    exposure = _load(exposure_path)
    if (
        exposure.get("schema") != "STAGE_V_COUNTERFACTUAL_EXPOSURE_UNION_V4" or exposure.get("status") != "PASS"
        or exposure.get("branch_results_read") is not False or exposure.get("protected_counters") != COUNTERS
    ):
        raise ValueError("EXPOSURE_MANIFEST_NOT_ADMISSIBLE")
    keys = exposure.get("excluded_parent_keys")
    if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("EXPOSURE_KEYS_INVALID")
    exposure_keys = set(keys)

    taxonomy_path = taxonomy_audit.resolve()
    taxonomy = _load(taxonomy_path)
    if (
        taxonomy.get("schema") != "STAGE_V_M3_5_SELECTION_TAXONOMY_ELIGIBILITY_AUDIT_V1"
        or taxonomy.get("status") != "PASS_WITH_EXPLICIT_INELIGIBLE_PARENTS"
        or taxonomy.get("branch_results_read") is not False
        or taxonomy.get("protected_counters") != COUNTERS
    ):
        raise ValueError("TAXONOMY_AUDIT_NOT_ADMISSIBLE")
    ineligible = set(taxonomy.get("ineligible_parent_keys", []))
    if int(taxonomy.get("selected_count", -1)) != len(exposure_keys) or int(taxonomy.get("eligible_count", -1)) + len(ineligible) != len(exposure_keys):
        raise ValueError("TAXONOMY_ACCOUNTING_INVALID")
    if not ineligible.issubset(exposure_keys):
        raise ValueError("TAXONOMY_INELIGIBLE_NOT_EXPOSED")

    coverage = _coverage_rows(coverage_roots, exposure_keys)
    if set(coverage) != exposure_keys - ineligible:
        missing = sorted((exposure_keys - ineligible) - set(coverage))
        extra = sorted(set(coverage) - (exposure_keys - ineligible))
        raise ValueError(f"CLEAN_COVERAGE_ACCOUNTING_INVALID:missing={missing}:extra={extra}")

    candidate_rows = [
        row for row in coverage.values()
        if row["clean_success"] and row["prospective_probe_plan_status"] == "PASS"
    ]
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    eligible_counts: dict[str, int] = {}
    for suite in SUITES:
        ranked = sorted((_rank(salt, row["canonical_parent_key"]), row) for row in candidate_rows if row["suite"] == suite)
        eligible_counts[suite] = len(ranked)
        if len(ranked) < per_suite:
            raise ValueError(f"DIAGNOSTIC_CORRIDOR_SHORTFALL:{suite}:{len(ranked)}/{per_suite}")
        for rank, row in ranked[:per_suite]:
            _suite, task_index, state_index = _identity(row["canonical_parent_key"])
            selected.append({
                **row,
                "task_index": task_index,
                "state_index": state_index,
                "selection_rank_sha256": rank,
            })
        counts[suite] = per_suite

    report = {
        "schema": SCHEMA,
        "version": "V2",
        "status": "FROZEN_FOR_VALIDATION",
        "selection_role": "outcome_blind_m3_5_diagnostic_only; exposed identities; sealed clean-only corridor evidence",
        "source_exposure_manifest": {"path": str(exposure_path), "sha256": _sha256_file(exposure_path), "count": len(exposure_keys)},
        "source_taxonomy_audit": {"path": str(taxonomy_path), "sha256": _sha256_file(taxonomy_path), "explicit_ineligible_parent_keys": sorted(ineligible)},
        "source_coverage_roots": [str(path.resolve()) for path in coverage_roots],
        "taxonomy_rule": "fixture-only goals without a declared In/On source object are prospectively excluded; fixture inference is forbidden",
        "selection_algorithm": "eligible clean-success parents with a valid V2 24-quantile contact corridor, then sha256(salt + '::' + canonical_parent_key) ascending within suite",
        "selection_salt": salt,
        "target_per_suite": per_suite,
        "corridor_eligible_counts_by_suite": eligible_counts,
        "selected_count": len(selected),
        "selected_counts_by_suite": counts,
        "selected_parents": sorted(selected, key=lambda row: (row["suite"], row["selection_rank_sha256"])),
        "candidate_accounting": {
            "exposure_count": len(exposure_keys),
            "taxonomy_ineligible_count": len(ineligible),
            "clean_coverage_count": len(coverage),
            "clean_success_corridor_eligible_count": len(candidate_rows),
        },
        "selection_reads": {
            "clean_trajectory_rows_read": True,
            "clean_success_read": True,
            "branch_results_read": False,
            "counterfactual_outcomes_read": False,
            "teacher_student_predictions_read": False,
            "protected_reads": 0,
        },
        "source_artifacts_modified": False,
        "old_artifacts_reused_for_runtime": False,
        "protected_counters": dict(COUNTERS),
        "runtime_authorized": False,
        "builder_source_commit": source_commit,
        "builder_source_tree": source_tree,
    }
    _write(output, report)
    digest = _sha256_file(output)
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exposure-manifest", type=Path, required=True)
    parser.add_argument("--taxonomy-audit", type=Path, required=True)
    parser.add_argument("--coverage-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-suite", type=int, default=2)
    parser.add_argument("--salt", default="STAGE_V_M3_5_DIAGNOSTIC_SELECTION_V2")
    parser.add_argument("--source-commit")
    parser.add_argument("--source-tree")
    args = parser.parse_args(argv)
    report = select(
        args.exposure_manifest, args.taxonomy_audit, args.coverage_root, args.output,
        per_suite=args.per_suite, salt=args.salt,
        source_commit=args.source_commit, source_tree=args.source_tree,
    )
    print(json.dumps({"status": report["status"], "selected_count": report["selected_count"], "selected_counts_by_suite": report["selected_counts_by_suite"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
