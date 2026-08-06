"""Independent read-only audit for the Stage V2 enrichment artifact."""
from __future__ import annotations

import argparse
import datetime as _datetime
import json
from pathlib import Path
import sys
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.monitoring.audit_stage_v_closure import atomic_write_json, sha256_file, verify_sha_manifest
try:
    from .stage_v2_teacher_enrichment import (
        CONFIG_SCHEMA,
        REPORT_SCHEMA,
        StageV2PreconditionError,
        canonical_json,
        compute_report,
        diagnostic_binding,
        finite,
        formal_binding,
        load_json,
        load_observations,
    )
except ImportError:  # pragma: no cover - direct script execution on server.
    from stage_v2_teacher_enrichment import (
        CONFIG_SCHEMA,
        REPORT_SCHEMA,
        StageV2PreconditionError,
        canonical_json,
        compute_report,
        diagnostic_binding,
        finite,
        formal_binding,
        load_json,
        load_observations,
    )


SCHEMA = "STAGE_V2_INDEPENDENT_AUDIT_V1"


def _write_v2_seal(root: Path) -> dict[str, Any]:
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            continue
        entries.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n")
    sums = root / "SHA256SUMS"
    sums.write_text("".join(entries), encoding="utf-8")
    with sums.open("r+b") as handle:
        import os

        os.fsync(handle.fileno())
    sidecar = root / "SHA256SUMS.sha256"
    sidecar.write_text(f"{sha256_file(sums)}  SHA256SUMS\n", encoding="utf-8")
    with sidecar.open("r+b") as handle:
        import os

        os.fsync(handle.fileno())
    return {"files": len(entries), "sha256sums_sha256": sha256_file(sums)}


def utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _report_core(value: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {"generated_utc"}
    return {key: item for key, item in value.items() if key not in ignored}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row is not an object")
                rows.append(value)
    return rows


def audit(
    stage_v_root: Path,
    v2_root: Path,
    config_path: Path,
    *,
    expected_source_commit: str,
    expected_source_tree: str,
    diagnostic_canary: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    if (v2_root / "SHA256SUMS").is_file() or (v2_root / "SHA256SUMS.sha256").is_file():
        seal_ok, seal_errors, _ = verify_sha_manifest(v2_root)
        if not seal_ok:
            errors.extend(f"v2_{item}" for item in seal_errors)
    config = load_json(config_path)
    if not isinstance(config, Mapping) or config.get("schema") != CONFIG_SCHEMA:
        errors.append("config_invalid")
        config = {}
    try:
        binding = (
            diagnostic_binding(stage_v_root, expected_source_commit, expected_source_tree)
            if diagnostic_canary
            else formal_binding(stage_v_root, expected_source_commit=expected_source_commit, expected_source_tree=expected_source_tree)
        )
    except StageV2PreconditionError as exc:
        binding = {}
        errors.append(str(exc))
    input_receipt = load_json(v2_root / "STAGE_V2_INPUT_RECEIPT.json")
    report = load_json(v2_root / "STAGE_V2_TEACHER_ENRICHMENT_REPORT.json")
    if not isinstance(input_receipt, Mapping):
        errors.append("input_receipt_missing_or_invalid")
    elif input_receipt.get("source_binding") != binding:
        errors.append("input_binding_mismatch")
    if not isinstance(report, Mapping) or report.get("schema") != REPORT_SCHEMA:
        errors.append("producer_report_missing_or_invalid")
        report = {}
    if not finite(report):
        errors.append("producer_report_non_finite")
    try:
        observed, observed_summary = load_observations(stage_v_root)
        stored_rows = _read_rows(v2_root / "STAGE_V2_TEACHER_ENRICHMENT_ROWS.jsonl")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        observed, observed_summary, stored_rows = [], {}, []
        errors.append(f"rows_unreadable:{exc}")
    if [canonical_json(row) for row in observed] != [canonical_json(row) for row in stored_rows]:
        errors.append("producer_rows_do_not_match_sealed_input")
    recomputed = compute_report(
        observed,
        config=config,
        execution_class="DIAGNOSTIC_CANARY_ONLY" if diagnostic_canary else "FORMAL",
        binding=binding,
        input_summary={
            **observed_summary,
            "execution_class": "DIAGNOSTIC_CANARY_ONLY" if diagnostic_canary else "FORMAL",
        },
    )
    if _report_core(recomputed) != _report_core(report):
        errors.append("independent_recompute_disagrees")
    for key in ("eval160_reads", "protected_eval_reads", "attack_rollouts", "vis_rollouts", "pgd_rollouts"):
        if report.get(key, 0) != 0:
            errors.append(f"boundary_nonzero:{key}")
    root_seal_status = "NOT_REQUIRED_DIAGNOSTIC"
    if not diagnostic_canary:
        root_seal_ok, root_seal_errors, _ = verify_sha_manifest(stage_v_root)
        root_seal_status = "PASS" if root_seal_ok else "FAIL"
        errors.extend(root_seal_errors)
    audit_report = {
        "schema": SCHEMA,
        "verdict": "PASS" if not errors else "FAIL",
        "execution_class": "DIAGNOSTIC_CANARY_ONLY" if diagnostic_canary else "FORMAL",
        "formal": not diagnostic_canary,
        "for_gate": not diagnostic_canary,
        "stage_v_root": str(stage_v_root),
        "v2_root": str(v2_root),
        "source_binding": binding,
        "root_seal_status": root_seal_status,
        "producer_report_sha256": sha256_file(v2_root / "STAGE_V2_TEACHER_ENRICHMENT_REPORT.json") if (v2_root / "STAGE_V2_TEACHER_ENRICHMENT_REPORT.json").is_file() else None,
        "recomputed_status": recomputed.get("status"),
        "observed_row_count": len(observed),
        "stored_row_count": len(stored_rows),
        "errors": sorted(set(errors)),
        "audited_utc": utc_now(),
    }
    atomic_write_json(v2_root / "STAGE_V2_INDEPENDENT_AUDIT.json", audit_report)
    _write_v2_seal(v2_root)
    return audit_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-v-root", required=True, type=Path)
    parser.add_argument("--v2-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-tree", required=True)
    parser.add_argument("--diagnostic-canary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit(
            args.stage_v_root.resolve(),
            args.v2_root.resolve(),
            args.config.resolve(),
            expected_source_commit=args.expected_source_commit,
            expected_source_tree=args.expected_source_tree,
            diagnostic_canary=args.diagnostic_canary,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"STAGE_V2_AUDIT_FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"verdict": report["verdict"], "errors": report["errors"]}, sort_keys=True))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
