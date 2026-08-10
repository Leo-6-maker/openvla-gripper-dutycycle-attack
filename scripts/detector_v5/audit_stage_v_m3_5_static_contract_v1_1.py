"""Independent static audit for the prospective M3.5 V1.1 contract.

The auditor intentionally does not import either manifest producer.  It reads
the frozen artifacts independently and emits a machine-readable receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_keys(keys: list[str]) -> str:
    payload = json.dumps(sorted(keys), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _check(checks: list[dict[str, Any]], name: str, condition: bool, detail: str) -> None:
    checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})


def _all_zero(value: Any) -> bool:
    return isinstance(value, dict) and all(value.get(key, 0) == 0 for key in value)


def audit(
    label_contract_path: Path,
    fresh_contract_path: Path,
    exposure_manifest_path: Path,
    clean_manifest_path: Path,
    *,
    protocol_path: Path | None = None,
    output_path: Path | None = None,
    auditor_source_commit: str | None = None,
    auditor_source_tree: str | None = None,
) -> dict[str, Any]:
    label_path = Path(label_contract_path).resolve()
    fresh_path = Path(fresh_contract_path).resolve()
    exposure_path = Path(exposure_manifest_path).resolve()
    clean_path = Path(clean_manifest_path).resolve()
    label = _load(label_path)
    fresh = _load(fresh_path)
    exposure = _load(exposure_path)
    clean = _load(clean_path)
    protocol = _load(Path(protocol_path).resolve()) if protocol_path else None
    checks: list[dict[str, Any]] = []

    _check(checks, "label_schema", label.get("schema") == "STAGE_V_M3_5_LABEL_AND_QUALIFICATION_CONTRACT_V1_1", str(label.get("schema")))
    _check(checks, "fresh_schema", fresh.get("schema") == "STAGE_V_FRESH_SCIENCE_PARENT_QUALIFICATION_CONTRACT_V1_1", str(fresh.get("schema")))
    _check(checks, "label_runtime_hold", "NO_RUNTIME_AUTHORIZATION" in str(label.get("status")), str(label.get("status")))
    _check(checks, "fresh_runtime_hold", "NO_RUNTIME_AUTHORIZATION" in str(fresh.get("status")), str(fresh.get("status")))

    label_freshness = label.get("authoritative_freshness_bindings", {})
    fresh_freshness = fresh.get("authoritative_freshness", {})
    exposure_binding = label_freshness.get("latest_counterfactual_exposure_union", {})
    clean_binding = label_freshness.get("cumulative_clean_attempt_union", {})
    _check(checks, "exposure_manifest_sha_bound", exposure_binding.get("sha256") == _sha256(exposure_path), f"bound={exposure_binding.get('sha256')} actual={_sha256(exposure_path)}")
    _check(checks, "clean_manifest_sha_bound", clean_binding.get("sha256") == _sha256(clean_path), f"bound={clean_binding.get('sha256')} actual={_sha256(clean_path)}")
    _check(checks, "fresh_exposure_binding_matches_label", fresh_freshness.get("latest_counterfactual_exposure_union_sha256") == exposure_binding.get("sha256"), "fresh/label exposure SHA")
    _check(checks, "fresh_clean_binding_matches_label", fresh_freshness.get("cumulative_clean_attempt_union_sha256") == clean_binding.get("sha256"), "fresh/label clean SHA")

    for name, manifest, expected_schema, binding in (
        ("exposure", exposure, "STAGE_V_COUNTERFACTUAL_EXPOSURE_UNION_V4", exposure_binding),
        ("clean", clean, "STAGE_V_CUMULATIVE_CLEAN_ATTEMPT_EXCLUSION_V2", clean_binding),
    ):
        keys = manifest.get("excluded_parent_keys")
        unique = isinstance(keys, list) and len(keys) == len(set(keys)) and all(isinstance(key, str) and key for key in keys)
        key_count = len(keys) if isinstance(keys, list) else -1
        _check(checks, f"{name}_schema", manifest.get("schema") == expected_schema, str(manifest.get("schema")))
        _check(checks, f"{name}_status", manifest.get("status") == "PASS", str(manifest.get("status")))
        _check(checks, f"{name}_unique_keys", unique, f"count={key_count if key_count >= 0 else 'invalid'}")
        _check(checks, f"{name}_count", manifest.get("excluded_parent_count") == key_count == binding.get("excluded_parent_count"), f"manifest={manifest.get('excluded_parent_count')} bound={binding.get('excluded_parent_count')}")
        _check(checks, f"{name}_union_hash", bool(unique) and manifest.get("union_sha256") == _sha256_keys(keys), str(manifest.get("union_sha256")))
        _check(checks, f"{name}_protected_counters", _all_zero(manifest.get("protected_counters")), str(manifest.get("protected_counters")))
        _check(checks, f"{name}_source_immutable", manifest.get("source_artifacts_modified") is False and manifest.get("old_artifacts_reused") is False, "source_artifacts_modified=false; old_artifacts_reused=false")
    _check(checks, "exposure_branch_results_unread", exposure.get("branch_results_read") is False, str(exposure.get("branch_results_read")))

    accounting = label.get("execution_accounting", {})
    per_parent = accounting.get("per_parent", {})
    totals = accounting.get("formal_m4_totals", {})
    _check(checks, "accounting_per_parent", per_parent == {"probe_locations": 24, "shared_control_executions": 24, "treatment_executions": 72, "physical_branch_executions": 96, "treatment_label_rows": 72}, str(per_parent))
    _check(checks, "accounting_formal_totals", totals == {"parents": 40, "treatment_label_rows": 2880, "physical_branch_executions": 3840}, str(totals))
    _check(checks, "accounting_dose_balance", accounting.get("treatment_breakdown") == {"T3": 24, "T5": 24, "T10": 24}, str(accounting.get("treatment_breakdown")))

    probes = label.get("probe_selection_algorithm", {})
    _check(checks, "probe_count", probes.get("probe_count_per_parent") == 24 and probes.get("probes_per_stratum") == 6 and len(probes.get("phase_strata", [])) == 4, str(probes))
    _check(checks, "probe_outcome_blind", probes.get("outcome_blind_before_runtime") is True and "outcome" in " ".join(probes.get("forbidden_inputs", [])).lower(), "clean-only and forbidden outcomes")
    _check(checks, "probe_no_backfill", "do not backfill" in " ".join(probes.get("selection_procedure", [])).lower(), "insufficient strata do not backfill")

    intervention = label.get("intervention_contract", {})
    window = intervention.get("treatment_window", {})
    horizon = label.get("horizon_contract", {})
    _check(checks, "surgical_arm_isolation", window.get("arm_delta_must_equal_zero") is True and intervention.get("control_replay_after_treatment") is False, "arm delta zero; no control replay")
    _check(checks, "dose_horizon", intervention.get("dose_steps") == {"T3": 3, "T5": 5, "T10": 10} and horizon.get("physical_outcome_horizon_steps") == 10 and horizon.get("global_remaining_horizon_is_not_parent_gate") is True, "dose-specific T and H_phys")
    compliance = label.get("treatment_compliance", {})
    _check(checks, "compliance_receipt", compliance.get("aperture_delta_alone_is_not_sufficient") is True and len(compliance.get("receipt_must_bind", [])) >= 5, "actual delivery receipt required")
    taxonomy = label.get("mediator_and_physical_failure_contract", {})
    predicates = taxonomy.get("physical_failure_predicates", {})
    _check(checks, "physical_predicates", len(predicates) >= 3 and all(isinstance(value, str) and value for value in predicates.values()), str(predicates))
    _check(checks, "unknown_not_negative", taxonomy.get("unknown_is_not_negative") is True and "OPEN_COMMAND_DELIVERED" in taxonomy.get("not_physical_failure", []), "mediator separated from physical failure")
    repeat = label.get("repeatability_gate", {})
    _check(checks, "repeatability_3_of_3", repeat.get("control_repetitions") == 3 and repeat.get("treatment_repetitions_each") == 3 and repeat.get("hard_gate_name") == "INTERVENTION_REPEATABILITY_3_OF_3", str(repeat))
    _check(checks, "truth_table", len(label.get("vulnerability_label_contract", {}).get("truth_table", [])) >= 6, "explicit physical truth table")
    _check(checks, "all_four_suites", label.get("diagnostic_parent_universe", {}).get("all_four_suites_required") is True, "all four suites required")

    if protocol is not None:
        _check(checks, "protocol_schema", protocol.get("schema") == "STAGE_V_M3_5_DIAGNOSTIC_PROTOCOL_V1", str(protocol.get("schema")))
        _check(checks, "protocol_frozen", protocol.get("status") == "FROZEN_FOR_VALIDATION", str(protocol.get("status")))
        _check(checks, "protocol_runtime_boundary", protocol.get("runtime_authorized") is False, str(protocol.get("runtime_authorized")))
        protocol_contracts = protocol.get("contract_bindings", {})
        _check(checks, "protocol_contract_sha", protocol_contracts.get("label_contract", {}).get("sha256") == _sha256(label_path) and protocol_contracts.get("fresh_parent_contract", {}).get("sha256") == _sha256(fresh_path), "protocol binds both contract SHAs")

    failures = [item for item in checks if item["status"] != "PASS"]
    report = {
        "schema": "STAGE_V_M3_5_STATIC_INDEPENDENT_AUDIT_V1",
        "status": "PASS" if not failures else "FAIL",
        "auditor_role": "independent_static_auditor; does_not_import_producer_helpers",
        "audited_files": {
            "label_contract": {"path": str(label_path), "sha256": _sha256(label_path)},
            "fresh_contract": {"path": str(fresh_path), "sha256": _sha256(fresh_path)},
            "exposure_manifest": {"path": str(exposure_path), "sha256": _sha256(exposure_path)},
            "clean_manifest": {"path": str(clean_path), "sha256": _sha256(clean_path)},
            "protocol": {"path": str(Path(protocol_path).resolve()), "sha256": _sha256(Path(protocol_path))} if protocol_path else None,
        },
        "checks": checks,
        "failure_count": len(failures),
        "protected_counters": {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0},
        "auditor_source_commit": auditor_source_commit,
        "auditor_source_tree": auditor_source_tree,
    }
    if output_path:
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        report["audit_sha256"] = _sha256(output)
        output.with_name(output.name + ".sha256").write_text(f"{report['audit_sha256']}  {output.name}\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-contract", type=Path, required=True)
    parser.add_argument("--fresh-contract", type=Path, required=True)
    parser.add_argument("--exposure-manifest", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--auditor-source-commit")
    parser.add_argument("--auditor-source-tree")
    args = parser.parse_args(argv)
    report = audit(
        args.label_contract,
        args.fresh_contract,
        args.exposure_manifest,
        args.clean_manifest,
        protocol_path=args.protocol,
        output_path=args.output,
        auditor_source_commit=args.auditor_source_commit,
        auditor_source_tree=args.auditor_source_tree,
    )
    print(json.dumps({"status": report["status"], "failure_count": report["failure_count"], "output": str(Path(args.output).resolve()) if args.output else None}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
