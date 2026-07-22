#!/usr/bin/env python3
"""Fail-closed identity/OOF feasibility audit for Factorized V2.

The plan is metadata only.  This command does not load a checkpoint or run a
predictor.  It refuses to call an identity set OOF unless the corresponding
sealed prediction source and checkpoint training manifest are both present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Mapping

EXPECTED_SPLITS = tuple(f"o{outer}_i{inner}" for outer in range(4) for inner in range(3))
VERDICTS = {
    "GROUP_CROSS_FITTED_OOF_FEASIBLE",
    "NESTED_RETRAIN_REQUIRED",
    "INDEPENDENT_IDENTITIES_REQUIRED",
    "BLOCKED_ROOTS_NOT_MOUNTED",
    "BLOCKED_MANIFEST_INCOMPLETE",
}
SHA64 = re.compile(r"^[0-9a-fA-F]{64}$")
SHA40 = re.compile(r"^[0-9a-fA-F]{40}$")


class IdentityAuditError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise IdentityAuditError(f"DUPLICATE_JSON_KEY:{path}:{key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)


def _ids(path: Path) -> set[str]:
    if not path.is_file():
        raise IdentityAuditError(f"IDENTITY_MANIFEST_MISSING:{path}")
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
        if isinstance(value, list):
            values = value
        elif isinstance(value, dict):
            values = next((value[key] for key in ("identities", "identity_list", "episodes", "canonical_parent_keys") if isinstance(value.get(key), list)), None)
            if values is None:
                raise IdentityAuditError(f"IDENTITY_LIST_MISSING:{path}")
        else:
            raise IdentityAuditError(f"IDENTITY_MANIFEST_SCHEMA:{path}")
    except json.JSONDecodeError:
        values = []
        for line in text.splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    values.append(row.get("identity") or row.get("episode") or row.get("canonical_parent_key"))
    values = [item for item in values if isinstance(item, str)]
    if not values or len(values) != len(set(values)):
        raise IdentityAuditError(f"IDENTITY_MANIFEST_EMPTY_OR_DUPLICATE:{path}")
    return set(values)


def _root_sealed(path: Path) -> tuple[bool, str, str | None]:
    if not path.is_dir():
        return False, "ROOT_MISSING", None
    sums = path / "SHA256SUMS"
    sidecar = path / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        return False, "SEAL_MISSING", None
    digest = sha256_file(sums)
    if sidecar.read_text(encoding="utf-8").strip() != f"{digest}  SHA256SUMS":
        return False, "SEAL_SIDECAR_MISMATCH", digest
    return True, "PASS", digest


def _prediction_identities(root: Path, manifest_key: str = "identity_manifest_path") -> tuple[set[str], str]:
    ok, reason, seal = _root_sealed(root)
    if not ok:
        raise IdentityAuditError(f"PREDICTION_ROOT_{reason}:{root}")
    manifest = root / "prediction_manifest.json"
    if manifest.is_file():
        value = _strict_json(manifest)
        if value.get("formal_selection_eligible") is not False:
            raise IdentityAuditError(f"PREDICTION_FORMAL_FLAG_INVALID:{root}")
        for key in ("identities", "identity_list", "heldout_identities", "training_identities"):
            if isinstance(value.get(key), list):
                return set(str(item) for item in value[key]), seal or ""
    candidates = list(root.glob("*identity*.json")) + list(root.glob("*identit*.jsonl"))
    if len(candidates) == 1:
        return _ids(candidates[0]), seal or ""
    raise IdentityAuditError(f"PREDICTION_IDENTITY_MANIFEST_MISSING:{root}")


def _binding_values(job: Mapping[str, Any]) -> tuple[str, str, str]:
    source_commit = job.get("source_commit")
    feature_order = job.get("feature_order_sha256")
    predictor_source = job.get("predictor_source_sha256")
    if not isinstance(source_commit, str) or not SHA40.fullmatch(source_commit):
        raise IdentityAuditError("SOURCE_COMMIT_INVALID")
    if not isinstance(feature_order, str) or not SHA64.fullmatch(feature_order):
        raise IdentityAuditError("FEATURE_ORDER_SHA_INVALID")
    if not isinstance(predictor_source, str) or not SHA64.fullmatch(predictor_source):
        raise IdentityAuditError("PREDICTOR_SOURCE_SHA_INVALID")
    return source_commit.lower(), feature_order.lower(), predictor_source.lower()


def _audit_prediction_binding(root: Path, job: Mapping[str, Any]) -> list[str]:
    """Check source metadata when the sealed root records it.

    Historical W32 manifests omit some fields, so absence is reported for the
    plan owner to close; a present conflicting value is always a hard hold.
    """
    source_commit, feature_order, _ = _binding_values(job)
    errors: list[str] = []
    manifests: list[Path] = [path for path in (root / "prediction_manifest.json", root / "source_binding.json") if path.is_file()]
    for path in manifests:
        value = _strict_json(path)
        for key, expected in (
            ("source_commit", source_commit),
            ("student_source_commit", source_commit),
            ("feature_order_sha256", feature_order),
            ("student_feature_order_sha256", feature_order),
        ):
            if key in value and str(value[key]).lower() != expected:
                errors.append(f"{path.name}:{key}_MISMATCH")
    return errors


def _audit_checkpoint_and_feature_binding(job: Mapping[str, Any]) -> list[str]:
    source_commit, feature_order, predictor_source = _binding_values(job)
    errors: list[str] = []
    checkpoint_root = Path(str(job.get("checkpoint_root", "")))
    checkpoint = checkpoint_root / "checkpoint.pt"
    expected_checkpoint = job.get("checkpoint_sha256")
    if not isinstance(expected_checkpoint, str) or not SHA64.fullmatch(expected_checkpoint):
        errors.append("CHECKPOINT_SHA_MISSING")
    elif checkpoint.is_file() and sha256_file(checkpoint) != expected_checkpoint.lower():
        errors.append("CHECKPOINT_SHA_MISMATCH")
    feature_root = Path(str(job.get("feature_root", "")))
    if not feature_root.is_dir():
        errors.append("FEATURE_ROOT_MISSING")
    else:
        observed: set[str] = set()
        for path in feature_root.rglob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, Mapping) and value.get("feature_order_sha256") is not None:
                    observed.add(str(value["feature_order_sha256"]).lower())
        if observed and observed != {feature_order}:
            errors.append("FEATURE_ORDER_MISMATCH")
    if not SHA40.fullmatch(source_commit) or not SHA64.fullmatch(predictor_source):
        errors.append("SOURCE_BINDING_INVALID")
    return errors


def _validate_path_sha(job: Mapping[str, Any], path_key: str, sha_key: str, *, directory_sealed: bool = False) -> tuple[bool, str]:
    value = job.get(path_key)
    if not isinstance(value, str) or not value:
        return False, f"{path_key}_MISSING"
    path = Path(value)
    if not path.exists():
        return False, f"{path_key}_NOT_MOUNTED"
    expected = job.get(sha_key)
    if not isinstance(expected, str) or not SHA64.fullmatch(expected):
        return False, f"{sha_key}_MISSING"
    actual = sha256_file(path / "SHA256SUMS") if directory_sealed and path.is_dir() else sha256_file(path)
    return (actual == expected.lower(), f"{path_key}_SHA_MISMATCH" if actual != expected.lower() else "PASS")


def audit_v2(plan_path: Path) -> dict[str, Any]:
    base = {
        "schema": "FACTORIZED_CALIBRATION_IDENTITY_FEASIBILITY_AUDIT_V2",
        "production_inference": False,
        "training": False,
        "cal_check_read": False,
        "attack": False,
    }
    if not plan_path.is_file():
        return {**base, "status": "BLOCKED_ROOTS_NOT_MOUNTED", "verdict": "BLOCKED_ROOTS_NOT_MOUNTED", "reason": "PLAN_NOT_MOUNTED", "rows": []}
    try:
        value = _strict_json(plan_path)
    except Exception as exc:
        return {**base, "status": "BLOCKED_MANIFEST_INCOMPLETE", "verdict": "BLOCKED_MANIFEST_INCOMPLETE", "reason": f"PLAN_UNREADABLE:{exc}", "rows": []}
    jobs = value.get("splits") or value.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 12:
        return {**base, "status": "BLOCKED_MANIFEST_INCOMPLETE", "verdict": "BLOCKED_MANIFEST_INCOMPLETE", "reason": "EXACT_12_SPLIT_PLAN_REQUIRED", "rows": []}
    by_split: dict[str, Mapping[str, Any]] = {}
    for job in jobs:
        if not isinstance(job, Mapping) or str(job.get("split", "")) not in EXPECTED_SPLITS or job["split"] in by_split:
            return {**base, "status": "BLOCKED_MANIFEST_INCOMPLETE", "verdict": "BLOCKED_MANIFEST_INCOMPLETE", "reason": "DUPLICATE_OR_INVALID_SPLIT", "rows": []}
        by_split[str(job["split"])] = job
    if set(by_split) != set(EXPECTED_SPLITS):
        return {**base, "status": "BLOCKED_MANIFEST_INCOMPLETE", "verdict": "BLOCKED_MANIFEST_INCOMPLETE", "reason": "EXACT_SPLIT_CLOSURE_MISSING", "rows": []}

    rows: list[dict[str, Any]] = []
    blocked_roots = False
    manifest_incomplete = False
    leakage = False
    all_oof = True
    source_assignments: dict[str, list[str]] = {}
    for split in EXPECTED_SPLITS:
        job = by_split[split]
        row: dict[str, Any] = {"split": split}
        try:
            binding_errors = _audit_checkpoint_and_feature_binding(job)
        except IdentityAuditError as exc:
            binding_errors = [str(exc)]
        row["binding_errors"] = binding_errors
        if binding_errors:
            manifest_incomplete = True
        sets: dict[str, set[str]] = {}
        for name, path_key in (
            ("train", "training_identity_manifest_path"),
            ("heldout", "heldout_identity_manifest_path"),
            ("calibrator_fit", "calibrator_fit_manifest_path"),
            ("policy_selection", "policy_selection_manifest_path"),
        ):
            try:
                sets[name] = _ids(Path(str(job.get(path_key, ""))))
            except IdentityAuditError as exc:
                sets[name] = set()
                row[f"{name}_error"] = str(exc)
                manifest_incomplete = True
        pairwise = {
            "train_heldout": not bool(sets["train"] & sets["heldout"]),
            "train_calibrator_fit": not bool(sets["train"] & sets["calibrator_fit"]),
            "train_policy_selection": not bool(sets["train"] & sets["policy_selection"]),
            "calibrator_fit_policy_selection": not bool(sets["calibrator_fit"] & sets["policy_selection"]),
            "calibrator_fit_heldout": not bool(sets["calibrator_fit"] & sets["heldout"]),
            "policy_selection_heldout": not bool(sets["policy_selection"] & sets["heldout"]),
        }
        leakage |= not all(pairwise.values())
        row["counts"] = {key: len(value) for key, value in sets.items()}
        row["pairwise_disjoint"] = pairwise
        for root_key in ("checkpoint_root", "prediction_root", "calibration_prediction_root", "policy_prediction_root", "feature_root"):
            if job.get(root_key):
                ok, reason, _ = _root_sealed(Path(str(job[root_key])))
                row[f"{root_key}_seal"] = reason
                blocked_roots |= not ok
            else:
                row[f"{root_key}_seal"] = "MISSING"
                manifest_incomplete = True
        train = sets["train"]
        for source_name, source_key, identity_set in (("calibrator", "calibration_prediction_root", sets["calibrator_fit"]), ("policy", "policy_prediction_root", sets["policy_selection"])):
            try:
                predicted, source_seal = _prediction_identities(Path(str(job.get(source_key, ""))))
                binding_errors = _audit_prediction_binding(Path(str(job.get(source_key, ""))), job)
                if binding_errors:
                    row[f"{source_name}_binding_errors"] = binding_errors
                    manifest_incomplete = True
                row[f"{source_name}_prediction_identity_count"] = len(predicted)
                row[f"{source_name}_prediction_seal"] = source_seal
                row[f"{source_name}_prediction_exact"] = predicted == identity_set
                if predicted != identity_set:
                    manifest_incomplete = True
                for identity in identity_set:
                    source_assignments.setdefault(identity, []).append(f"{split}:{source_name}")
                    if identity in train:
                        leakage = True
            except Exception as exc:
                row[f"{source_name}_prediction_error"] = str(exc)
                all_oof = False
                blocked_roots = True
        rows.append(row)

    for identity, sources in source_assignments.items():
        if len(sources) != 1:
            all_oof = False
    if blocked_roots:
        verdict = "BLOCKED_ROOTS_NOT_MOUNTED"
    elif manifest_incomplete:
        verdict = "BLOCKED_MANIFEST_INCOMPLETE"
    elif leakage:
        verdict = "INDEPENDENT_IDENTITIES_REQUIRED"
    elif not all_oof:
        verdict = "NESTED_RETRAIN_REQUIRED"
    else:
        verdict = "GROUP_CROSS_FITTED_OOF_FEASIBLE"
    return {**base, "status": verdict, "verdict": verdict, "reason": "READ_ONLY_IDENTITY_AND_OOF_AUDIT", "split_names": list(EXPECTED_SPLITS), "rows": rows, "all_pairwise_disjoint": not leakage, "unique_prediction_source_assignment": all_oof, "source_plan_sha256": sha256_file(plan_path)}


def _atomic(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.staging")
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()
    result = audit_v2(args.plan.resolve())
    _atomic(args.json_output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv_output:
        fields = ["split", "counts", "pairwise_disjoint", "checkpoint_root_seal", "prediction_root_seal", "calibration_prediction_root_seal", "policy_prediction_root_seal", "calibrator_prediction_exact", "policy_prediction_exact"]
        if args.csv_output.exists():
            raise FileExistsError(f"OUTPUT_EXISTS:{args.csv_output}")
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        staging = args.csv_output.with_name(f".{args.csv_output.name}.{uuid.uuid4().hex}.staging")
        with staging.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in result.get("rows", []):
                writer.writerow({field: json.dumps(row.get(field), sort_keys=True) if isinstance(row.get(field), (dict, list)) else row.get(field, "") for field in fields})
        os.replace(staging, args.csv_output)
    print(json.dumps({"status": result["status"], "json_output": str(args.json_output)}, sort_keys=True))
    return 0 if result["status"] == "GROUP_CROSS_FITTED_OOF_FEASIBLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
