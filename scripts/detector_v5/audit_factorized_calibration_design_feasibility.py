#!/usr/bin/env python3
"""Read-only identity feasibility audit for independent Factorized calibration.

The audit consumes manifests and inline identity declarations only.  It never
loads a checkpoint, runs a predictor, or reads CAL/CHECK data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable

EXPECTED = tuple(f"o{outer}_i{inner}" for outer in range(4) for inner in range(3))
VERDICTS = {
    "GROUP_CROSS_FITTED_OOF_FEASIBLE",
    "NESTED_GROUP_HOLDOUT_REQUIRED",
    "INDEPENDENT_IDENTITIES_REQUIRED",
    "BLOCKED_ROOTS_NOT_MOUNTED",
    "BLOCKED_MANIFEST_INCOMPLETE",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity_list(value: Any) -> list[str] | None:
    if isinstance(value, list):
        return sorted({str(item) for item in value if isinstance(item, str)})
    if isinstance(value, dict):
        for key in ("identities", "identity_list", "episodes", "heldout_identities", "inner_train_identities"):
            result = _identity_list(value.get(key))
            if result is not None:
                return result
        return None
    return None


def _load_identities(path_value: Any) -> list[str] | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value)
    if not path.is_file():
        return None
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        values = []
        for row in rows:
            value = row.get("identity") or row.get("episode") or row.get("canonical_parent_key")
            if isinstance(value, str):
                values.append(value)
        return sorted(set(values))
    text = path.read_text(encoding="utf-8")
    try:
        return _identity_list(json.loads(text))
    except json.JSONDecodeError:
        values = []
        for line in text.splitlines():
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    value = item.get("identity") or item.get("episode") or item.get("canonical_parent_key")
                    if isinstance(value, str):
                        values.append(value)
        return sorted(set(values))


def _job_ids(job: dict[str, Any], inline_key: str, path_keys: Iterable[str]) -> list[str] | None:
    direct = _identity_list(job.get(inline_key))
    if direct is not None:
        return direct
    for key in path_keys:
        direct = _load_identities(job.get(key))
        if direct is not None:
            return direct
    return None


def _base_result(status: str, reason: str, *, plan_path: Path | None = None) -> dict[str, Any]:
    result = {
        "schema": "FACTORIZED_CALIBRATION_IDENTITY_FEASIBILITY_AUDIT_V1",
        "status": status,
        "verdict": status if status in VERDICTS else "BLOCKED_MANIFEST_INCOMPLETE",
        "reason": reason,
        "production_inference": False,
        "training": False,
        "cal_check_read": False,
        "rows": [],
    }
    if plan_path is not None and plan_path.is_file():
        result["source_plan_sha256"] = _sha(plan_path)
    return result


def audit(plan_path: Path | None) -> dict[str, Any]:
    if plan_path is None or not plan_path.is_file():
        return _base_result("BLOCKED_ROOTS_NOT_MOUNTED", "12 checkpoint and identity manifest roots are not mounted")
    try:
        value = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _base_result("BLOCKED_MANIFEST_INCOMPLETE", f"plan unreadable: {exc}", plan_path=plan_path)
    jobs = value.get("jobs", value.get("checkpoints")) if isinstance(value, dict) else None
    if not isinstance(jobs, list) or len(jobs) != 12:
        return _base_result("BLOCKED_MANIFEST_INCOMPLETE", "exactly 12 checkpoint jobs are required", plan_path=plan_path)
    by_split: dict[str, dict[str, Any]] = {}
    missing_fields: list[str] = []
    for job in jobs:
        if not isinstance(job, dict):
            return _base_result("BLOCKED_MANIFEST_INCOMPLETE", "job must be an object", plan_path=plan_path)
        split = str(job.get("split", f"o{job.get('outer_fold')}_i{job.get('inner_fold')}"))
        if split not in EXPECTED or split in by_split:
            return _base_result("BLOCKED_MANIFEST_INCOMPLETE", f"invalid or duplicate split: {split}", plan_path=plan_path)
        train = _job_ids(job, "inner_train_identities", ("identity_manifest_path", "training_identity_manifest_path"))
        heldout = _job_ids(job, "heldout_identities", ("heldout_identity_manifest_path", "validation_identity_manifest_path"))
        calibrator = _job_ids(job, "calibrator_fit_identities", ("calibrator_fit_manifest_path",))
        policy = _job_ids(job, "policy_selection_identities", ("policy_selection_manifest_path",))
        for label, ids in (("train", train), ("heldout", heldout)):
            if not ids:
                missing_fields.append(f"{split}:{label}")
        if calibrator is None:
            missing_fields.append(f"{split}:calibrator_fit")
        if policy is None:
            missing_fields.append(f"{split}:policy_selection")
        by_split[split] = {
            "split": split,
            "train": train or [],
            "heldout": heldout or [],
            "calibrator_fit": calibrator or [],
            "policy_selection": policy or [],
            "checkpoint_path": str(job.get("checkpoint_path", "")),
        }
    if set(by_split) != set(EXPECTED):
        return _base_result("BLOCKED_MANIFEST_INCOMPLETE", "exact split closure is missing", plan_path=plan_path)
    if missing_fields:
        status = "BLOCKED_ROOTS_NOT_MOUNTED" if any(not Path(row["checkpoint_path"]).is_file() for row in by_split.values()) else "BLOCKED_MANIFEST_INCOMPLETE"
        result = _base_result(status, "required identity manifests or independent calibration/policy-selection manifests are unavailable", plan_path=plan_path)
        result["missing_fields"] = sorted(missing_fields)
        result["rows"] = [{"split": key, "train_count": len(row["train"]), "heldout_count": len(row["heldout"]), "calibrator_fit_count": len(row["calibrator_fit"]), "policy_selection_count": len(row["policy_selection"])} for key, row in sorted(by_split.items())]
        return result

    rows: list[dict[str, Any]] = []
    complete_oof = True
    all_disjoint = True
    independent_sources = True
    for split in EXPECTED:
        row = by_split[split]
        train = set(row["train"])
        heldout = set(row["heldout"])
        calibrator = set(row["calibrator_fit"])
        policy = set(row["policy_selection"])
        disjoint = not (heldout & calibrator or heldout & policy or calibrator & policy)
        all_disjoint &= disjoint and not train & heldout
        assignment = {
            identity: [candidate for candidate in EXPECTED if identity in set(by_split[candidate]["heldout"]) and identity not in set(by_split[candidate]["train"])]
            for identity in sorted(heldout)
        }
        oof_complete = all(bool(candidates) for candidates in assignment.values())
        complete_oof &= oof_complete
        independent = bool(calibrator) and not train & calibrator and not heldout & calibrator
        independent_sources &= independent
        rows.append({
            "split": split,
            "heldout_count": len(heldout),
            "calibrator_fit_count": len(calibrator),
            "policy_selection_count": len(policy),
            "pairwise_disjoint": disjoint,
            "train_heldout_disjoint": not bool(train & heldout),
            "oof_complete": oof_complete,
            "independent_calibration_source": independent,
            "heldout_identity_prediction_sources": json.dumps(assignment, sort_keys=True),
        })
    if complete_oof and all_disjoint and independent_sources:
        verdict = "GROUP_CROSS_FITTED_OOF_FEASIBLE"
    elif not all_disjoint:
        verdict = "INDEPENDENT_IDENTITIES_REQUIRED"
    elif not complete_oof:
        verdict = "NESTED_GROUP_HOLDOUT_REQUIRED"
    else:
        verdict = "INDEPENDENT_IDENTITIES_REQUIRED"
    return {
        "schema": "FACTORIZED_CALIBRATION_IDENTITY_FEASIBILITY_AUDIT_V1",
        "status": verdict,
        "verdict": verdict,
        "split_names": list(EXPECTED),
        "rows": rows,
        "production_inference": False,
        "training": False,
        "cal_check_read": False,
        "source_plan_sha256": _sha(plan_path),
    }


def _write_atomic(path: Path, data: str) -> None:
    if path.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.staging")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()
    json_output = args.json_output or args.output
    if json_output is None:
        raise SystemExit("JSON_OUTPUT_REQUIRED")
    result = audit(args.plan)
    _write_atomic(json_output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.csv_output is not None:
        columns = ["split", "heldout_count", "calibrator_fit_count", "policy_selection_count", "pairwise_disjoint", "train_heldout_disjoint", "oof_complete", "independent_calibration_source", "heldout_identity_prediction_sources"]
        lines = []
        for row in result.get("rows", []):
            lines.append({key: row.get(key, "") for key in columns})
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        if args.csv_output.exists():
            raise FileExistsError(f"OUTPUT_EXISTS:{args.csv_output}")
        temporary = args.csv_output.with_name(f".{args.csv_output.name}.{uuid.uuid4().hex}.staging")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(lines)
        os.replace(temporary, args.csv_output)
    print(json.dumps({"status": result["status"], "json_output": str(json_output), "csv_output": str(args.csv_output) if args.csv_output else None}, sort_keys=True))
    return 0 if result["status"] == "GROUP_CROSS_FITTED_OOF_FEASIBLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
