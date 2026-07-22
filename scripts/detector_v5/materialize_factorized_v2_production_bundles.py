#!/usr/bin/env python3
"""Materialize the three Factorized V2 offline/runtime bundles.

The command consumes a sealed, explicit plan.  It does not infer paths from a
checkpoint name and it never performs inference or calibration fitting.  Any
missing top-level seal, independent identity source, or exact join aborts
before the output staging directory is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.factorized_runtime import build_runtime_record  # noqa: E402
from gripper_attack.b3_training_protocol import seal_directory  # noqa: E402
try:  # package import for pytest; direct import for the CLI file mode
    from scripts.detector_v5.audit_factorized_v2_production_inputs import (  # noqa: E402
        EXPECTED_SPLITS,
        _index_rows,
        _rows,
        _strict_json,
        verify_sealed_root,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by direct CLI use
    from audit_factorized_v2_production_inputs import (  # noqa: E402
        EXPECTED_SPLITS,
        _index_rows,
        _rows,
        _strict_json,
        verify_sealed_root,
    )


class ProductionBundleError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _verify_artifact_manifest(root: Path) -> str:
    """Verify one sealed CLEAN artifact without modifying it."""
    manifest_path = root / "artifact_sha256.json"
    if not manifest_path.is_file():
        raise ProductionBundleError(f"ARTIFACT_SEAL_MISSING:{root}")
    manifest = _strict_json(manifest_path)
    rows = manifest.get("files")
    if not isinstance(rows, list) or manifest.get("recursive_sha256") != _json_sha(rows):
        raise ProductionBundleError(f"ARTIFACT_SEAL_INVALID:{root}")
    listed: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str) or not isinstance(row.get("sha256"), str):
            raise ProductionBundleError(f"ARTIFACT_SEAL_ROW_INVALID:{root}")
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts or rel.as_posix() == "artifact_sha256.json" or rel.as_posix() in listed:
            raise ProductionBundleError(f"ARTIFACT_SEAL_PATH_INVALID:{root}")
        digest = _sha_field(row["sha256"], "ARTIFACT_FILE_SHA_INVALID")
        target = root / rel
        if not target.is_file() or target.is_symlink() or sha256_file(target) != digest:
            raise ProductionBundleError(f"ARTIFACT_FILE_MISMATCH:{root}/{rel.as_posix()}")
        if "size" in row and int(row["size"]) != target.stat().st_size:
            raise ProductionBundleError(f"ARTIFACT_FILE_SIZE_MISMATCH:{root}/{rel.as_posix()}")
        listed.add(rel.as_posix())
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_sha256.json"
    }
    if actual != listed:
        raise ProductionBundleError(
            f"ARTIFACT_SEAL_COVERAGE_MISMATCH:{root}:extra={sorted(actual - listed)}:missing={sorted(listed - actual)}"
        )
    return str(manifest["recursive_sha256"])


def _feature_order_values(root: Path) -> set[str]:
    values: set[str] = set()
    for path in root.rglob("*.jsonl"):
        for row in _rows(path):
            value = row.get("feature_order_sha256")
            if value is not None:
                if not isinstance(value, str):
                    raise ProductionBundleError(f"FEATURE_ORDER_VALUE_INVALID:{path}")
                values.add(value.lower())
    return values


def _identity_list(path: Path) -> set[str]:
    if not path.is_file():
        raise ProductionBundleError(f"IDENTITY_MANIFEST_MISSING:{path}")
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
        if isinstance(value, list):
            values = value
        elif isinstance(value, dict):
            values = next((value[key] for key in ("identities", "identity_list", "episodes", "canonical_parent_keys") if isinstance(value.get(key), list)), None)
            if values is None:
                raise ProductionBundleError(f"IDENTITY_MANIFEST_SCHEMA:{path}")
        else:
            raise ProductionBundleError(f"IDENTITY_MANIFEST_SCHEMA:{path}")
    except json.JSONDecodeError:
        values = []
        for line in text.splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    values.append(row.get("identity") or row.get("episode") or row.get("canonical_parent_key"))
    values = [item for item in values if isinstance(item, str)]
    if not values or len(values) != len(set(values)):
        raise ProductionBundleError(f"IDENTITY_MANIFEST_EMPTY_OR_DUPLICATE:{path}")
    return set(values)


def _sha_field(value: Any, code: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ProductionBundleError(code)
    try:
        int(value, 16)
    except ValueError as exc:
        raise ProductionBundleError(code) from exc
    return value.lower()


def _load_prediction(root: Path) -> list[dict[str, Any]]:
    return _rows(root / "heldout_step_predictions.jsonl")


def _assert_step_closure(rows: list[Mapping[str, Any]], identities: set[str], label: str) -> None:
    by_identity: dict[str, list[int]] = {identity: [] for identity in identities}
    for row in rows:
        identity = _row_identity(row)
        step = _row_step(row)
        if identity not in by_identity:
            raise ProductionBundleError(f"{label}_IDENTITY_SET_MISMATCH:{identity}")
        by_identity[identity].append(step)
    if set(by_identity) != identities or any(
        not steps or sorted(steps) != list(range(len(steps)))
        for steps in by_identity.values()
    ):
        raise ProductionBundleError(f"{label}_STEP_CLOSURE")


def _row_identity(row: Mapping[str, Any]) -> str:
    value = row.get("episode") or row.get("canonical_parent_key") or row.get("identity")
    if not isinstance(value, str):
        raise ProductionBundleError("IDENTITY_INVALID")
    return value


def _row_step(row: Mapping[str, Any]) -> int:
    value = row.get("step", row.get("step_index"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProductionBundleError("STEP_INVALID")
    return value


def _runtime_row(prediction: Mapping[str, Any], student: Mapping[str, Any], runtime: Mapping[str, Any], job: Mapping[str, Any], *, prediction_seal: str, runtime_seal: str) -> dict[str, Any]:
    return build_runtime_record(
        prediction,
        student,
        runtime,
        checkpoint_sha256=_sha_field(job["checkpoint_sha256"], "CHECKPOINT_SHA_INVALID"),
        source_commit=str(job["source_commit"]),
        prediction_artifact_seal=prediction_seal,
        runtime_artifact_seal=runtime_seal,
        feature_order_sha256=_sha_field(job["feature_order_sha256"], "FEATURE_ORDER_SHA_INVALID"),
        runtime_manifest=_strict_json(Path(str(job["runtime_manifest_path"]))) if job.get("runtime_manifest_path") else None,
        scheduler_source_sha256=job.get("scheduler_source_sha256"),
        structural_config_sha256=job.get("structural_config_sha256"),
    ) | {"split": job["split"]}


def _prediction_head_values(row: Mapping[str, Any], name: str) -> tuple[Any, Any]:
    probability = row.get(f"{name}_probability", row.get(f"{name}_prob"))
    logit = row.get(f"{name}_logit")
    if logit is None or probability is None:
        raise ProductionBundleError(f"CALIBRATION_HEAD_INCOMPLETE:{name}")
    return logit, probability


def _calibration_row(
    prediction: Mapping[str, Any],
    teacher: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    prediction_seal: str,
    teacher_seal: str,
    provenance: str,
) -> dict[str, Any]:
    """Join Student scores to independently sealed Teacher supervision.

    Prediction artifacts are intentionally not trusted as a label source.  A
    prediction row may carry diagnostic labels for older consumers, but the
    calibration stream must be built from the exact identity/step-matched
    Teacher row and fail closed when any head is incomplete.
    """
    output: dict[str, Any] = {"episode": _row_identity(prediction), "step": _row_step(prediction)}
    teacher_fields = {
        "grasp": ("grasp_established", "grasp_established_known_mask"),
        "manipulation": ("manipulation_active", "manipulation_active_known_mask"),
        "release": ("release_or_instability", "release_or_instability_known_mask"),
    }
    for name in ("grasp", "manipulation", "release"):
        logit, probability = _prediction_head_values(prediction, name)
        target_field, known_field = teacher_fields[name]
        if target_field not in teacher or known_field not in teacher:
            raise ProductionBundleError(f"TEACHER_HEAD_INCOMPLETE:{name}")
        target = teacher[target_field]
        known = teacher[known_field]
        if not isinstance(target, bool) or not isinstance(known, bool):
            raise ProductionBundleError(f"TEACHER_HEAD_TYPE_INVALID:{name}")
        output.update({f"{name}_logit": logit, f"{name}_probability": probability, f"{name}_target": target, f"{name}_known_mask": known})
    output.update({
        "checkpoint_sha256": job["checkpoint_sha256"],
        "source_commit": job["source_commit"],
        "feature_order_sha256": job["feature_order_sha256"],
        "prediction_artifact_seal": prediction_seal,
        "teacher_label_seal": teacher_seal,
        "identity_provenance": provenance,
    })
    return output


def _policy_row(row: Mapping[str, Any], runtime: Mapping[str, Any], job: Mapping[str, Any], *, prediction_seal: str) -> dict[str, Any]:
    """Build only the runtime side of the policy-selection bundle.

    Offline labels are written to the sibling evaluation-label stream.  They
    must not be copied into this stream because DeepSeek consumes this record
    set as scheduler input.
    """
    output = dict(runtime)
    output["prediction_artifact_seal"] = prediction_seal
    output["identity_provenance"] = "INDEPENDENT_POLICY_SELECTION"
    required_runtime = {
        "episode", "step", "route", "route_supported", "student_valid", "candidate_close",
        "action_known", "grasp_logit", "manipulation_logit", "release_logit", "split",
        "scheduler_source_sha256", "structural_config_sha256",
    }
    if not required_runtime.issubset(output) or any(output.get(key) in (None, "") for key in required_runtime):
        raise ProductionBundleError("POLICY_SELECTION_RUNTIME_FIELD_MISSING")
    return output


def _evaluation_row(row: Mapping[str, Any], *, teacher_seal: str) -> dict[str, Any]:
    required = ("strict_k10_feasible", "strict_k10_known_mask")
    if any(key not in row for key in required):
        raise ProductionBundleError("EVALUATION_LABEL_MISSING")
    return {
        "episode": _row_identity(row),
        "step": _row_step(row),
        "strict_k10_feasible": row["strict_k10_feasible"],
        "strict_k10_known_mask": row["strict_k10_known_mask"],
        "event_id": row.get("event_id"),
        "event_role": row.get("event_role"),
        "teacher_label_seal": teacher_seal,
        "eligible_start": row.get("eligible_start"),
    }


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _recursive_seal(root: Path) -> str:
    files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not (path.parent == root and path.name in {"SHA256SUMS", "SHA256SUMS.sha256"})
    )
    sums = "".join(f"{sha256_file(root / name)}  {name}\n" for name in files)
    (root / "SHA256SUMS").write_text(sums, encoding="utf-8")
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def _require_exact_plan(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if plan.get("schema") != "FACTORIZED_V2_PRODUCTION_INPUT_PLAN_V1":
        raise ProductionBundleError("PRODUCTION_PLAN_SCHEMA")
    for field in ("formal_selection_eligible", "training_authorized", "attack_authorized"):
        if plan.get(field) is not False:
            raise ProductionBundleError(f"PRODUCTION_PLAN_{field.upper()}_MUST_BE_FALSE")
    jobs = plan.get("splits")
    if not isinstance(jobs, list) or len(jobs) != 12 or {str(job.get("split")) for job in jobs if isinstance(job, Mapping)} != set(EXPECTED_SPLITS):
        raise ProductionBundleError("EXACT_12_SPLIT_CLOSURE_REQUIRED")
    required = {
        "split", "prediction_root", "run_root", "s1_root", "clean_root", "teacher_root",
        "checkpoint_root", "feature_root",
        "training_identity_manifest_path", "heldout_identity_manifest_path", "calibrator_fit_manifest_path",
        "policy_selection_manifest_path", "calibration_prediction_root", "policy_prediction_root",
        "checkpoint_sha256", "source_commit", "feature_order_sha256", "predictor_source_sha256",
        "scheduler_source_sha256", "structural_config_sha256",
    }
    result: list[Mapping[str, Any]] = []
    allowed = required | {"runtime_manifest_path"}
    for job in jobs:
        if (
            not isinstance(job, Mapping)
            or not required.issubset(set(job))
            or bool(set(job) - allowed)
        ):
            raise ProductionBundleError("PRODUCTION_PLAN_SCHEMA")
        result.append(job)
    return sorted(result, key=lambda item: EXPECTED_SPLITS.index(str(item["split"])))


def _validate_job(job: Mapping[str, Any]) -> tuple[set[str], set[str], set[str], set[str], dict[str, Any]]:
    manifest_keys = {
        "training": "training_identity_manifest_path",
        "heldout": "heldout_identity_manifest_path",
        "calibrator_fit": "calibrator_fit_manifest_path",
        "policy_selection": "policy_selection_manifest_path",
    }
    sets = {name: _identity_list(Path(str(job[manifest_keys[name]]))) for name in manifest_keys}
    if any(not sets[name] for name in sets):
        raise ProductionBundleError("IDENTITY_MANIFEST_EMPTY")
    train, heldout, calibrator, policy = sets.values()
    if train & heldout or train & calibrator or train & policy or calibrator & policy or calibrator & heldout or policy & heldout:
        raise ProductionBundleError("IDENTITY_LEAKAGE")
    seals: dict[str, Any] = {}
    for key in ("prediction_root", "run_root", "checkpoint_root", "s1_root", "feature_root", "clean_root", "teacher_root", "calibration_prediction_root", "policy_prediction_root"):
        status = verify_sealed_root(Path(str(job[key])))
        seals[key] = status
        if not status["pass"]:
            raise ProductionBundleError(f"{key.upper()}_SEAL_HOLD:{status.get('reason')}")
    feature_values = _feature_order_values(Path(str(job["feature_root"]))) if job.get("feature_root") else set()
    if feature_values and feature_values != {str(job["feature_order_sha256"]).lower()}:
        raise ProductionBundleError("FEATURE_ORDER_ROOT_MISMATCH")
    if not isinstance(job["source_commit"], str) or len(job["source_commit"]) != 40:
        raise ProductionBundleError("SOURCE_COMMIT_INVALID")
    checkpoint_sha = _sha_field(job["checkpoint_sha256"], "CHECKPOINT_SHA_INVALID")
    checkpoint_path = Path(str(job["run_root"])) / "checkpoint.pt"
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != checkpoint_sha:
        raise ProductionBundleError("CHECKPOINT_SHA_MISMATCH")
    source_binding_path = Path(str(job["run_root"])) / "source_binding.json"
    if source_binding_path.is_file():
        source_binding = _strict_json(source_binding_path)
        if source_binding.get("source_commit") not in (None, job["source_commit"]):
            raise ProductionBundleError("SOURCE_COMMIT_MISMATCH")
    _sha_field(job["feature_order_sha256"], "FEATURE_ORDER_SHA_INVALID")
    _sha_field(job["predictor_source_sha256"], "PREDICTOR_SOURCE_SHA_INVALID")
    _sha_field(job["scheduler_source_sha256"], "SCHEDULER_SOURCE_SHA_INVALID")
    _sha_field(job["structural_config_sha256"], "STRUCTURAL_CONFIG_SHA_INVALID")
    return train, heldout, calibrator, policy, seals


def materialize(plan_path: Path, output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"OUTPUT_EXISTS:{output_root}")
    plan = _strict_json(plan_path)
    jobs = _require_exact_plan(plan)
    for job in jobs:
        _validate_job(job)
    staging = output_root.with_name(f".{output_root.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        split_bindings = []
        for job in jobs:
            split = str(job["split"])
            runtime_out = staging / "runtime" / split
            calibration_out = staging / "calibration" / split
            policy_out = staging / "policy_selection" / split
            evaluation_out = staging / "evaluation" / split
            for bundle_dir in (runtime_out, calibration_out, policy_out, evaluation_out):
                bundle_dir.mkdir(parents=True)
            prediction_root = Path(str(job["prediction_root"]))
            policy_root = Path(str(job["policy_prediction_root"]))
            prediction_seal = sha256_file(prediction_root / "SHA256SUMS")
            policy_seal = sha256_file(policy_root / "SHA256SUMS")
            teacher_root = Path(str(job["teacher_root"]))
            teacher_seal = sha256_file(teacher_root / "SHA256SUMS")
            predictions = _load_prediction(prediction_root)
            calibration_predictions = _load_prediction(Path(str(job["calibration_prediction_root"])))
            policy_predictions = _load_prediction(policy_root)
            heldout = _identity_list(Path(str(job["heldout_identity_manifest_path"])))
            calibrator = _identity_list(Path(str(job["calibrator_fit_manifest_path"])))
            policy_selection = _identity_list(Path(str(job["policy_selection_manifest_path"])))
            _assert_step_closure(predictions, heldout, "HELDOUT_PREDICTION")
            _assert_step_closure(calibration_predictions, calibrator, "CALIBRATION_PREDICTION")
            _assert_step_closure(policy_predictions, policy_selection, "POLICY_PREDICTION")
            if { _row_identity(row) for row in predictions } != heldout:
                raise ProductionBundleError("HELDOUT_PREDICTION_IDENTITY_SET_MISMATCH")
            if { _row_identity(row) for row in policy_predictions } != policy_selection:
                raise ProductionBundleError("POLICY_PREDICTION_IDENTITY_SET_MISMATCH")
            checkpoint_path = Path(str(job["run_root"])) / "checkpoint.pt"
            if sha256_file(checkpoint_path) != _sha_field(job["checkpoint_sha256"], "CHECKPOINT_SHA_INVALID"):
                raise ProductionBundleError("CHECKPOINT_SHA_MISMATCH")
            student_root = Path(str(job["s1_root"]))
            clean_root = Path(str(job["clean_root"]))
            teacher_rows: dict[tuple[str, int], dict[str, Any]] = {}
            for path in teacher_root.rglob("*.jsonl"):
                if "factorized_teacher" not in path.name:
                    continue
                for item in _rows(path):
                    key = (_row_identity(item), _row_step(item))
                    if key in teacher_rows:
                        raise ProductionBundleError(f"DUPLICATE_TEACHER_STEP:{key}")
                    teacher_rows[key] = item
            runtime_rows = []
            calibration_rows = []
            evaluation_rows = []
            policy_rows = []
            policy_labels = []
            for row in predictions:
                identity = _row_identity(row)
                student = _index_rows(student_root.joinpath(*identity.split("/")) / "student_input_records.jsonl")[(identity, _row_step(row))]
                if student.get("feature_order_sha256") not in (None, job["feature_order_sha256"]):
                    raise ProductionBundleError("FEATURE_ORDER_SHA_MISMATCH")
                runtime_path = clean_root.joinpath(*identity.split("/")) / "step_records.jsonl"
                runtime_seal = _verify_artifact_manifest(runtime_path.parent)
                runtime = dict(_index_rows(runtime_path, require_identity=False)[_row_step(row)])
                runtime.setdefault("canonical_parent_key", identity)
                runtime_record = _runtime_row(row, student, runtime, job, prediction_seal=prediction_seal, runtime_seal=sha256_file(runtime_path.parent / "artifact_sha256.json"))
                if runtime_seal != json.loads((runtime_path.parent / "artifact_sha256.json").read_text(encoding="utf-8"))["recursive_sha256"]:
                    raise ProductionBundleError("RUNTIME_ARTIFACT_SEAL_MISMATCH")
                runtime_rows.append(runtime_record)
                key = (identity, _row_step(row))
                if key not in teacher_rows:
                    raise ProductionBundleError(f"TEACHER_STEP_MISSING:{key}")
                evaluation_rows.append(_evaluation_row(teacher_rows[key], teacher_seal=teacher_seal))
            calibration_prediction_seal = sha256_file(Path(str(job["calibration_prediction_root"])) / "SHA256SUMS")
            for row in calibration_predictions:
                if (_row_identity(row), _row_step(row)) not in teacher_rows:
                    raise ProductionBundleError(f"TEACHER_STEP_MISSING:{_row_identity(row), _row_step(row)}")
                calibration_rows.append(_calibration_row(row, teacher_rows[(_row_identity(row), _row_step(row))], job, prediction_seal=calibration_prediction_seal, teacher_seal=teacher_seal, provenance="INDEPENDENT_CALIBRATION"))
            policy_prediction_index = {( _row_identity(row), _row_step(row)): row for row in policy_predictions}
            if len(policy_prediction_index) != len(policy_predictions):
                raise ProductionBundleError("DUPLICATE_POLICY_PREDICTION_STEP")
            expected_policy = _identity_list(Path(str(job["policy_selection_manifest_path"])))
            if {key[0] for key in policy_prediction_index} != expected_policy:
                raise ProductionBundleError("POLICY_IDENTITY_SET_MISMATCH")
            for row in policy_predictions:
                key = (_row_identity(row), _row_step(row))
                student = _index_rows(student_root.joinpath(*key[0].split("/")) / "student_input_records.jsonl")[(key[0], key[1])]
                if student.get("feature_order_sha256") not in (None, job["feature_order_sha256"]):
                    raise ProductionBundleError("FEATURE_ORDER_SHA_MISMATCH")
                runtime_path = clean_root.joinpath(*key[0].split("/")) / "step_records.jsonl"
                _verify_artifact_manifest(runtime_path.parent)
                runtime = dict(_index_rows(runtime_path, require_identity=False)[key[1]])
                runtime.setdefault("canonical_parent_key", key[0])
                runtime_record = _runtime_row(row, student, runtime, job, prediction_seal=policy_seal, runtime_seal=sha256_file(runtime_path.parent / "artifact_sha256.json"))
                policy_rows.append(_policy_row(row, runtime_record, job, prediction_seal=policy_seal))
                policy_labels.append(_evaluation_row(teacher_rows[key], teacher_seal=teacher_seal))
            _write_jsonl(runtime_out / "runtime_scheduler_inputs.jsonl", runtime_rows)
            (runtime_out / "manifest.json").write_text(json.dumps({
                "schema": "FACTORIZED_V2_RUNTIME_SCHEDULER_INPUT_BUNDLE_V2",
                "split": split,
                "data_filename": "runtime_scheduler_inputs.jsonl",
                "record_count": len(runtime_rows),
                "checkpoint_sha256": job["checkpoint_sha256"],
                "source_commit": job["source_commit"],
                "feature_order_sha256": job["feature_order_sha256"],
                "prediction_seal": prediction_seal,
                "formal_selection_eligible": False,
                "training_authorized": False,
                "attack_authorized": False,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _write_jsonl(calibration_out / "calibration_records.jsonl", calibration_rows)
            (calibration_out / "manifest.json").write_text(json.dumps({
                "schema": "FACTORIZED_V2_OFFLINE_CALIBRATION_BUNDLE_V1",
                "split": split,
                "data_filename": "calibration_records.jsonl",
                "record_stream": "calibration_records.jsonl",
                "record_count": len(calibration_rows),
                "checkpoint_sha256": job["checkpoint_sha256"],
                "source_commit": job["source_commit"],
                "feature_order_sha256": job["feature_order_sha256"],
                "fit_identity_manifest_sha256": sha256_file(Path(str(job["calibrator_fit_manifest_path"]))),
                "checkpoint_training_identity_manifest_sha256": sha256_file(Path(str(job["training_identity_manifest_path"]))),
                "heldout_identity_manifest_sha256": sha256_file(Path(str(job["heldout_identity_manifest_path"]))),
                "feature_input_seal_sha256": sha256_file(Path(str(job["feature_root"])) / "SHA256SUMS"),
                "predictor_source_sha256": job["predictor_source_sha256"],
                "prediction_artifact_seal_sha256": calibration_prediction_seal,
                "fields": ["raw_logits", "head_targets", "head_known_masks", "identity", "step", "checkpoint_binding", "training_identity_provenance"],
                "teacher_label_seal": teacher_seal,
                "formal_selection_eligible": False,
                "training_authorized": False,
                "attack_authorized": False,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _write_jsonl(policy_out / "policy_selection_runtime_records.jsonl", policy_rows)
            _write_jsonl(policy_out / "policy_selection_evaluation_labels.jsonl", policy_labels)
            (policy_out / "manifest.json").write_text(json.dumps({
                "schema": "FACTORIZED_V2_OFFLINE_POLICY_SELECTION_BUNDLE_V1",
                "split": split,
                "runtime_record_stream": "policy_selection_runtime_records.jsonl",
                "evaluation_label_stream": "policy_selection_evaluation_labels.jsonl",
                "checkpoint_sha256": job["checkpoint_sha256"],
                "source_commit": job["source_commit"],
                "feature_order_sha256": job["feature_order_sha256"],
                "scheduler_source_sha256": job.get("scheduler_source_sha256"),
                "structural_config_sha256": job.get("structural_config_sha256"),
                "policy_selection_identities": sorted(policy_selection),
                "policy_selection_identity_manifest_sha256": sha256_file(Path(str(job["policy_selection_manifest_path"]))),
                "calibrator_identity_manifest_sha256": sha256_file(Path(str(job["calibrator_fit_manifest_path"]))),
                "formal_selection_eligible": False,
                "training_authorized": False,
                "attack_authorized": False,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _write_jsonl(evaluation_out / "evaluation_records.jsonl", evaluation_rows)
            (evaluation_out / "manifest.json").write_text(json.dumps({
                "schema": "FACTORIZED_V2_OFFLINE_EVALUATION_BUNDLE_V1",
                "split": split,
                "data_filename": "evaluation_records.jsonl",
                "record_count": len(evaluation_rows),
                "checkpoint_sha256": job["checkpoint_sha256"],
                "teacher_label_seal": teacher_seal,
                "teacher_label_seal_sha256": teacher_seal,
                "feature_input_seal_sha256": sha256_file(Path(str(job["feature_root"])) / "SHA256SUMS"),
                "record_stream": "evaluation_records.jsonl",
                "fields": [
                    "strict_k10_feasible",
                    "strict_k10_known_mask",
                    "identity",
                    "step",
                    "teacher_label_seal",
                    "eligible_start_contract",
                ],
                "formal_selection_eligible": False,
                "training_authorized": False,
                "attack_authorized": False,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            for bundle_name, bundle_dir in (("runtime", runtime_out), ("calibration", calibration_out), ("policy_selection", policy_out), ("evaluation", evaluation_out)):
                seal_directory(bundle_dir)
            split_bindings.append({
                "split": split,
                "runtime_path": f"runtime/{split}",
                "runtime_seal_sha256": sha256_file(runtime_out / "SHA256SUMS"),
                "calibration_path": f"calibration/{split}",
                "calibration_seal_sha256": sha256_file(calibration_out / "SHA256SUMS"),
                "policy_selection_path": f"policy_selection/{split}",
                "policy_selection_seal_sha256": sha256_file(policy_out / "SHA256SUMS"),
                "evaluation_path": f"evaluation/{split}",
                "evaluation_seal_sha256": sha256_file(evaluation_out / "SHA256SUMS"),
                "record_count": len(runtime_rows),
            })
        (staging / "split_bindings.json").write_text(json.dumps({"schema": "FACTORIZED_V2_PRODUCTION_SPLIT_BINDINGS_V1", "splits": split_bindings}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "manifest.json").write_text(json.dumps({
            "schema": "FACTORIZED_V2_PRODUCTION_INPUT_BUNDLE_V1",
            "split_keys": list(EXPECTED_SPLITS),
            "split_count": 12,
            "bundle_directories": ["runtime", "calibration", "policy_selection", "evaluation"],
            "formal_selection_eligible": False,
            "training_authorized": False,
            "attack_authorized": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _recursive_seal(staging)
        os.replace(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"status": "PASS", "output_root": str(output_root), "split_count": 12, "sha256s_sha256": sha256_file(output_root / "SHA256SUMS"), "formal_selection_eligible": False, "training_authorized": False, "attack_authorized": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(materialize(args.plan.resolve(), args.output_root.resolve()), sort_keys=True))
    except Exception as exc:
        print(f"HOLD:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
