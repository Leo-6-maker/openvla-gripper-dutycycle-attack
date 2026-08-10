#!/usr/bin/env python3
"""Independent runtime closeout for the frozen M3.5 V1.3 diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import statistics
from typing import Any, Mapping


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
DOSES = {"T3": 3, "T5": 5, "T10": 10}
H_PHYS = 10
MAX_DOSE_NONMONOTONIC_TRIPLET_RATE = 0.25
FAILURES = {"GRIPPER_CONTACT_LOSS", "PREMATURE_OBJECT_RELEASE", "OBJECT_DROP"}
BINARY = {"V_PHYS", "NO_PHYSICAL_VULNERABILITY"}
REGISTERED_PHYSICAL = FAILURES | {"NO_PHYSICAL_FAILURE", "PHYSICAL_AMBIGUITY_ABSTAIN", "HORIZON_CENSORED_ABSTAIN", "CONTROL_REPLAY_DIVERGENCE_ABSTAIN"}


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"JSONL_OBJECT_REQUIRED:{path}:{line_number}")
        rows.append(dict(value))
    return rows


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _inside(root: Path, relative: Any) -> Path | None:
    try:
        path = (root.resolve() / str(relative)).resolve()
        path.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return path


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "count": len(finite),
        "min": min(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def _verify_seal(root: Path) -> list[str]:
    errors = []
    sums, header = root / "SHA256SUMS", root / "SHA256SUMS.sha256"
    if not sums.is_file() or not header.is_file():
        return ["SEAL_FILES_MISSING"]
    parts = header.read_text(encoding="utf-8").split()
    if len(parts) != 2 or parts[1] != "SHA256SUMS" or parts[0] != _sha_file(sums):
        errors.append("SEAL_HEADER_INVALID")
    sealed: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        target = _inside(root, relative)
        valid_digest = len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)
        if not separator or relative in sealed or not valid_digest or target is None or not target.is_file() or _sha_file(target) != digest:
            errors.append(f"SEAL_ROW_INVALID:{relative}")
        sealed.add(relative)
    return errors


def _counter_errors(value: Any, path: str = "root") -> list[str]:
    errors = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            current = f"{path}.{key}"
            if key in COUNTERS and item != 0:
                errors.append(f"PROTECTED_COUNTER_NONZERO:{current}:{item}")
            errors.extend(_counter_errors(item, current))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_counter_errors(item, f"{path}[{index}]"))
    return errors


def _position(row: Mapping[str, Any]) -> list[float] | None:
    value = row.get("post_object_position")
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _telemetry_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: row.get(field)
        for field in (
            "relative_step", "post_contact_telemetry_valid", "post_object_identity",
            "post_object_position", "post_eef_position", "post_object_eef_distance_m",
            "post_object_gripper_contact", "post_object_support_contact", "task_success",
        )
    }


def _loss_step(rows: list[Mapping[str, Any]], reference: list[Mapping[str, Any]] | None = None) -> int | None:
    count = 0
    for index, row in enumerate(rows):
        lost = row.get("post_object_gripper_contact") is not True
        if reference is not None:
            lost = reference[index].get("post_object_gripper_contact") is True and lost
        count = count + 1 if lost else 0
        if count >= 2:
            return index - 1
    return None


def _physical(branch: Mapping[str, Any], required: int, reference: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result = {"class": "PHYSICAL_AMBIGUITY_ABSTAIN", "failure_latency_steps": None, "required_horizon_steps": required, "predicate_evidence": {}, "telemetry_evidence": []}
    available = _integer(branch.get("available_horizon_steps"))
    if branch.get("status") != "PASS" or branch.get("state_restore_exact") is not True or branch.get("causal_input_binding_pass") is not True or available is None or available < required:
        return {**result, "class": "HORIZON_CENSORED_ABSTAIN"}
    raw_rows = branch.get("rows")
    rows = list(raw_rows)[:required] if isinstance(raw_rows, list) else []
    if len(rows) != required or any(not isinstance(row, Mapping) or row.get("post_contact_telemetry_valid") is not True for row in rows):
        return result
    if reference is None:
        if branch.get("control_clean_action_equivalence") is not True:
            return {**result, "class": "CONTROL_REPLAY_DIVERGENCE_ABSTAIN"}
        latency = _loss_step(rows)
        return ({**result, "class": "GRIPPER_CONTACT_LOSS", "failure_latency_steps": latency, "predicate_evidence": {"consecutive_absent_frames": 2, "normal_release_window": False}, "telemetry_evidence": [_telemetry_evidence(row) for row in rows[latency:latency + 2]]} if latency is not None else {**result, "class": "NO_PHYSICAL_FAILURE"})
    raw_control_rows = reference.get("rows")
    control_rows = list(raw_control_rows)[:required] if isinstance(raw_control_rows, list) else []
    if len(control_rows) != required or any(not isinstance(row, Mapping) or row.get("post_contact_telemetry_valid") is not True for row in control_rows):
        return result
    for index, (control, treatment) in enumerate(zip(control_rows, rows)):
        left, right = _position(control), _position(treatment)
        if left is None or right is None:
            return result
        displacement = math.sqrt(sum((right[axis] - left[axis]) ** 2 for axis in range(3)))
        z_drop = left[2] - right[2]
        if z_drop >= 0.02 and treatment.get("post_object_gripper_contact") is not True and treatment.get("post_object_support_contact") is not True:
            return {**result, "class": "OBJECT_DROP", "failure_latency_steps": index, "predicate_evidence": {"matched_z_drop_m": z_drop, "treatment_gripper_contact": False, "treatment_support_contact": False}, "telemetry_evidence": [{"control": _telemetry_evidence(control), "treatment": _telemetry_evidence(treatment)}]}
        if displacement >= 0.01 and control.get("post_object_gripper_contact") is True and treatment.get("post_object_gripper_contact") is not True:
            return {**result, "class": "PREMATURE_OBJECT_RELEASE", "failure_latency_steps": index, "predicate_evidence": {"matched_displacement_m": displacement, "normal_release_window": False}, "telemetry_evidence": [{"control": _telemetry_evidence(control), "treatment": _telemetry_evidence(treatment)}]}
    latency = _loss_step(rows, control_rows)
    if latency is not None:
        return {**result, "class": "GRIPPER_CONTACT_LOSS", "failure_latency_steps": latency, "predicate_evidence": {"consecutive_matched_contact_loss_frames": 2, "normal_release_window": False}, "telemetry_evidence": [{"control": _telemetry_evidence(control_rows[index]), "treatment": _telemetry_evidence(rows[index])} for index in range(latency, latency + 2)]}
    return {**result, "class": "NO_PHYSICAL_FAILURE"}


def _compliance(branch: Mapping[str, Any], expected: int) -> dict[str, Any]:
    receipts = branch.get("treatment_receipts") if isinstance(branch.get("treatment_receipts"), list) else []
    failures = []
    pre, post = [], []
    if len(receipts) != expected:
        failures.append(f"DELIVERED_STEP_COUNT:{len(receipts)}/{expected}")
    for index, row in enumerate(receipts):
        if not isinstance(row, Mapping):
            failures.append(f"STEP_{index}_RECEIPT_INVALID")
            continue
        for name, expected_gripper in (("raw_policy_action", 1.0), ("normalized_action", 1.0), ("env_action", -1.0)):
            vector = row.get(name)
            try:
                valid = isinstance(vector, list) and len(vector) == 7 and all(math.isfinite(float(item)) for item in vector) and abs(float(vector[-1]) - expected_gripper) <= 1e-7
            except (TypeError, ValueError):
                valid = False
            if not valid:
                failures.append(f"STEP_{index}_{name}_INVALID")
        try:
            if not math.isfinite(float(row.get("arm_delta_linf"))) or float(row["arm_delta_linf"]) > 1e-7:
                failures.append(f"STEP_{index}_ARM_DELTA_NONZERO")
            if row.get("pre_aperture") is not None and math.isfinite(float(row["pre_aperture"])):
                pre.append(float(row["pre_aperture"]))
            if row.get("post_aperture") is not None and math.isfinite(float(row["post_aperture"])):
                post.append(float(row["post_aperture"]))
        except (TypeError, ValueError):
            failures.append(f"STEP_{index}_APERTURE_INVALID")
    already_open = bool(pre and pre[0] >= 0.03)
    delta = max(post) - pre[0] if pre and post else None
    response = delta is not None and delta >= 0.005
    compliant = not failures and (already_open or response)
    return {"compliant": compliant, "command_delivery_valid": not failures, "already_open": already_open, "aperture_response": response, "aperture_delta": delta, "failures": failures, "delivered": len(receipts)}


def _label(control_outcome: Mapping[str, Any], treatment_outcome: Mapping[str, Any], treatment_compliant: bool) -> dict[str, Any]:
    control_class, treatment_class = str(control_outcome["class"]), str(treatment_outcome["class"])
    control_valid = control_class == "NO_PHYSICAL_FAILURE"
    treatment_valid = treatment_compliant and treatment_class in FAILURES | {"NO_PHYSICAL_FAILURE"}
    f_control = 1 if control_class in FAILURES else (0 if control_valid else None)
    f_open = 1 if treatment_class in FAILURES else (0 if treatment_valid else None)
    if f_control == 1:
        label = "CONTROL_CONTAMINATION_ABSTAIN" if f_open == 1 else "CONTROL_PHYSICAL_FAILURE_ABSTAIN"
    elif not control_valid:
        label = "CONTROL_INVALID_ABSTAIN"
    elif not treatment_valid:
        label = "TREATMENT_INVALID_ABSTAIN"
    elif f_control is None or f_open is None:
        label = "PHYSICAL_AMBIGUITY_ABSTAIN"
    else:
        label = "V_PHYS" if f_open == 1 else "NO_PHYSICAL_VULNERABILITY"
    return {"control_valid": control_valid, "treatment_valid": treatment_valid, "f_control": f_control, "f_open": f_open, "label_class": label}


def _repeat(rows: list[Mapping[str, Any]]) -> tuple[str, str | None, bool]:
    repetitions = [_integer(row.get("repetition")) for row in rows]
    if len(rows) != 3 or None in repetitions or sorted(repetitions) != [0, 1, 2]:
        return "HOLD_STOCHASTIC_INTERVENTION_OUTCOME", None, False
    classes = [str(row.get("label_class")) for row in rows]
    if len(set(classes)) != 1:
        return "HOLD_STOCHASTIC_INTERVENTION_OUTCOME", None, False
    if classes[0].endswith("_ABSTAIN") or classes[0] in {"UNKNOWN", "HORIZON_CENSORED"}:
        return "STABLE_ABSTAIN", classes[0], False
    if not all(row.get("treatment_compliant") is True for row in rows):
        return "TREATMENT_NONCOMPLIANCE_ABSTAIN", None, False
    return "PASS_REPEATABILITY_3_OF_3", classes[0], classes[0] in BINARY


def _gate(gates: dict[str, list[str]], name: str, passed: bool, reason: str) -> None:
    if not passed:
        gates.setdefault(name, []).append(reason)


def _audit_parent(
    root: Path,
    expected: Mapping[str, Any],
    protocol: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    integrity = _verify_seal(root)
    gates: dict[str, list[str]] = {}
    required = {name: root / name for name in (
        "PARENT_RESULT.json", "M35_RUNTIME_BINDING_RECEIPT.json", "CLEAN_TRAJECTORY.json", "PROBE_PLAN.json",
        "CORRIDOR_COVERAGE.json", "COUNTERFACTUAL_BRANCHES.jsonl", "TREATMENT_REPETITION_OBSERVATIONS.jsonl",
        "COLLAPSED_PROBE_DOSE_LABELS.jsonl", "REPEATABILITY_SUMMARY.json", "BLINDED_TAXONOMY_EVIDENCE_MANIFEST.json",
        "PROGRESS.json", "RESOURCE_PRE.json", "RESOURCE_POST.json",
    )}
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {"integrity_errors": integrity + [f"REQUIRED_FILES_MISSING:{','.join(missing)}"], "gates": gates}
    try:
        result = _load(required["PARENT_RESULT.json"])
        runtime = _load(required["M35_RUNTIME_BINDING_RECEIPT.json"])
        clean = _load(required["CLEAN_TRAJECTORY.json"])
        plan = _load(required["PROBE_PLAN.json"])
        corridor = _load(required["CORRIDOR_COVERAGE.json"])
        branches = _jsonl(required["COUNTERFACTUAL_BRANCHES.jsonl"])
        observations = _jsonl(required["TREATMENT_REPETITION_OBSERVATIONS.jsonl"])
        labels = _jsonl(required["COLLAPSED_PROBE_DOSE_LABELS.jsonl"])
        repeatability = _load(required["REPEATABILITY_SUMMARY.json"])
        evidence = _load(required["BLINDED_TAXONOMY_EVIDENCE_MANIFEST.json"])
        progress = _load(required["PROGRESS.json"])
        resource_pre, resource_post = _load(required["RESOURCE_PRE.json"]), _load(required["RESOURCE_POST.json"])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"integrity_errors": integrity + [f"ARTIFACT_READ_FAIL:{type(exc).__name__}:{exc}"], "gates": gates}
    key = str(expected.get("canonical_parent_key"))
    if result.get("schema") != "STAGE_V_M3_5_PARENT_RESULT_V2" or result.get("canonical_parent_key") != key or result.get("status") != "COMPLETE_VALID":
        integrity.append("PARENT_RESULT_SCHEMA_STATUS_IDENTITY_INVALID")
    for field in ("suite", "task_index", "state_index"):
        if result.get(field) != expected.get(field):
            integrity.append(f"PARENT_IDENTITY_MISMATCH:{field}")
    _gate(
        gates,
        "provenance",
        result.get("source_commit") == authorization.get("source_commit")
        and result.get("source_tree") == authorization.get("source_tree"),
        "SOURCE_BINDING_MISMATCH",
    )
    _gate(
        gates,
        "provenance",
        runtime.get("status") == "PASS"
        and runtime.get("parent_key") == key
        and runtime.get("source_commit") == result.get("source_commit")
        and runtime.get("source_tree") == result.get("source_tree")
        and runtime.get("runtime_input_binding", {}).get("actual_source_status") in ("", None),
        "RUNTIME_RECEIPT_INVALID",
    )
    expected_gpu = str(protocol.get("resource_contract", {}).get("gpu_uuid_by_index", {}).get(str(result.get("gpu")), "")).lower().removeprefix("gpu-")
    gpu_values = [str(item.get("gpu_uuid", "")).lower().removeprefix("gpu-") for item in (runtime, resource_pre, resource_post)]
    _gate(gates, "provenance", bool(expected_gpu) and all(value == expected_gpu for value in gpu_values), "GPU_UUID_MISMATCH")
    _gate(gates, "provenance", resource_pre.get("gpu_id") == result.get("gpu") == resource_post.get("gpu_id"), "GPU_INDEX_MISMATCH")
    _gate(
        gates,
        "provenance",
        str(runtime.get("torch_device_uuid", "")).lower().removeprefix("gpu-") == expected_gpu
        and runtime.get("torch_device_uuid_source") in {
            "torch.cuda.get_device_properties.uuid+nvidia-smi",
            "CUDA_VISIBLE_DEVICES[0]+nvidia-smi_physical_index",
        }
        and runtime.get("torch_current_device") == 0
        and runtime.get("cuda_visible_devices") == str(result.get("gpu"))
        and runtime.get("mujoco_egl_device_id") == str(result.get("gpu"))
        and runtime.get("env_render_gpu_device_id") == result.get("gpu"),
        "CUDA_LOGICAL_EGL_PHYSICAL_BINDING_INVALID",
    )
    model = result.get("runtime_input_binding", {}).get("model", {})
    expected_model = protocol.get("runtime_inputs", {}).get("models", {}).get(result.get("suite"), {})
    _gate(gates, "provenance", all(model.get(field) == expected_model.get(field) for field in ("path", "algorithm", "tree_sha256", "file_count", "total_bytes")), "MODEL_BINDING_MISMATCH")

    clean_rows = clean.get("rows") if isinstance(clean.get("rows"), list) else []
    clean_steps = [row.get("step") for row in clean_rows if isinstance(row, Mapping)]
    _gate(gates, "probe_dose_horizon", clean.get("schema") == "STAGE_V_M3_5_CLEAN_TRAJECTORY_V1" and clean.get("outcomes_read") is False and clean.get("trajectory_sha256") == _sha_json(clean_rows) and clean_steps == list(range(len(clean_rows))), "CLEAN_TRAJECTORY_INVALID")
    candidates = []
    for index, row in enumerate(clean_rows):
        try:
            finite = all(math.isfinite(float(item)) for field in ("object_position", "eef_position") for item in row.get(field, [])) and math.isfinite(float(row.get("object_eef_distance_m")))
            eligible = row.get("clean_record_valid") is True and row.get("clean_terminal") is not True and row.get("phase_eligible") is True and row.get("contact_telemetry_valid") is True and row.get("object_gripper_contact") is True and isinstance(row.get("object_support_contact"), bool) and bool(row.get("object_identity")) and finite and int(row.get("remaining_horizon")) >= 20 and len(clean_rows) - index >= 20
        except (TypeError, ValueError):
            eligible = False
        if eligible:
            candidates.append(row)
    indices = [((ordinal * (len(candidates) - 1)) + 11) // 23 for ordinal in range(24)] if len(candidates) >= 24 else []
    expected_steps = [int(candidates[index]["step"]) for index in indices] if len(set(indices)) == 24 else []
    actual_probes = plan.get("probe_steps") if isinstance(plan.get("probe_steps"), list) else []
    actual_steps = [row.get("step") for row in actual_probes if isinstance(row, Mapping)]
    _gate(gates, "probe_dose_horizon", plan.get("schema") == "STAGE_V_M3_5_PROBE_PLAN_V2" and plan.get("outcomes_read") is False and plan.get("trajectory_sha256") == _sha_json(clean_rows) and actual_steps == expected_steps and corridor.get("corridor_qualified") is True, "PROBE_PLAN_RECOMPUTE_MISMATCH")
    probe_by_id = {row.get("probe_id"): row for row in actual_probes if isinstance(row, Mapping)}
    if len(probe_by_id) != 24 or set(probe_by_id) != {f"Q{index:02d}" for index in range(24)}:
        integrity.append("PROBE_ID_COVERAGE_INVALID")

    branch_by_id: dict[str, dict[str, Any]] = {}
    controls: dict[tuple[Any, Any], dict[str, Any]] = {}
    treatments: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    if len(branches) != 288 or len(observations) != 216 or len(labels) != 72:
        integrity.append(f"ACCOUNTING_COUNT_INVALID:{len(branches)}/{len(observations)}/{len(labels)}")
    if progress.get("schema") != "STAGE_V_M3_5_PROGRESS_V1" or progress.get("stage") != "COMPLETE" or progress.get("branch_progress") != 288 or progress.get("current_branch") is not None or progress.get("protected_counters") != COUNTERS:
        integrity.append("FINAL_PROGRESS_RECEIPT_INVALID")
    valid_probe_ids = {f"Q{index:02d}" for index in range(24)}
    for row in branches:
        branch_id, probe_id, repetition, arm = str(row.get("branch_id", "")), row.get("probe_id"), row.get("repetition"), row.get("arm")
        branch = row.get("branch") if isinstance(row.get("branch"), Mapping) else {}
        expected_id = "m35-" + hashlib.sha256(f"M35_V1_3::{key}::{probe_id}::R{repetition}::{arm}".encode()).hexdigest()
        identity_valid = (
            probe_id in valid_probe_ids
            and _integer(repetition) in range(3)
            and arm in {"CONTROL", *DOSES}
            and row.get("probe_step") == probe_by_id.get(probe_id, {}).get("step")
        )
        if row.get("schema") != "STAGE_V_M3_5_PHYSICAL_EXECUTION_V2" or not identity_valid or branch_id != expected_id or branch_id in branch_by_id or row.get("branch_result_sha256") != _sha_json(branch):
            integrity.append("BRANCH_ID_SCHEMA_OR_SHA_INVALID")
        branch_by_id[branch_id] = row
        identity = (probe_id, repetition) if arm == "CONTROL" else (probe_id, repetition, arm)
        target = controls if arm == "CONTROL" else treatments
        if identity in target:
            integrity.append("BRANCH_IDENTITY_DUPLICATED")
        target[identity] = row
        probe = probe_by_id.get(probe_id, {})
        state_ok = branch.get("status") == "PASS" and branch.get("state_restore_exact") is True and branch.get("restored_state_sha256") == branch.get("expected_probe_state_sha256") == probe.get("state_sha256") and branch.get("causal_input_binding_pass") is True and branch.get("probe_policy_input_sha256") == probe.get("policy_input_sha256") and branch.get("probe_policy_rgb_224_sha256") == probe.get("policy_rgb_224_sha256") and branch.get("target_object_id") == probe.get("object_identity")
        _gate(gates, "causal_state", state_ok, f"CAUSAL_STATE_INVALID:{branch_id}")
        if arm == "CONTROL":
            available = _integer(branch.get("available_horizon_steps"))
            _gate(gates, "causal_state", branch.get("control_clean_action_equivalence") is True and available is not None and available >= 20, f"CONTROL_REPLAY_INVALID:{branch_id}")
        if row.get("protected_counters") != COUNTERS:
            integrity.append("BRANCH_PROTECTED_COUNTERS_INVALID")

    expected_controls = {(probe_id, repetition) for probe_id in valid_probe_ids for repetition in range(3)}
    expected_treatments = {(probe_id, repetition, dose) for probe_id in valid_probe_ids for repetition in range(3) for dose in DOSES}
    if set(controls) != expected_controls or set(treatments) != expected_treatments:
        integrity.append("BRANCH_IDENTITY_COVERAGE_INVALID")

    recomputed_observations: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    auto_by_branch_dose: dict[tuple[str, str], str] = {}
    dose_sanity = {
        dose: {"execution_count": 0, "command_delivery_valid_count": 0, "compliant_count": 0, "already_open_count": 0, "aperture_response_count": 0, "delivered_open_steps": [], "aperture_deltas": [], "physical_failure_count": 0}
        for dose in DOSES
    }
    dose_failures: dict[tuple[Any, Any, str], bool | None] = {}
    for identity, treatment_row in treatments.items():
        probe_id, repetition, arm = identity
        control_row = controls.get((probe_id, repetition))
        if control_row is None:
            integrity.append("ORPHAN_TREATMENT_CONTROL_MISSING")
            continue
        if treatment_row.get("shared_control_branch_id") != control_row.get("branch_id") or treatment_row.get("shared_control_result_sha256") != control_row.get("branch_result_sha256"):
            integrity.append("MATCHED_CONTROL_LINEAGE_INVALID")
        control = control_row.get("branch") if isinstance(control_row.get("branch"), Mapping) else {}
        treatment = treatment_row.get("branch") if isinstance(treatment_row.get("branch"), Mapping) else {}
        dose = DOSES.get(str(arm), -1)
        compliance = _compliance(treatment, dose)
        producer_compliance = treatment.get("treatment_compliance", {})
        compliance_match = treatment.get("treatment_compliant") is compliance["compliant"] and producer_compliance.get("command_delivery_valid") is compliance["command_delivery_valid"] and producer_compliance.get("delivered_open_steps") == compliance["delivered"]
        _gate(gates, "treatment_semantics", compliance["command_delivery_valid"], f"COMMAND_SEMANTICS_INVALID:{treatment_row.get('branch_id')}")
        _gate(gates, "treatment_compliance", compliance["compliant"] and compliance_match, f"TREATMENT_NONCOMPLIANT:{treatment_row.get('branch_id')}")
        control_actions = control.get("actions") if isinstance(control.get("actions"), list) else []
        treatment_actions = treatment.get("actions") if isinstance(treatment.get("actions"), list) else []
        surgical = len(control_actions) >= dose and len(treatment_actions) >= dose
        for index in range(max(0, dose)):
            if not surgical:
                break
            left, right = control_actions[index], treatment_actions[index]
            surgical = isinstance(left, Mapping) and isinstance(right, Mapping) and left.get("raw_policy_action", [])[:6] == right.get("raw_policy_action", [])[:6] and left.get("env_action", [])[:6] == right.get("env_action", [])[:6] and all(left.get(field) == right.get(field) for field in ("policy_input_sha256", "policy_rgb_224_sha256", "prompt_sha256", "input_ids_sha256", "pixel_values_sha256", "decode_config_sha256"))
        _gate(gates, "surgical_isolation", surgical, f"SURGICAL_ARM_MISMATCH:{treatment_row.get('branch_id')}")
        required_steps = dose + H_PHYS
        available = _integer(treatment.get("available_horizon_steps"))
        _gate(gates, "probe_dose_horizon", dose in DOSES.values() and available is not None and available >= required_steps, f"TREATMENT_HORIZON_INVALID:{treatment_row.get('branch_id')}")
        control_outcome, treatment_outcome = _physical(control, required_steps), _physical(treatment, required_steps, control)
        label = _label(control_outcome, treatment_outcome, compliance["compliant"])
        pair = treatment_row.get("pair") if isinstance(treatment_row.get("pair"), Mapping) else {}
        pair_match = all(pair.get(field) == label.get(field) for field in ("control_valid", "treatment_valid", "f_control", "f_open", "label_class")) and pair.get("control_physical_outcome") == control_outcome and pair.get("treatment_physical_outcome") == treatment_outcome and pair.get("required_horizon_steps") == required_steps
        _gate(gates, "physical_taxonomy", pair_match, f"PHYSICAL_CLASSIFIER_RECOMPUTE_MISMATCH:{treatment_row.get('branch_id')}")
        auto_by_branch_dose[(str(control_row.get("branch_id")), str(arm))] = str(control_outcome["class"])
        auto_by_branch_dose[(str(treatment_row.get("branch_id")), str(arm))] = str(treatment_outcome["class"])
        metrics = dose_sanity[str(arm)]
        metrics["execution_count"] += 1
        metrics["command_delivery_valid_count"] += int(compliance["command_delivery_valid"])
        metrics["compliant_count"] += int(compliance["compliant"])
        metrics["already_open_count"] += int(compliance["already_open"])
        metrics["aperture_response_count"] += int(compliance["aperture_response"])
        metrics["physical_failure_count"] += int(treatment_outcome["class"] in FAILURES)
        metrics["delivered_open_steps"].append(compliance["delivered"])
        dose_failures[(probe_id, repetition, str(arm))] = (
            True if treatment_outcome["class"] in FAILURES
            else (False if treatment_outcome["class"] == "NO_PHYSICAL_FAILURE" else None)
        )
        if isinstance(compliance["aperture_delta"], (int, float)) and math.isfinite(float(compliance["aperture_delta"])):
            metrics["aperture_deltas"].append(float(compliance["aperture_delta"]))
        recomputed_observations[identity] = {"label_class": label["label_class"], "treatment_compliant": compliance["compliant"], "control": control_row, "treatment": treatment_row}

    observation_by_identity = {(row.get("probe_id"), row.get("repetition"), row.get("dose")): row for row in observations}
    if len(observation_by_identity) != len(observations):
        integrity.append("OBSERVATION_IDENTITY_DUPLICATED")
    for identity, recomputed in recomputed_observations.items():
        row = observation_by_identity.get(identity)
        if row is None:
            integrity.append("TREATMENT_OBSERVATION_MISSING")
            continue
        control_row, treatment_row = recomputed["control"], recomputed["treatment"]
        valid = row.get("treatment_branch_id") == treatment_row.get("branch_id") and row.get("treatment_result_sha256") == treatment_row.get("branch_result_sha256") and row.get("shared_control_branch_id") == control_row.get("branch_id") and row.get("shared_control_result_sha256") == control_row.get("branch_result_sha256") and row.get("label_class") == recomputed["label_class"] and row.get("treatment_compliant") is recomputed["treatment_compliant"]
        if not valid:
            integrity.append("TREATMENT_OBSERVATION_RECONCILIATION_INVALID")

    labels_by_identity = {(row.get("probe_id"), row.get("dose")): row for row in labels}
    all_binary = True
    for probe_id in {f"Q{i:02d}" for i in range(24)}:
        for dose in DOSES:
            rows = [observation_by_identity.get((probe_id, repetition, dose), {}) for repetition in range(3)]
            status, collapsed, binary = _repeat(rows)
            label = labels_by_identity.get((probe_id, dose), {})
            valid = label.get("repeatability_status") == status and label.get("collapsed_label_class") == collapsed and label.get("binary_label_consumable") is binary and set(label.get("treatment_observation_ids", [])) == {row.get("observation_id") for row in rows}
            _gate(gates, "repeatability", valid and binary, f"REPEATABILITY_INVALID:{probe_id}:{dose}")
            all_binary = all_binary and valid and binary
    _gate(gates, "repeatability", repeatability.get("collapsed_labels") == labels and repeatability.get("collapsed_label_count") == 72, "REPEATABILITY_SUMMARY_INVALID")
    expected_parent_label = "PASS" if result.get("clean_success") is True and evidence.get("complete") is True and all_binary and all(row.get("branch", {}).get("status") == "PASS" for row in branches) else "FAIL"
    _gate(gates, "producer_auditor_reconciliation", result.get("label_validation_status") == expected_parent_label, "PARENT_LABEL_STATUS_MISMATCH")

    binding = result.get("taxonomy_binding") if isinstance(result.get("taxonomy_binding"), Mapping) else {}
    registered = set(binding.get("target_object_ids", []))
    _gate(gates, "physical_taxonomy", binding.get("status") == "PASS" and bool(registered) and binding.get("taxonomy_eligibility", {}).get("fixture_binding_inference_allowed") is False, "TAXONOMY_BINDING_INVALID")
    _gate(gates, "physical_taxonomy", all(row.get("branch", {}).get("target_object_id") in registered for row in branches), "BRANCH_TARGET_OBJECT_UNREGISTERED")
    manual_rows = protocol.get("blinded_manual_taxonomy_audit", {}).get("selected_pairs", [])
    manual_pair = next((row for row in manual_rows if isinstance(row, Mapping) and row.get("canonical_parent_key") == key), {}) if isinstance(manual_rows, list) else {}
    manual_control = controls.get((manual_pair.get("probe_id"), manual_pair.get("repetition")), {})
    manual_treatment = treatments.get((manual_pair.get("probe_id"), manual_pair.get("repetition"), manual_pair.get("dose")), {})
    manual_dose = str(manual_pair.get("dose", ""))
    expected_evidence_ids = {str(manual_control.get("branch_id", "")), str(manual_treatment.get("branch_id", ""))}
    evidence_records = evidence.get("records") if isinstance(evidence.get("records"), list) else []
    record_ids = [str(record.get("branch_id", "")) for record in evidence_records if isinstance(record, Mapping)]
    evidence_sha_by_id = {
        str(record.get("branch_id", "")): [frame.get("sha256") for frame in record.get("frames", []) if isinstance(frame, Mapping)]
        for record in evidence_records
        if isinstance(record, Mapping) and isinstance(record.get("frames"), list)
    }
    expected_frames = DOSES.get(manual_dose, -H_PHYS) + H_PHYS + 1
    evidence_ok = (
        evidence.get("schema") == "STAGE_V_M3_5_BLINDED_TAXONOMY_EVIDENCE_MANIFEST_V1"
        and evidence.get("canonical_parent_key") == key
        and evidence.get("preregistered_pair") == manual_pair
        and evidence.get("evidence_steps") == expected_frames - 1
        and evidence.get("complete") is True
        and len(record_ids) == 2
        and set(record_ids) == expected_evidence_ids
        and "" not in expected_evidence_ids
    )
    for record in evidence_records:
        if not isinstance(record, Mapping):
            evidence_ok = False
            continue
        frames = record.get("frames") if isinstance(record.get("frames"), list) else []
        evidence_ok = evidence_ok and record.get("complete") is True and record.get("frame_count") == expected_frames and record.get("expected_frame_count") == expected_frames and len(frames) == expected_frames
        for frame in frames:
            path = _inside(root, frame.get("path")) if isinstance(frame, Mapping) else None
            evidence_ok = evidence_ok and path is not None and path.is_file() and _sha_file(path) == frame.get("sha256")
    _gate(gates, "physical_taxonomy", evidence_ok, "BLINDED_EVIDENCE_INVALID")
    protected_errors = _counter_errors({"result": result, "runtime": runtime, "clean": clean, "plan": plan, "corridor": corridor, "branches": branches, "observations": observations, "labels": labels, "evidence": evidence, "progress": progress})
    _gate(gates, "protected_zero", not protected_errors, ";".join(protected_errors[:10]) or "PASS")

    dose_summary = {
        dose: {
            **{name: value for name, value in metrics.items() if name not in {"aperture_deltas", "delivered_open_steps"}},
            "delivered_open_steps": _numeric_summary([float(value) for value in metrics["delivered_open_steps"]]),
            "aperture_delta": _numeric_summary(metrics["aperture_deltas"]),
            "physical_failure_rate": metrics["physical_failure_count"] / metrics["execution_count"] if metrics["execution_count"] else None,
        }
        for dose, metrics in dose_sanity.items()
    }
    comparable_triplets = 0
    nonmonotonic_triplets = 0
    for probe_id in valid_probe_ids:
        for repetition in range(3):
            values = [dose_failures.get((probe_id, repetition, dose)) for dose in DOSES]
            if None in values:
                continue
            comparable_triplets += 1
            nonmonotonic_triplets += int(any(left and not right for left, right in zip(values, values[1:])))
    nonmonotonic_rate = nonmonotonic_triplets / comparable_triplets if comparable_triplets else None
    dose_summary["cross_dose"] = {
        "comparable_triplets": comparable_triplets,
        "nonmonotonic_triplets": nonmonotonic_triplets,
        "nonmonotonic_triplet_rate": nonmonotonic_rate,
        "preregistered_hold_threshold": MAX_DOSE_NONMONOTONIC_TRIPLET_RATE,
    }
    _gate(
        gates,
        "probe_dose_horizon",
        comparable_triplets == 72 and nonmonotonic_rate is not None and nonmonotonic_rate <= MAX_DOSE_NONMONOTONIC_TRIPLET_RATE,
        f"DOSE_NONMONOTONICITY_HOLD:{nonmonotonic_triplets}/{comparable_triplets}",
    )
    return {
        "canonical_parent_key": key, "root": str(root), "integrity_errors": sorted(set(integrity)),
        "gates": {name: sorted(set(reasons)) for name, reasons in gates.items()},
        "artifact_bindings": {
            name: {"path": str(path), "sha256": _sha_file(path)}
            for name, path in {**required, "SHA256SUMS": root / "SHA256SUMS", "SHA256SUMS.sha256": root / "SHA256SUMS.sha256"}.items()
            if path.is_file()
        },
        "producer_label_status": result.get("label_validation_status"), "auditor_label_status": expected_parent_label,
        "manual_pair": manual_pair,
        "manual_branches": [
            {"branch_id": manual_control.get("branch_id"), "automatic_class": auto_by_branch_dose.get((str(manual_control.get("branch_id")), manual_dose)), "source_frame_sha256s": evidence_sha_by_id.get(str(manual_control.get("branch_id")), [])},
            {"branch_id": manual_treatment.get("branch_id"), "automatic_class": auto_by_branch_dose.get((str(manual_treatment.get("branch_id")), manual_dose)), "source_frame_sha256s": evidence_sha_by_id.get(str(manual_treatment.get("branch_id")), [])},
        ],
        "dose_sanity": dose_summary,
        "evidence_manifest": str(required["BLINDED_TAXONOMY_EVIDENCE_MANIFEST.json"]),
    }


def _audit_parent_guarded(
    root: Path,
    expected: Mapping[str, Any],
    protocol: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        return _audit_parent(root, expected, protocol, authorization)
    except (IndexError, KeyError, OverflowError, TypeError, ValueError) as exc:
        return {
            "canonical_parent_key": expected.get("canonical_parent_key"),
            "root": str(root),
            "integrity_errors": [f"PARENT_AUDIT_MALFORMED_ARTIFACT:{type(exc).__name__}:{exc}"],
            "gates": {},
        }


def _find_parent_roots(runtime_root: Path) -> dict[str, Path]:
    roots = {}
    for path in runtime_root.resolve().rglob("PARENT_RESULT.json"):
        try:
            result = _load(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if result.get("schema") == "STAGE_V_M3_5_PARENT_RESULT_V2":
            key = str(result.get("canonical_parent_key", ""))
            if not key or key in roots:
                raise ValueError(f"PARENT_RESULT_DUPLICATED_OR_INVALID:{key}")
            roots[key] = path.parent
    return roots


def _audit_all(protocol_path: Path, authorization_path: Path, runtime_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol, authorization = _load(protocol_path), _load(authorization_path)
    selection_path = Path(str(protocol.get("diagnostic_parent_selection", {}).get("path", ""))).resolve()
    selection = _load(selection_path) if selection_path.is_file() else {}
    structural: list[str] = []
    if protocol.get("schema") != "STAGE_V_M3_5_DIAGNOSTIC_PROTOCOL_V1_3" or protocol.get("version") != "V1.3.3" or _sha_file(protocol_path) != authorization.get("protocol_sha256"):
        structural.append("PROTOCOL_OR_AUTHORIZATION_BINDING_INVALID")
    source_commit, source_tree = authorization.get("source_commit"), authorization.get("source_tree")
    if authorization.get("status") != "PASS" or not isinstance(source_commit, str) or len(source_commit) != 40 or not isinstance(source_tree, str) or len(source_tree) != 40 or authorization.get("source_status") not in ("", None) or authorization.get("protected_counters") != COUNTERS:
        structural.append("AUTHORIZATION_SOURCE_INVALID")
    selection_sha = _sha_file(selection_path) if selection_path.is_file() else None
    if selection_sha is None or selection_sha != protocol.get("diagnostic_parent_selection", {}).get("sha256") or authorization.get("selection_sha256") != selection_sha:
        structural.append("SELECTION_BINDING_INVALID")
    static_path = Path(str(authorization.get("static_audit_report", ""))).resolve()
    try:
        static = _load(static_path) if static_path.is_file() else {}
        static_valid = (
            static_path.is_file()
            and _sha_file(static_path) == authorization.get("static_audit_sha256")
            and static.get("schema") == protocol.get("static_audit_binding", {}).get("receipt_schema")
            and static.get("status") == "PASS"
            and static.get("protocol_sha256") == _sha_file(protocol_path)
            and static.get("actual_source_commit") == source_commit
            and static.get("actual_source_tree") == source_tree
            and static.get("actual_source_status") in ("", None)
            and static.get("protected_counters") == COUNTERS
        )
    except (OSError, ValueError, json.JSONDecodeError):
        static_valid = False
    if not static_valid:
        structural.append("STATIC_AUDIT_AUTHORIZATION_BINDING_INVALID")
    selected_rows = selection.get("selected_parents") if isinstance(selection.get("selected_parents"), list) else []
    expected_rows = {
        str(row.get("canonical_parent_key")): row
        for row in selected_rows
        if isinstance(row, Mapping) and row.get("canonical_parent_key")
    }
    if len(selected_rows) != 8 or len(expected_rows) != 8 or selection.get("selected_count") != 8:
        structural.append("SELECTION_PARENT_SET_INVALID")
    try:
        roots = _find_parent_roots(runtime_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        structural.append(f"RUNTIME_ROOT_DISCOVERY_INVALID:{type(exc).__name__}:{exc}")
        roots = {}
    if set(roots) != set(expected_rows):
        structural.append(f"PARENT_SET_MISMATCH:actual={sorted(roots)}:expected={sorted(expected_rows)}")
    parents = [_audit_parent_guarded(roots[key], expected_rows[key], protocol, authorization) for key in sorted(set(roots) & set(expected_rows))]
    structural.extend(error for parent in parents for error in parent.get("integrity_errors", []))
    return {"protocol": protocol, "selection": selection, "authorization": authorization, "structural_errors": sorted(set(structural))}, parents


def _contact_sheet(paths: list[Path], output: Path) -> None:
    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in paths]
    width, height, columns = 224, 224, 5
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    for index, image in enumerate(images):
        image.thumbnail((width, height))
        sheet.paste(image, ((index % columns) * width, (index // columns) * height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG")


def prepare_blinded(
    protocol_path: Path,
    authorization_path: Path,
    runtime_root: Path,
    output_root: Path,
    private_key_path: Path,
) -> dict[str, Any]:
    context, parents = _audit_all(protocol_path, authorization_path, runtime_root)
    if context["structural_errors"]:
        raise ValueError(";".join(context["structural_errors"]))
    if output_root.exists():
        raise FileExistsError(f"REFUSE_OVERWRITE:{output_root}")
    if private_key_path.exists() or private_key_path.resolve().is_relative_to(output_root.resolve()):
        raise FileExistsError(f"PRIVATE_KEY_MUST_BE_NEW_AND_OUTSIDE_PUBLIC_ROOT:{private_key_path}")
    output_root.mkdir(parents=True)
    salt = secrets.token_hex(32)
    packet_rows, key_rows = [], []
    for parent in parents:
        manifest = _load(Path(parent["evidence_manifest"]))
        records = {str(row.get("branch_id")): row for row in manifest.get("records", [])}
        branches = [row for row in parent["manual_branches"] if row.get("branch_id") in records]
        if len(branches) != 2 or any(row.get("automatic_class") not in REGISTERED_PHYSICAL for row in branches):
            raise ValueError(f"BLINDED_PAIR_EVIDENCE_INVALID:{parent['canonical_parent_key']}")
        branches.sort(key=lambda row: hashlib.sha256(f"{salt}::{row['branch_id']}".encode()).hexdigest())
        case_id = "case-" + hashlib.sha256(f"{salt}::{parent['canonical_parent_key']}".encode()).hexdigest()[:16]
        public_sequences, private_sequences = [], []
        for alias, branch in zip(("A", "B"), branches):
            source_frames = [Path(parent["root"]) / frame["path"] for frame in records[branch["branch_id"]]["frames"]]
            media_dir = output_root / "BLINDED_REVIEW_MEDIA" / case_id / alias
            media_dir.mkdir(parents=True)
            copied = []
            for index, source in enumerate(source_frames):
                target = media_dir / f"frame_{index:03d}.png"
                shutil.copy2(source, target)
                copied.append(target)
            sheet = media_dir / "contact_sheet.png"
            _contact_sheet(copied, sheet)
            public_sequences.append({
                "alias": alias,
                "frame_count": len(copied),
                "contact_sheet": {"path": sheet.relative_to(output_root).as_posix(), "sha256": _sha_file(sheet)},
                "frames": [{"path": path.relative_to(output_root).as_posix(), "sha256": _sha_file(path)} for path in copied],
            })
            private_sequences.append({"alias": alias, **branch})
        packet_rows.append({"case_id": case_id, "sequences": public_sequences})
        key_rows.append({"case_id": case_id, "sequences": private_sequences})
    packet = {
        "schema": "STAGE_V_M3_5_BLINDED_MANUAL_REVIEW_PACKET_V1", "status": "READY_FOR_BLINDED_REVIEW",
        "protocol_sha256": _sha_file(protocol_path), "condition_identity_visible": False,
        "registered_classes": sorted(REGISTERED_PHYSICAL), "cases": packet_rows, "protected_counters": dict(COUNTERS),
    }
    _write(output_root / "BLINDED_REVIEW_PACKET.json", packet)
    key = {
        "schema": "STAGE_V_M3_5_PRIVATE_BLINDING_KEY_V1",
        "packet_sha256": _sha_file(output_root / "BLINDED_REVIEW_PACKET.json"),
        "private_blinding_salt": salt,
        "cases": key_rows,
    }
    _write(private_key_path, key)
    os.chmod(private_key_path, 0o600)
    return {"status": "READY_FOR_BLINDED_REVIEW", "case_count": len(packet_rows), "packet": str(output_root / "BLINDED_REVIEW_PACKET.json"), "private_key": str(private_key_path)}


def _manual_state(value: str) -> str:
    if value in FAILURES:
        return "FAILURE"
    if value == "NO_PHYSICAL_FAILURE":
        return "NO_FAILURE"
    return "ABSTAIN"


def closeout(protocol_path: Path, authorization_path: Path, runtime_root: Path, packet_path: Path, key_path: Path, manual_path: Path, output_root: Path) -> dict[str, Any]:
    context, parents = _audit_all(protocol_path, authorization_path, runtime_root)
    packet, key, manual = _load(packet_path), _load(key_path), _load(manual_path)
    structural = list(context["structural_errors"])
    packet_sha = _sha_file(packet_path)
    packet_valid = (
        packet.get("schema") == "STAGE_V_M3_5_BLINDED_MANUAL_REVIEW_PACKET_V1"
        and packet.get("status") == "READY_FOR_BLINDED_REVIEW"
        and packet.get("protocol_sha256") == _sha_file(protocol_path)
        and packet.get("condition_identity_visible") is False
        and packet.get("protected_counters") == COUNTERS
    )
    manual_valid = (
        manual.get("schema") == "STAGE_V_M3_5_BLINDED_MANUAL_REVIEW_V1"
        and manual.get("status") == "COMPLETE_BLINDED_REVIEW"
        and manual.get("packet_sha256") == packet_sha
        and manual.get("condition_identity_visible") is False
    )
    salt = key.get("private_blinding_salt")
    key_valid = (
        key.get("schema") == "STAGE_V_M3_5_PRIVATE_BLINDING_KEY_V1"
        and key.get("packet_sha256") == packet_sha
        and isinstance(salt, str)
        and len(salt) == 64
        and all(character in "0123456789abcdef" for character in salt)
    )
    if not packet_valid or not key_valid or not manual_valid:
        structural.append("BLINDED_MANUAL_REVIEW_BINDING_INVALID")
    public_rows = packet.get("cases") if isinstance(packet.get("cases"), list) else []
    private_rows = key.get("cases") if isinstance(key.get("cases"), list) else []
    manual_rows = manual.get("cases") if isinstance(manual.get("cases"), list) else []
    public_cases = {row.get("case_id") for row in public_rows if isinstance(row, Mapping)}
    private_cases = {row.get("case_id"): row for row in private_rows if isinstance(row, Mapping)}
    manual_cases = {row.get("case_id"): row for row in manual_rows if isinstance(row, Mapping)}
    if public_cases != set(private_cases) or public_cases != set(manual_cases) or len(public_cases) != len(public_rows) or len(public_cases) != len(private_rows) or len(public_cases) != len(manual_rows) or len(public_cases) != 8:
        structural.append("BLINDED_MANUAL_REVIEW_CASE_SET_INVALID")
    if key_valid:
        expected_private_cases = {}
        for parent in parents:
            case_id = "case-" + hashlib.sha256(f"{salt}::{parent.get('canonical_parent_key')}".encode()).hexdigest()[:16]
            branches = sorted(
                parent.get("manual_branches", []),
                key=lambda row: hashlib.sha256(f"{salt}::{row.get('branch_id')}".encode()).hexdigest(),
            )
            expected_private_cases[case_id] = {
                "case_id": case_id,
                "sequences": [{"alias": alias, **branch} for alias, branch in zip(("A", "B"), branches)],
            }
        if private_cases != expected_private_cases:
            structural.append("PRIVATE_BLINDING_KEY_MAPPING_INVALID")
    for public in public_rows:
        if not isinstance(public, Mapping):
            structural.append("BLINDED_PUBLIC_CASE_INVALID")
            continue
        sequences = public.get("sequences") if isinstance(public.get("sequences"), list) else []
        if len(sequences) != 2 or {row.get("alias") for row in sequences if isinstance(row, Mapping)} != {"A", "B"}:
            structural.append(f"BLINDED_PUBLIC_SEQUENCE_SET_INVALID:{public.get('case_id')}")
            continue
        for sequence in sequences:
            frames = sequence.get("frames") if isinstance(sequence.get("frames"), list) else []
            sheet = sequence.get("contact_sheet") if isinstance(sequence.get("contact_sheet"), Mapping) else {}
            media = [*frames, sheet]
            if sequence.get("frame_count") != len(frames) or not media:
                structural.append(f"BLINDED_PUBLIC_MEDIA_COUNT_INVALID:{public.get('case_id')}:{sequence.get('alias')}")
            for item in media:
                path = _inside(packet_path.parent, item.get("path")) if isinstance(item, Mapping) else None
                if path is None or not path.is_file() or _sha_file(path) != item.get("sha256"):
                    structural.append(f"BLINDED_PUBLIC_MEDIA_HASH_INVALID:{public.get('case_id')}:{sequence.get('alias')}")
            private_sequences = private_cases.get(public.get("case_id"), {}).get("sequences", [])
            private_sequence = next((row for row in private_sequences if isinstance(row, Mapping) and row.get("alias") == sequence.get("alias")), {})
            if [row.get("sha256") for row in frames if isinstance(row, Mapping)] != private_sequence.get("source_frame_sha256s"):
                structural.append(f"BLINDED_PUBLIC_SOURCE_BINDING_INVALID:{public.get('case_id')}:{sequence.get('alias')}")
    major, exact, reviewed = [], 0, 0
    for case_id in sorted(public_cases & set(private_cases) & set(manual_cases)):
        private_sequences = private_cases[case_id].get("sequences", [])
        manual_sequences = manual_cases[case_id].get("sequences", [])
        private_by_alias = {row.get("alias"): row for row in private_sequences if isinstance(row, Mapping)}
        manual_by_alias = {row.get("alias"): row for row in manual_sequences if isinstance(row, Mapping)}
        if len(private_sequences) != 2 or len(manual_sequences) != 2 or set(private_by_alias) != {"A", "B"} or set(manual_by_alias) != {"A", "B"}:
            structural.append(f"BLINDED_SEQUENCE_SET_INVALID:{case_id}")
            continue
        for alias in ("A", "B"):
            automatic = str(private_by_alias[alias].get("automatic_class"))
            human = str(manual_by_alias[alias].get("manual_class"))
            if automatic not in REGISTERED_PHYSICAL:
                structural.append(f"AUTOMATIC_CLASS_UNREGISTERED:{case_id}:{alias}:{automatic}")
                continue
            if manual_by_alias[alias].get("review_complete") is not True or human not in REGISTERED_PHYSICAL:
                structural.append(f"MANUAL_CLASS_UNREGISTERED:{case_id}:{alias}:{human}")
                continue
            reviewed += 1
            exact += automatic == human
            if {_manual_state(automatic), _manual_state(human)} == {"FAILURE", "NO_FAILURE"}:
                major.append({"case_id": case_id, "alias": alias, "automatic_class": automatic, "manual_class": human})

    gate_names = ("treatment_semantics", "treatment_compliance", "causal_state", "surgical_isolation", "repeatability", "physical_taxonomy", "probe_dose_horizon", "producer_auditor_reconciliation", "provenance", "protected_zero")
    gate_failures = {name: sorted({reason for parent in parents for reason in parent.get("gates", {}).get(name, [])}) for name in gate_names}
    if major:
        gate_failures["physical_taxonomy"].append("MAJOR_BLINDED_MANUAL_DISAGREEMENT")
    if reviewed != 16:
        gate_failures["physical_taxonomy"].append(f"BLINDED_REVIEW_COVERAGE:{reviewed}/16")
    gate_status = {name: "PASS" if not reasons else "FAIL" for name, reasons in gate_failures.items()}
    parent_reconciliation = all(parent.get("producer_label_status") == parent.get("auditor_label_status") for parent in parents)
    if not parent_reconciliation:
        gate_status["producer_auditor_reconciliation"] = "FAIL"
    overall = "INVALID_HARD_STOP" if structural else ("PASS" if all(value == "PASS" for value in gate_status.values()) else "FAIL_AND_SEALED")
    if output_root.exists():
        raise FileExistsError(f"REFUSE_OVERWRITE:{output_root}")
    output_root.mkdir(parents=True)
    common = {
        "protocol": str(protocol_path),
        "protocol_sha256": _sha_file(protocol_path),
        "authorization_receipt": str(authorization_path),
        "authorization_receipt_sha256": _sha_file(authorization_path),
        "diagnostic_selection": str(context["protocol"].get("diagnostic_parent_selection", {}).get("path", "")),
        "diagnostic_selection_sha256": context["authorization"].get("selection_sha256"),
        "runtime_root": str(runtime_root),
        "blinded_packet": str(packet_path),
        "blinded_packet_sha256": packet_sha,
        "private_blinding_key_sha256": _sha_file(key_path),
        "manual_review": str(manual_path),
        "manual_review_sha256": _sha_file(manual_path),
        "protected_counters": dict(COUNTERS),
    }
    dose_sanity = {
        "preregistered_nonmonotonic_triplet_rate_hold_threshold": MAX_DOSE_NONMONOTONIC_TRIPLET_RATE,
        "parents": {parent.get("canonical_parent_key"): parent.get("dose_sanity") for parent in parents},
    }
    receipts = {
        "GRIPPER_EXECUTION_SEMANTICS.json": ("treatment_semantics", "STAGE_V_M3_5_GRIPPER_EXECUTION_SEMANTICS_V1"),
        "TREATMENT_COMPLIANCE_AUDIT.json": ("treatment_compliance", "STAGE_V_M3_5_TREATMENT_COMPLIANCE_AUDIT_V1"),
        "COUNTERFACTUAL_STATE_AUDIT.json": ("causal_state", "STAGE_V_M3_5_COUNTERFACTUAL_STATE_AUDIT_V1"),
        "SURGICAL_INTERVENTION_AUDIT.json": ("surgical_isolation", "STAGE_V_M3_5_SURGICAL_INTERVENTION_AUDIT_V1"),
        "REPEATABILITY_AUDIT.json": ("repeatability", "STAGE_V_M3_5_REPEATABILITY_AUDIT_V1"),
        "PROBE_AND_DOSE_PROTOCOL.json": ("probe_dose_horizon", "STAGE_V_M3_5_PROBE_AND_DOSE_AUDIT_V1"),
    }
    for name, (gate_name, schema) in receipts.items():
        payload = {"schema": schema, "status": gate_status[gate_name], "failures": gate_failures[gate_name], **common}
        if name == "PROBE_AND_DOSE_PROTOCOL.json":
            payload["dose_sanity"] = dose_sanity
        _write(output_root / name, payload)
    _write(output_root / "PHYSICAL_FAILURE_TAXONOMY.json", {
        "schema": "STAGE_V_M3_5_PHYSICAL_FAILURE_TAXONOMY_AUDIT_V1", "status": gate_status["physical_taxonomy"],
        "failures": gate_failures["physical_taxonomy"], "blinded_reviewed_sequences": reviewed,
        "exact_class_agreement_count": exact, "major_disagreements": major, **common,
    })
    independent = {
        "schema": "STAGE_V_M3_5_RUNTIME_INDEPENDENT_AUDIT_V1_3", "status": overall,
        "structural_errors": structural, "gate_status": gate_status, "gate_failures": gate_failures,
        "parent_audits": parents, "parent_count": len(parents), "producer_auditor_reconciliation": parent_reconciliation,
        **common,
    }
    _write(output_root / "M3_5_INDEPENDENT_AUDIT.json", independent)
    validation = {
        "schema": "STAGE_V_M3_5_VALIDATION_RECEIPT_V1_3", "status": overall,
        "M3_5_LABEL_VALIDATION": "PASS" if overall == "PASS" else "FAIL",
        "v7_authorized": overall == "PASS", "next_legal_gate": "V7_FRESH_QUALIFICATION" if overall == "PASS" else "HARD_STOP_SEALED_M3_5",
        "gate_status": gate_status, "structural_errors": structural, "protected_counters": dict(COUNTERS),
    }
    _write(output_root / "M3_5_VALIDATION_RECEIPT.json", validation)
    files = sorted(path for path in output_root.iterdir() if path.is_file())
    (output_root / "SHA256SUMS").write_text("".join(f"{_sha_file(path)}  {path.name}\n" for path in files), encoding="utf-8")
    (output_root / "SHA256SUMS.sha256").write_text(f"{_sha_file(output_root / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8")
    return validation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--prepare-blinded-root", type=Path)
    parser.add_argument("--private-key-output", type=Path)
    parser.add_argument("--blinded-packet", type=Path)
    parser.add_argument("--blinding-key", type=Path)
    parser.add_argument("--manual-review", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    if args.prepare_blinded_root:
        if args.private_key_output is None:
            raise SystemExit("PRIVATE_KEY_OUTPUT_REQUIRED_AND_MUST_BE_OUTSIDE_PUBLIC_ROOT")
        result = prepare_blinded(
            args.protocol.resolve(),
            args.authorization_receipt.resolve(),
            args.runtime_root.resolve(),
            args.prepare_blinded_root.resolve(),
            args.private_key_output.resolve(),
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    if not all((args.blinded_packet, args.blinding_key, args.manual_review, args.output_root)):
        raise SystemExit("FINAL_CLOSEOUT_ARGUMENTS_REQUIRED")
    result = closeout(
        args.protocol.resolve(), args.authorization_receipt.resolve(), args.runtime_root.resolve(),
        args.blinded_packet.resolve(), args.blinding_key.resolve(), args.manual_review.resolve(), args.output_root.resolve(),
    )
    print(json.dumps({"status": result["status"], "M3_5_LABEL_VALIDATION": result["M3_5_LABEL_VALIDATION"]}, sort_keys=True))
    return 2 if result["status"] == "INVALID_HARD_STOP" else 0


if __name__ == "__main__":
    raise SystemExit(main())
