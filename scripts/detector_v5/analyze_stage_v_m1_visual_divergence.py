#!/usr/bin/env python3
"""CPU-only forensic analysis for Stage V M1 visual/input determinism."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


TRACE_FILES = {
    "raw_observation": "observation_trace.jsonl",
    "physical_state": "physical_state_trace.jsonl",
    "full_sim_state": "full_sim_state_trace.jsonl",
    "policy_rgb": "policy_rgb_224_trace.jsonl",
    "model_input": "model_input_trace.jsonl",
    "token": "policy_token_trace.jsonl",
    "postprocessed_action": "postprocessed_action_trace.jsonl",
}
CLASSIFICATIONS = {
    "RAW_OBSERVATION_NON_POLICY_DIFFERENCE",
    "SAME_MODE_RENDER_OR_OBSERVATION_NONDETERMINISM",
    "MODE_PATH_SPECIFIC_VISUAL_DIVERGENCE",
    "PROCESSOR_OR_MODEL_INPUT_NONDETERMINISM",
    "SIMULATOR_RUNTIME_NONDETERMINISM",
    "POLICY_VISUAL_INPUT_NONDETERMINISM_ACTION_STABLE",
    "MULTI_LAYER_NONDETERMINISM",
    "UNCLASSIFIED",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        fields: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                fields.append(child)
            else:
                fields.extend(_diff_paths(left[key], right[key], child))
        return fields
    if isinstance(left, list) and isinstance(right, list):
        return [] if left == right else [prefix or "<list>"]
    return [] if left == right else [prefix or "<value>"]


def _trace_pair(left_root: Path, right_root: Path, name: str, filename: str) -> dict[str, Any]:
    left_path = left_root / "trace" / filename
    right_path = right_root / "trace" / filename
    left_rows = _load_jsonl(left_path)
    right_rows = _load_jsonl(right_path)
    mismatch_steps = 0
    fields = Counter()
    first_step: int | None = None
    examples: list[dict[str, Any]] = []
    for index in range(min(len(left_rows), len(right_rows))):
        changed = _diff_paths(left_rows[index], right_rows[index])
        if changed:
            mismatch_steps += 1
            first_step = left_rows[index].get("step", index) if first_step is None else first_step
            fields.update(changed)
            if len(examples) < 20:
                examples.append({"step": left_rows[index].get("step", index), "fields": changed[:40]})
    if len(left_rows) != len(right_rows):
        first_step = first_step if first_step is not None else min(len(left_rows), len(right_rows))
    return {
        "name": name,
        "filename": filename,
        "left_rows": len(left_rows),
        "right_rows": len(right_rows),
        "row_count_equal": len(left_rows) == len(right_rows),
        "equal": mismatch_steps == 0 and len(left_rows) == len(right_rows),
        "first_mismatch_step": first_step,
        "mismatch_step_count": mismatch_steps,
        "mismatch_field_counts": dict(fields),
        "left_sha256": _sha256(left_path),
        "right_sha256": _sha256(right_path),
        "first_mismatches": examples,
    }


def _first_component_steps(left_root: Path, right_root: Path) -> dict[str, int | None]:
    traces = {name: _trace_pair(left_root, right_root, name, filename) for name, filename in TRACE_FILES.items()}
    model = {"pixel_values": None, "input_ids": None, "attention_mask": None}
    left_rows = _load_jsonl(left_root / "trace" / TRACE_FILES["model_input"])
    right_rows = _load_jsonl(right_root / "trace" / TRACE_FILES["model_input"])
    for index in range(min(len(left_rows), len(right_rows))):
        for field in model:
            path = f"model_inputs.{field}"
            if _diff_paths(left_rows[index].get("model_inputs", {}).get(field), right_rows[index].get("model_inputs", {}).get(field), path):
                if model[field] is None:
                    model[field] = left_rows[index].get("step", index)
    return {
        "raw_observation": traces["raw_observation"]["first_mismatch_step"],
        "physical_state": traces["physical_state"]["first_mismatch_step"],
        "full_sim_state": traces["full_sim_state"]["first_mismatch_step"],
        "policy_rgb": traces["policy_rgb"]["first_mismatch_step"],
        "pixel_values": model["pixel_values"],
        "input_ids": model["input_ids"],
        "attention_mask": model["attention_mask"],
        "token": traces["token"]["first_mismatch_step"],
        "raw_action": next((item["step"] for item in traces["token"]["first_mismatches"] if any(field.endswith("raw_action") for field in item["fields"])), None),
        "postprocessed_action": traces["postprocessed_action"]["first_mismatch_step"],
    }


def _classify_pair(traces: Mapping[str, Mapping[str, Any]]) -> str:
    if not traces["full_sim_state"]["equal"]:
        return "SIMULATOR_RUNTIME_NONDETERMINISM"
    if traces["policy_rgb"]["equal"] and traces["model_input"]["equal"] and not traces["raw_observation"]["equal"]:
        return "RAW_OBSERVATION_NON_POLICY_DIFFERENCE"
    if traces["raw_observation"]["equal"] and traces["policy_rgb"]["equal"] and not traces["model_input"]["equal"]:
        return "PROCESSOR_OR_MODEL_INPUT_NONDETERMINISM"
    if not traces["policy_rgb"]["equal"] and not traces["model_input"]["equal"] and traces["token"]["equal"] and traces["postprocessed_action"]["equal"]:
        return "POLICY_VISUAL_INPUT_NONDETERMINISM_ACTION_STABLE"
    if not traces["raw_observation"]["equal"] or not traces["policy_rgb"]["equal"] or not traces["model_input"]["equal"]:
        return "MULTI_LAYER_NONDETERMINISM"
    return "UNCLASSIFIED"


def _observation_components(left_root: Path, right_root: Path) -> dict[str, dict[str, Any]]:
    left_rows = _load_jsonl(left_root / "trace" / TRACE_FILES["raw_observation"])
    right_rows = _load_jsonl(right_root / "trace" / TRACE_FILES["raw_observation"])
    keys = sorted({key for rows in (left_rows, right_rows) for row in rows for key in row.get("observation", {})})
    result: dict[str, dict[str, Any]] = {}
    for key in keys:
        left_values = [row.get("observation", {}).get(key) for row in left_rows]
        right_values = [row.get("observation", {}).get(key) for row in right_rows]
        mismatch = [index for index, (left, right) in enumerate(zip(left_values, right_values)) if left != right]
        sample = next((value for value in left_values + right_values if isinstance(value, Mapping)), {})
        shape = sample.get("shape", []) if isinstance(sample, Mapping) else []
        dtype = sample.get("dtype") if isinstance(sample, Mapping) else None
        lower = key.lower()
        if len(shape) == 3 or "image" in lower or "rgb" in lower or "camera" in lower:
            kind = "IMAGE_LIKE"
        elif "depth" in lower:
            kind = "DEPTH"
        elif "seg" in lower or "mask" in lower:
            kind = "SEGMENTATION"
        elif any(token in lower for token in ("qpos", "qvel", "proprio", "robot_state")):
            kind = "ROBOT_PROPRIO"
        elif len(shape) <= 2 and shape:
            kind = "LOW_DIM_STATE"
        else:
            kind = "UNKNOWN"
        result[key] = {
            "name": key,
            "dtype": dtype,
            "shape": shape,
            "classification": kind,
            "first_mismatch_step": mismatch[0] if mismatch else None,
            "mismatch_step_count": len(mismatch),
            "left_first_hash": left_values[mismatch[0]].get("raw_sha256") if mismatch and isinstance(left_values[mismatch[0]], Mapping) else None,
            "right_first_hash": right_values[mismatch[0]].get("raw_sha256") if mismatch and isinstance(right_values[mismatch[0]], Mapping) else None,
        }
    return result


def _raw_entries(run_root: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    manifest_path = run_root / "M1_RAW_CAPTURE_MANIFEST.json"
    if not manifest_path.is_file():
        return {}
    manifest = _load(manifest_path)
    return {(int(item["step"]), str(item["group"]), str(item["field"])): item for item in manifest.get("entries", [])}


def _numeric_array(run_root: Path, entry: Mapping[str, Any]):
    import numpy as np
    raw_root = run_root / "trace" / "raw_capture"
    raw = (raw_root / str(entry["binary_path"])).read_bytes()
    dtype = str(entry["dtype"])
    shape = tuple(int(item) for item in entry["shape"])
    if dtype.startswith("torch."):
        import torch
        torch_dtype = getattr(torch, dtype.split(".", 1)[1])
        return torch.frombuffer(bytearray(raw), dtype=torch_dtype).reshape(shape).float().numpy()
    return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape)


def _bbox(mask: Any) -> dict[str, int] | None:
    import numpy as np
    if not np.any(mask):
        return None
    rows, cols = np.where(mask)
    return {"min_row": int(rows.min()), "max_row": int(rows.max()), "min_col": int(cols.min()), "max_col": int(cols.max())}


def _numeric_diff(left: Any, right: Any, *, kind: str) -> dict[str, Any]:
    import numpy as np
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return {"exact_equal": False, "shape_equal": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    diff = left.astype(np.float64) - right.astype(np.float64)
    abs_diff = np.abs(diff)
    changed = diff != 0
    if left.ndim == 3:
        changed_pixels = np.any(changed, axis=-1)
    else:
        changed_pixels = changed
    result: dict[str, Any] = {
        "kind": kind,
        "dtype": str(left.dtype),
        "shape": list(left.shape),
        "exact_equal": bool(np.array_equal(left, right)),
        "num_elements": int(left.size),
        "num_different_elements": int(np.count_nonzero(changed)),
        "different_fraction": float(np.count_nonzero(changed) / left.size) if left.size else 0.0,
        "max_abs_diff": float(abs_diff.max()) if abs_diff.size else 0.0,
        "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else 0.0,
        "median_abs_diff": float(np.median(abs_diff)) if abs_diff.size else 0.0,
        "rms_diff": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
        "changed_pixel_count": int(np.count_nonzero(changed_pixels)),
        "changed_pixel_fraction": float(np.count_nonzero(changed_pixels) / changed_pixels.size) if changed_pixels.size else 0.0,
        "bounding_box": _bbox(changed_pixels) if changed_pixels.ndim == 2 else None,
    }
    if left.ndim == 3:
        result["per_channel"] = {
            str(channel): {
                "max_abs_diff": float(abs_diff[..., channel].max()),
                "mean_abs_diff": float(abs_diff[..., channel].mean()),
                "changed_fraction": float(np.count_nonzero(changed[..., channel]) / changed[..., channel].size),
            }
            for channel in range(left.shape[-1])
        }
    if kind == "rgb_uint8":
        integer_diff = np.abs(left.astype(np.int64) - right.astype(np.int64))
        result["diff_equal_1_count"] = int(np.count_nonzero(integer_diff == 1))
        result["diff_le_1_fraction"] = float(np.count_nonzero(integer_diff <= 1) / integer_diff.size)
        result["diff_le_2_fraction"] = float(np.count_nonzero(integer_diff <= 2) / integer_diff.size)
        result["diff_le_4_fraction"] = float(np.count_nonzero(integer_diff <= 4) / integer_diff.size)
    else:
        result["quantile_abs_diff"] = {name: float(np.quantile(abs_diff, quantile)) for name, quantile in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99), ("p999", 0.999))}
    return result


def numeric_pair(left_root: Path, right_root: Path) -> dict[str, Any]:
    left_entries = _raw_entries(left_root)
    right_entries = _raw_entries(right_root)
    common = sorted(set(left_entries) & set(right_entries))
    records: list[dict[str, Any]] = []
    for key in common:
        left_entry = left_entries[key]
        right_entry = right_entries[key]
        if left_entry.get("raw_sha256") == right_entry.get("raw_sha256"):
            continue
        kind = "rgb_uint8" if key[1] in {"raw_observation", "policy_rgb_224"} and str(left_entry.get("dtype")) == "uint8" else "pixel_values"
        records.append({
            "step": key[0],
            "group": key[1],
            "field": key[2],
            "left_raw_sha256": left_entry.get("raw_sha256"),
            "right_raw_sha256": right_entry.get("raw_sha256"),
            "metrics": _numeric_diff(_numeric_array(left_root, left_entry), _numeric_array(right_root, right_entry), kind=kind),
        })
    return {
        "numeric_difference_available": True,
        "capture_steps": sorted({key[0] for key in common}),
        "different_fields": records,
    }


def analyze_pair(left_root: Path, right_root: Path, pair_name: str) -> dict[str, Any]:
    traces = {name: _trace_pair(left_root, right_root, name, filename) for name, filename in TRACE_FILES.items()}
    left_receipt = _load(left_root / "RB1_INDEPENDENT_RECEIPT.json")
    right_receipt = _load(right_root / "RB1_INDEPENDENT_RECEIPT.json")
    raw_available = (left_root / "M1_RAW_CAPTURE_MANIFEST.json").is_file() and (right_root / "M1_RAW_CAPTURE_MANIFEST.json").is_file()
    return {
        "schema": "STAGE_V_M1_PAIR_FORENSIC_V1",
        "pair_name": pair_name,
        "left_mode": left_receipt.get("mode"),
        "right_mode": right_receipt.get("mode"),
        "canonical_parent_key": left_receipt.get("canonical_parent_key"),
        "initial_state_exact": left_receipt.get("initial_state_sha256") == right_receipt.get("initial_state_sha256"),
        "terminal_step_exact": left_receipt.get("termination_step") == right_receipt.get("termination_step"),
        "terminal_outcome_exact": left_receipt.get("terminal_outcome") == right_receipt.get("terminal_outcome"),
        "traces": traces,
        "first_mismatch_by_component": _first_component_steps(left_root, right_root),
        "observation_components": _observation_components(left_root, right_root),
        "numeric_difference_available": raw_available,
        "numeric_forensic": numeric_pair(left_root, right_root) if raw_available else None,
        "classification": _classify_pair(traces),
    }


def classify_repeatability(pairs: Mapping[str, Mapping[str, Any]]) -> str:
    same_mode = [pairs["SAME_MODE_Q"], pairs["SAME_MODE_C"]]
    if any(not pair["traces"]["full_sim_state"]["equal"] for pair in same_mode):
        return "SIMULATOR_RUNTIME_NONDETERMINISM"
    q_exact = pairs["SAME_MODE_Q"]["initial_state_exact"] and all(item["equal"] for item in pairs["SAME_MODE_Q"]["traces"].values())
    c_exact = pairs["SAME_MODE_C"]["initial_state_exact"] and all(item["equal"] for item in pairs["SAME_MODE_C"]["traces"].values())
    cross_exact = all(pairs[name]["initial_state_exact"] and all(item["equal"] for item in pairs[name]["traces"].values()) for name in ("CROSS_MODE_R1", "CROSS_MODE_R2"))
    if q_exact and c_exact and not cross_exact:
        return "MODE_PATH_SPECIFIC_VISUAL_DIVERGENCE"
    if not q_exact or not c_exact:
        if all(pair["traces"]["full_sim_state"]["equal"] for pair in same_mode):
            if any(pair["traces"]["policy_rgb"]["equal"] is False and pair["traces"]["model_input"]["equal"] is False and pair["traces"]["token"]["equal"] and pair["traces"]["postprocessed_action"]["equal"] for pair in same_mode):
                return "POLICY_VISUAL_INPUT_NONDETERMINISM_ACTION_STABLE"
            if any(pair["traces"]["raw_observation"]["equal"] is False for pair in same_mode):
                return "SAME_MODE_RENDER_OR_OBSERVATION_NONDETERMINISM"
            if any(pair["traces"]["model_input"]["equal"] is False for pair in same_mode):
                return "PROCESSOR_OR_MODEL_INPUT_NONDETERMINISM"
    if all(pair["traces"]["full_sim_state"]["equal"] for pair in pairs.values()):
        return "MULTI_LAYER_NONDETERMINISM" if not cross_exact else "UNCLASSIFIED"
    return "SIMULATOR_RUNTIME_NONDETERMINISM"


def _find_failing_identity(root: Path) -> Path:
    for identity_root in sorted((root / "runs").glob("*") if (root / "runs").is_dir() else []):
        pair = identity_root / "RB1A_PAIR_AUDIT.json"
        if pair.is_file() and _load(pair).get("verdict") != "PASS":
            return identity_root
    raise ValueError("FAILING_IDENTITY_NOT_FOUND")


def verify_source_root(root: Path) -> dict[str, Any]:
    sums = root / "SHA256SUMS"
    lines = [line for line in sums.read_text(encoding="utf-8").splitlines() if line.strip()]
    bad: list[str] = []
    for line in lines:
        digest, relative = line.split("  ", 1)
        if _sha256(root / relative) != digest:
            bad.append(relative)
    manifest = _load(root / "RB1_DIAGNOSTIC_MANIFEST.json")
    manifest_without_self = dict(manifest)
    manifest_without_self.pop("manifest_sha256", None)
    recomputed = hashlib.sha256(json.dumps(manifest_without_self, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    sums_digest = _sha256(sums)
    recorded_sums_digest = (root / "SHA256SUMS.sha256").read_text(encoding="utf-8").split()[0]
    return {
        "source_root": str(root),
        "listed_file_count": len(lines),
        "all_artifact_sha256_pass": not bad,
        "bad_files": bad,
        "sha256s_sha256": sums_digest,
        "sha256s_sha256_matches": sums_digest == recorded_sums_digest,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "manifest_sha256_recomputed": recomputed,
        "manifest_sha256_matches": manifest.get("manifest_sha256") == recomputed,
        "status": _load(root / "RB1A_STATUS.json").get("status"),
    }


def write_r0(source_root: Path, output_json: Path, output_md: Path) -> dict[str, Any]:
    identity_root = _find_failing_identity(source_root)
    pair = analyze_pair(
        identity_root / "CLEAN_QUALIFICATION", identity_root / "COUNTERFACTUAL_CLEAN_PREFIX", "RB1A_EXISTING_ROOT",
    )
    report = {
        "schema": "STAGE_V_M1_READ_ONLY_FORENSIC_REPORT_V1",
        "source_root_audit": verify_source_root(source_root),
        "failing_identity_root": str(identity_root),
        "pair": pair,
        "numeric_difference_available": False,
        "classification_status": "M1_R0_PASS_LOCALIZATION_SUFFICIENT_FOR_REPEATABILITY_TEST",
        "interpretation": "Existing sidecars prove exact or non-exact byte identity only; numeric magnitude is unavailable until prospective raw capture.",
    }
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    pair_summary = pair["first_mismatch_by_component"]
    lines = [
        "# M1 R0 read-only forensic report", "",
        f"- source root: {source_root}",
        f"- failing identity: {pair['canonical_parent_key']}",
        f"- source root SHA audit: {report['source_root_audit']['all_artifact_sha256_pass']}",
        f"- numeric_difference_available: {report['numeric_difference_available']}",
        "", "| component | first mismatch step | exact |", "|---|---:|---:|",
    ]
    for component, step in pair_summary.items():
        trace_name = {"raw_observation": "raw_observation", "physical_state": "physical_state", "full_sim_state": "full_sim_state", "policy_rgb": "policy_rgb", "pixel_values": "model_input", "token": "token", "postprocessed_action": "postprocessed_action"}.get(component)
        exact = pair["traces"].get(trace_name, {}).get("equal") if trace_name else None
        lines.append(f"| {component} | {step} | {exact} |")
    lines.extend(["", "R0 is diagnostic-only and does not modify the frozen RB1 V1 decision."])
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def write_repeatability(root: Path, identity: str, run_base: Path, output_dir: Path | None = None) -> dict[str, Any]:
    identity_root = run_base / identity.replace("/", "__")
    dirs = {
        "Q1": identity_root / "CLEAN_QUALIFICATION" / "rep_01",
        "Q2": identity_root / "CLEAN_QUALIFICATION" / "rep_02",
        "C1": identity_root / "COUNTERFACTUAL_CLEAN_PREFIX" / "rep_01",
        "C2": identity_root / "COUNTERFACTUAL_CLEAN_PREFIX" / "rep_02",
    }
    pairs = {
        "SAME_MODE_Q": analyze_pair(dirs["Q1"], dirs["Q2"], "SAME_MODE_Q"),
        "SAME_MODE_C": analyze_pair(dirs["C1"], dirs["C2"], "SAME_MODE_C"),
        "CROSS_MODE_R1": analyze_pair(dirs["Q1"], dirs["C1"], "CROSS_MODE_R1"),
        "CROSS_MODE_R2": analyze_pair(dirs["Q2"], dirs["C2"], "CROSS_MODE_R2"),
    }
    classification = classify_repeatability(pairs)
    result = {
        "schema": "STAGE_V_M1_REPEATABILITY_PAIR_MATRIX_V1",
        "identity": identity,
        "pairs": pairs,
        "classification": classification,
        "numeric_difference_available": any(pair["numeric_difference_available"] for pair in pairs.values()),
    }
    target = output_dir or root
    (target / "M1_REPEATABILITY_PAIR_MATRIX.json").write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "schema": "STAGE_V_M1_REPEATABILITY_REPORT_V1",
        "identity": identity,
        "classification": classification,
        "pair_exact": {name: pair["initial_state_exact"] and pair["terminal_step_exact"] and pair["terminal_outcome_exact"] and all(trace["equal"] for trace in pair["traces"].values()) for name, pair in pairs.items()},
        "first_mismatch_by_pair": {name: pair["first_mismatch_by_component"] for name, pair in pairs.items()},
        "numeric_difference_available": result["numeric_difference_available"],
    }
    (target / "M1_REPEATABILITY_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# M1 repeatability report", "", f"- identity: {identity}", f"- classification: {classification}", "", "| pair | exact |", "|---|---:|"]
    for name, exact in report["pair_exact"].items():
        lines.append(f"| {name} | {exact} |")
    lines.extend(["", f"- numeric_difference_available: {result['numeric_difference_available']}", "- RB1A V1 remains frozen and unchanged."])
    (target / "M1_REPEATABILITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if result["numeric_difference_available"]:
        numeric = {name: pair["numeric_forensic"] for name, pair in pairs.items()}
        (target / "M1_NUMERIC_VISUAL_FORENSIC.json").write_text(json.dumps({"schema": "STAGE_V_M1_NUMERIC_VISUAL_FORENSIC_V1", "identity": identity, "pairs": numeric}, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def write_r1(root: Path, identity: str, output_dir: Path | None = None) -> dict[str, Any]:
    return write_repeatability(root, identity, root / "runs", output_dir)


def write_r2(root: Path, identity: str, output_dir: Path | None = None) -> dict[str, Any]:
    return write_repeatability(root, identity, root / "raw_runs", output_dir)


def make_capture_plan(repeatability_report: Path, identity: str, output: Path) -> dict[str, Any]:
    report = _load(repeatability_report)
    candidates: list[int] = []
    matrix_path = repeatability_report.parent / "M1_REPEATABILITY_PAIR_MATRIX.json"
    matrix = _load(matrix_path)
    for pair in matrix.get("pairs", {}).values():
        for component in ("policy_rgb", "pixel_values"):
            step = pair.get("first_mismatch_by_component", {}).get(component)
            if step is not None:
                candidates.append(int(step))
    if not candidates:
        raise ValueError("RAW_CAPTURE_T_STAR_NOT_FOUND")
    t_star = min(candidates)
    horizon = 520
    capture_steps = sorted({0, *range(max(0, t_star - 2), min(horizon - 1, t_star + 2) + 1)})
    plan = {
        "schema": "STAGE_V_M1_RAW_CAPTURE_PLAN_V1",
        "status": "FROZEN_BEFORE_RAW_CAPTURE_RUN",
        "source_repeatability_report_sha256": _sha256(repeatability_report),
        "identity": identity,
        "t_star": t_star,
        "capture_steps": capture_steps,
        "fields": ["raw_observation_image_like", "policy_rgb_224", "model_inputs.pixel_values", "model_inputs.input_ids", "model_inputs.attention_mask"],
        "generation_rule": "step_0_plus_earliest_policy_rgb_or_pixel_values_mismatch_minus2_through_plus2_clamped_to_horizon",
    }
    output.write_text(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0-source-root", type=Path)
    parser.add_argument("--r0-output-json", type=Path)
    parser.add_argument("--r0-output-md", type=Path)
    parser.add_argument("--r1-root", type=Path)
    parser.add_argument("--r2-root", type=Path)
    parser.add_argument("--make-capture-plan-report", type=Path)
    parser.add_argument("--make-capture-plan-output", type=Path)
    parser.add_argument("--identity")
    args = parser.parse_args(argv)
    try:
        if args.r0_source_root is not None:
            if not args.r0_output_json or not args.r0_output_md:
                raise ValueError("R0_OUTPUTS_REQUIRED")
            result = write_r0(args.r0_source_root.resolve(), args.r0_output_json.resolve(), args.r0_output_md.resolve())
        elif args.r1_root is not None and args.identity:
            result = write_r1(args.r1_root.resolve(), args.identity)
        elif args.r2_root is not None and args.identity:
            result = write_r2(args.r2_root.resolve(), args.identity)
        elif args.make_capture_plan_report is not None and args.make_capture_plan_output is not None and args.identity:
            result = make_capture_plan(args.make_capture_plan_report.resolve(), args.identity, args.make_capture_plan_output.resolve())
        else:
            raise ValueError("R0_OR_R1_INPUT_REQUIRED")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"verdict": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"verdict": "PASS", "classification": result.get("classification_status", result.get("classification"))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
