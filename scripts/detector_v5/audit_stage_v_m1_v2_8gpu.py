#!/usr/bin/env python3
"""Independent read-only audit for an M1-V2.1 eight-GPU root."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

try:
    from .run_stage_v_m1_v2_8gpu import (
        BOUNDARIES, GPU_IDS, IDENTITY, LABELS, REPO_ROOT, V2Error,
        _load, _write, canonical_gpu_uuid, sha256_file, validate_binding_receipt,
        validate_manifest_authorization, validate_protocol, validate_runtime_binding_receipt,
        validate_uuid_binding,
    )
except ImportError:  # direct script execution
    from run_stage_v_m1_v2_8gpu import (
        BOUNDARIES, GPU_IDS, IDENTITY, LABELS, REPO_ROOT, V2Error,
        _load, _write, canonical_gpu_uuid, sha256_file, validate_binding_receipt,
        validate_manifest_authorization, validate_protocol, validate_runtime_binding_receipt,
        validate_uuid_binding,
    )


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()


def _pair_exact(pair: Mapping[str, Any]) -> bool:
    return bool(pair.get("initial_state_exact") and pair.get("terminal_step_exact") and pair.get("terminal_outcome_exact") and all(item.get("equal") for item in pair.get("traces", {}).values()))


def _trace_equal(pair: Mapping[str, Any], field: str) -> bool:
    return pair.get("traces", {}).get(field, {}).get("equal") is True


def _render_visual_diff(pair: Mapping[str, Any]) -> bool:
    traces = pair.get("traces", {})
    return traces.get("policy_rgb", {}).get("equal") is False


def _pipeline_visual_diff(pair: Mapping[str, Any]) -> bool:
    traces = pair.get("traces", {})
    return _render_visual_diff(pair) or traces.get("model_input", {}).get("equal") is False


def _visual_diff(pair: Mapping[str, Any]) -> bool:
    return _render_visual_diff(pair)


def _full_sim_exact(pair: Mapping[str, Any]) -> bool:
    return pair.get("traces", {}).get("full_sim_state", {}).get("equal") is True


def _independent_profile(local: Mapping[str, Any], cross: Mapping[str, Any]) -> dict[str, Any]:
    same_mode: list[tuple[str, Mapping[str, Any]]] = []
    cross_mode: list[tuple[str, Mapping[str, Any]]] = []
    for gpu in GPU_IDS:
        pairs = local["gpus"][f"gpu_{gpu:02d}"]["pairs"]
        cross_mode.extend(((f"CROSS_MODE_R1_GPU{gpu}", pairs[f"CROSS_MODE_R1_GPU{gpu}"]), (f"CROSS_MODE_R2_GPU{gpu}", pairs[f"CROSS_MODE_R2_GPU{gpu}"])))
        same_mode.append((f"SAME_MODE_Q_GPU{gpu}", pairs[f"SAME_MODE_Q_GPU{gpu}"]))
    for gpu in GPU_IDS:
        pairs = local["gpus"][f"gpu_{gpu:02d}"]["pairs"]
        same_mode.append((f"SAME_MODE_C_GPU{gpu}", pairs[f"SAME_MODE_C_GPU{gpu}"]))
    cross_gpu = [(f"CROSS_GPU_{label}_{name}", pair) for label, pairs in cross["labels"].items() for name, pair in pairs.items()]
    raw_only = [name for name, pair in same_mode if not _trace_equal(pair, "raw_observation") and _trace_equal(pair, "policy_rgb") and _trace_equal(pair, "model_input") and _full_sim_exact(pair)]
    processor_only = [name for name, pair in same_mode if _trace_equal(pair, "raw_observation") and _trace_equal(pair, "policy_rgb") and not _trace_equal(pair, "model_input") and _full_sim_exact(pair)]
    same_visual = [name for name, pair in same_mode if _render_visual_diff(pair) and _full_sim_exact(pair)]
    mode_specific_visual = [name for name, pair in cross_mode if not _pair_exact(pair) and _render_visual_diff(pair) and _full_sim_exact(pair)]
    mode_specific_nonvisual = [name for name, pair in cross_mode if not _pair_exact(pair) and not _pipeline_visual_diff(pair) and _full_sim_exact(pair)]
    gpu_visual = [name for name, pair in cross_gpu if _render_visual_diff(pair) and _full_sim_exact(pair)]
    simulator = [name for name, pair in same_mode + cross_mode + cross_gpu if not _full_sim_exact(pair)]
    visual_pairs = [pair for _name, pair in same_mode + cross_mode + cross_gpu if _pipeline_visual_diff(pair) and _full_sim_exact(pair)]
    action_divergent = [name for name, pair in same_mode + cross_mode + cross_gpu if _full_sim_exact(pair) and (not _trace_equal(pair, "token") or not _trace_equal(pair, "postprocessed_action"))]
    mechanisms = set()
    if raw_only:
        mechanisms.add("RAW_OBSERVATION_NON_POLICY_DIFFERENCE")
    if processor_only:
        mechanisms.add("PROCESSOR_OR_MODEL_INPUT_NONDETERMINISM")
    if same_visual:
        mechanisms.add("SAME_MODE_RENDER_OR_OBSERVATION_NONDETERMINISM")
    if mode_specific_visual:
        mechanisms.add("MODE_PATH_SPECIFIC_VISUAL_DIVERGENCE")
    if mode_specific_nonvisual:
        mechanisms.add("MODE_SPECIFIC_NONVISUAL")
    if gpu_visual:
        mechanisms.add("GPU_CONTEXT_DEPENDENT_VISUAL_DIVERGENCE")
    if simulator:
        mechanisms.add("SIMULATOR_RUNTIME_NONDETERMINISM")
    return {
        "raw_only_pairs": raw_only,
        "processor_only_pairs": processor_only,
        "same_mode_visual_pairs": same_visual,
        "mode_specific_pairs": mode_specific_visual,
        "mode_specific_visual_pairs": mode_specific_visual,
        "mode_specific_nonvisual_pairs": mode_specific_nonvisual,
        "cross_gpu_visual_pairs": gpu_visual,
        "simulator_pairs": simulator,
        "action_divergent_pairs": action_divergent,
        "same_mode_visual_mismatch_gpus": sorted({int(name.rsplit("GPU", 1)[1]) for name in same_visual}),
        "mode_specific_mismatch_gpus": sorted({int(name.rsplit("GPU", 1)[1]) for name in mode_specific_visual}),
        "mode_specific_nonvisual_mismatch_gpus": sorted({int(name.rsplit("GPU", 1)[1]) for name in mode_specific_nonvisual}),
        "action_stable": bool(visual_pairs) and all(_trace_equal(pair, "token") and _trace_equal(pair, "postprocessed_action") for pair in visual_pairs),
        "mechanisms": sorted(mechanisms),
        "mixed_mechanisms": len(mechanisms) > 1,
    }


def _independent_classification(local: Mapping[str, Any], cross: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    profile = _independent_profile(local, cross)
    same_mode_exact = all(
        _pair_exact(local["gpus"][f"gpu_{gpu:02d}"]["pairs"][name])
        for gpu in GPU_IDS
        for name in (f"SAME_MODE_Q_GPU{gpu}", f"SAME_MODE_C_GPU{gpu}")
    )
    if profile["simulator_pairs"]:
        classification = "SIMULATOR_RUNTIME_NONDETERMINISM"
    elif profile["mixed_mechanisms"]:
        classification = "HETEROGENEOUS_MULTI_GPU_DIVERGENCE"
    elif same_mode_exact and profile["cross_gpu_visual_pairs"]:
        classification = "GPU_CONTEXT_DEPENDENT_VISUAL_DIVERGENCE"
    elif profile["raw_only_pairs"]:
        classification = "RAW_OBSERVATION_NON_POLICY_DIFFERENCE"
    elif profile["processor_only_pairs"]:
        classification = "PROCESSOR_OR_MODEL_INPUT_NONDETERMINISM"
    elif profile["same_mode_visual_pairs"]:
        classification = "SAME_MODE_RENDER_OR_OBSERVATION_NONDETERMINISM"
    elif same_mode_exact and len(profile["mode_specific_visual_pairs"]) >= 8:
        classification = "MODE_PATH_SPECIFIC_VISUAL_DIVERGENCE"
    elif profile["mode_specific_visual_pairs"] or profile["cross_gpu_visual_pairs"]:
        classification = "HETEROGENEOUS_MULTI_GPU_DIVERGENCE"
    else:
        classification = "UNCLASSIFIED"
    return classification, profile


def _verify_runs(root: Path, run_set: str, manifest: Mapping[str, Any]) -> None:
    base = root / run_set
    for gpu in GPU_IDS:
        for label in LABELS:
            run = base / f"gpu_{gpu:02d}" / label
            receipt_path = run / "RB1_INDEPENDENT_RECEIPT.json"
            if not receipt_path.is_file():
                raise V2Error(f"V2_RECEIPT_MISSING:{run_set}:gpu_{gpu:02d}:{label}")
            receipt = _load(receipt_path)
            if receipt.get("canonical_parent_key") != IDENTITY:
                raise V2Error(f"V2_IDENTITY_MISMATCH:{run_set}:gpu_{gpu:02d}:{label}")
            if any(receipt.get(field, 0) != 0 for field in BOUNDARIES):
                raise V2Error(f"V2_PROTECTED_BOUNDARY_NONZERO:{run_set}:gpu_{gpu:02d}:{label}")
            runtime = _load(run / "M1_V2_RUNTIME_BINDING_RECEIPT.json")
            validate_runtime_binding_receipt(
                runtime, gpu, run_set=run_set, phase=label,
                source_commit=str(manifest["source_commit"]), source_tree=str(manifest["source_tree"]),
            )
            gate = f"PRE_{run_set.upper()}_{label}"
            preflight = _load(root / f"M1_V2_1_GPU_PREFLIGHT_{gate}.json")
            canary = _load(root / "M1_V2_RENDERER_CANARY.json")
            validate_uuid_binding(
                gpu=gpu,
                preflight_uuid=preflight.get("uuid_by_gpu", {}).get(str(gpu)),
                canary_uuid=canary.get("uuid_by_gpu", {}).get(str(gpu)),
                runtime_uuid=runtime.get("torch_device_uuid_canonical", runtime.get("torch_device_uuid")),
                phase=f"{run_set.upper()}_{label}",
            )


def _verify_canary(root: Path) -> None:
    aggregate = _load(root / "M1_V2_RENDERER_CANARY.json")
    if aggregate.get("status") != "PASS" or aggregate.get("gpu_ids") != list(GPU_IDS):
        raise V2Error("V2_CANARY_NOT_PASS")
    preflight = _load(root / "M1_V2_1_GPU_PREFLIGHT_PRE_CANARY.json")
    for gpu in GPU_IDS:
        canary = _load(root / "renderer_canary" / f"gpu_{gpu:02d}.json")
        validate_binding_receipt(canary, gpu, require_canonical_uuid=True)
        validate_uuid_binding(
            gpu=gpu,
            preflight_uuid=preflight.get("uuid_by_gpu", {}).get(str(gpu)),
            canary_uuid=canary.get("gpu_uuid_canonical"),
            phase="PRE_CANARY",
        )
        if aggregate.get("uuid_by_gpu", {}).get(str(gpu)) != canary.get("gpu_uuid_canonical"):
            raise V2Error(f"HOLD_GPU_UUID_BINDING_MISMATCH:gpu_{gpu:02d}:CANARY_AGGREGATE")


def _verify_preflight_bindings(root: Path, run_set: str, manifest: Mapping[str, Any], protocol_sha256: str, graphics_contract: Mapping[str, Any]) -> None:
    gates = ["PRE_CANARY"] + [f"PRE_{run_set.upper()}_{label}" for label in LABELS]
    for gate in gates:
        value = _load(root / f"M1_V2_1_GPU_PREFLIGHT_{gate}.json")
        if value.get("status") != "PASS" or value.get("all_8_safe") is not True:
            raise V2Error(f"V2_PREFLIGHT_NOT_PASS:{gate}")
        expected_run_set = "canary" if gate == "PRE_CANARY" else run_set
        if value.get("gate") != gate or value.get("run_set") != expected_run_set or value.get("protocol_sha256") != protocol_sha256 or value.get("source_commit") != manifest.get("source_commit") or value.get("source_tree") != manifest.get("source_tree") or value.get("graphics_contract") != dict(graphics_contract):
            raise V2Error(f"V2_PREFLIGHT_CONTEXT_MISMATCH:{gate}")
        if value.get("unmapped_processes") or value.get("foreign_user_workloads") or any(row.get("foreign_processes") for row in value.get("gpu_rows", [])):
            raise V2Error(f"V2_PREFLIGHT_FOREIGN_WORKLOAD_PRESENT:{gate}")
        if any(process.get("classification") != "SYSTEM_GRAPHICS_BASELINE" for process in value.get("baseline_system_graphics", [])):
            raise V2Error(f"V2_PREFLIGHT_GRAPHICS_BASELINE_INVALID:{gate}")
        rows = {int(row.get("index")): row for row in value.get("gpu_rows", [])}
        for gpu in GPU_IDS:
            row = rows.get(gpu)
            if row is None or value.get("uuid_by_gpu", {}).get(str(gpu)) != canonical_gpu_uuid(row.get("uuid")):
                raise V2Error(f"HOLD_GPU_UUID_BINDING_MISMATCH:gpu_{gpu:02d}:{gate}")
        receipt_path = value.get("phase_receipt_path")
        receipt_sha = value.get("phase_receipt_sha256")
        if not receipt_path or not receipt_sha:
            raise V2Error(f"V2_PREFLIGHT_NOT_BOUND:{gate}")
        receipt = root / str(receipt_path)
        if not receipt.is_file() or sha256_file(receipt) != receipt_sha:
            raise V2Error(f"V2_PREFLIGHT_RECEIPT_SHA_MISMATCH:{gate}")


def _seal(root: Path, name: str) -> None:
    excluded = {name, f"{name}.sha256"}
    files = [path for path in root.rglob("*") if path.is_file() and path.name not in excluded and not path.name.startswith(".")]
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in sorted(files)]
    target = root / name
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / f"{name}.sha256").write_text(f"{sha256_file(target)}  {name}\n", encoding="utf-8")


def audit_root(root: Path, *, final: bool) -> dict[str, Any]:
    protocol_path = REPO_ROOT / "configs/stage_v_m1_visual_determinism_protocol_v2_1_1_8gpu.json"
    protocol = validate_protocol(protocol_path)
    manifest = _load(root / "M1_V2_MANIFEST.json")
    if manifest.get("schema") != "STAGE_V_M1_V2_1_1_8GPU_MANIFEST_V1":
        raise V2Error("V2_MANIFEST_SCHEMA_INVALID")
    if manifest.get("status") != "PREPARED_NO_RUNTIME_STARTED":
        raise V2Error("V2_ROOT_ALREADY_CONSUMED")
    if manifest.get("diagnostic_identity") != IDENTITY:
        raise V2Error("V2_IDENTITY_MISMATCH")
    validate_manifest_authorization(manifest)
    if manifest.get("source_commit") != _git("rev-parse", "HEAD") or manifest.get("source_tree") != _git("rev-parse", "HEAD^{tree}"):
        raise V2Error("V2_SOURCE_BINDING_MISMATCH")
    if _git("status", "--porcelain"):
        raise V2Error("V2_AUDITOR_WORKTREE_DIRTY")
    if manifest.get("protocol_sha256") != sha256_file(protocol_path) or manifest.get("protocol_schema") != protocol["schema"]:
        raise V2Error("V2_PROTOCOL_SHA256_MISMATCH")
    _verify_canary(root)
    protocol_sha256 = sha256_file(protocol_path)
    _verify_preflight_bindings(root, "r1", manifest, protocol_sha256, protocol["system_graphics_baseline"])
    _verify_runs(root, "runs", manifest)
    local = _load(root / "M1_V2_R1_GPU_LOCAL_PAIR_MATRIX.json")
    cross = _load(root / "M1_V2_R1_CROSS_GPU_PAIR_MATRIX.json")
    aggregate = _load(root / "M1_V2_R1_AGGREGATE_REPORT.json")
    classification_receipt = _load(root / "M1_V2_CLASSIFICATION_RECEIPT.json")
    local_count = sum(len(value["pairs"]) for value in local["gpus"].values())
    cross_count = sum(len(value) for value in cross["labels"].values())
    if local_count != 32 or cross_count != 112:
        raise V2Error(f"V2_PAIR_COUNT_MISMATCH:{local_count}:{cross_count}")
    classification, profile = _independent_classification(local, cross)
    if aggregate.get("classification") != classification or aggregate.get("evidence_profile") != profile:
        raise V2Error("V2_AUDITOR_CLASSIFICATION_DISAGREEMENT")
    if classification_receipt.get("classification") != classification or classification_receipt.get("evidence_profile") != profile:
        raise V2Error("V2_CLASSIFICATION_RECEIPT_DISAGREEMENT")
    receipt = {
        "schema": "STAGE_V_M1_V2_1_INDEPENDENT_AUDIT_V1",
        "verdict": "PASS",
        "final": final,
        "source_commit": manifest.get("source_commit"),
        "source_tree": manifest.get("source_tree"),
        "protocol_sha256": sha256_file(protocol_path),
        "r1_run_count": 32,
        "gpu_local_pair_count": 32,
        "cross_gpu_pair_count": 112,
        "classification": classification,
        "evidence_profile": profile,
        "protected_boundaries": {field: 0 for field in BOUNDARIES},
    }
    if final:
        _verify_preflight_bindings(root, "r2", manifest, protocol_sha256, protocol["system_graphics_baseline"])
        _verify_runs(root, "raw_runs", manifest)
        plan_path = root / "M1_V2_RAW_CAPTURE_PLAN.json"
        if not plan_path.is_file():
            raise V2Error("V2_RAW_CAPTURE_PLAN_MISSING")
        if _load(plan_path).get("schema") != "STAGE_V_M1_V2_1_1_RAW_CAPTURE_PLAN_V1":
            raise V2Error("V2_1_RAW_CAPTURE_PLAN_SCHEMA_INVALID")
        producer = _load(root / "M1_V2_PRODUCER_ANALYSIS.json")
        if producer.get("status") != "PASS_PENDING_INDEPENDENT_AUDIT":
            raise V2Error("V2_PRODUCER_COMPLETION_OWNERSHIP_INVALID")
        if producer.get("classification") != classification or producer.get("evidence_profile") != profile:
            raise V2Error("V2_PRODUCER_ANALYSIS_DISAGREEMENT")
        receipt["raw_capture_plan_sha256"] = sha256_file(plan_path)
        _write(root / "M1_V2_COMPLETE.json", {"schema": "STAGE_V_M1_V2_COMPLETE_V1", "status": "PASS_CLASSIFIED", "completion_owner": "INDEPENDENT_AUDITOR", "classification": classification, "evidence_profile": profile, "rb1a_status": "HOLD"})
    status = _load(root / "M1_V2_STATUS.json")
    status.update({"status": "PASS_CLASSIFIED" if final else "R1_AUDITED", "classification": classification, "evidence_profile": profile, "independent_audit": "PASS", "protected_boundaries": {field: 0 for field in BOUNDARIES}})
    _write(root / "M1_V2_STATUS.json", status)
    _write(root / "M1_V2_INDEPENDENT_AUDIT.json", receipt)
    _seal(root, "M1_V2_SHA256SUMS_FINAL" if final else "M1_V2_SHA256SUMS_R1")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = audit_root(args.root.resolve(), final=args.final)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.SubprocessError, V2Error) as exc:
        print(json.dumps({"verdict": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"verdict": result["verdict"], "classification": result["classification"], "final": result["final"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
