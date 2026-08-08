#!/usr/bin/env python3
"""CPU-only local/cross-GPU M1-V2 pair analysis and raw forensic helpers."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from .analyze_stage_v_m1_visual_divergence import _load, _sha256, analyze_pair
except ImportError:  # direct script execution
    from analyze_stage_v_m1_visual_divergence import _load, _sha256, analyze_pair


GPU_IDS = tuple(range(8))
LABELS = ("Q1", "C1", "Q2", "C2")
CLASSIFICATIONS = {
    "RAW_OBSERVATION_NON_POLICY_DIFFERENCE",
    "SAME_MODE_RENDER_OR_OBSERVATION_NONDETERMINISM",
    "MODE_PATH_SPECIFIC_VISUAL_DIVERGENCE",
    "GPU_CONTEXT_DEPENDENT_VISUAL_DIVERGENCE",
    "PROCESSOR_OR_MODEL_INPUT_NONDETERMINISM",
    "SIMULATOR_RUNTIME_NONDETERMINISM",
    "POLICY_VISUAL_INPUT_NONDETERMINISM_ACTION_STABLE",
    "HETEROGENEOUS_MULTI_GPU_DIVERGENCE",
    "MULTI_LAYER_NONDETERMINISM",
    "UNCLASSIFIED",
}


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _run(root: Path, gpu: int, label: str, run_set: str = "runs") -> Path:
    return root / run_set / f"gpu_{gpu:02d}" / label


def _pair_exact(pair: Mapping[str, Any]) -> bool:
    return bool(pair.get("initial_state_exact") and pair.get("terminal_step_exact") and pair.get("terminal_outcome_exact") and all(item.get("equal") for item in pair.get("traces", {}).values()))


def _visual_diff(pair: Mapping[str, Any]) -> bool:
    traces = pair.get("traces", {})
    return traces.get("policy_rgb", {}).get("equal") is False or traces.get("model_input", {}).get("equal") is False


def _full_sim_exact(pair: Mapping[str, Any]) -> bool:
    return pair.get("traces", {}).get("full_sim_state", {}).get("equal") is True


def local_pairs(root: Path, identity: str, run_set: str = "runs") -> dict[str, Any]:
    result: dict[str, Any] = {f"gpu_{gpu:02d}": {} for gpu in GPU_IDS}
    for gpu in GPU_IDS:
        runs = {label: _run(root, gpu, label, run_set) for label in LABELS}
        result[f"gpu_{gpu:02d}"] = {
            "gpu_id": gpu,
            "identity": identity,
            "pairs": {
                f"SAME_MODE_Q_GPU{gpu}": analyze_pair(runs["Q1"], runs["Q2"], f"SAME_MODE_Q_GPU{gpu}"),
                f"SAME_MODE_C_GPU{gpu}": analyze_pair(runs["C1"], runs["C2"], f"SAME_MODE_C_GPU{gpu}"),
                f"CROSS_MODE_R1_GPU{gpu}": analyze_pair(runs["Q1"], runs["C1"], f"CROSS_MODE_R1_GPU{gpu}"),
                f"CROSS_MODE_R2_GPU{gpu}": analyze_pair(runs["Q2"], runs["C2"], f"CROSS_MODE_R2_GPU{gpu}"),
            },
        }
    return {"schema": "STAGE_V_M1_V2_GPU_LOCAL_PAIR_MATRIX_V1", "identity": identity, "gpu_count": 8, "pair_count": 32, "gpus": result}


def cross_gpu_pairs(root: Path, identity: str, run_set: str = "runs") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in LABELS:
        pairs: dict[str, Any] = {}
        for left, right in itertools.combinations(GPU_IDS, 2):
            name = f"CROSS_GPU_{label}_GPU{left}_GPU{right}"
            pairs[name] = analyze_pair(_run(root, left, label, run_set), _run(root, right, label, run_set), name)
        result[label] = pairs
    return {"schema": "STAGE_V_M1_V2_CROSS_GPU_PAIR_MATRIX_V1", "identity": identity, "gpu_count": 8, "pair_count": 112, "labels": result}


def _all_local_pairs(local: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [pair for gpu in local["gpus"].values() for pair in gpu["pairs"].values()]


def classify_v2(local: Mapping[str, Any], cross: Mapping[str, Any]) -> str:
    local_all = _all_local_pairs(local)
    if any(not _full_sim_exact(pair) for pair in local_all):
        return "SIMULATOR_RUNTIME_NONDETERMINISM"
    same_q = [local["gpus"][f"gpu_{gpu:02d}"]["pairs"][f"SAME_MODE_Q_GPU{gpu}"] for gpu in GPU_IDS]
    same_c = [local["gpus"][f"gpu_{gpu:02d}"]["pairs"][f"SAME_MODE_C_GPU{gpu}"] for gpu in GPU_IDS]
    cross_mode = [local["gpus"][f"gpu_{gpu:02d}"]["pairs"][name] for gpu in GPU_IDS for name in (f"CROSS_MODE_R1_GPU{gpu}", f"CROSS_MODE_R2_GPU{gpu}")]
    same_mode_pairs = same_q + same_c
    if any(not pair["traces"]["raw_observation"]["equal"] and pair["traces"]["policy_rgb"]["equal"] and pair["traces"]["model_input"]["equal"] for pair in same_mode_pairs):
        return "RAW_OBSERVATION_NON_POLICY_DIFFERENCE"
    if any(pair["traces"]["raw_observation"]["equal"] and pair["traces"]["policy_rgb"]["equal"] and not pair["traces"]["model_input"]["equal"] for pair in same_mode_pairs):
        return "PROCESSOR_OR_MODEL_INPUT_NONDETERMINISM"
    if any((not _pair_exact(pair)) and _visual_diff(pair) and pair.get("traces", {}).get("token", {}).get("equal") and pair.get("traces", {}).get("postprocessed_action", {}).get("equal") for pair in same_q + same_c):
        return "POLICY_VISUAL_INPUT_NONDETERMINISM_ACTION_STABLE"
    same_mode_exact = all(_pair_exact(pair) for pair in same_q + same_c)
    cross_gpu_visual = [pair for pairs in cross["labels"].values() for pair in pairs.values() if _visual_diff(pair) and _full_sim_exact(pair)]
    if same_mode_exact and cross_gpu_visual:
        return "GPU_CONTEXT_DEPENDENT_VISUAL_DIVERGENCE"
    same_mode_mismatch_gpus = {gpu for gpu in GPU_IDS if any(_visual_diff(pair) for pair in (same_q[gpu], same_c[gpu]))}
    mode_mismatch_gpus = {gpu for gpu in GPU_IDS if any(not _pair_exact(pair) for pair in (cross_mode[gpu * 2], cross_mode[gpu * 2 + 1]))}
    if same_mode_mismatch_gpus and mode_mismatch_gpus:
        return "HETEROGENEOUS_MULTI_GPU_DIVERGENCE"
    if not same_mode_exact and any(_visual_diff(pair) for pair in same_q + same_c):
        return "SAME_MODE_RENDER_OR_OBSERVATION_NONDETERMINISM"
    mode_diff_count = sum(not _pair_exact(pair) for pair in cross_mode)
    if same_mode_exact and mode_diff_count >= 8:
        return "MODE_PATH_SPECIFIC_VISUAL_DIVERGENCE"
    if any(not _pair_exact(pair) for pair in local_all) or cross_gpu_visual:
        return "HETEROGENEOUS_MULTI_GPU_DIVERGENCE"
    return "UNCLASSIFIED"


def _first_steps(pairs: Mapping[str, Mapping[str, Any]]) -> dict[str, int | None]:
    values: list[int] = []
    for pair in pairs.values():
        first = pair.get("first_mismatch_by_component", {})
        for field in ("policy_rgb", "pixel_values"):
            if first.get(field) is not None:
                values.append(int(first[field]))
    return {"t_star": min(values) if values else None}


def make_r2_plan(root: Path, identity: str, local: Mapping[str, Any], cross: Mapping[str, Any], local_sha: str, cross_sha: str) -> dict[str, Any]:
    local_t: dict[str, int | None] = {}
    for gpu in GPU_IDS:
        local_t[f"gpu_{gpu:02d}"] = _first_steps(local["gpus"][f"gpu_{gpu:02d}"]["pairs"])["t_star"]
    cross_t = {label: _first_steps(pairs)["t_star"] for label, pairs in cross["labels"].items()}
    all_steps = [value for value in local_t.values() if value is not None] + [value for value in cross_t.values() if value is not None]
    global_t = min(all_steps) if all_steps else None

    def n2(value: int | None) -> set[int]:
        return set() if value is None else set(range(max(0, value - 2), value + 3))

    by_gpu = {str(gpu): sorted({0, *n2(global_t), *n2(local_t[f"gpu_{gpu:02d}"])}) for gpu in GPU_IDS}
    return {
        "schema": "STAGE_V_M1_V2_RAW_CAPTURE_PLAN_V1",
        "status": "FROZEN_BEFORE_RAW_CAPTURE_RUN",
        "identity": identity,
        "global_t_star": global_t,
        "local_t_star": local_t,
        "cross_gpu_t_star": cross_t,
        "capture_steps_by_gpu": by_gpu,
        "capture_steps_union": sorted({step for values in by_gpu.values() for step in values}),
        "generation_rule": "step_0_union_N2_global_t_star_union_N2_local_t_star_clamped_at_zero",
        "source_gpu_local_pair_matrix_sha256": local_sha,
        "source_cross_gpu_pair_matrix_sha256": cross_sha,
    }


def _raw_audit(run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "M1_RAW_CAPTURE_MANIFEST.json"
    if not manifest_path.is_file():
        return {"verdict": "FAIL", "reason": "RAW_MANIFEST_MISSING"}
    manifest = _load(manifest_path)
    bad: list[str] = []
    for entry in manifest.get("entries", []):
        binary = run_root / "trace/raw_capture" / str(entry["binary_path"])
        descriptor = run_root / "trace/raw_capture" / str(entry["descriptor_path"])
        descriptor_value = _load(descriptor) if descriptor.is_file() else {}
        if (not binary.is_file() or not descriptor.is_file() or _sha256(binary) != entry.get("raw_sha256")
                or descriptor_value.get("raw_sha256") != entry.get("raw_sha256")
                or descriptor_value.get("byte_length") != binary.stat().st_size):
            bad.append(str(entry.get("binary_path")))
    return {"verdict": "PASS" if not bad else "FAIL", "entry_count": len(manifest.get("entries", [])), "bad_entries": bad}


def numeric_forensics(root: Path, local: Mapping[str, Any], cross: Mapping[str, Any]) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    for gpu in GPU_IDS:
        for name, pair in local["gpus"][f"gpu_{gpu:02d}"]["pairs"].items():
            pairs[name] = pair.get("numeric_forensic")
    for label, label_pairs in cross["labels"].items():
        for name, pair in label_pairs.items():
            pairs[name] = pair.get("numeric_forensic")
    return {"schema": "STAGE_V_M1_V2_NUMERIC_VISUAL_FORENSIC_V1", "pair_count": len(pairs), "pairs": pairs}


def hash_clusters(root: Path) -> dict[str, Any]:
    clusters: dict[str, dict[str, list[str]]] = {}
    for gpu in GPU_IDS:
        for label in LABELS:
            manifest_path = _run(root, gpu, label, "raw_runs") / "M1_RAW_CAPTURE_MANIFEST.json"
            if not manifest_path.is_file():
                continue
            for entry in _load(manifest_path).get("entries", []):
                key = f"{entry.get('step')}:{entry.get('group')}:{entry.get('field')}"
                clusters.setdefault(key, {}).setdefault(str(entry.get("raw_sha256")), []).append(f"gpu_{gpu:02d}/{label}")
    return {"schema": "STAGE_V_M1_V2_VISUAL_HASH_CLUSTER_REPORT_V1", "fields": {key: {"unique_hash_count": len(values), "clusters": values} for key, values in clusters.items()}}


def analyze_root(root: Path, *, final: bool = False) -> dict[str, Any]:
    manifest = _load(root / "M1_V2_MANIFEST.json")
    identity = str(manifest["diagnostic_identity"])
    local = local_pairs(root, identity)
    cross = cross_gpu_pairs(root, identity)
    local_path = root / "M1_V2_R1_GPU_LOCAL_PAIR_MATRIX.json"
    cross_path = root / "M1_V2_R1_CROSS_GPU_PAIR_MATRIX.json"
    _write(local_path, local)
    _write(cross_path, cross)
    classification = classify_v2(local, cross)
    aggregate = {
        "schema": "STAGE_V_M1_V2_R1_AGGREGATE_REPORT_V1", "identity": identity,
        "gpu_local_pair_count": 32, "cross_gpu_pair_count": 112,
        "classification": classification,
        "same_mode_q_visual_mismatch_gpus": sum(not _pair_exact(local["gpus"][f"gpu_{gpu:02d}"]["pairs"][f"SAME_MODE_Q_GPU{gpu}"]) for gpu in GPU_IDS),
        "same_mode_c_visual_mismatch_gpus": sum(not _pair_exact(local["gpus"][f"gpu_{gpu:02d}"]["pairs"][f"SAME_MODE_C_GPU{gpu}"]) for gpu in GPU_IDS),
        "full_sim_exact_pairs": sum(_full_sim_exact(pair) for pair in _all_local_pairs(local)),
        "token_action_exact_pairs": sum(pair.get("traces", {}).get("token", {}).get("equal") and pair.get("traces", {}).get("postprocessed_action", {}).get("equal") for pair in _all_local_pairs(local)),
    }
    aggregate_path = root / "M1_V2_R1_AGGREGATE_REPORT.json"
    _write(aggregate_path, aggregate)
    (root / "M1_V2_R1_AGGREGATE_REPORT.md").write_text("# M1-V2 R1 aggregate report\n\n" + "\n".join(f"- {key}: {value}" for key, value in aggregate.items() if key != "schema") + "\n", encoding="utf-8")
    plan = make_r2_plan(root, identity, local, cross, _sha256(local_path), _sha256(cross_path))
    _write(root / "M1_V2_RAW_CAPTURE_PLAN.json", plan)
    _write(root / "M1_V2_CLASSIFICATION_RECEIPT.json", {"schema": "STAGE_V_M1_V2_CLASSIFICATION_RECEIPT_V1", "status": "PASS_CLASSIFIED" if final else "PENDING_R2_RAW_CAPTURE", "classification": classification, "rb1a_status": "HOLD", "identity": identity, "source_commit": manifest.get("source_commit"), "source_tree": manifest.get("source_tree")})
    if final:
        raw_audits = {f"gpu_{gpu:02d}/{label}": _raw_audit(_run(root, gpu, label, "raw_runs")) for gpu in GPU_IDS for label in LABELS}
        if any(value["verdict"] != "PASS" for value in raw_audits.values()):
            raise ValueError("M1_V2_RAW_AUDIT_FAIL")
        raw_local = local_pairs(root, identity, "raw_runs")
        raw_cross = cross_gpu_pairs(root, identity, "raw_runs")
        numeric = numeric_forensics(root, raw_local, raw_cross)
        clusters = hash_clusters(root)
        _write(root / "M1_V2_NUMERIC_VISUAL_FORENSIC.json", numeric)
        _write(root / "M1_V2_VISUAL_HASH_CLUSTER_REPORT.json", clusters)
        _write(root / "M1_V2_INDEPENDENT_AUDIT.json", {"schema": "STAGE_V_M1_V2_INDEPENDENT_AUDIT_V1", "verdict": "PASS", "r1_run_count": 32, "gpu_local_pair_count": 32, "cross_gpu_pair_count": 112, "raw_audits": raw_audits, "classification": classification})
        _write(root / "M1_V2_COMPLETE.json", {"schema": "STAGE_V_M1_V2_COMPLETE_V1", "status": "PASS_CLASSIFIED", "classification": classification, "rb1a_status": "HOLD"})
    return {"classification": classification, "local": local, "cross": cross, "plan": plan, "aggregate": aggregate}


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = analyze_root(args.root.resolve(), final=args.final)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"verdict": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"verdict": "PASS", "classification": result["classification"], "gpu_local_pair_count": 32, "cross_gpu_pair_count": 112}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
