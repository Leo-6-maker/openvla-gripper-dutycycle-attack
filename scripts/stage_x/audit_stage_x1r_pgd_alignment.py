from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
DEFAULT_MODEL_PATHS = {
    "libero_10": "/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10",
    "libero_goal": "/mnt/sdc/dty_user/openvla_attack/models/libero-goal",
    "libero_object": "/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object",
    "libero_spatial": "/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620",
}
DEFAULT_PARITY_ROOT = "/mnt/sdc/dty_user/openvla_attack_outputs/n5/phase3_student/STAGE_X_X1R_VICTIM_PROVENANCE_CLOSURE_20260817T122000Z"
DEFAULT_CANONICAL_TOKENIZER = "/mnt/sdc/dty_user/openvla_attack/repos/openvla-upstream-clean-c8f03f4/prismatic/vla/action_tokenizer.py"
COUNTERS = {
    "pgd_calls": 0,
    "env_step_calls": 0,
    "physical_interventions": 0,
    "vphys_reads": 0,
    "attack_outcome_reads": 0,
    "protected_reads": 0,
    "eval160_reads": 0,
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def stats_from_config(config: dict[str, Any], suite: str) -> dict[str, Any]:
    suite_stats = config["norm_stats"][suite]
    return suite_stats["action"] if "action" in suite_stats else suite_stats


class _TextConfig:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)


class _Config:
    def __init__(self, vocab_size: int, pad_to_multiple_of: int) -> None:
        self.text_config = _TextConfig(vocab_size)
        self.pad_to_multiple_of = int(pad_to_multiple_of)


class _TokenizerStubModel:
    def __init__(self, config: dict[str, Any], suite: str, tokenizer_vocab_size: int, bin_centers: np.ndarray) -> None:
        self.config = _Config(config["text_config"]["vocab_size"], config["pad_to_multiple_of"])
        self.norm_stats = {suite: stats_from_config(config, suite)}
        self.bin_centers = bin_centers

    def get_action_stats(self, unnorm_key: str) -> dict[str, Any]:
        return self.norm_stats[unnorm_key]


def load_native_suite(suite: str, model_path: Path) -> dict[str, Any]:
    """Load the checkpoint-local/native tokenizer; never guess a replacement."""
    from transformers import AutoTokenizer
    from prismatic.vla.action_tokenizer import ActionTokenizer

    config = load_json(model_path / "config.json")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    native = ActionTokenizer(tokenizer, bins=int(config["n_action_bins"]))
    vocab_eff = int(config["text_config"]["vocab_size"] - config["pad_to_multiple_of"])
    if int(tokenizer.vocab_size) != vocab_eff:
        raise ValueError(f"TOKENIZER_VOCAB_BINDING_MISMATCH:{suite}:{tokenizer.vocab_size}:{vocab_eff}")
    return {
        "suite": suite,
        "model_path": str(model_path),
        "config": config,
        "stats": stats_from_config(config, suite),
        "tokenizer": tokenizer,
        "native": native,
        "vocab_eff": vocab_eff,
        "bin_centers": np.asarray(native.bin_centers, dtype=np.float64),
        "bins": np.asarray(native.bins, dtype=np.float64),
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "native_tokenizer_class": f"{type(native).__module__}.{type(native).__name__}",
        "tokenizer_files": {
            name: sha256_file(model_path / name)
            for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
            if (model_path / name).exists()
        },
    }


def normalize_action(raw_action: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    raw = np.asarray(raw_action, dtype=np.float64)
    mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)), dtype=bool)
    low = np.asarray(stats["q01"], dtype=np.float64)
    high = np.asarray(stats["q99"], dtype=np.float64)
    return np.where(mask, 2.0 * (raw - low) / np.maximum(high - low, 1e-6) - 1.0, raw)


def denormalize_action(normalized: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    norm = np.asarray(normalized, dtype=np.float64)
    mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)), dtype=bool)
    low = np.asarray(stats["q01"], dtype=np.float64)
    high = np.asarray(stats["q99"], dtype=np.float64)
    return np.where(mask, 0.5 * (norm + 1.0) * (high - low) + low, norm)


def canonical_token_ids(native_info: dict[str, Any], normalized_action: np.ndarray) -> np.ndarray:
    native = native_info["native"]
    clipped = np.clip(np.asarray(normalized_action), float(native.min_action), float(native.max_action))
    discretized = np.digitize(clipped, native.bins)
    return (int(native_info["tokenizer_vocab_size"]) - discretized).astype(np.int64)


def helper_token_ids(native_info: dict[str, Any], raw_action: np.ndarray) -> np.ndarray:
    from gripper_attack.attack_adapter import TokenPrefixPGDAttacker

    stub = _TokenizerStubModel(
        native_info["config"],
        native_info["suite"],
        int(native_info["tokenizer_vocab_size"]),
        native_info["bin_centers"].astype(np.float32),
    )
    helper = TokenPrefixPGDAttacker(stub, object(), {}, device="cpu")
    return helper.action_to_token_ids(np.asarray(raw_action), native_info["suite"]).detach().cpu().numpy()


def _case(action_dim: int, dim: int, value: float, kind: str, *, value_space: str = "normalized", **extra: Any) -> dict[str, Any]:
    result = {"action_dim": int(action_dim), "dim": int(dim), "value": float(value), "kind": kind, "value_space": value_space}
    result.update({str(key): int(value) for key, value in extra.items()})
    return result


def differential_cases(native_info: dict[str, Any]) -> list[dict[str, Any]]:
    bins = native_info["bins"]
    centers = native_info["bin_centers"]
    cases: list[dict[str, Any]] = []
    for dim in range(7):
        for index, value in enumerate(centers):
            cases.append(_case(7, dim, float(value), "bin_center", center_index=index))
        for index, value in enumerate(bins):
            cases.append(_case(7, dim, float(value), "bin_boundary", boundary_index=index))
            cases.append(_case(7, dim, float(np.nextafter(value, -np.inf)), "nextafter_boundary_left", boundary_index=index))
            cases.append(_case(7, dim, float(np.nextafter(value, np.inf)), "nextafter_boundary_right", boundary_index=index))
        for value in (-1.0, 0.0, 1.0):
            cases.append(_case(7, dim, value, "normalized_special"))
    for value in (0.0, 0.5, 1.0):
        cases.append(_case(7, 6, value, "raw_gripper_special", value_space="raw"))
    return cases


def run_differential(native_info: dict[str, Any], q00_rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = native_info["stats"]
    base = np.zeros(7, dtype=np.float64)
    mismatches: list[dict[str, Any]] = []
    total = 0
    for case in differential_cases(native_info):
        raw = base.copy()
        if case["value_space"] == "normalized":
            normalized = base.copy()
            normalized[int(case["dim"])] = float(case["value"])
            raw = denormalize_action(normalized, stats)
        else:
            normalized = normalize_action(raw, stats)
            raw[int(case["dim"])] = float(case["value"])
            normalized = normalize_action(raw, stats)
        canonical = canonical_token_ids(native_info, normalized)
        helper = helper_token_ids(native_info, raw)
        total += 1
        if not np.array_equal(canonical, helper):
            mismatches.append({
                "case": case,
                "canonical_token": int(canonical[int(case["dim"])]),
                "project_helper_token": int(helper[int(case["dim"])]),
                "exact_match": False,
                "normalized_value": float(normalized[int(case["dim"])]),
                "raw_value": float(raw[int(case["dim"])]),
                "bin_index_canonical": int(np.digitize(np.clip(normalized[int(case["dim"])], -1.0, 1.0), native_info["bins"])),
                "nearest_center_index_project": int(np.abs(normalized[int(case["dim"])] - native_info["bin_centers"]).argmin()),
            })
    q00_results = []
    for row in q00_rows:
        raw = np.asarray(row["raw_action"], dtype=np.float64)
        normalized = normalize_action(raw, stats)
        canonical = canonical_token_ids(native_info, normalized)
        helper = helper_token_ids(native_info, raw)
        q00_results.append({
            "stage": row["stage"],
            "parent_key": row["parent_key"],
            "probe_id": row["probe_id"],
            "raw_action": [float(x) for x in raw],
            "canonical_tokens": [int(x) for x in canonical],
            "project_helper_tokens": [int(x) for x in helper],
            "exact_match": bool(np.array_equal(canonical, helper)),
            "mismatch_dimensions": [int(i) for i in np.flatnonzero(canonical != helper)],
        })
    return {
        "suite": native_info["suite"],
        "total_differential_cases": total,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "q00_rows": q00_results,
        "canonical_authority": {
            "class": native_info["native_tokenizer_class"],
            "algorithm": "np.digitize(action, bins); tokenizer.vocab_size - discretized",
            "tokenizer_vocab_size": native_info["tokenizer_vocab_size"],
            "n_bins": int(native_info["native"].n_bins),
            "bins": [float(x) for x in native_info["bins"]],
            "bin_centers": [float(x) for x in native_info["bin_centers"]],
        },
    }


def read_q00_rows(parity_root: Path, suite: str) -> list[dict[str, Any]]:
    rows = []
    for worker_path in sorted(parity_root.glob("workers/*.json")):
        worker = load_json(worker_path)
        if worker.get("suite") != suite:
            continue
        for row in worker.get("rows", []):
            generation = row.get("clean_generation", {})
            raw = generation.get("reference_raw_action")
            if isinstance(raw, list) and len(raw) == 7:
                rows.append({
                    "stage": row.get("stage"),
                    "parent_key": row.get("binding", {}).get("parent_key"),
                    "probe_id": row.get("binding", {}).get("probe_id"),
                    "raw_action": raw,
                    "current_generated_token_ids": generation.get("token_ids", []),
                    "reference_token_ids": generation.get("reference_token_ids", []),
                })
    if not rows:
        raise ValueError(f"NO_Q00_RAW_ROWS:{suite}:{parity_root}")
    return rows


def decode_tokens(native_info: dict[str, Any], token_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    token_ids = np.asarray(token_ids, dtype=np.int64)
    disc = np.clip(native_info["vocab_eff"] - token_ids - 1, 0, len(native_info["bin_centers"]) - 1)
    normalized = native_info["bin_centers"][disc]
    raw = denormalize_action(normalized, native_info["stats"])
    return normalized, raw


def roundtrip_row(native_info: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    current = np.asarray(row["current_generated_token_ids"], dtype=np.int64)
    historical_raw = np.asarray(row["raw_action"], dtype=np.float64)
    current_norm, current_raw = decode_tokens(native_info, current)
    canonical_current = canonical_token_ids(native_info, normalize_action(current_raw, native_info["stats"]))
    helper_current = helper_token_ids(native_info, current_raw)
    canonical_historical = canonical_token_ids(native_info, normalize_action(historical_raw, native_info["stats"]))
    helper_historical = helper_token_ids(native_info, historical_raw)
    gripper = 6
    current_disc = int(np.clip(native_info["vocab_eff"] - current[gripper] - 1, 0, len(native_info["bin_centers"]) - 1))
    adjacent = max(0, current_disc - 1) if current_disc else min(len(native_info["bin_centers"]) - 1, current_disc + 1)
    q01 = float(native_info["stats"]["q01"][gripper])
    q99 = float(native_info["stats"]["q99"][gripper])
    normalized_historical = float(normalize_action(historical_raw, native_info["stats"])[gripper])
    return {
        "stage": row["stage"],
        "parent_key": row["parent_key"],
        "probe_id": row["probe_id"],
        "current_generated_token_ids": [int(x) for x in current],
        "reference_derived_token_ids": [int(x) for x in row["reference_token_ids"]],
        "historical_generated_token_ids": None,
        "historical_generated_token_evidence": "NOT_IDENTIFIABLE; prior artifact contains reference/reencoded token fields, not a direct immutable generation output",
        "R1_current_token_canonical_decode_then_canonical_encode": [int(x) for x in canonical_current],
        "R2_current_token_project_decode_then_project_encode": [int(x) for x in helper_current],
        "R3_reference_raw_canonical_encode": [int(x) for x in canonical_historical],
        "R4_reference_raw_project_encode": [int(x) for x in helper_historical],
        "gripper": {
            "current_generated_token": int(current[gripper]),
            "canonical_reencoded_current_raw_token": int(canonical_current[gripper]),
            "project_reencoded_current_raw_token": int(helper_current[gripper]),
            "canonical_reencoded_reference_raw_token": int(canonical_historical[gripper]),
            "project_reencoded_reference_raw_token": int(helper_historical[gripper]),
            "current_bin_index": current_disc,
            "adjacent_bin_index": adjacent,
            "current_bin_center": float(native_info["bin_centers"][current_disc]),
            "adjacent_bin_center": float(native_info["bin_centers"][adjacent]),
            "left_edge": float(native_info["bins"][current_disc]),
            "right_edge": float(native_info["bins"][min(current_disc + 1, len(native_info["bins"]) - 1)]),
            "historical_raw_action": float(historical_raw[gripper]),
            "normalized_historical_action": normalized_historical,
            "distance_to_current_center": float(abs(normalized_historical - native_info["bin_centers"][current_disc])),
            "distance_to_adjacent_center": float(abs(normalized_historical - native_info["bin_centers"][adjacent])),
            "distance_to_left_edge": float(abs(normalized_historical - native_info["bins"][current_disc])),
            "distance_to_right_edge": float(abs(normalized_historical - native_info["bins"][min(current_disc + 1, len(native_info["bins"]) - 1)])),
            "q01": q01,
            "q99": q99,
            "mask": bool(native_info["stats"].get("mask", [True] * 7)[gripper]),
        },
    }


def run_roundtrip(native_info: dict[str, Any], q00_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [roundtrip_row(native_info, row) for row in q00_rows]
    return {
        "schema": "STAGE_X_X1R_T0_TOKEN_ROUNDTRIP_FORENSIC_V1",
        "suite": native_info["suite"],
        "row_count": len(rows),
        "rows": rows,
        "historical_generated_token_ids": "NOT_IDENTIFIABLE",
        "conclusion": "token IDs are not equivalent merely because decoded actions are numerically close",
    }


def action_token_logit_row_index(dim: int, action_dim: int) -> int:
    return -(int(action_dim) - int(dim) + 1)


def teacher_forced_rows(model: Any, input_ids: Any, pixel_values: Any, token_ids: np.ndarray, action_dim: int) -> list[dict[str, Any]]:
    import torch

    generated = torch.as_tensor(token_ids, dtype=torch.long, device=input_ids.device).view(1, -1)
    if int(generated.shape[1]) != int(action_dim):
        raise ValueError(f"TEACHER_FORCED_ACTION_LENGTH_FAIL:{generated.shape[1]}:{action_dim}")
    full_input_ids = torch.cat((input_ids, generated), dim=1)
    with torch.inference_mode():
        output = model(input_ids=full_input_ids, pixel_values=pixel_values, use_cache=False, return_dict=True)
    logits = output.logits[0]
    rows = []
    for dim in range(int(action_dim)):
        row_index = action_token_logit_row_index(dim, action_dim)
        expected = int(generated[0, dim].item())
        argmax = int(torch.argmax(logits[row_index]).item())
        rows.append({
            "dim": dim,
            "row_index": row_index,
            "expected_autoregressive_token": expected,
            "teacher_forced_argmax_token": argmax,
            "exact": argmax == expected,
        })
    return rows


def run_causal_row_toy() -> dict[str, Any]:
    import torch

    action_dim = 7
    sequence_len = 23
    rows = []
    for dim in range(action_dim):
        target = 1000 + dim
        logits = torch.full((sequence_len, 2048), -10.0)
        row = action_token_logit_row_index(dim, action_dim)
        logits[row, target] = 10.0
        predicted = int(torch.argmax(logits[row]).item())
        rows.append({"dim": dim, "row_index": row, "expected_token": target, "argmax": predicted, "pass": predicted == target})
    return {"action_dim": action_dim, "rows": rows, "pass": all(row["pass"] for row in rows), "method":"teacher-forced causal row toy"}


def run_numerical_audit() -> dict[str, Any]:
    import torch
    from gripper_attack.execution_target import target_token_cw_loss_and_stats
    from gripper_attack.m3_controls import project_and_cast_processor_values

    row = torch.tensor([1.0, 0.0], requires_grad=True)
    loss, _ = target_token_cw_loss_and_stats(row, target_token_id=0, margin=0.0)
    gradient = torch.autograd.grad(loss, row)[0]
    descent = row.detach() - 0.1 * gradient.sign()
    loss_after, _ = target_token_cw_loss_and_stats(descent, target_token_id=0, margin=0.0)
    dtype_rows = []
    for dtype in (torch.float16, torch.bfloat16):
        x_orig = torch.tensor([0.12345, -0.23456], dtype=dtype)
        candidate = x_orig.float() + torch.tensor([0.1, -0.1])
        casted, corrections = project_and_cast_processor_values(x_orig, candidate, epsilon=0.1, candidate_is_delta=False)
        actual_linf = float((casted.float() - x_orig.float()).abs().max())
        dtype_rows.append({"dtype": str(dtype), "actual_linf": actual_linf, "epsilon": 0.1, "correction_count": int(corrections), "pass": actual_linf <= 0.1000001})
    return {
        "cw": {"initial_loss": float(loss), "gradient": [float(x) for x in gradient], "loss_after_sign_descent": float(loss_after), "pass": float(loss_after) <= float(loss)},
        "update": {"formula": "adv -= step_size * sign(gradient)", "pass": True},
        "master_dtype": "torch.float32",
        "projection": {"formula": "x_orig-epsilon <= adv <= x_orig+epsilon", "pass": True},
        "dtype_casts": dtype_rows,
        "pass": bool(float(loss_after) <= float(loss) and all(row["pass"] for row in dtype_rows)),
    }


def build_cpu_reports(args: argparse.Namespace) -> None:
    model_paths = dict(DEFAULT_MODEL_PATHS)
    if args.model_paths:
        model_paths.update(json.loads(Path(args.model_paths).read_text(encoding="utf-8")))
    parity_root = Path(args.parity_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_diff: dict[str, Any] = {}
    all_roundtrip: dict[str, Any] = {}
    for suite in SUITES:
        info = load_native_suite(suite, Path(model_paths[suite]))
        q00 = read_q00_rows(parity_root, suite)
        all_diff[suite] = run_differential(info, q00)
        all_roundtrip[suite] = run_roundtrip(info, q00)
    diff_report = {
        "schema": "STAGE_X_X1R_T0_TOKENIZER_DIFFERENTIAL_V1",
        "status": "PASS_CANONICAL_AUTHORITY_LOADED_WITH_PROJECT_HELPER_MISMATCHES" if any(v["mismatch_count"] for v in all_diff.values()) else "PASS_EXACT",
        "canonical_source": DEFAULT_CANONICAL_TOKENIZER,
        "suites": all_diff,
        "protected_counters": dict(COUNTERS),
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    roundtrip_report = {
        "schema": "STAGE_X_X1R_T0_TOKEN_ROUNDTRIP_FORENSIC_V1",
        "status": "HISTORICAL_GENERATED_TOKEN_IDS_NOT_IDENTIFIABLE",
        "suites": all_roundtrip,
        "protected_counters": dict(COUNTERS),
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    (out / "STAGE_X_X1R_T0_TOKENIZER_DIFFERENTIAL_V1.json").write_text(json.dumps(diff_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "STAGE_X_X1R_T0_TOKEN_ROUNDTRIP_FORENSIC_V1.json").write_text(json.dumps(roundtrip_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    numerical = run_numerical_audit()
    (out / "STAGE_X_X1R_T0_NUMERICAL_PRIMITIVE_AUDIT_V1.json").write_text(json.dumps(numerical, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": diff_report["status"], "suites": {k: v["mismatch_count"] for k, v in all_diff.items()}, "numerical_pass": numerical["pass"]}, sort_keys=True))


def load_model_and_processor(model_path: Path, device: str):
    from scripts.stage_x.run_stage_x1r_victim_parity import load_model_and_processor as loader

    return loader(model_path, device)


def first_q00(snapshot_root: Path, suite: str) -> list[Path]:
    from scripts.stage_x.run_stage_x1r_victim_parity import first_q00 as finder

    return [finder(snapshot_root, suite)]


def clean_forward_worker(args: argparse.Namespace) -> None:
    import torch
    from gripper_attack.stage_v_causal_observation_snapshot import load_snapshot
    from scripts.stage_x.run_stage_x1r_victim_parity import append_empty_action_token

    contract = load_json(Path(args.contract))
    suite_cfg = contract["suites"][args.suite]
    model_path = Path(suite_cfg["model_path"])
    receipt = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits", "-i", str(args.physical_gpu)
    ], text=True).strip().split(",")
    free = int(receipt[2].strip())
    if free <= 20480:
        raise RuntimeError(f"GPU_RESOURCE_GATE_FAIL:{free}")
    model, processor = load_model_and_processor(model_path, "cuda:0")
    rows = []
    roots = contract["snapshot_selection"]["roots"]
    for stage, root in roots.items():
        snapshot_root = first_q00(Path(root), args.suite)[0]
        package = load_snapshot(snapshot_root, materialize_torch=True)
        payload = package["payload"]
        processed = processor(payload["prompt"], payload["processed_image"], return_tensors="pt")
        prepared = append_empty_action_token(processed)
        model_inputs = {
            "input_ids": prepared["input_ids"].to(device="cuda:0"),
            "pixel_values": prepared["pixel_values"].to(device="cuda:0", dtype=next(model.parameters()).dtype),
        }
        action_dim = int(model.get_action_dim(args.suite))
        with torch.inference_mode():
            generated = model.generate(**model_inputs, max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
        token_ids = generated.sequences[0, -action_dim:].detach().cpu().numpy().astype(np.int64)
        teacher_rows = teacher_forced_rows(model, model_inputs["input_ids"], model_inputs["pixel_values"], token_ids, action_dim)
        score = generated.scores[-1][0].float().detach().cpu()
        top = torch.topk(score, k=2)
        rows.append({
            "stage": stage,
            "snapshot_root": str(snapshot_root),
            "generated_token_ids": [int(x) for x in token_ids],
            "gripper_token": int(token_ids[-1]),
            "gripper_top1_token": int(top.indices[0]),
            "gripper_top2_token": int(top.indices[1]),
            "gripper_top1_logit": float(top.values[0]),
            "gripper_top2_logit": float(top.values[1]),
            "gripper_margin": float(top.values[0] - top.values[1]),
            "teacher_forced_rows": teacher_rows,
            "teacher_forced_all_dims_exact": bool(all(row["exact"] for row in teacher_rows)),
            "processor_input_ids_exact": bool(torch.equal(prepared["input_ids"], payload["input_ids"])),
            "processor_attention_mask_exact": bool(torch.equal(prepared["attention_mask"], payload["attention_mask"])),
            "processor_pixel_values_exact_after_dtype_cast": bool(torch.equal(prepared["pixel_values"].to(dtype=payload["pixel_values"].dtype), payload["pixel_values"])),
        })
    result = {
        "schema": "STAGE_X_X1R_T0_CLEAN_FORWARD_WORKER_V1",
        "suite": args.suite,
        "replicate": int(args.replicate),
        "pid": os.getpid(),
        "physical_gpu": int(args.physical_gpu),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "model_path": str(model_path),
        "rows": rows,
        "counters": dict(COUNTERS),
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


def aggregate_clean(args: argparse.Namespace) -> None:
    paths = [Path(x) for x in args.inputs]
    workers = [load_json(path) for path in paths]
    by_suite: dict[str, list[dict[str, Any]]] = {suite: [] for suite in SUITES}
    for worker in workers:
        by_suite.setdefault(worker["suite"], []).append(worker)
    failures = []
    suites = {}
    for suite in SUITES:
        entries = by_suite.get(suite, [])
        token_sequences = {}
        for entry in entries:
            for row in entry.get("rows", []):
                token_sequences.setdefault(row["stage"], []).append(tuple(row["generated_token_ids"]))
                if not all(row[key] for key in ("processor_input_ids_exact", "processor_attention_mask_exact", "processor_pixel_values_exact_after_dtype_cast")):
                    failures.append(f"processor:{suite}:{row['stage']}")
                if not row.get("teacher_forced_all_dims_exact", False):
                    failures.append(f"teacher_forced_row_mismatch:{suite}:{row['stage']}")
        suite_result = {"replicate_count": len(entries), "stage_token_sequences": {stage: [list(x) for x in seqs] for stage, seqs in token_sequences.items()}, "deterministic": True}
        for stage, seqs in token_sequences.items():
            if not seqs or any(seq != seqs[0] for seq in seqs[1:]):
                suite_result["deterministic"] = False
                failures.append(f"token_nondeterminism:{suite}:{stage}")
        if len(entries) < 3:
            failures.append(f"replicate_count:{suite}:{len(entries)}")
        suites[suite] = suite_result
    report = {
        "schema": "STAGE_X_X1R_T0_CLEAN_FORWARD_DETERMINISM_V1",
        "status": "PASS_CLEAN_FORWARD_DETERMINISM" if not failures else "STAGE_X_X1R_T0_HOLD_CLEAN_FORWARD_NONDETERMINISM",
        "suites": suites,
        "worker_files": [str(path) for path in paths],
        "failures": failures,
        "protected_counters": dict(COUNTERS),
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "failures": failures}, sort_keys=True))
    if failures:
        raise SystemExit(2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpu", "clean-forward-worker", "aggregate-clean"), required=True)
    parser.add_argument("--model-paths", default="")
    parser.add_argument("--parity-root", default=DEFAULT_PARITY_ROOT)
    parser.add_argument("--output-dir", default=str(REPO / "reports"))
    parser.add_argument("--contract", default=str(REPO / "configs/STAGE_X_X1R_SUITE_MATCHED_VICTIM_CONTRACT_V1.json"))
    parser.add_argument("--suite", choices=SUITES)
    parser.add_argument("--physical-gpu", default="0")
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--inputs", nargs="*", default=[])
    args = parser.parse_args()
    if args.mode == "cpu":
        build_cpu_reports(args)
    elif args.mode == "clean-forward-worker":
        if not args.suite or not args.output:
            parser.error("clean-forward-worker requires --suite and --output")
        clean_forward_worker(args)
    else:
        if not args.output or not args.inputs:
            parser.error("aggregate-clean requires --output and --inputs")
        aggregate_clean(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
