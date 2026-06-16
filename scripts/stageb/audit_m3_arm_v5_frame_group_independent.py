#!/usr/bin/env python3
"""Independent auditor for M3 arm-v5.2 fixed-frame candidate artifacts.

This script intentionally does not import producer-side frame-group selection
helpers. It recomputes frozen-input bindings, candidate feasibility, selected
candidate ordering, paired control margins, and aggregate scientific gates from
the files on disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping


V5_2_CONDITIONS = (
    "TRUE_PGD21_SELECTIVE",
    "RAND21_SELECTIVE",
    "SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE",
)
V5_2_CANDIDATE_COUNT = 21
V5_2_FROZEN_SEED = 428198
V5_2_FORBIDDEN_SEEDS = {85, 86}
V5_2_TARGET_TOKEN = 31744
V5_2_ARM_MATCH_MIN = 5
V5_2_ARM_LEN = 6
V5_2_EPSILON = 6.0 / 255.0
V5_2_PANEL_SIZE = 8

REQUIRED_CANDIDATE_FIELDS = (
    "frame_id",
    "condition",
    "candidate_index",
    "seed",
    "raw_image_path",
    "raw_image_sha256",
    "processed_tensor_path",
    "processed_tensor_sha256",
    "prompt_token_ids",
    "prompt_token_ids_sha256",
    "frozen_input_row_sha256",
    "clean_exact_7_tokens",
    "attacked_exact_7_tokens",
    "clean_arm_prefix",
    "attacked_arm_prefix",
    "official_gripper_token",
    "target_token_score",
    "best_competitor_score",
    "official_target_margin",
    "perturbation_tensor_path",
    "perturbation_tensor_sha256",
    "perturbation_linf",
    "model_checkpoint_sha256",
    "processor_config_sha256",
    "preprocess_config_sha256",
    "commit",
    "algorithm_config_sha256",
    "libero_rollout_used",
    "score_invariant_status",
    "route_status",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def safe_rel(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_int_list(value: Any, *, field: str, length: int | None = None) -> list[int]:
    obj = json.loads(value) if isinstance(value, str) else value
    if not isinstance(obj, list):
        raise ValueError(f"{field} must be a list")
    out = [int(x) for x in obj]
    if length is not None and len(out) != length:
        raise ValueError(f"{field} length must be {length}, got {len(out)}")
    return out


def finite_float(value: Any, *, field: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field} must be finite")
    return out


def row_sha256(row: Mapping[str, Any]) -> str:
    return canonical_sha(dict(row))


def load_frozen_manifest(root: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], str]:
    manifest = root / "frozen_input_manifest.csv"
    if not manifest.is_file():
        raise ValueError("frozen_input_manifest.csv is required")
    rows = read_csv(manifest)
    if len(rows) != V5_2_PANEL_SIZE:
        raise ValueError(f"frozen input manifest must contain exactly {V5_2_PANEL_SIZE} rows")
    by_frame: dict[str, dict[str, str]] = {}
    for row in rows:
        frame_id = str(row.get("frame_id", ""))
        if not frame_id:
            raise ValueError("frozen manifest row missing frame_id")
        if frame_id in by_frame:
            raise ValueError(f"duplicate frozen frame_id: {frame_id}")
        by_frame[frame_id] = row
        for rel_key, sha_key in (("raw_image_path", "raw_image_sha256"), ("processed_tensor_path", "processed_tensor_sha256")):
            path = root / safe_rel(str(row.get(rel_key, "")))
            if not path.is_file():
                raise ValueError(f"frozen input artifact missing: {frame_id}:{rel_key}")
            if sha256_file(path) != str(row.get(sha_key, "")):
                raise ValueError(f"frozen input sha mismatch: {frame_id}:{rel_key}")
        if sha256_text(str(row.get("prompt_token_ids", ""))) != str(row.get("prompt_token_ids_sha256", "")):
            raise ValueError(f"prompt token sha mismatch in frozen manifest: {frame_id}")
    return rows, by_frame, sha256_file(manifest)


def perturbation_linf_from_file(path: Path) -> float:
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        values = obj.get("values", obj)
        flat: list[float] = []

        def walk(item: Any) -> None:
            if isinstance(item, list):
                for child in item:
                    walk(child)
            else:
                flat.append(float(item))

        walk(values)
        if not flat:
            raise ValueError(f"empty perturbation tensor: {path}")
        return max(abs(x) for x in flat)
    if path.suffix.lower() == ".pt":
        import torch

        tensor = torch.load(path, map_location="cpu")
        if isinstance(tensor, Mapping):
            tensor = tensor.get("delta", tensor.get("perturbation", tensor.get("values")))
        if tensor is None:
            raise ValueError(f"cannot locate tensor in {path}")
        return float(tensor.detach().abs().max().item())
    raise ValueError(f"unsupported perturbation tensor format: {path}")


def candidate_path(root: Path, *, frame_id: str, condition: str, index: int) -> Path:
    return root / "frames" / frame_id / condition / f"candidate_{index:02d}.json"


def verify_candidate(root: Path, payload: Mapping[str, Any], *, frame_id: str, condition: str, index: int, seed: int, frozen_row: Mapping[str, str]) -> dict[str, Any]:
    for field in REQUIRED_CANDIDATE_FIELDS:
        if payload.get(field, "") in ("", None):
            raise ValueError(f"{frame_id}:{condition}:{index} missing {field}")
    if str(payload["frame_id"]) != frame_id:
        raise ValueError("candidate frame_id mismatch")
    if str(payload["condition"]) != condition:
        raise ValueError("candidate condition mismatch")
    if int(payload["candidate_index"]) != index:
        raise ValueError("candidate index mismatch")
    if int(payload["seed"]) != seed:
        raise ValueError("candidate seed mismatch")
    if parse_bool(payload["libero_rollout_used"]):
        raise ValueError("LIBERO rollout evidence is forbidden in fixed-frame artifact")
    if str(payload["route_status"]) != "PASS":
        raise ValueError("route_status must be PASS")
    if str(payload["score_invariant_status"]) != "PASS":
        raise ValueError("score_invariant_status must be PASS")
    if str(payload["frozen_input_row_sha256"]) != row_sha256(frozen_row):
        raise ValueError("frozen input row binding mismatch")
    for key in ("raw_image_path", "raw_image_sha256", "processed_tensor_path", "processed_tensor_sha256"):
        if str(payload[key]) != str(frozen_row.get(key, "")):
            raise ValueError(f"candidate frozen input field mismatch: {key}")
    if sha256_text(str(payload["prompt_token_ids"])) != str(payload["prompt_token_ids_sha256"]):
        raise ValueError("candidate prompt token sha mismatch")
    if str(payload["prompt_token_ids_sha256"]) != str(frozen_row.get("prompt_token_ids_sha256", "")):
        raise ValueError("candidate prompt binding mismatch")
    for rel_key, sha_key in (("raw_image_path", "raw_image_sha256"), ("processed_tensor_path", "processed_tensor_sha256"), ("perturbation_tensor_path", "perturbation_tensor_sha256")):
        path = root / safe_rel(str(payload[rel_key]))
        if not path.is_file():
            raise ValueError(f"candidate artifact missing: {rel_key}")
        if sha256_file(path) != str(payload[sha_key]):
            raise ValueError(f"candidate sha mismatch: {rel_key}")
    clean_exact = parse_int_list(payload["clean_exact_7_tokens"], field="clean_exact_7_tokens", length=7)
    attacked_exact = parse_int_list(payload["attacked_exact_7_tokens"], field="attacked_exact_7_tokens", length=7)
    clean_arm = parse_int_list(payload["clean_arm_prefix"], field="clean_arm_prefix", length=V5_2_ARM_LEN)
    attacked_arm = parse_int_list(payload["attacked_arm_prefix"], field="attacked_arm_prefix", length=V5_2_ARM_LEN)
    official_token = int(payload["official_gripper_token"])
    if attacked_exact[-1] != official_token:
        raise ValueError("official gripper token does not match attacked exact generation")
    if clean_exact[:V5_2_ARM_LEN] != clean_arm:
        raise ValueError("clean arm prefix does not match clean exact generation")
    if attacked_exact[:V5_2_ARM_LEN] != attacked_arm:
        raise ValueError("attacked arm prefix does not match attacked exact generation")
    arm_match = sum(1 for a, b in zip(clean_arm, attacked_arm) if a == b)
    target_score = finite_float(payload["target_token_score"], field="target_token_score")
    competitor = finite_float(payload["best_competitor_score"], field="best_competitor_score")
    margin = finite_float(payload["official_target_margin"], field="official_target_margin")
    if abs((target_score - competitor) - margin) > 1e-6:
        raise ValueError("official target margin is not target_score - best_competitor_score")
    linf = finite_float(payload["perturbation_linf"], field="perturbation_linf")
    recomputed_linf = perturbation_linf_from_file(root / safe_rel(str(payload["perturbation_tensor_path"])))
    if abs(recomputed_linf - linf) > 1e-7:
        raise ValueError("perturbation Linf mismatch")
    feasible = (
        official_token == V5_2_TARGET_TOKEN
        and arm_match >= V5_2_ARM_MATCH_MIN
        and linf <= V5_2_EPSILON + 1e-8
    )
    return {
        "condition": condition,
        "candidate_index": index,
        "official_gripper_token": official_token,
        "arm_match_count": arm_match,
        "official_target_margin": margin,
        "perturbation_linf": linf,
        "feasible": feasible,
    }


def load_condition(root: Path, *, frame_id: str, condition: str, seed: int, frozen_row: Mapping[str, str]) -> list[dict[str, Any]]:
    out = []
    for idx in range(V5_2_CANDIDATE_COUNT):
        path = candidate_path(root, frame_id=frame_id, condition=condition, index=idx)
        if not path.is_file():
            raise ValueError(f"missing candidate artifact: {path}")
        out.append(verify_candidate(root, json.loads(path.read_text(encoding="utf-8")), frame_id=frame_id, condition=condition, index=idx, seed=seed, frozen_row=frozen_row))
    return out


def select_best(candidates: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    feasible = [row for row in candidates if bool(row["feasible"])]
    if not feasible:
        return None
    return sorted(feasible, key=lambda row: (-float(row["official_target_margin"]), float(row["perturbation_linf"]), int(row["candidate_index"])))[0]


def audit_frame_group(root: Path, *, frame_ids: list[str], seed: int) -> dict[str, Any]:
    if int(seed) in V5_2_FORBIDDEN_SEEDS:
        raise ValueError(f"legacy seed is forbidden for V5.2: {seed}")
    if int(seed) != V5_2_FROZEN_SEED:
        raise ValueError(f"V5.2 seed must be frozen seed {V5_2_FROZEN_SEED}, got {seed}")
    manifest_rows, frozen_by_frame, frozen_manifest_sha = load_frozen_manifest(root)
    manifest_ids = [row["frame_id"] for row in manifest_rows]
    if sorted(frame_ids) != sorted(manifest_ids) or len(set(frame_ids)) != V5_2_PANEL_SIZE:
        raise ValueError("frame_ids must exactly match the 8-frame frozen input manifest")

    frames: list[dict[str, Any]] = []
    rand_paired: list[float] = []
    shuffled_paired: list[float] = []
    for frame_id in frame_ids:
        selected: dict[str, Mapping[str, Any] | None] = {}
        feasible_counts: dict[str, int] = {}
        for condition in V5_2_CONDITIONS:
            candidates = load_condition(root, frame_id=frame_id, condition=condition, seed=seed, frozen_row=frozen_by_frame[frame_id])
            feasible_counts[condition] = sum(1 for row in candidates if row["feasible"])
            selected[condition] = select_best(candidates)
        true = selected["TRUE_PGD21_SELECTIVE"]
        rand = selected["RAND21_SELECTIVE"]
        shuffled = selected["SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE"]
        true_feasible = true is not None
        rand_status = "TRUE_INFEASIBLE"
        shuffled_status = "TRUE_INFEASIBLE"
        rand_delta: float | None = None
        shuffled_delta: float | None = None
        if true_feasible:
            if rand is None:
                rand_status = "WIN_CONTROL_INFEASIBLE"
            else:
                rand_delta = float(true["official_target_margin"]) - float(rand["official_target_margin"])
                rand_status = "WIN_FINITE" if rand_delta > 0 else "CONTROL_NOT_BEATEN"
                rand_paired.append(rand_delta)
            if shuffled is None:
                shuffled_status = "WIN_CONTROL_INFEASIBLE"
            else:
                shuffled_delta = float(true["official_target_margin"]) - float(shuffled["official_target_margin"])
                shuffled_status = "WIN_FINITE" if shuffled_delta > 0 else "CONTROL_NOT_BEATEN"
                shuffled_paired.append(shuffled_delta)
        full_pass = true_feasible and rand_status.startswith("WIN") and shuffled_status.startswith("WIN")
        frames.append(
            {
                "frame_id": frame_id,
                "frame_full_selective_pass": full_pass,
                "rand_comparison_status": rand_status,
                "shuffled_comparison_status": shuffled_status,
                "true_selected_margin": "" if true is None else true["official_target_margin"],
                "rand_selected_margin": "" if rand is None else rand["official_target_margin"],
                "shuffled_selected_margin": "" if shuffled is None else shuffled["official_target_margin"],
                "true_minus_rand_margin": "" if rand_delta is None else rand_delta,
                "true_minus_shuffled_margin": "" if shuffled_delta is None else shuffled_delta,
                "true_feasible_count": feasible_counts["TRUE_PGD21_SELECTIVE"],
                "rand_feasible_count": feasible_counts["RAND21_SELECTIVE"],
                "shuffled_feasible_count": feasible_counts["SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE"],
            }
        )

    pass_count = sum(1 for row in frames if row["frame_full_selective_pass"])
    rand_median = statistics.median(rand_paired) if rand_paired else None
    shuffled_median = statistics.median(shuffled_paired) if shuffled_paired else None
    scientific_pass = (
        pass_count >= 6
        and len(rand_paired) >= 4
        and len(shuffled_paired) >= 4
        and rand_median is not None
        and rand_median > 0
        and shuffled_median is not None
        and shuffled_median > 0
    )
    return {
        "artifact_audit_status": "PASS",
        "scientific_gate_status": "PASS" if scientific_pass else "FAIL",
        "audit_status": "PASS",
        "seed": int(seed),
        "frame_count": len(frames),
        "frozen_input_manifest_sha256": frozen_manifest_sha,
        "frame_full_selective_pass_count": pass_count,
        "rand_finite_paired_frame_count": len(rand_paired),
        "shuffled_finite_paired_frame_count": len(shuffled_paired),
        "median_true_minus_rand_margin": "" if rand_median is None else rand_median,
        "median_true_minus_shuffled_margin": "" if shuffled_median is None else shuffled_median,
        "frames": frames,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_root", required=True)
    ap.add_argument("--frame_ids", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--audit_output", default="")
    args = ap.parse_args()
    frames = [item.strip() for item in args.frame_ids.split(",") if item.strip()]
    try:
        result = audit_frame_group(Path(args.artifact_root), frame_ids=frames, seed=int(args.seed))
    except Exception as exc:
        result = {
            "artifact_audit_status": "FAIL",
            "scientific_gate_status": "NOT_EVALUATED",
            "audit_status": "FAIL",
            "failure_reason": repr(exc),
            "artifact_root": str(args.artifact_root),
            "seed": int(args.seed),
            "frame_ids": frames,
        }
    output = Path(args.audit_output) if args.audit_output else Path(args.artifact_root) / "m3_arm_v5_frame_group_audit.json"
    write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["artifact_audit_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
