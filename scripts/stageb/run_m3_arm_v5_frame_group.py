#!/usr/bin/env python3
"""CPU/mock entrypoint for M3 arm-v5.2 frame-group artifact contracts.

This script does not run OpenVLA, PGD, RAND, shuffled-gradient, or LIBERO. It is
used to smoke-test the fixed-frame artifact layout before real GPU execution is
authorized after V5.1 input freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gripper_attack.m3_v5_attack_harness import (  # noqa: E402
    V5_2_CANDIDATE_COUNT,
    V5_2_CONDITIONS,
    V5_2_FROZEN_SEED,
    sha256_file,
    sha256_text,
    write_candidate_artifact,
    write_csv,
    write_json,
)
from scripts.stageb.audit_m3_arm_v5_frame_group_independent import audit_frame_group  # noqa: E402


def canonical_sha(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_text_artifact(root: Path, rel: str, text: str) -> tuple[str, str]:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return rel, sha256_file(path)


def mock_payload(*, root: Path, frame_row: dict[str, str], condition: str, candidate_index: int, seed: int) -> dict[str, object]:
    base_margin = {
        "TRUE_PGD21_SELECTIVE": 10.0,
        "RAND21_SELECTIVE": 1.0,
        "SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE": 0.5,
    }[condition]
    frame_id = frame_row["frame_id"]
    perturb_rel, perturb_sha = _write_text_artifact(
        root,
        f"frames/{frame_id}/{condition}/perturbations/delta_{candidate_index:02d}.json",
        json.dumps({"values": [0.0, 0.0, 0.0]}),
    )
    margin = base_margin + float(candidate_index) / 100.0
    return {
        "seed": int(seed),
        "raw_image_path": frame_row["raw_image_path"],
        "raw_image_sha256": frame_row["raw_image_sha256"],
        "processed_tensor_path": frame_row["processed_tensor_path"],
        "processed_tensor_sha256": frame_row["processed_tensor_sha256"],
        "prompt_token_ids": frame_row["prompt_token_ids"],
        "prompt_token_ids_sha256": frame_row["prompt_token_ids_sha256"],
        "frozen_input_row_sha256": canonical_sha(frame_row),
        "clean_exact_7_tokens": [1, 2, 3, 4, 5, 6, 31872],
        "attacked_exact_7_tokens": [1, 2, 3, 4, 5, 6, 31744],
        "clean_arm_prefix": [1, 2, 3, 4, 5, 6],
        "attacked_arm_prefix": [1, 2, 3, 4, 5, 6],
        "official_gripper_token": 31744,
        "target_token_score": 10.0 + margin,
        "best_competitor_score": 10.0,
        "official_target_margin": margin,
        "perturbation_tensor_path": perturb_rel,
        "perturbation_tensor_sha256": perturb_sha,
        "perturbation_linf": 0.0,
        "model_checkpoint_sha256": "mock_model_sha",
        "processor_config_sha256": "mock_processor_sha",
        "preprocess_config_sha256": "mock_preprocess_sha",
        "commit": "mock_commit",
        "algorithm_config_sha256": "mock_algorithm_sha",
        "score_invariant_status": "PASS",
        "route_status": "PASS",
        "libero_rollout_used": False,
        "candidate_mode": "CPU_MOCK_ZERO_PERTURBATION",
    }


def write_frozen_manifest(root: Path, frames: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for frame_id in frames:
        raw_rel, raw_sha = _write_text_artifact(root, f"frozen_inputs/{frame_id}/raw.bin", f"raw-{frame_id}")
        tensor_rel, tensor_sha = _write_text_artifact(root, f"frozen_inputs/{frame_id}/processor.bin", f"tensor-{frame_id}")
        prompt = "[1,2,3,29871]"
        rows.append(
            {
                "frame_id": frame_id,
                "raw_image_path": raw_rel,
                "raw_image_sha256": raw_sha,
                "processed_tensor_path": tensor_rel,
                "processed_tensor_sha256": tensor_sha,
                "prompt_token_ids": prompt,
                "prompt_token_ids_sha256": sha256_text(prompt),
            }
        )
    write_csv(
        root / "frozen_input_manifest.csv",
        rows,
        ["frame_id", "raw_image_path", "raw_image_sha256", "processed_tensor_path", "processed_tensor_sha256", "prompt_token_ids", "prompt_token_ids_sha256"],
    )
    return rows


def run_mock_zero(args: argparse.Namespace) -> None:
    if int(args.seed) != V5_2_FROZEN_SEED:
        raise SystemExit(f"mock V5.2 harness requires frozen seed {V5_2_FROZEN_SEED}")
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        raise SystemExit("--output_dir must be new or empty")
    frames = [item.strip() for item in args.frame_ids.split(",") if item.strip()]
    frame_rows = {row["frame_id"]: row for row in write_frozen_manifest(root, frames)}
    for frame_id in frames:
        for condition in V5_2_CONDITIONS:
            for idx in range(V5_2_CANDIDATE_COUNT):
                write_candidate_artifact(
                    root,
                    frame_id=frame_id,
                    condition=condition,
                    candidate_index=idx,
                    payload=mock_payload(root=root, frame_row=frame_rows[frame_id], condition=condition, candidate_index=idx, seed=int(args.seed)),
                )
    result = audit_frame_group(root, frame_ids=frames, seed=int(args.seed))
    write_json(root / "m3_arm_v5_frame_group_mock_summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mock_zero_perturbation"], required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--frame_ids", required=True)
    ap.add_argument("--seed", type=int, default=V5_2_FROZEN_SEED)
    args = ap.parse_args()
    if args.mode == "mock_zero_perturbation":
        run_mock_zero(args)


if __name__ == "__main__":
    main()
