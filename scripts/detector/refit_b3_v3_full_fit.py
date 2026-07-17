#!/usr/bin/env python3
"""Full-FIT refit entrypoint, gated by a sealed viability PASS.

The fold trainer never silently changes from 600 to 800 identities.  This
entrypoint is the only path that may create a FULL_FIT_REFIT_CANDIDATE, and it
requires a distinct machine-built ``fit_scope=FULL_FIT`` authorization.
"""

from __future__ import annotations

import argparse
import json
import importlib.util
from pathlib import Path

from gripper_attack.b3_formal import load_b3_checkpoint_bundle, save_b3_checkpoint_bundle
from gripper_attack.b3_training_protocol import load_training_authorization_bundle, sha256_file, verify_sealed_directory
from gripper_attack.b3_v3_dataset import compute_fit_normalization, load_episode, load_formal_registry_csv


def _load_viability(path: Path) -> dict:
    root = path
    if root.is_dir():
        verify_sealed_directory(root)
        root = root / "viability_aggregate.json"
    value = json.loads(root.read_text(encoding="utf-8"))
    if value.get("schema") != "B3_OFFICIAL_V3_FIT_VIABILITY_AGGREGATE_V1" or value.get("status") != "PASS":
        raise ValueError("full-FIT refit requires an independently sealed viability PASS")
    if value.get("run_count") != 24 or value.get("formal_training_authorized") is not False:
        raise ValueError("viability matrix closure is incomplete")
    return value


def load_full_fit_episodes(registry_csv: Path, s1_root: Path, *, policy_intent_root: Path | None = None):
    rows = load_formal_registry_csv(registry_csv, require_a_only=True)
    episodes = []
    for row in rows:
        root = s1_root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
        nine_d = None if policy_intent_root is None else policy_intent_root / row["suite"] / f"task_{int(row['task_idx']):02d}" / f"state_{int(row['state_id']):02d}"
        episodes.append(load_episode(root, row, include_9d_root=nine_d))
    if len(episodes) != 800:
        raise ValueError("full-FIT refit requires exactly 800 episodes")
    return episodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viability-root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--s1-root", type=Path, required=True)
    parser.add_argument("--output-checkpoint-bundle", type=Path, required=True)
    parser.add_argument("--variant", choices=("B3_25D", "B3_25D9D"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy-intent-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--execute-formal", action="store_true")
    args = parser.parse_args()
    viability = _load_viability(args.viability_root)
    if not args.execute_formal:
        print(json.dumps({"status": "FULL_FIT_REFIT_HOLD", "reason": "explicit formal execution flag is required", "viability_runs": viability["run_count"], "formal_attack_authorized": False}, sort_keys=True))
        return 0
    authorization = load_training_authorization_bundle(args.authorization) if args.authorization.is_dir() else json.loads(args.authorization.read_text(encoding="utf-8"))
    if authorization.get("fit_scope") != "FULL_FIT" or authorization.get("variant") != args.variant or authorization.get("seed") != args.seed:
        raise SystemExit("FULL_FIT_REFIT_HOLD: authorization scope/variant/seed mismatch")
    episodes = load_full_fit_episodes(args.registry_csv, args.s1_root, policy_intent_root=args.policy_intent_root)
    trainer_path = Path(__file__).with_name("train_b3_v3_detector.py")
    spec = importlib.util.spec_from_file_location("b3_v3_formal_trainer", trainer_path)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load formal trainer")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    normalization = compute_fit_normalization(episodes, include_9d=args.variant == "B3_25D9D")
    if authorization.get("normalization_sha256") != normalization.sha256:
        raise SystemExit("FULL_FIT_REFIT_HOLD: normalization is not recomputed from all 800 FIT episodes")
    model, losses = trainer.train_model(episodes, variant=args.variant, normalization=normalization, seed=args.seed, device=args.device)
    save_b3_checkpoint_bundle(
        args.output_checkpoint_bundle, model, normalization, authorization=authorization,
        checkpoint_status="FULL_FIT_REFIT_CANDIDATE",
        extra={"fit_scope": "FULL_FIT", "fit_episode_count": 800, "seed": args.seed, "loss_history": losses, "viability_report_sha256": sha256_file(args.viability_root / "SHA256SUMS") if args.viability_root.is_dir() else sha256_file(args.viability_root)},
    )
    print(json.dumps({"status": "FULL_FIT_REFIT_CANDIDATE", "fit_episode_count": 800, "formal_attack_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
