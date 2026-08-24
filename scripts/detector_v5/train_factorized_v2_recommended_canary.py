#!/usr/bin/env python3
"""Train one recommended V2B exact-W32 engineering sidecar run.

Frozen sidecar config:
  V2B exact causal TCN, W=32, H=64, dropout=0.1, wd=1e-4,
  AdamW lr=1e-3, batch=8, epochs=30, 25D-only.

This script deliberately writes a separate engineering namespace and marks every
artifact ``formal_selection_eligible=false``. It must not be mixed into the
frozen 864-job Stage-1 selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.b3_training_protocol import sha256_file, verify_sealed_directory
from gripper_attack.v5_factorized_dataset import (
    SUPPORTED_ROUTES,
    compute_factorized_normalization,
    load_factorized_episodes,
    verify_factorized_source_roots,
)
from gripper_attack.v5_factorized_student_v2_recommended import (
    RecommendedEventBalancedLoss,
    RecommendedFactorizedStudentV2,
)
from gripper_attack.v5_factorized_v2_splits import resolve_inner_train_val_ids

BASE_TRAINER_PATH = ROOT / "scripts/detector_v5/train_factorized_v2_inner_cv.py"
_spec = importlib.util.spec_from_file_location("v2_base_trainer_helpers", BASE_TRAINER_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot import helper module: {BASE_TRAINER_PATH}")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

S1 = _base.S1
TEACHER = _base.TEACHER
REGISTRY = _base.REGISTRY

CONFIG = {
    "candidate": "V2B_RECOMMENDED_EXACT",
    "context_steps": 32,
    "hidden_dim": 64,
    "dropout": 0.1,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 30,
    "batch_size": 8,
    "input": "25D_ONLY",
    "formal_selection_eligible": False,
}


def _atomic_text(path: Path, value: str) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("x") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_seal(root: Path) -> None:
    excluded = {"SHA256SUMS", "SHA256SUMS.sha256"}
    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.name not in excluded),
        key=lambda p: p.relative_to(root).as_posix(),
    )
    content = "".join(
        f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in files
    )
    _atomic_text(root / "SHA256SUMS", content)
    _atomic_text(
        root / "SHA256SUMS.sha256",
        f"{sha256_file(root / 'SHA256SUMS')}  SHA256SUMS\n",
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", type=int, required=True, choices=range(4))
    parser.add_argument("--inner-fold", type=int, required=True, choices=range(3))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--inner-cv-splits-root", type=Path, required=True)
    parser.add_argument("--reference-authorization-root", type=Path, default=None)
    args = parser.parse_args()

    output = args.output_root.resolve()
    if output.exists():
        raise SystemExit(f"OUTPUT EXISTS: {output}")
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    verify_sealed_directory(args.inner_cv_splits_root)
    splits = json.loads((args.inner_cv_splits_root / "inner_cv_splits.json").read_text())
    inner_train_ids, inner_val_ids = resolve_inner_train_val_ids(
        splits, args.outer_fold, args.inner_fold
    )

    rows = list(csv.DictReader(REGISTRY.open()))
    fit_rows = [row for row in rows if row.get("split") == "FIT_TRAIN"]
    id_to_row = {row["canonical_parent_key"]: row for row in fit_rows}
    train_rows = [id_to_row[i] for i in inner_train_ids if i in id_to_row]
    val_rows = [id_to_row[i] for i in inner_val_ids if i in id_to_row]

    verify_factorized_source_roots(S1, TEACHER)
    train_eps = load_factorized_episodes(S1, TEACHER, train_rows)
    val_eps = load_factorized_episodes(S1, TEACHER, val_rows)
    print(f"Train: {len(train_eps)} episodes, Val: {len(val_eps)} episodes", flush=True)

    mean_25d, std_25d = compute_factorized_normalization(train_eps)
    class_weights = _base.compute_class_weights(train_eps)
    rng_seed = args.seed + args.outer_fold * 100 + args.inner_fold * 10
    train_batches = _base.build_route_balanced_batches(
        train_eps, CONFIG["batch_size"], rng_seed
    )

    single_val = [
        ep for ep in val_eps if ep.mechanism_route == "single_object_pick_place"
    ]
    multi_val = [
        ep for ep in val_eps if ep.mechanism_route == "multi_object_transfer"
    ]
    val_batches = []
    for route, episodes in (
        ("single_object_pick_place", single_val),
        ("multi_object_transfer", multi_val),
    ):
        for i in range(0, len(episodes), CONFIG["batch_size"]):
            val_batches.append((route, episodes[i:i + CONFIG["batch_size"]]))

    model = RecommendedFactorizedStudentV2(
        input_dim_25d=25,
        hidden_dim=CONFIG["hidden_dim"],
        context_steps=CONFIG["context_steps"],
        dropout=CONFIG["dropout"],
    ).to(device)
    loss_fn = RecommendedEventBalancedLoss(consistency_weight=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )

    jitter_rng = __import__("random").Random(args.seed + 999)
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_grasp": [],
        "val_manipulation": [],
        "val_release": [],
    }
    sampling_audit = []

    def batch_to_device(batch_eps, training: bool):
        batch_size = len(batch_eps)
        max_t = max(len(ep.features_25d) for ep in batch_eps)
        x25 = torch.zeros(batch_size, max_t, 25, device=device)
        valid = torch.zeros(batch_size, max_t, dtype=torch.bool, device=device)
        for b, episode in enumerate(batch_eps):
            current = episode
            if training:
                jitter = jitter_rng.randint(0, 4)
                current = _base.apply_temporal_jitter(episode, jitter, jitter_rng)
            t = len(current.features_25d)
            x25[b, :t] = ((current.features_25d - mean_25d) / std_25d).to(device)
            valid[b, :t] = current.valid_mask.to(device)
        return x25, valid

    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_losses = []
        epoch_audit = {"duration_buckets": {}, "identity_count": 0}

        for route, batch_eps in train_batches:
            x25, valid = batch_to_device(batch_eps, training=True)
            optimizer.zero_grad()
            identity_map = _base.compute_identity_weights(batch_eps, route)
            identity_weights = torch.tensor(
                [identity_map.get(id(ep), 1.0) for ep in batch_eps], device=device
            )
            logits = model.forward_logits(x25, None, valid, None, route)
            route_weights = class_weights.get(route, {})
            loss, _, audit = loss_fn(
                logits,
                batch_eps,
                valid_mask=valid,
                class_weights=route_weights,
                identity_weights=identity_weights,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
            for key, value in audit.get("duration_buckets", {}).items():
                epoch_audit["duration_buckets"].setdefault(
                    key, {"event_count": 0, "loss_sum": 0.0}
                )
                epoch_audit["duration_buckets"][key]["event_count"] += value["event_count"]
                epoch_audit["duration_buckets"][key]["loss_sum"] += value["loss_sum"]

        model.eval()
        route_losses = defaultdict(list)
        head_metrics = defaultdict(list)
        with torch.no_grad():
            for route, batch_eps in val_batches:
                if not batch_eps:
                    continue
                x25, valid = batch_to_device(batch_eps, training=False)
                logits = model.forward_logits(x25, None, valid, None, route)
                route_weights = class_weights.get(route, {})
                loss, metrics, _ = loss_fn(
                    logits,
                    batch_eps,
                    valid_mask=valid,
                    class_weights=route_weights,
                )
                route_losses[route].append(float(loss.item()))
                for head in ("grasp", "manipulation", "release"):
                    head_metrics[head].append(float(metrics[head]))

        avg_train = sum(train_losses) / max(1, len(train_losses))
        single_loss = sum(route_losses["single_object_pick_place"]) / max(
            1, len(route_losses["single_object_pick_place"])
        )
        multi_loss = sum(route_losses["multi_object_transfer"]) / max(
            1, len(route_losses["multi_object_transfer"])
        )
        avg_val = (single_loss + multi_loss) / 2

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        for head in ("grasp", "manipulation", "release"):
            history[f"val_{head}"].append(
                sum(head_metrics[head]) / max(1, len(head_metrics[head]))
            )
        sampling_audit.append(epoch_audit)

        if epoch % 5 == 0:
            print(
                f"epoch {epoch:2d}: train={avg_train:.4f} val={avg_val:.4f} "
                f"g={history['val_grasp'][-1]:.4f} "
                f"m={history['val_manipulation'][-1]:.4f} "
                f"r={history['val_release'][-1]:.4f}",
                flush=True,
            )

    checkpoint = {
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "config": {
            **CONFIG,
            "outer_fold": args.outer_fold,
            "inner_fold": args.inner_fold,
            "seed": args.seed,
            "actual_receptive_field": model.encoder_25d.actual_receptive_field,
        },
        "epoch": CONFIG["epochs"],
    }
    torch.save(checkpoint, staging / "checkpoint.pt")

    run_config = {
        **CONFIG,
        "outer_fold": args.outer_fold,
        "inner_fold": args.inner_fold,
        "seed": args.seed,
        "parameter_count": model.parameter_count(),
        "actual_receptive_field": model.encoder_25d.actual_receptive_field,
        "sidecar_status": "ENGINEERING_RECOMMENDED_CONFIG_CANARY",
    }
    _atomic_text(staging / "run_config.json", json.dumps(run_config, indent=2))
    _atomic_text(staging / "history.json", json.dumps(history, indent=2))
    _atomic_text(
        staging / "normalization.json",
        json.dumps({"mean_25d": mean_25d.tolist(), "std_25d": std_25d.tolist()}),
    )
    _atomic_text(staging / "class_weights.json", json.dumps(class_weights, indent=2))
    _atomic_text(staging / "sampling_audit.json", json.dumps(sampling_audit, indent=2))
    _atomic_text(
        staging / "environment.json",
        json.dumps({
            "python": sys.executable,
            "python_version": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "host": platform.node(),
        }, indent=2),
    )

    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    source_binding = {
        "source_commit": source_commit,
        "recommended_module_sha": _sha(
            ROOT / "src/gripper_attack/v5_factorized_student_v2_recommended.py"
        ),
        "trainer_sha": _sha(Path(__file__)),
        "base_trainer_helper_sha": _sha(BASE_TRAINER_PATH),
        "dataset_sha": _sha(ROOT / "src/gripper_attack/v5_factorized_dataset.py"),
        "inner_cv_splits_root": str(args.inner_cv_splits_root.resolve()),
        "formal_selection_eligible": False,
    }
    _atomic_text(staging / "source_binding.json", json.dumps(source_binding, indent=2))

    reference_receipt = {
        "status": "NO_REFERENCE_AUTHORIZATION",
        "formal_selection_eligible": False,
    }
    if args.reference_authorization_root is not None:
        auth_root = args.reference_authorization_root.resolve()
        verify_sealed_directory(auth_root)
        reference_receipt = {
            "status": "REFERENCE_ONLY_NOT_AUTHORIZATION",
            "reference_authorization_root": str(auth_root),
            "reference_authorization_seal": sha256_file(auth_root / "SHA256SUMS"),
            "formal_selection_eligible": False,
        }
    _atomic_text(
        staging / "sidecar_reference_receipt.json",
        json.dumps(reference_receipt, indent=2),
    )

    _write_seal(staging)
    os.replace(staging, output)
    print(f"Sealed engineering sidecar: {output}", flush=True)


if __name__ == "__main__":
    main()
