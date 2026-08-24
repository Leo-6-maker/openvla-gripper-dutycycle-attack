"""Formal fold-0 GPU engineering smoke using the shared D8 training core."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "detector_v5"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from audit_r3_contact_input import sha256_file, verify_seal
from d8_source_contract import (
    P5_REQUIRED_SOURCE_FILES,
    load_and_validate_source_snapshot,
)
from d8_train_core import (
    FEATURE_DIM,
    SEED,
    apply_normalization,
    audit_effective_mask,
    checkpoint_roundtrip_parity,
    compute_loss,
    compute_normalization,
    continuation_parity,
    create_model,
    identity_digest,
    save_checkpoint,
)
from gripper_attack.seal_utils import rename_noreplace

FOLD = 0
ALLOWED_KEYS = {
    "episode_id",
    "step",
    "features_25d_raw",
    "physical_target",
    "effective_mask",
    "D8_weight",
    "fold_id",
    "right_censored",
    "geometry_not_applicable",
    "articulated",
}


def _write_seal(root: Path) -> str:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}
    )
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(
        f"{digest}  SHA256SUMS\n", encoding="utf-8"
    )
    return digest


def _load_cache(cache_root: Path) -> tuple[list[dict], dict, dict]:
    seal = verify_seal(cache_root)
    manifest_path = cache_root / "CACHE_MANIFEST.json"
    if not manifest_path.is_file():
        raise RuntimeError("CACHE_MANIFEST.json missing")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("schema") != "DETECTOR_V3_D8_25D_CACHE_V3":
        raise RuntimeError(f"unsupported cache schema: {manifest.get('schema')!r}")
    if manifest.get("status") != "BUILT_PENDING_H1" or manifest.get("consumer_eligible") is not False:
        raise RuntimeError("P5 requires a non-consumer-eligible BUILT_PENDING_H1 cache")

    per_episode = cache_root / "per_episode"
    files = sorted(per_episode.glob("*.json"))
    if len(files) != 670:
        raise RuntimeError(f"expected 670 per-episode cache files, found {len(files)}")
    entries = []
    for path in files:
        episode_entries = json.loads(path.read_text("utf-8"))
        if not isinstance(episode_entries, list) or not episode_entries:
            raise RuntimeError(f"invalid per-episode cache file: {path}")
        entries.extend(episode_entries)
    if len(entries) != 196_483:
        raise RuntimeError(f"expected 196483 cache entries, found {len(entries)}")
    return entries, manifest, seal


def _environment(device: torch.device) -> dict:
    result = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        result["device_name"] = torch.cuda.get_device_name(device)
        result["device_capability"] = list(torch.cuda.get_device_capability(device))
    return result


def run_smoke(
    cache_root: Path,
    source_snapshot_path: Path,
    output_root: Path,
    run_label: str,
) -> dict:
    if output_root.exists():
        raise FileExistsError(f"output_root must not exist: {output_root}")

    source = load_and_validate_source_snapshot(
        source_snapshot_path, ROOT, P5_REQUIRED_SOURCE_FILES
    )
    entries, cache_manifest, cache_seal = _load_cache(cache_root)

    cache_source = cache_manifest.get("code_snapshot", {})
    expected_source_binding = {
        "executable_source_commit": source["executable_source_commit"],
        "executable_source_tree": source["executable_source_tree"],
        "source_snapshot_sha256": source["source_snapshot_sha256"],
    }
    actual_source_binding = {
        key: cache_source.get(key) for key in expected_source_binding
    }
    if actual_source_binding != expected_source_binding:
        raise RuntimeError(
            f"cache/source binding mismatch: expected={expected_source_binding} "
            f"actual={actual_source_binding}"
        )

    mask_audit = audit_effective_mask(entries)
    if not mask_audit["taxonomy"]["pass"]:
        raise RuntimeError(f"effective-mask contract failed: {mask_audit['issues'][:10]}")

    extra_keys = set()
    for entry in entries:
        extra_keys.update(set(entry) - ALLOWED_KEYS)
    if extra_keys:
        raise RuntimeError(f"cache contains non-deployment keys: {sorted(extra_keys)}")

    train = [entry for entry in entries if entry["fold_id"] != FOLD and entry["effective_mask"]]
    val = [entry for entry in entries if entry["fold_id"] == FOLD and entry["effective_mask"]]
    train_ids = sorted({entry["episode_id"] for entry in train})
    val_ids = sorted({entry["episode_id"] for entry in val})
    if set(train_ids) & set(val_ids):
        raise RuntimeError("train/validation identity overlap")
    if len(train) != 141_694 or len(val) != 37_980:
        raise RuntimeError(f"unexpected fold-0 sizes: train={len(train)} val={len(val)}")
    if len(train_ids) != 507 or len(val_ids) != 136:
        raise RuntimeError(
            f"unexpected fold-0 identity counts: train={len(train_ids)} val={len(val_ids)}"
        )

    x_train_cpu = torch.tensor(
        [entry["features_25d_raw"] for entry in train], dtype=torch.float32
    )
    y_train_cpu = torch.tensor(
        [entry["physical_target"] for entry in train], dtype=torch.float32
    )
    w_train_cpu = torch.tensor(
        [entry["D8_weight"] for entry in train], dtype=torch.float32
    )
    x_val_cpu = torch.tensor(
        [entry["features_25d_raw"] for entry in val], dtype=torch.float32
    )
    y_val_cpu = torch.tensor(
        [entry["physical_target"] for entry in val], dtype=torch.float32
    )
    w_val_cpu = torch.tensor(
        [entry["D8_weight"] for entry in val], dtype=torch.float32
    )
    if x_train_cpu.shape[1] != FEATURE_DIM or x_val_cpu.shape[1] != FEATURE_DIM:
        raise RuntimeError("cache feature dimension is not 25")

    train_id_digest = identity_digest(train_ids)
    val_id_digest = identity_digest(val_ids)
    norm = compute_normalization(
        x_train_cpu, source_identity_digest=train_id_digest
    )
    if norm["train_sample_count"] != len(train):
        raise RuntimeError("normalization train sample count mismatch")
    if norm["source_identity_digest"] != train_id_digest:
        raise RuntimeError("normalization train identity digest mismatch")

    norm_recomputed = compute_normalization(
        x_train_cpu.clone(), source_identity_digest=train_id_digest
    )
    x_val_mutated = x_val_cpu.clone()
    x_val_mutated[0, 0] += 100.0
    if norm_recomputed != norm:
        raise RuntimeError("train-only normalization is not deterministic")
    combined_norm = compute_normalization(
        torch.cat((x_train_cpu, x_val_mutated), dim=0),
        source_identity_digest="INTENTIONALLY_INVALID_COMBINED_SET",
    )
    if combined_norm["mean"] == norm["mean"]:
        raise RuntimeError("normalization leakage mutation test is non-discriminating")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    x_train = x_train_cpu.to(device)
    y_train = y_train_cpu.to(device)
    w_train = w_train_cpu.to(device)
    x_val = x_val_cpu.to(device)
    y_val = y_val_cpu.to(device)
    w_val = w_val_cpu.to(device)

    torch.manual_seed(SEED)
    epoch_losses = []
    first_grad_norm = None
    first_logits_finite = False
    first_grad_finite = False
    for _epoch in range(5):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(apply_normalization(x_train, norm))
        loss = compute_loss(logits, y_train, w_train)
        loss.backward()
        grad_sq = torch.zeros((), dtype=torch.float32, device=device)
        for parameter in model.parameters():
            if parameter.grad is not None:
                grad_sq = grad_sq + parameter.grad.detach().float().pow(2).sum()
        grad_norm = grad_sq.sqrt()
        if first_grad_norm is None:
            first_grad_norm = float(grad_norm.item())
            first_logits_finite = bool(torch.isfinite(logits).all().item())
            first_grad_finite = bool(torch.isfinite(grad_norm).item())
        epoch_losses.append(float(loss.detach().item()))
        optimizer.step()

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    staging.mkdir(parents=True, exist_ok=False)
    parity_dir = staging / "parity"
    parity_dir.mkdir()

    checkpoint_metadata = {
        "feature_schema_sha256": cache_manifest["script_provenance"]["feature_schema_sha256"],
        "source_snapshot_sha256": source["source_snapshot_sha256"],
        "executable_source_commit": source["executable_source_commit"],
        "executable_source_tree": source["executable_source_tree"],
    }
    roundtrip = checkpoint_roundtrip_parity(
        model,
        optimizer,
        x_val[:32],
        y_val[:32],
        w_val[:32],
        norm,
        parity_dir / "checkpoint_roundtrip.pt",
        device,
        checkpoint_metadata,
    )
    continuation = continuation_parity(
        model,
        optimizer,
        x_val[:32],
        y_val[:32],
        w_val[:32],
        norm,
        parity_dir / "checkpoint_continuation.pt",
        device,
        checkpoint_metadata,
    )

    model.eval()
    with torch.no_grad():
        val_logits = model(apply_normalization(x_val, norm))
        val_loss = compute_loss(val_logits, y_val, w_val)

    gates = {
        "source_snapshot_contract": True,
        "cache_source_binding": True,
        "cache_seal": True,
        "input_dim_25": x_train.shape[1] == FEATURE_DIM,
        "train_val_identity_disjoint": not bool(set(train_ids) & set(val_ids)),
        "norm_from_train_only": (
            norm["fit_on"] == "outer_training_fold_only"
            and norm["source_identity_digest"] == train_id_digest
            and combined_norm["mean"] != norm["mean"]
        ),
        "effective_mask_contract": mask_audit["taxonomy"]["pass"],
        "no_privileged_keys": not extra_keys,
        "finite_loss": all(np.isfinite(value) for value in epoch_losses),
        "finite_logits": first_logits_finite,
        "finite_gradients": first_grad_finite,
        "grad_nonzero": bool(first_grad_norm and first_grad_norm > 0.0),
        "loss_decreases": epoch_losses[-1] < epoch_losses[0],
        "checkpoint_restore": all(
            roundtrip[key]
            for key in (
                "pre_post_logits_match",
                "pre_post_loss_match",
                "params_match",
                "optimizer_match",
                "normalization_match",
            )
        ),
        "continuation_parity": all(
            continuation[key]
            for key in (
                "pre_step_logits_match",
                "pre_step_loss_match",
                "post_step_params_match",
                "post_step_optimizer_match",
                "post_step_logits_match",
                "post_step_loss_match",
                "post_step_rng_match",
            )
        ),
        "validation_completes": val_logits.shape[0] == len(val),
        "val_loss_finite": bool(torch.isfinite(val_loss).item()),
    }
    all_pass = all(gates.values())

    report = {
        "schema": "D8_P5_25D_GPU_SMOKE_V2",
        "status": "PASS_ENGINEERING_NONCONSUMABLE" if all_pass else "FAIL",
        "consumer_eligible": False,
        "run_label": run_label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_binding": expected_source_binding,
        "cache_binding": {
            "cache_root": str(cache_root),
            "cache_sha256sums_sha256": cache_seal["sha256sums_sha256"],
            "cache_manifest_sha256": sha256_file(cache_root / "CACHE_MANIFEST.json"),
        },
        "script_provenance": {
            "p5_script_sha256": sha256_file(Path(__file__)),
            "train_core_sha256": sha256_file(ROOT / "scripts/detector_v5/d8_train_core.py"),
            "source_contract_sha256": sha256_file(ROOT / "scripts/detector_v5/d8_source_contract.py"),
        },
        "environment": _environment(device),
        "command": list(sys.argv),
        "seed": SEED,
        "fold": FOLD,
        "feature_dim": FEATURE_DIM,
        "train_samples": len(train),
        "val_samples": len(val),
        "train_identities": len(train_ids),
        "val_identities": len(val_ids),
        "train_identity_digest": train_id_digest,
        "val_identity_digest": val_id_digest,
        "train_TRUE": int((y_train_cpu == 1.0).sum().item()),
        "train_FALSE": int((y_train_cpu == 0.0).sum().item()),
        "epoch_losses": epoch_losses,
        "val_loss": float(val_loss.item()),
        "first_grad_norm": first_grad_norm,
        "gates": dict(sorted(gates.items())),
        "all_gates_pass": all_pass,
        "mask_audit": mask_audit,
        "checkpoint_roundtrip": roundtrip,
        "continuation_parity": continuation,
        "test_reads": 0,
        "protected_reads": 0,
        "eval160_reads": 0,
    }
    (staging / "P5_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    checkpoint_sha = save_checkpoint(
        model,
        optimizer,
        5,
        len(train),
        norm,
        staging / "CHECKPOINT.pt",
        checkpoint_metadata,
    )
    (staging / "NORMALIZATION.json").write_text(
        json.dumps(norm, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (staging / "BATCH_SCHEMA.json").write_text(
        json.dumps(
            {
                "schema": "D8_BATCH_SCHEMA_V1",
                "feature_dim": FEATURE_DIM,
                "target": "physical_target",
                "weight": "D8_weight",
                "effective_mask_required": True,
                "train_fold_rule": f"fold_id != {FOLD}",
                "val_fold_rule": f"fold_id == {FOLD}",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    access = {
        "test_reads": 0,
        "protected_reads": 0,
        "eval160_reads": 0,
        "teacher_records_accessed": False,
        "sidecar_accessed": False,
        "relation_data_accessed": False,
        "telemetry_raw_accessed": False,
        "cache_only_training": True,
    }
    (staging / "ACCESS_AUDIT.json").write_text(
        json.dumps(access, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt = {
        "schema": "D8_P5_EXECUTION_RECEIPT_V2",
        "status": "COMPLETED" if all_pass else "FAILED_GATE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "return_code": 0 if all_pass else 1,
        "source_binding": expected_source_binding,
        "cache_binding": report["cache_binding"],
        "script_provenance": report["script_provenance"],
        "environment": report["environment"],
        "command": report["command"],
        "checkpoint_sha256": checkpoint_sha,
    }
    (staging / "EXECUTION_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    digest = _write_seal(staging)
    rename_noreplace(staging, output_root)
    report["sha256sums_sha256"] = digest
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-label", type=str, required=True)
    args = parser.parse_args()
    report = run_smoke(
        args.cache_root.resolve(strict=True),
        args.source_snapshot.resolve(strict=True),
        args.output_root,
        args.run_label,
    )
    for gate, passed in report["gates"].items():
        print(f"{gate}: {'PASS' if passed else 'FAIL'}")
    print(f"All gates: {'PASS' if report['all_gates_pass'] else 'FAIL'}")
    print(f"Seal: {report.get('sha256sums_sha256', 'UNSEALED')}")
    return 0 if report["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
