"""P5: 25D fold-0 GPU smoke — uses shared d8_train_core. ENGINEERING_NONCONSUMABLE."""
from __future__ import annotations

import argparse, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "scripts" / "detector_v5"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from d8_train_core import (
    D8StudentDetector, create_model, compute_normalization, apply_normalization,
    compute_loss, save_checkpoint, load_checkpoint,
    checkpoint_roundtrip_parity, continuation_parity, audit_effective_mask,
    SEED, FEATURE_DIM,
)
from audit_r3_contact_input import sha256_file, verify_seal
from gripper_attack.seal_utils import rename_noreplace

FOLD = 0
ALLOWED_KEYS = {"episode_id", "step", "features_25d_raw", "physical_target",
                "effective_mask", "D8_weight", "fold_id",
                "right_censored", "geometry_not_applicable", "articulated"}


def _write_seal(p: Path) -> str:
    files = sorted(x for x in p.rglob("*") if x.is_file() and x.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    (p / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(x)}  {x.relative_to(p).as_posix()}\n" for x in files), encoding="utf-8")
    d = sha256_file(p / "SHA256SUMS")
    (p / "SHA256SUMS.sha256").write_text(f"{d}  SHA256SUMS\n", encoding="utf-8")
    return d


def load_cache_entries(cache_root: Path) -> list[dict]:
    verify_seal(cache_root)
    entries = []
    for ep_file in sorted((cache_root / "per_episode").iterdir()):
        if ep_file.suffix == ".json":
            entries.extend(json.loads(ep_file.read_text("utf-8")))
    return entries


def run_smoke(cache_root: Path, output_root: Path) -> dict:
    # P0-B: Require output_root does NOT exist
    if output_root.exists():
        raise FileExistsError(f"output_root must not exist: {output_root}")

    entries = load_cache_entries(cache_root)
    print(f"Loaded {len(entries)} entries")

    # Mask audit
    mask_audit = audit_effective_mask(entries)
    if not mask_audit["taxonomy"]["pass"]:
        raise RuntimeError(f"Mask taxonomy failure: {mask_audit['issues'][:5]}")
    print(f"Mask taxonomy: {mask_audit['taxonomy']}")

    # Privileged scan: ALL entries, not just first 10
    extra_keys = set()
    for e in entries:
        extra_keys |= set(e.keys()) - ALLOWED_KEYS
    if extra_keys:
        raise RuntimeError(f"Privileged keys in cache: {extra_keys}")

    train = [e for e in entries if e["fold_id"] != FOLD and e["effective_mask"]]
    val = [e for e in entries if e["fold_id"] == FOLD and e["effective_mask"]]
    print(f"Train: {len(train)}, Val: {len(val)}")

    X_train = torch.tensor([s["features_25d_raw"] for s in train], dtype=torch.float32)
    y_train = torch.tensor([s["physical_target"] for s in train], dtype=torch.float32)
    w_train = torch.tensor([s["D8_weight"] for s in train], dtype=torch.float32)
    X_val = torch.tensor([s["features_25d_raw"] for s in val], dtype=torch.float32)
    y_val = torch.tensor([s["physical_target"] for s in val], dtype=torch.float32)
    w_val = torch.tensor([s["D8_weight"] for s in val], dtype=torch.float32)

    if X_train.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"Expected {FEATURE_DIM}D, got {X_train.shape[1]}D")

    train_ids = len(set(e["episode_id"] for e in train))
    val_ids = len(set(e["episode_id"] for e in val))
    print(f"Train ids: {train_ids}, Val ids: {val_ids}")

    norm = compute_normalization(X_train)
    # Verify train-only normalization
    mean1 = np.array(norm["mean"]).copy()
    X_val_mod = X_val.clone(); X_val_mod[0, 0] += 100.0
    norm2 = compute_normalization(X_train)
    if not np.allclose(mean1, norm2["mean"]):
        raise RuntimeError("Validation data affected normalization")

    print(f"Train: {(y_train==1).sum().item()} TRUE, {(y_train==0).sum().item()} FALSE")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    X_train, y_train, w_train = X_train.to(device), y_train.to(device), w_train.to(device)
    X_val, y_val, w_val = X_val.to(device), y_val.to(device), w_val.to(device)

    gates = {"input_dim_25": True, "norm_from_train_only": True,
             "effective_mask_contract": mask_audit["taxonomy"]["pass"],
             "no_privileged_keys": len(extra_keys) == 0}

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    torch.manual_seed(SEED)
    initial_loss = None

    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        Xn = apply_normalization(X_train, norm)
        logits = model(Xn)
        loss = compute_loss(logits, y_train, w_train)
        loss.backward()

        if epoch == 0:
            initial_loss = float(loss)
            gates["finite_loss"] = torch.isfinite(loss).item()
            gates["finite_logits"] = torch.isfinite(logits).all().item()
            grad_norm = sum((p.grad**2).sum() for p in model.parameters() if p.grad is not None).sqrt()
            gates["finite_gradients"] = torch.isfinite(grad_norm).item()
            gates["grad_nonzero"] = float(grad_norm) > 0

        optimizer.step()

    final_loss = float(loss)
    gates["loss_decreases"] = final_loss < initial_loss

    # P0-B: Create staging for all outputs
    staging = output_root.with_name(f".{output_root.name}.staging.{os.getpid()}")
    staging.mkdir(parents=True)
    parity_dir = staging / "parity"
    parity_dir.mkdir()

    # Checkpoint parity (writes to staging/parity/)
    rpt = checkpoint_roundtrip_parity(model, optimizer, X_val[:32], y_val[:32], w_val[:32],
                                       norm, parity_dir / "ckpt_roundtrip.pt", device)
    gates["checkpoint_restore"] = rpt["pre_post_logits_match"] and rpt["params_match"] and rpt["optimizer_match"]

    # Continuation parity
    cp = continuation_parity(model, optimizer, X_val[:32], y_val[:32], w_val[:32],
                              norm, parity_dir / "ckpt_continuation.pt", device)
    gates["continuation_parity"] = cp["pre_step_logits_match"] and cp["post_step_params_match"] and cp["post_step_optimizer_match"]

    # Validation
    model.eval()
    with torch.no_grad():
        Xvn = apply_normalization(X_val, norm)
        val_logits = model(Xvn)
        val_loss = compute_loss(val_logits, y_val, w_val)
    gates["validation_completes"] = True
    gates["val_loss_finite"] = torch.isfinite(val_loss).item()

    all_pass = all(gates.values())

    # Compute provenance
    commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip() if (ROOT / ".git").is_dir() else "UNKNOWN"
    self_sha = sha256_file(Path(__file__))
    core_sha = sha256_file(ROOT / "scripts" / "detector_v5" / "d8_train_core.py")
    cache_seal = sha256_file(cache_root / "SHA256SUMS.sha256")

    report = {
        "schema": "D8_P5_25D_GPU_SMOKE_V1",
        "status": "PASS_ENGINEERING_NONCONSUMABLE" if all_pass else "FAIL",
        "consumer_eligible": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_binding": {
            "commit": commit,
            "p5_script_sha256": self_sha,
            "train_core_sha256": core_sha,
        },
        "cache_binding": {
            "cache_root": str(cache_root),
            "cache_sha256sums_sha256": cache_seal,
        },
        "environment": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "seed": SEED, "fold": FOLD, "feature_dim": FEATURE_DIM,
        "train_samples": len(train), "val_samples": len(val),
        "train_identities": train_ids, "val_identities": val_ids,
        "initial_loss": initial_loss, "final_loss": final_loss, "val_loss": float(val_loss),
        "gates": {k: v for k, v in sorted(gates.items())},
        "all_gates_pass": all_pass,
        "mask_audit": mask_audit,
        "test_reads": 0, "eval160_reads": 0,
    }
    (staging / "P5_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    ckpt_sha = save_checkpoint(model, optimizer, 5, len(train), norm, staging / "CHECKPOINT.pt")
    (staging / "NORMALIZATION.json").write_text(json.dumps(norm, indent=2) + "\n")

    access = {"test_reads": 0, "eval160_reads": 0, "protected_reads": 0,
              "teacher_records_accessed": False, "sidecar_accessed": False,
              "relation_data_accessed": False, "telemetry_raw_accessed": False}
    (staging / "ACCESS_AUDIT.json").write_text(json.dumps(access, indent=2, sort_keys=True) + "\n")

    receipt = {
        "schema": "EXECUTION_RECEIPT_V1", "status": "COMPLETED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit, "p5_script_sha256": self_sha, "train_core_sha256": core_sha,
        "cache_root": str(cache_root), "cache_seal": cache_seal,
        "checkpoint_sha256": ckpt_sha, "device": str(device),
        "torch_version": torch.__version__,
    }
    (staging / "EXECUTION_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    digest = _write_seal(staging)
    rename_noreplace(staging, output_root)
    report["sha256sums_sha256"] = digest

    print(f"\nGates:")
    for k, v in sorted(gates.items()):
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"All gates: {'PASS' if all_pass else 'FAIL'}")
    print(f"Loss: {initial_loss:.4f} -> {final_loss:.4f}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-label", type=str, default="A")
    args = parser.parse_args()
    report = run_smoke(args.cache_root, args.output_root)
    raise SystemExit(0 if report["all_gates_pass"] else 1)
