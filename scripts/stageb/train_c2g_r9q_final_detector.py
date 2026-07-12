"""Train the corrected R9Q B2 detector on an explicit FIT/CAL view."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from scripts.stageb import train_c2g_r9p_preview_detector as base
from tools.multisuite_detector.build_c2g_r9q_training_manifests import (
    sha256_file,
    read_jsonl,
)


POS_WEIGHT_CAP = 20.0
MODEL_SCHEMA = "c2g.r9q.final_detector_checkpoint.2026-07-13.v1"


def _git_state(expected_head: str) -> None:
    actual = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
    if actual != expected_head or dirty:
        raise RuntimeError(f"repository provenance mismatch: head={actual} dirty={bool(dirty)}")


def _verify_view(view_root: Path, expected_manifest_sha: str, expected_sums_sha: str | None = None) -> dict[str, Any]:
    sums = view_root / "SHA256SUMS"
    sidecar = view_root / "SHA256SUMS.sha256"
    if not sums.is_file() or not sidecar.is_file():
        raise FileNotFoundError("training view checksum closure is missing")
    if expected_sums_sha and sha256_file(sums) != expected_sums_sha:
        raise ValueError("training view SHA256SUMS mismatch")
    listed: set[str] = set()
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        name = name.strip()
        if name in listed or Path(name).name != name or name in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            raise ValueError(f"invalid or duplicate checksum entry: {name}")
        path = view_root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"training view checksum mismatch: {name}")
        listed.add(name)
    actual = {p.name for p in view_root.iterdir() if p.is_file()} - {"SHA256SUMS", "SHA256SUMS.sha256"}
    if actual != listed:
        raise ValueError(f"training view fileset mismatch: actual={actual ^ listed}")
    manifest_path = view_root / "training_manifest.json"
    if sha256_file(manifest_path) != expected_manifest_sha:
        raise ValueError("training manifest SHA mismatch")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _verify_source_closure(combined_root: Path, view_root: Path) -> dict[str, Any]:
    closure_path = view_root / "dataset_source_closure.json"
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    for item in closure["files"]:
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"source closure mismatch: {path}")
    if Path(closure["combined_root"]).resolve() != combined_root.resolve():
        raise ValueError("source closure combined root mismatch")
    return {"path": str(closure_path), "sha256": sha256_file(closure_path), "file_count": closure["source_file_count"]}


def compute_fit_class_balance(fit_rows: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    for head in base.R9P_HEAD_NAMES:
        counts[head] = {"positive_rows": 0, "negative_rows": 0, "positive_episodes": 0, "negative_episodes": 0}
    for row in fit_rows:
        data = np.load(Path(row["npz_path"]), allow_pickle=False)
        for head in base.R9P_HEAD_NAMES:
            mask = data[f"m_{head}"].astype(bool)
            target = data[f"y_{head}"]
            positive = mask & (target > 0.5)
            negative = mask & ~positive
            counts[head]["positive_rows"] += int(positive.sum())
            counts[head]["negative_rows"] += int(negative.sum())
            counts[head]["positive_episodes"] += int(positive.any())
            counts[head]["negative_episodes"] += int((not positive.any()) and mask.all())
    for head, item in counts.items():
        if item["positive_rows"] == 0 or item["negative_rows"] == 0:
            raise ValueError(f"no positive/negative FIT support for {head}: {item}")
        item["raw_pos_weight"] = item["negative_rows"] / item["positive_rows"]
        item["pos_weight"] = min(POS_WEIGHT_CAP, max(1.0, item["raw_pos_weight"]))
    return {
        "schema": "c2g.r9q.fit_class_balance.2026-07-13.v1",
        "source": "FIT_ONLY",
        "pos_weight_cap": POS_WEIGHT_CAP,
        "heads": counts,
    }


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    model_label: str,
    seed: int,
    epoch: int,
    config: dict[str, Any],
    provenance: dict[str, Any],
    history: list[dict[str, Any]],
) -> None:
    torch.save({
        "schema_version": MODEL_SCHEMA,
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "model_config": config,
        "model_label": model_label,
        "seed": seed,
        "epoch": epoch,
        "history": history,
        "provenance": provenance,
    }, path)


def train_b2(
    *,
    combined_root: Path,
    view_root: Path,
    output_root: Path,
    model_label: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    expected_head: str,
    expected_manifest_sha: str,
    device_str: str,
) -> dict[str, Any]:
    if model_label not in {"a", "b"}:
        raise ValueError("model_label must be a or b")
    _git_state(expected_head)
    manifest = _verify_view(view_root, expected_manifest_sha)
    if manifest["mode"] != "B2" or manifest["fit_count"] != 960:
        raise ValueError(f"B2 manifest mismatch: {manifest}")
    source_closure = _verify_source_closure(combined_root, view_root)
    fit_rows = read_jsonl(view_root / "fit_manifest.jsonl")
    cal_rows = read_jsonl(view_root / "cal_manifest.jsonl")
    if any(row["preview_split"] != "FIT" for row in fit_rows) or any(row["preview_split"] != "CAL" for row in cal_rows):
        raise ValueError("training manifests contain a forbidden split")
    if any(row["suite"] == "libero_10" for row in fit_rows) is False:
        raise ValueError("B2 FIT has no partial L10 support")
    class_balance = compute_fit_class_balance(fit_rows, combined_root)
    class_balance_path = output_root / f"b2_seed{seed}_class_balance.json"
    output_root.mkdir(parents=True, exist_ok=True)
    class_balance_path.write_text(json.dumps(class_balance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fit_manifest_sha = sha256_file(view_root / "fit_manifest.jsonl")
    cal_manifest_sha = sha256_file(view_root / "cal_manifest.jsonl")
    device = torch.device(device_str if device_str == "cpu" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds = base.R9PEpisodeDataset(fit_rows, combined_root)
    cal_ds = base.R9PEpisodeDataset(cal_rows, combined_root)
    if not train_ds or not cal_ds:
        raise ValueError("empty FIT or CAL dataset")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=base.collate_episodes)
    cal_loader = DataLoader(cal_ds, batch_size=batch_size, shuffle=False, collate_fn=base.collate_episodes)
    use_policy = model_label == "b"
    model_config = base.C2gDetectorConfig(
        visual_dim=base.VISUAL_DIM, language_dim=base.LANGUAGE_DIM, policy_intent_dim=9,
        hidden=128, dropout=0.1, use_policy_intent=use_policy,
        use_visual=False, use_language_conditioning=False, head_names=base.R9P_HEAD_NAMES,
    )
    model = base.C2gGripperCriticalWindowDetector(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    norm = base.load_normalization(combined_root)
    if norm is None:
        raise FileNotFoundError("combined normalization.json missing")
    p_mean = torch.from_numpy(norm["proprio_mean"]).to(device)
    p_std = torch.from_numpy(norm["proprio_std"]).to(device).clamp_min(1e-8)
    pi_mean = torch.from_numpy(norm["policy_intent_mean"]).to(device)
    pi_std = torch.from_numpy(norm["policy_intent_std"]).to(device).clamp_min(1e-8)
    pos_weights = {h: v["pos_weight"] for h, v in class_balance["heads"].items()}
    run_root = output_root / f"b2_seed{seed}"
    if run_root.exists():
        raise FileExistsError(f"training run exists: {run_root}")
    run_root.mkdir(parents=True)
    provenance = {
        "git_commit": expected_head,
        "combined_root": str(combined_root.resolve()),
        "dataset_index_sha256": sha256_file(combined_root / "dataset_index.jsonl"),
        "normalization_sha256": norm["sha256"],
        "training_manifest_sha256": expected_manifest_sha,
        "fit_manifest_sha256": fit_manifest_sha,
        "cal_manifest_sha256": cal_manifest_sha,
        "class_balance_sha256": sha256_file(class_balance_path),
        "source_closure_sha256": source_closure["sha256"],
        "source_closure_file_count": source_closure["file_count"],
        "check_reads": 0,
        "runtime_gate_contract": {
            "heads": ["critical_window", "release_safe", "grounding_confidence"],
            "positive_heads": ["critical_window", "grounding_confidence"],
            "veto_head": "release_safe",
            "burst_length": 10,
            "persistence_window": 3,
            "persistence_required": 2,
        },
        "language_embedding_contract": "NOT_USED_B2_25D_PLUS_9D",
    }
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        totals: list[float] = []
        for batch in train_loader:
            proprio_raw = batch["proprio_25d"].to(device)
            policy_raw = batch["policy_intent"].to(device)
            # B2 is explicitly 25D+9D.  Keep the detector interface shape but
            # disable language conditioning so missing language cannot leak in.
            language = torch.zeros(batch["language"].shape, device=device)
            outputs = model(
                (proprio_raw - p_mean) / p_std,
                language,
                policy_intent=((policy_raw - pi_mean) / pi_std) if use_policy else None,
                return_sequence=True,
            )
            pad_mask = batch["padding_mask"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}
            masks = {k: v.to(device) for k, v in batch["masks"].items()}
            for head in base.R9P_HEAD_NAMES:
                outputs[head] = outputs[head] * pad_mask.float()
            losses = base.r9p_preview_loss(
                outputs, targets, masks, sample_weight=pad_mask.float(),
                head_pos_weight=pos_weights,
                weight_start=base.LOSS_WEIGHTS["start"],
                weight_burst=base.LOSS_WEIGHTS["burst"],
                weight_critical=base.LOSS_WEIGHTS["critical"],
                weight_release=base.LOSS_WEIGHTS["release"],
                weight_contact=base.LOSS_WEIGHTS["contact"],
                weight_grounding=base.LOSS_WEIGHTS["grounding"],
                weight_early_emit=base.LOSS_WEIGHTS["early_emit"],
                weight_episode_miss=base.LOSS_WEIGHTS["episode_miss"],
                weight_negative_any_emit=base.LOSS_WEIGHTS["negative_any_emit"],
                weight_release_safe_emit=base.LOSS_WEIGHTS["release_safe_emit"],
            )
            loss = losses["total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for head in base.R9P_HEAD_NAMES:
                if not any(p.grad is not None for p in model.heads[head].parameters()):
                    raise FloatingPointError(f"missing gradient for head {head}")
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError("non-finite gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise FloatingPointError("non-finite parameter")
            totals.append(float(loss.detach().cpu()))
        cal_metrics = base._evaluate_model(model, cal_loader, device, use_policy, norm)
        if not all(np.isfinite(float(v)) for v in cal_metrics.values() if isinstance(v, (int, float))):
            raise FloatingPointError(f"non-finite CAL metrics at epoch {epoch}")
        history.append({"epoch": epoch, "train_loss": float(np.mean(totals)), "cal_metrics": cal_metrics})
        _save_checkpoint(
            run_root / f"epoch_{epoch:03d}.pt", model,
            model_label=model_label, seed=seed, epoch=epoch,
            config={
                "visual_dim": base.VISUAL_DIM, "language_dim": base.LANGUAGE_DIM,
                "policy_intent_dim": 9, "hidden": 128, "dropout": 0.1,
                "use_policy_intent": use_policy, "use_visual": False,
                "use_language_conditioning": False, "head_names": list(base.R9P_HEAD_NAMES),
            }, provenance=provenance, history=history,
        )
    _save_checkpoint(
        run_root / "checkpoint.pt", model, model_label=model_label, seed=seed,
        epoch=epochs, config={
            "visual_dim": base.VISUAL_DIM, "language_dim": base.LANGUAGE_DIM,
            "policy_intent_dim": 9, "hidden": 128, "dropout": 0.1,
            "use_policy_intent": use_policy, "use_visual": False,
            "use_language_conditioning": False, "head_names": list(base.R9P_HEAD_NAMES),
        }, provenance=provenance, history=history,
    )
    report = {
        "schema": MODEL_SCHEMA,
        "status": "PASS_C2G_R9Q_B2_TRAINING_RUN",
        "model_label": model_label, "seed": seed, "epochs": epochs,
        "fit_count": len(fit_rows), "cal_count": len(cal_rows),
        "history": history, "class_balance": class_balance,
        "checkpoint_sha256": sha256_file(run_root / "checkpoint.pt"),
        "provenance": provenance,
    }
    (run_root / "training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = [p for p in run_root.iterdir() if p.is_file()]
    (run_root / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in sorted(files, key=lambda p: p.name)), encoding="utf-8")
    sums_sha = sha256_file(run_root / "SHA256SUMS")
    (run_root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    report["sha256sums_sha256"] = sums_sha
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train corrected R9Q B2 detector")
    parser.add_argument("--combined-root", required=True, type=Path)
    parser.add_argument("--view-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model", choices=["a", "b"], default="b")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-manifest-sha", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        view = json.loads((args.view_root / "training_manifest.json").read_text(encoding="utf-8"))
        report = train_b2(
            combined_root=args.combined_root, view_root=args.view_root,
            output_root=args.output_root, model_label=args.model, seed=args.seed,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            expected_head=args.expected_head, expected_manifest_sha=args.expected_manifest_sha,
            device_str=args.device,
        )
    except Exception as exc:
        print(f"HOLD_C2G_R9Q_TRAINING: {exc}")
        return 1
    print(json.dumps({"mode": view.get("mode"), "run": report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
