#!/usr/bin/env python3
"""Export a frozen detector bundle for downstream attack validation.

Bundle includes: checkpoint, config, normalization, data contract, SHAs.
NO live paths. All paths must be relative to bundle root.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys
from pathlib import Path

BUNDLE_FILES = [
    "detector_config.json",
    "feature_contract.json",
    "data_contract.json",
    "normalization.json",
    "checkpoint.pt",
    "checkpoint_metadata.json",
    "FILES.json",
    "SHA256SUMS.txt",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Export frozen detector bundle")
    ap.add_argument("--checkpoint", required=True, help="Trained checkpoint .pt")
    ap.add_argument("--feature_contract", required=True, help="Feature contract JSON")
    ap.add_argument("--config", required=True, help="Detector config JSON/YAML")
    ap.add_argument("--output_dir", required=True, help="Bundle output directory")
    ap.add_argument("--metadata", help="Optional metadata JSON")
    args = ap.parse_args()

    out = Path(args.output_dir)
    if out.exists():
        sys.exit(f"Output exists, refusing to overwrite: {out}")
    out.mkdir(parents=True)

    ckpt_src = Path(args.checkpoint)
    shutil.copy2(ckpt_src, out / "checkpoint.pt")

    fc_src = Path(args.feature_contract)
    shutil.copy2(fc_src, out / "feature_contract.json")

    import torch
    ckpt = torch.load(str(ckpt_src), map_location="cpu", weights_only=False)
    meta = {
        "feature_names": ckpt.get("feature_names", []),
        "phase_classes": ckpt.get("phase_classes", []),
        "dataset_sha256": ckpt.get("dataset_sha256", ""),
        "n_train": ckpt.get("n_train", 0),
        "n_val": ckpt.get("n_val", 0),
        "seed": ckpt.get("seed", -1),
        "epoch": ckpt.get("epoch", -1),
        "val_loss": ckpt.get("val_loss", -1.0),
    }
    meta.update(json.loads(Path(args.metadata).read_text()) if args.metadata else {})
    with open(out / "checkpoint_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    config = json.loads(Path(args.config).read_text()) if args.config.endswith(".json") else {}
    config["checkpoint_sha256"] = sha256_file(out / "checkpoint.pt")
    config["feature_contract_sha256"] = sha256_file(out / "feature_contract.json")
    with open(out / "detector_config.json", "w") as f:
        json.dump(config, f, indent=2)

    files = {}
    for name in BUNDLE_FILES:
        p = out / name
        if p.exists():
            files[name] = sha256_file(p)
    with open(out / "FILES.json", "w") as f:
        json.dump({"files": files, "total": len(files)}, f, indent=2)

    with open(out / "SHA256SUMS.txt", "w") as f:
        for name, sha in sorted(files.items()):
            f.write(f"{sha}  {name}\n")

    print(f"Bundle exported to {out}")
    print(f"Files: {len(files)}")


if __name__ == "__main__":
    main()
