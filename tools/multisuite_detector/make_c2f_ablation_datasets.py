#!/usr/bin/env python3
"""Create A/B/C/D C2f modality-ablation NPZ datasets.

This utility avoids changing the trainer while enforcing clean ablation inputs.
It copies a materialized C2f dataset and zeros disabled modalities:

A_25d_only:          visual=0, language=0, context kept by default
B_25d_language:      visual=0, language kept, context kept
C_25d_rgb:           visual kept, language=0, context kept
D_full_rgb_language: visual kept, language kept, context kept

Context is kept by default because the existing C2e3/D8F detectors used suite/task
context. Pass --drop-context to also create no-context variants.

No LIBERO/OpenVLA runtime is used. No D7 outcome is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def save_variant(src: Dict[str, Any], out_path: Path, *, use_visual: bool, use_language: bool, use_context: bool) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: src[k] for k in src.keys()}
    arrays = dict(arrays)
    if not use_visual:
        arrays["X_visual"] = np.zeros_like(arrays["X_visual"], dtype=np.float32)
    if not use_language:
        arrays["X_language"] = np.zeros_like(arrays["X_language"], dtype=np.float32)
    if not use_context:
        arrays["X_context"] = np.zeros_like(arrays["X_context"], dtype=np.float32)
    np.savez_compressed(out_path, **arrays)
    return {
        "path": str(out_path),
        "sha256": sha256_file(out_path),
        "use_visual": use_visual,
        "use_language": use_language,
        "use_context": use_context,
        "n_windows": int(arrays["X_temporal"].shape[0]),
        "temporal_shape": list(arrays["X_temporal"].shape),
        "visual_shape": list(arrays["X_visual"].shape),
        "language_shape": list(arrays["X_language"].shape),
        "context_shape": list(arrays["X_context"].shape),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Create C2f A/B/C/D modality ablation datasets")
    ap.add_argument("--dataset", required=True, help="NPZ produced by materialize_c2f_frozen_embeddings.py")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--drop-context", action="store_true", help="also zero 108D context in all variants")
    ap.add_argument("--git-commit", required=True)
    args = ap.parse_args()

    t0 = time.time()
    src_path = Path(args.dataset)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz = np.load(src_path, allow_pickle=True)
    src = {k: npz[k] for k in npz.files}

    required = ["X_temporal", "X_visual", "X_language", "X_context", "y_hazard", "y_primary", "split"]
    missing = [k for k in required if k not in src]
    if missing:
        raise RuntimeError(f"Dataset missing required arrays: {missing}")

    use_context = not args.drop_context
    variants = {
        "A_25d_only": dict(use_visual=False, use_language=False, use_context=use_context),
        "B_25d_language": dict(use_visual=False, use_language=True, use_context=use_context),
        "C_25d_rgb": dict(use_visual=True, use_language=False, use_context=use_context),
        "D_full_rgb_language": dict(use_visual=True, use_language=True, use_context=use_context),
    }
    summaries = {}
    for name, cfg in variants.items():
        summaries[name] = save_variant(src, out / f"{name}.npz", **cfg)

    report = {
        "gate": "C2F_MODALITY_ABLATION_DATASETS",
        "status": "PASS_WRITTEN",
        "source_dataset": str(src_path),
        "source_dataset_sha256": sha256_file(src_path),
        "output_dir": str(out),
        "drop_context": bool(args.drop_context),
        "variants": summaries,
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - t0,
        "git_commit": args.git_commit,
        "boundaries": {
            "attack": "NOT_PERFORMED",
            "LIBERO_runtime": "NOT_PERFORMED",
            "D7_Table1": "NOT_MODIFIED",
            "d7b2_outcome_read": False,
        },
    }
    write_json(out / "c2f_ablation_datasets_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
