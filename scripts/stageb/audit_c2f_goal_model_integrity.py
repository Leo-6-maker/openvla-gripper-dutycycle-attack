#!/usr/bin/env python3
"""CPU/static integrity audit for the C2f Goal OpenVLA model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_stats_keys(model: Any) -> List[str]:
    stats = getattr(model, "norm_stats", None)
    if stats is None and hasattr(model, "config"):
        stats = getattr(model.config, "norm_stats", None)
    return sorted(str(k) for k in stats.keys()) if isinstance(stats, dict) else []


def resolve_goal_key(keys: List[str]) -> str:
    if "libero_goal" in keys:
        return "libero_goal"
    matches = [k for k in keys if "goal" in k.lower()]
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(f"Cannot resolve Goal unnorm_key from norm_stats keys: {keys}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/mnt/sdc/dty_user/openvla_attack/models/libero-goal")
    ap.add_argument("--report", default="reports/C2F_GOAL_MODEL_INTEGRITY_AUDIT_20260710.md")
    ap.add_argument("--manifest", default="artifacts/goal_model_manifest.json")
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    model_path = Path(args.model_path).resolve()
    files = [p for p in sorted(model_path.rglob("*")) if p.is_file()]
    file_rows: List[Dict[str, Any]] = []
    for p in files:
        file_rows.append({
            "path": str(p),
            "relative_path": str(p.relative_to(model_path)),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })

    index_path = model_path / "model.safetensors.index.json"
    referenced = []
    missing_referenced = []
    if index_path.exists():
        index = json.loads(index_path.read_text())
        referenced = sorted(set(index.get("weight_map", {}).values()))
        missing_referenced = [name for name in referenced if not (model_path / name).exists()]

    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForVision2Seq as AutoModelCls
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    import torch

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    model = AutoModelCls.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).eval()
    keys = norm_stats_keys(model)
    unnorm_key = resolve_goal_key(keys)

    manifest = {
        "status": "PASS_C2F_GOAL_MODEL_INTEGRITY_AUDITED" if not missing_referenced else "HOLD_C2F_GOAL_MODEL_SHARDS_MISSING",
        "model_path": str(model_path),
        "file_count": len(file_rows),
        "files": file_rows,
        "referenced_shards": referenced,
        "missing_referenced_shards": missing_referenced,
        "processor_class": type(processor).__name__,
        "model_class": type(model).__name__,
        "norm_stats_keys": keys,
        "unnorm_key": unnorm_key,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# C2F Goal Model Integrity Audit - 2026-07-10\n\n"
        f"STATUS: {manifest['status']}\n\n"
        "CPU/static audit only. No LIBERO env, rollout, intervention, attack, or GPU episode was launched.\n\n"
        f"- model_path: `{model_path}`\n"
        f"- file_count: {len(file_rows)}\n"
        f"- referenced_shards: {len(referenced)}\n"
        f"- missing_referenced_shards: {len(missing_referenced)}\n"
        f"- processor_class: `{manifest['processor_class']}`\n"
        f"- model_class: `{manifest['model_class']}`\n"
        f"- norm_stats_keys: `{keys}`\n"
        f"- resolved unnorm_key: `{unnorm_key}`\n\n"
        f"Manifest: `{manifest_path}`\n",
        encoding="utf-8",
    )
    print(json.dumps({k: manifest[k] for k in ["status", "file_count", "referenced_shards", "missing_referenced_shards", "norm_stats_keys", "unnorm_key"]}, indent=2, default=str))
    return 0 if not missing_referenced else 2


if __name__ == "__main__":
    raise SystemExit(main())