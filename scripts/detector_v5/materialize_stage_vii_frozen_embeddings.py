#!/usr/bin/env python3
"""Materialize fixed clean visual/language embeddings from sealed snapshots."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def model_inventory(path: Path) -> list[dict[str, Any]]:
    return [
        {"relative_path": item.relative_to(path).as_posix(), "size": item.stat().st_size, "sha256": sha256_file(item)}
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]


def load_model(model_path: Path, device: str) -> Any:
    import torch
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    for register, key, value in (
        (AutoConfig.register, "openvla", OpenVLAConfig),
        (AutoImageProcessor.register, OpenVLAConfig, PrismaticImageProcessor),
        (AutoProcessor.register, OpenVLAConfig, PrismaticProcessor),
        (AutoModelForVision2Seq.register, OpenVLAConfig, OpenVLAForActionPrediction),
    ):
        try:
            register(key, value)
        except ValueError:
            pass
    model = AutoModelForVision2Seq.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        device_map={"": int(device.split(":")[-1])},
    )
    model.eval()
    return model


def snapshot_rows(roots: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    from gripper_attack.stage_v_causal_observation_snapshot import load_snapshot

    result = []
    for root in roots:
        for manifest_path in sorted(root.glob("**/CAUSAL_PROBE_SNAPSHOT_V2.json")):
            package = load_snapshot(manifest_path.parent, materialize_torch=True)
            manifest = package["manifest"]
            payload = package["payload"]
            if manifest.get("status") != "SEALED_PROSPECTIVE_SNAPSHOT":
                raise ValueError(f"SNAPSHOT_NOT_SEALED:{manifest_path}")
            for field in ("pixel_values", "input_ids", "attention_mask"):
                if field not in payload:
                    raise ValueError(f"SNAPSHOT_FIELD_MISSING:{field}:{manifest_path}")
            result.append((manifest_path, {"manifest": manifest, "payload": payload}))
    if not result:
        raise ValueError("NO_SNAPSHOTS")
    return result


def materialize(model: Any, rows: list[tuple[Path, dict[str, Any]]], device: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    import torch

    visual = []
    language = []
    metadata = []
    for index, (path, package) in enumerate(rows):
        payload = package["payload"]
        pixel_values = payload["pixel_values"].to(device=device)
        input_ids = payload["input_ids"].to(device=device)
        attention_mask = payload["attention_mask"].to(device=device)
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        with torch.inference_mode():
            patch_features = model.vision_backbone(pixel_values)
            token_features = model.get_input_embeddings()(input_ids)
        mask = attention_mask.to(dtype=token_features.dtype).unsqueeze(-1)
        pooled_language = (token_features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled_visual = patch_features.mean(dim=1)
        visual.append(pooled_visual[0].float().cpu().numpy())
        language.append(pooled_language[0].float().cpu().numpy())
        binding = package["manifest"]["binding"]
        metadata.append({
            "row_index": index,
            "snapshot_path": str(path),
            "snapshot_manifest_sha256": sha256_file(path),
            "stage": "STAGE_VI_B2" if "STAGE_VI" in str(path) else "STAGE_V",
            "canonical_parent_key": str(binding["parent_key"]),
            "probe_id": str(binding["probe_id"]),
            "probe_step": int(binding["step"]),
        })
    return np.asarray(visual, dtype=np.float32), np.asarray(language, dtype=np.float32), metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", action="append", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"REFUSING_TO_OVERWRITE:{args.output_root}")
    protocol = load_json(args.protocol)
    if protocol.get("schema") != "STAGE_VII_FROZEN_CONTEXT_ENCODER_PROTOCOL_V1" or protocol.get("status") != "FROZEN_BEFORE_EMBEDDING_MATERIALIZATION":
        raise SystemExit("ENCODER_PROTOCOL_NOT_FROZEN")
    if str(args.model_path) != protocol["encoder"]["model_path"]:
        raise SystemExit("ENCODER_MODEL_BINDING_MISMATCH")
    rows = snapshot_rows([root.resolve() for root in args.snapshot_root])
    model = load_model(args.model_path.resolve(), args.device)
    visual, language, metadata = materialize(model, rows, args.device)
    args.output_root.mkdir(parents=True)
    np.savez_compressed(args.output_root / "FROZEN_CONTEXT_EMBEDDINGS.npz", visual=visual, language=language)
    (args.output_root / "ROWS.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in metadata), encoding="utf-8")
    inventory = model_inventory(args.model_path.resolve())
    summary = {
        "schema": "STAGE_VII_FROZEN_CONTEXT_EMBEDDINGS_V1",
        "status": "PASS_STAGE_VII_FROZEN_CONTEXT_EMBEDDINGS",
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "encoder_protocol": str(args.protocol.resolve()),
        "encoder_protocol_sha256": sha256_file(args.protocol.resolve()),
        "encoder_model_path": str(args.model_path.resolve()),
        "encoder_model_inventory": inventory,
        "snapshot_roots": [str(root.resolve()) for root in args.snapshot_root],
        "snapshot_count": len(metadata),
        "visual_shape": list(visual.shape),
        "language_shape": list(language.shape),
        "finite": bool(np.isfinite(visual).all() and np.isfinite(language).all()),
        "labels_or_outcomes_read": False,
        "suite_or_task_id_input": False,
        "formal_m4_executed": False,
        "protected_counters": {"protected_reads": 0, "eval160_reads": 0},
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    (args.output_root / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output_root": str(args.output_root), "snapshot_count": len(metadata), "visual_shape": visual.shape, "language_shape": language.shape}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
