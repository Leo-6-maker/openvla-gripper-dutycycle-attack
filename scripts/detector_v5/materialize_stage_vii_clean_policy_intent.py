#!/usr/bin/env python3
"""Replay sealed clean model inputs and export deployment-safe policy intent."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
SEAL_EXCLUDED = {"SHA256SUMS", "SHA256SUMS.sha256", "ROOT_SEAL.json", "ROOT_SEAL.sha256"}
COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def load_protocol(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schema") != "STAGE_VII_CLEAN_POLICY_INTENT_PROTOCOL_V1":
        raise ValueError("POLICY_PROTOCOL_SCHEMA_INVALID")
    if value.get("status") != "FROZEN_BEFORE_MATERIALIZATION":
        raise ValueError("POLICY_PROTOCOL_NOT_FROZEN")
    return value


def verify_official_source(protocol: dict[str, Any]) -> dict[str, Any]:
    binding = protocol["source_binding"]
    root = Path(binding["official_source_root"]).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"OFFICIAL_SOURCE_ROOT_INVALID:{root}")
    checks = {
        "root": str(root),
        "commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "tree": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True).strip(),
        "status": subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True),
        "files": {},
    }
    if checks["commit"] != binding["official_source_commit"] or checks["tree"] != binding["official_source_tree"] or checks["status"]:
        raise ValueError(f"OFFICIAL_SOURCE_BINDING_MISMATCH:{checks}")
    for relative, expected in binding["files"].items():
        path = root / relative
        actual = sha256_file(path) if path.is_file() else None
        checks["files"][relative] = {"path": str(path), "sha256": actual, "expected": expected}
        if actual != expected:
            raise ValueError(f"OFFICIAL_SOURCE_FILE_MISMATCH:{relative}:{actual}:{expected}")
    return checks


def load_snapshot(path: Path) -> dict[str, Any]:
    from gripper_attack.stage_v_causal_observation_snapshot import load_snapshot as loader

    package = loader(path.parent, materialize_torch=True)
    manifest = package["manifest"]
    payload = package["payload"]
    binding = manifest.get("binding") or {}
    for field in ("parent_key", "probe_id", "step", "source_commit", "source_tree"):
        if field not in binding:
            raise ValueError(f"SNAPSHOT_BINDING_MISSING:{field}:{path}")
    for field in ("input_ids", "pixel_values", "attention_mask", "decode_config", "prompt"):
        if field not in payload:
            raise ValueError(f"SNAPSHOT_INPUT_MISSING:{field}:{path}")
    if not bool(payload.get("attention_mask_present", True)):
        raise ValueError(f"SNAPSHOT_ATTENTION_MASK_MISSING:{path}")
    return {"manifest": manifest, "payload": payload, "path": path}


def register_openvla() -> None:
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


def load_bound_model(model_path: str, device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForVision2Seq

    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        device_map={"": int(device.split(":")[-1])},
    )
    model.eval()
    return model, torch.device(device)


def patch_predict_action(model: Any) -> None:
    import torch

    original = model.predict_action

    def consistent(*args: Any, **kwargs: Any) -> Any:
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        attention_mask = kwargs.get("attention_mask")
        if input_ids is not None and attention_mask is not None:
            if int(input_ids.shape[1]) == int(attention_mask.shape[1]) and not torch.all(input_ids[:, -1] == 29871):
                kwargs["attention_mask"] = torch.cat([attention_mask, torch.ones_like(attention_mask[:, :1])], dim=1)
        return original(*args, **kwargs)

    model.predict_action = consistent


def score_snapshot(package: dict[str, Any], model: Any, device: Any, adapter: Any) -> dict[str, Any]:
    import torch
    from gripper_attack.official_libero_protocol import decode_official_generated_action, generated_action_tokens

    payload = package["payload"]
    config = payload["decode_config"]
    key = str(config["unnorm_key"])
    expected = payload.get("controller_and_wrapper_runtime_state", {}).get("adapter_execution", {})
    if list(adapter.open_token_ids) != list(expected.get("open_token_ids", adapter.open_token_ids)):
        raise ValueError(f"OPEN_TOKEN_BINDING_MISMATCH:{package['path']}")
    if list(adapter.close_token_ids) != list(expected.get("close_token_ids", adapter.close_token_ids)):
        raise ValueError(f"CLOSE_TOKEN_BINDING_MISMATCH:{package['path']}")
    inputs = {name: payload[name].to(device=device) for name in ("input_ids", "pixel_values", "attention_mask")}
    captured: dict[str, Any] = {}
    original_generate = model.generate

    def capture(*args: Any, **kwargs: Any) -> Any:
        captured["count"] = int(captured.get("count", 0)) + 1
        kwargs["return_dict_in_generate"] = True
        kwargs["output_scores"] = True
        generation = original_generate(*args, **kwargs)
        captured["generation"] = generation
        return generation.sequences

    model.generate = capture
    try:
        action = model.predict_action(**inputs, unnorm_key=key, do_sample=False)
    finally:
        model.generate = original_generate
    generation = captured.get("generation")
    if generation is None:
        raise ValueError(f"GENERATION_CAPTURE_MISSING:{package['path']}")
    score_action = decode_official_generated_action(model, generation.sequences, key)
    action_error = float(np.max(np.abs(np.asarray(action, dtype=np.float32) - score_action)))
    scores = list(getattr(generation, "scores", []) or [])
    if captured.get("count") != 1 or len(scores) != 7 or action_error > 1e-6:
        raise ValueError(f"GENERATION_PARITY_FAIL:{package['path']}:{captured.get('count')}:{len(scores)}:{action_error}")
    intent, top_ids, top_logits = adapter.detector_policy_features(generation)
    score_summary = []
    for score in scores:
        probs = torch.softmax(score[0].float(), dim=-1)
        top_probability, top_id = torch.max(probs, dim=-1)
        score_summary.append({"top_token": int(top_id), "top_probability": float(top_probability)})
    binding = package["manifest"]["binding"]
    stage = "STAGE_VI_B2" if "STAGE_VI" in str(package["path"]) else "STAGE_V"
    return {
        "schema": "STAGE_VII_CLEAN_POLICY_INTENT_ROW_V1",
        "stage": stage,
        "canonical_parent_key": str(binding["parent_key"]),
        "probe_id": str(binding["probe_id"]),
        "probe_step": int(binding["step"]),
        "snapshot_path": str(package["path"]),
        "snapshot_manifest_sha256": sha256_file(package["path"]),
        "snapshot_source_commit": str(binding["source_commit"]),
        "snapshot_source_tree": str(binding["source_tree"]),
        "policy_input_sha256": str(payload.get("policy_input_sha256", "")),
        "model_path": str(config["base_vla_name"]),
        "unnorm_key": key,
        "clean_policy_intent_9d": intent,
        "action_token_ids": generated_action_tokens(model, generation, key),
        "clean_action_token_top_ids": [int(x) for x in top_ids],
        "clean_action_token_top_logits": [float(x) for x in top_logits],
        "score_head_summary": score_summary,
        "score_adapter_parity_pass": True,
        "single_generation_parity_pass": True,
        "generation_passes_per_step": 1,
        "labels_or_outcomes_read": False,
        "privileged_state_consumed": False,
    }


def seal(root: Path, summary: dict[str, Any]) -> None:
    entries = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in SEAL_EXCLUDED
    ]
    (root / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")
    sums_sha = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    seal_value = {
        "schema": "STAGE_VII_CLEAN_POLICY_INTENT_ROOT_SEAL_V1",
        "status": summary["status"],
        "summary_sha256": hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "sha256sums_sha256": sums_sha,
        "snapshot_count": summary["snapshot_count"],
        "labels_or_outcomes_read": False,
        "formal_m4_executed": False,
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    (root / "ROOT_SEAL.json").write_text(json.dumps(seal_value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "ROOT_SEAL.sha256").write_text(f"{sha256_file(root / 'ROOT_SEAL.json')}  ROOT_SEAL.json\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", action="append", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--official-source-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_root.exists():
        raise SystemExit(f"REFUSING_TO_OVERWRITE:{args.output_root}")
    protocol = load_protocol(args.protocol.resolve())
    source = verify_official_source(protocol)
    if args.official_source_root.resolve() != Path(protocol["source_binding"]["official_source_root"]).resolve():
        raise SystemExit("OFFICIAL_SOURCE_ARG_MISMATCH")
    snapshots = []
    for root in args.snapshot_root:
        snapshots.extend(load_snapshot(path) for path in sorted(root.resolve().glob("**/CAUSAL_PROBE_SNAPSHOT_V2.json")))
    if not snapshots:
        raise SystemExit("NO_SNAPSHOTS")
    import gripper_attack
    external_package = args.official_source_root.resolve() / "src" / "gripper_attack"
    gripper_attack.__path__.append(str(external_package))
    sys.path.insert(0, str(args.official_source_root.resolve() / "src"))
    register_openvla()
    from gripper_attack.official_openvla_adapter import OfficialOpenVLAActionAdapter
    models: dict[str, tuple[Any, Any, Any]] = {}
    rows = []
    for package in snapshots:
        config = package["payload"]["decode_config"]
        model_path = str(config["base_vla_name"])
        if model_path not in models:
            model, device = load_bound_model(model_path, args.device)
            patch_predict_action(model)
            adapter = OfficialOpenVLAActionAdapter(model, None, device, str(config["unnorm_key"]), center_crop=True, base_vla_name=model_path)
            models[model_path] = (model, device, adapter)
        model, device, adapter = models[model_path]
        rows.append(score_snapshot(package, model, device, adapter))
        if len(rows) % 32 == 0:
            print(json.dumps({"processed": len(rows), "total": len(snapshots)}, sort_keys=True), flush=True)
    output = args.output_root.resolve()
    output.mkdir(parents=True)
    (output / "POLICY_INTENT_ROWS.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "schema": "STAGE_VII_CLEAN_POLICY_INTENT_MATERIALIZATION_V1",
        "status": "PASS_STAGE_VII_CLEAN_POLICY_INTENT_MATERIALIZATION",
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "source_worktree_status": git("status", "--porcelain"),
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol.resolve()),
        "official_source": source,
        "snapshot_roots": [str(root.resolve()) for root in args.snapshot_root],
        "snapshot_count": len(rows),
        "row_count": len(rows),
        "stage_counts": dict(Counter(row["stage"] for row in rows)),
        "parent_count": len({row["canonical_parent_key"] for row in rows}),
        "model_paths": sorted(models),
        "all_generation_passes_one": all(row["generation_passes_per_step"] == 1 for row in rows),
        "all_single_generation_parity": all(row["single_generation_parity_pass"] is True for row in rows),
        "all_score_adapter_parity": all(row["score_adapter_parity_pass"] is True for row in rows),
        "labels_or_outcomes_read": False,
        "privileged_state_consumed": False,
        "formal_m4_executed": False,
        "protected_counters": COUNTERS,
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    (output / "STAGE_VII_CLEAN_POLICY_INTENT_MATERIALIZATION.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal(output, summary)
    print(json.dumps({"status": summary["status"], "output_root": str(output), "snapshot_count": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
