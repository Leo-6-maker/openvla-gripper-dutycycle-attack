#!/usr/bin/env python3
"""Materialize C2f frozen RGB/language embeddings from observation-rich clean rollouts.

Input: C2f collection root written by scripts/stageb/collect_c2f_observation_clean_rollouts.py
Output: windowed NPZ dataset for C2f student training.

The script is intentionally deterministic and provenance-heavy. It does not read
D7B2 outcomes and does not run LIBERO/OpenVLA.

Embedding backends
------------------
1. `--backend clip` uses transformers CLIPModel/CLIPProcessor if available.
2. `--backend stats` uses deterministic lightweight RGB statistics + text hash.
   This is a fallback/smoke-test backend only; do not claim it as C2f final.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

EVENT_ROLE_TO_ID = {
    "primary_attackable": 0,
    "auxiliary_manipulation": 1,
    "distractor_or_setup": 2,
    "unsupported_or_abstain": 3,
}
SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_hash_vec(text: str, dim: int = 128) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().replace("/", " ").replace("_", " ").split()
    if not tokens:
        tokens = [text]
    for tok in tokens:
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    n = np.linalg.norm(vec)
    return vec / max(n, 1e-8)


def image_stats_embedding(path: Path, dim: int = 128) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((64, 64))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    feats = []
    feats.extend(arr.mean(axis=(0, 1)).tolist())
    feats.extend(arr.std(axis=(0, 1)).tolist())
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        feats.extend(np.quantile(arr.reshape(-1, 3), q, axis=0).tolist())
    # simple gradient energy
    gx = np.diff(arr, axis=1)
    gy = np.diff(arr, axis=0)
    feats.extend(np.mean(np.abs(gx), axis=(0, 1)).tolist())
    feats.extend(np.mean(np.abs(gy), axis=(0, 1)).tolist())
    base = np.asarray(feats, dtype=np.float32)
    out = np.zeros(dim, dtype=np.float32)
    out[: min(dim, len(base))] = base[:dim]
    if len(base) < dim:
        # deterministic expansion from path + statistics
        extra = stable_hash_vec(path.as_posix() + str(float(base.sum())), dim)
        out[len(base):] = extra[len(base):]
    return out


class Embedder:
    def __init__(self, backend: str, device: str, model_name: str, emb_dim: int,
                 openvla_model_path: str = ""):
        self.backend = backend
        self.device = device
        self.model_name = model_name
        self.emb_dim = emb_dim
        self.model = None
        self.processor = None
        self._openvla_text_emb = None  # cached Llama embedding layer
        if backend == "clip":
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor
            except Exception as e:
                raise RuntimeError("--backend clip requires torch + transformers") from e
            self.torch = torch
            self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
            self.processor = CLIPProcessor.from_pretrained(model_name)
        elif backend == "openvla_siglip":
            import torch
            self.torch = torch
            if not openvla_model_path:
                raise ValueError("--openvla-model-path required for backend=openvla_siglip")
            try:
                from transformers import AutoModelForVision2Seq as _AutoModelCls
            except ImportError:
                from transformers import AutoModelForImageTextToText as _AutoModelCls
            from transformers import AutoProcessor as _AutoProcessor
            self.model = _AutoModelCls.from_pretrained(
                openvla_model_path, trust_remote_code=True, local_files_only=True,
                torch_dtype=torch.bfloat16, device_map=device,
            ).eval()
            self.processor = _AutoProcessor.from_pretrained(
                openvla_model_path, trust_remote_code=True, local_files_only=True,
            )
            # Infer native vision embedding dimension from SigLIP config
            if hasattr(self.model, "vision_backbone") and hasattr(self.model.vision_backbone, "config"):
                self.emb_dim = self.model.vision_backbone.config.hidden_size
            else:
                self.emb_dim = 1152  # SigLIP ViT-SO400M default
        elif backend == "stats":
            pass
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def encode_image(self, path: Path) -> np.ndarray:
        if self.backend == "stats":
            return image_stats_embedding(path, self.emb_dim)
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.backend == "openvla_siglip":
            return self._encode_image_siglip(img)
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            emb = self.model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return emb.cpu().numpy()[0].astype(np.float32)

    def _encode_image_siglip(self, img) -> np.ndarray:
        """OpenVLA SigLIP vision backbone: preprocess → vision_backbone → pooled."""
        # Use the processor's image_processor (PrismaticImageProcessor)
        img_proc = self.processor.image_processor
        pixel_values = img_proc(images=img, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(device=self.device, dtype=self.torch.bfloat16)
        with self.torch.no_grad():
            outputs = self.model.vision_backbone(pixel_values)
            # OpenVLA's vision_backbone may return a plain Tensor or a model output
            if self.torch.is_tensor(outputs):
                emb = outputs  # already pooled [B, D]
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            elif hasattr(outputs, "last_hidden_state"):
                emb = outputs.last_hidden_state.mean(dim=1)
            else:
                emb = outputs[0]  # fallback: first element
            if emb.dim() > 2:
                emb = emb.mean(dim=1)  # mean pool if still 3D
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return emb.float().cpu().numpy()[0].astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        if self.backend == "stats":
            return stable_hash_vec(text, self.emb_dim)
        if self.backend == "openvla_siglip":
            return self._encode_text_openvla(text)
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            emb = self.model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return emb.cpu().numpy()[0].astype(np.float32)

    def _encode_text_openvla(self, text: str) -> np.ndarray:
        """OpenVLA Llama embedding layer: tokenize → embed → mean pool.

        This does NOT run the full language model; it only uses the frozen
        token embedding table.  It is a lightweight deterministic text encoder
        that lives in the same token space as the OpenVLA policy, without
        requiring a forward pass through the 7B-param LLM.
        """
        if self._openvla_text_emb is None:
            self._openvla_text_emb = self.model.language_model.get_input_embeddings()
        tokenizer = self.processor.tokenizer
        tokens = tokenizer(text, return_tensors="pt", padding=False, truncation=True,
                           max_length=64)
        input_ids = tokens["input_ids"].to(self.device)
        with self.torch.no_grad():
            tok_emb = self._openvla_text_emb(input_ids)  # [1, L, D]
            emb = tok_emb.float().mean(dim=1)  # mean pool over tokens
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return emb.cpu().numpy()[0].astype(np.float32)


def context_108d(suite: str, task_index: int) -> np.ndarray:
    vec = np.zeros(108, dtype=np.float32)
    if suite in SUITES:
        si = SUITES.index(suite)
        vec[si] = 1.0
        off = 68 + si * 10 + int(task_index)
        if 0 <= off < 108:
            vec[off] = 1.0
    key = f"{suite}|{task_index}"
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    vec[4 + (h % 32)] = 1.0
    vec[36 + ((h // 32) % 32)] = 1.0
    return vec


def collect_episodes(root: Path) -> List[Path]:
    return sorted((root / "episodes").rglob("episode_metadata.json"))


def make_split_keys(episode_ids: List[str], seed: int) -> Dict[str, str]:
    keys = sorted(set(episode_ids))
    rng = random.Random(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    split = {}
    for i, k in enumerate(keys):
        split[k] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return split


def main() -> int:
    ap = argparse.ArgumentParser(description="Materialize C2f frozen embedding dataset")
    ap.add_argument("--c2f-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--backend", choices=["clip", "stats", "openvla_siglip"], default="stats")
    ap.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    ap.add_argument("--openvla-model-path", default="",
                    help="Path to OpenVLA model dir for openvla_siglip backend")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--embedding-dim", type=int, default=128, help="Used by stats backend; CLIP dim inferred")
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--episode-offset", type=int, default=0, help="Skip first N episodes (for parallel sharding)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--git-commit", required=True)
    args = ap.parse_args()

    t0 = time.time()
    root = Path(args.c2f_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ep_meta_paths = collect_episodes(root)
    if args.episode_offset > 0:
        ep_meta_paths = ep_meta_paths[args.episode_offset:]
    if args.max_episodes > 0:
        ep_meta_paths = ep_meta_paths[: args.max_episodes]
    if not ep_meta_paths:
        raise RuntimeError(f"No C2f episodes found under {root / 'episodes'}")

    embedder = Embedder(args.backend, args.device, args.model_name, args.embedding_dim,
                        openvla_model_path=args.openvla_model_path)

    episode_ids = []
    episode_records = []
    for meta_path in ep_meta_paths:
        meta = json.loads(meta_path.read_text())
        eid = f"{meta['suite']}|{meta['parent_key']}"
        episode_ids.append(eid)
        episode_records.append((eid, meta_path, meta))
    split_by_episode = make_split_keys(episode_ids, args.seed)

    X_temporal, X_visual, X_language, X_context = [], [], [], []
    y_hazard, y_primary, y_release, y_role = [], [], [], []
    suite_rows, task_rows, episode_rows, step_rows, split_rows = [], [], [], [], []

    image_cache: Dict[str, np.ndarray] = {}
    text_cache: Dict[str, np.ndarray] = {}

    for eid, meta_path, meta in episode_records:
        ep_dir = meta_path.parent
        rows = read_jsonl(ep_dir / "step_records.jsonl")
        if len(rows) < args.window:
            continue
        text = str(meta.get("task_language") or rows[0].get("task_language", ""))
        if text not in text_cache:
            text_cache[text] = embedder.encode_text(text)
        lang_emb = text_cache[text]
        ctx = context_108d(str(meta["suite"]), int(meta.get("task_index", -1)))

        for end_idx in range(args.window - 1, len(rows)):
            win = rows[end_idx - args.window + 1 : end_idx + 1]
            feat = np.asarray([r["features_25d"] for r in win], dtype=np.float32)
            cur = rows[end_idx]
            rgb_rel = cur.get("rgb_path", "")
            rgb_path = (ep_dir / rgb_rel).resolve()
            key = rgb_path.as_posix()
            if key not in image_cache:
                image_cache[key] = embedder.encode_image(rgb_path)
            X_temporal.append(feat)
            X_visual.append(image_cache[key])
            X_language.append(lang_emb)
            X_context.append(ctx)
            y_hazard.append(int(cur.get("teacher_hazard", 0)))
            y_primary.append(int(cur.get("teacher_primary_attackable", 0)))
            y_release.append(int(cur.get("teacher_release_safe", 0)))
            y_role.append(EVENT_ROLE_TO_ID.get(str(cur.get("teacher_event_role", "unsupported_or_abstain")), 3))
            suite_rows.append(str(meta["suite"]))
            task_rows.append(int(meta.get("task_index", -1)))
            episode_rows.append(eid)
            step_rows.append(int(cur.get("step", end_idx)))
            split_rows.append(split_by_episode[eid])

    if not X_temporal:
        raise RuntimeError("No windows materialized")

    dataset_path = out / f"c2f_w{args.window:02d}_{args.backend}_dataset.npz"
    # Use float16 for large visual/language embeddings to keep NPZ manageable
    vis_dtype = np.float16 if args.backend == "openvla_siglip" else np.float32
    lang_dtype = np.float16 if args.backend == "openvla_siglip" else np.float32
    np.savez_compressed(
        dataset_path,
        X_temporal=np.asarray(X_temporal, dtype=np.float32),
        X_visual=np.asarray(X_visual, dtype=vis_dtype),
        X_language=np.asarray(X_language, dtype=lang_dtype),
        X_context=np.asarray(X_context, dtype=np.float32),
        y_hazard=np.asarray(y_hazard, dtype=np.int64),
        y_primary=np.asarray(y_primary, dtype=np.int64),
        y_release=np.asarray(y_release, dtype=np.int64),
        y_role=np.asarray(y_role, dtype=np.int64),
        suite=np.asarray(suite_rows),
        task_index=np.asarray(task_rows, dtype=np.int64),
        episode_id=np.asarray(episode_rows),
        step=np.asarray(step_rows, dtype=np.int64),
        split=np.asarray(split_rows),
    )

    report = {
        "gate": "C2F_FROZEN_EMBEDDING_MATERIALIZATION",
        "status": "PASS_MATERIALIZED",
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "n_windows": len(X_temporal),
        "n_episodes": len(set(episode_rows)),
        "window": args.window,
        "backend": args.backend,
        "model_name": args.model_name,
        "openvla_model_path": args.openvla_model_path if args.backend == "openvla_siglip" else "",
        "visual_dim": int(np.asarray(X_visual).shape[1]),
        "language_dim": int(np.asarray(X_language).shape[1]),
        "context_dim": 108,
        "created_at_unix": time.time(),
        "runtime_seconds": time.time() - t0,
        "git_commit": args.git_commit,
        "boundaries": {"attack": "NOT_PERFORMED", "d7b2_outcome_read": False,
                        "privileged_state": False, "post_attack_hidden_state": False},
    }
    write_json(out / "c2f_materialization_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
