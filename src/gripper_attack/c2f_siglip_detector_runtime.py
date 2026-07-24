#!/usr/bin/env python3
"""C2f SigLIP detector deployment runtime for online canary.

Binds the trained C2fDetector checkpoint with OpenVLA's frozen vision_backbone
and Llama embedding layer for online feature extraction.  Follows the same
predict() interface as C2e3GRUDetectorRuntime where possible.

GPU required (OpenVLA vision_backbone inference).
"""
from __future__ import annotations

import hashlib, json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

CANONICAL_25D_FEATURES = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close", "eef_speed", "eef_z_delta_since_close",
    "qpos_delta_1", "qpos_delta_3", "opening_proxy_delta_3",
    "opening_proxy_variance_5", "eef_speed_variance_5",
]
SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]


class C2fSigLIPDetectorRuntime:
    """Online C2f detector: 25D GRU + SigLIP visual + Llama-emb language.

    Parameters
    ----------
    checkpoint_path : str
        Path to the C2fDetector .pt file.
    openvla_model : transformers model
        Loaded OpenVLA model. Must have ``vision_backbone`` and ``language_model``.
    openvla_processor : transformers processor
        OpenVLA processor (for image preprocessing and tokenization).
    device : str, default "cuda"
    window : int, default 16
    tau_emit : float, default 0.33
    tau_suppress : float, default 0.67
    """

    def __init__(
        self,
        checkpoint_path: str,
        openvla_model,
        openvla_processor,
        device: str = "cuda",
        window: int = 16,
        tau_emit: float = 0.33,
        tau_suppress: float = 0.67,
    ):
        self.device = device
        self.window = window
        self.tau_emit = tau_emit
        self.tau_suppress = tau_suppress
        self.tau_abstain = 0.5
        self.tau_primary = 0.5
        self._vla = openvla_model
        self._processor = openvla_processor
        self._text_emb = None  # cached Llama embedding layer

        # Load detector — checkpoint is a wrapper dict with model_state_dict + dims + config
        raw = torch.load(checkpoint_path, map_location="cpu")
        state = raw.get("model_state_dict", raw)
        dims = raw.get("dims", None)
        config = raw.get("config", {})
        if dims is None:
            raise KeyError("Checkpoint missing 'dims' key — cannot infer feature dimensions")
        nf = dims["temporal"]
        nv = dims["visual"]
        nl = dims["language"]
        nc = dims["context"]
        hidden = config.get("hidden", 128)
        proj = config.get("proj", 128)

        from tools.multisuite_detector.train_c2f_rgb_lang_temporal_detector_v0 import C2fDetector
        self._model = C2fDetector(nf=nf, nv=nv, nl=nl, nc=nc, hidden=hidden, proj=proj, dropout=0.0).to(device)
        self._model.load_state_dict(state)
        self._model.eval()

        # Buffers
        self._buffer_25d: List[np.ndarray] = []
        self._lang_emb_cache: Dict[str, np.ndarray] = {}
        self._context_cache: Dict[Tuple[str, int], np.ndarray] = {}

    def _context_108d(self, suite: str, task_index: int) -> np.ndarray:
        key = (suite, task_index)
        if key in self._context_cache:
            return self._context_cache[key]
        vec = np.zeros(108, dtype=np.float32)
        if suite in SUITES:
            si = SUITES.index(suite)
            vec[si] = 1.0
            off = 68 + si * 10 + int(task_index)
            if 0 <= off < 108:
                vec[off] = 1.0
        h = int(hashlib.sha256(f"{suite}|{task_index}".encode()).hexdigest(), 16)
        vec[4 + (h % 32)] = 1.0
        vec[36 + ((h // 32) % 32)] = 1.0
        self._context_cache[key] = vec
        return vec

    def _encode_text(self, text: str) -> np.ndarray:
        if text in self._lang_emb_cache:
            return self._lang_emb_cache[text]
        if self._text_emb is None:
            self._text_emb = self._vla.language_model.get_input_embeddings()
        tokenizer = self._processor.tokenizer
        tokens = tokenizer(text, return_tensors="pt", padding=False, truncation=True, max_length=64)
        input_ids = tokens["input_ids"].to(self.device)
        with torch.no_grad():
            emb = self._text_emb(input_ids).float().mean(dim=1)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        result = emb.cpu().numpy()[0].astype(np.float16)
        self._lang_emb_cache[text] = result
        return result

    def _encode_image(self, rgb: np.ndarray) -> np.ndarray:
        """Extract SigLIP visual features from an RGB frame."""
        from PIL import Image
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(rgb).convert("RGB")
        img_proc = self._processor.image_processor
        pixel_values = img_proc(images=img, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(device=self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            outputs = self._vla.vision_backbone(pixel_values)
            if torch.is_tensor(outputs):
                emb = outputs
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            else:
                emb = outputs.last_hidden_state.mean(dim=1)
            if emb.dim() > 2:
                emb = emb.mean(dim=1)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return emb.float().cpu().numpy()[0].astype(np.float16)

    def predict(self, features_25d: List[np.ndarray], rgb: np.ndarray,
                task_language: str, suite: str, task_index: int,
                ) -> Dict[str, Any]:
        """Predict detector outputs for the current step.

        Returns dict with emit_p, suppress_p, abstain_p, primary_p, role_logits,
        and emitted (bool per the full 4-condition gate matching offline evaluator).
        """
        if len(features_25d) < self.window:
            return {"emit_p": 0.0, "suppress_p": 1.0, "abstain_p": 0.0,
                    "primary_p": 0.0, "emitted": False, "ready": False}

        win = np.asarray(features_25d[-self.window:], dtype=np.float32)
        vis = self._encode_image(rgb).astype(np.float32)
        lang = self._encode_text(task_language).astype(np.float32)
        ctx = self._context_108d(suite, task_index)

        xt = torch.tensor(win[None, ...]).float().to(self.device)
        xv = torch.tensor(vis[None, ...]).float().to(self.device)
        xl = torch.tensor(lang[None, ...]).float().to(self.device)
        xc = torch.tensor(ctx[None, ...]).float().to(self.device)

        with torch.no_grad():
            out = self._model(xt, xv, xl, xc)

        emit_p = float(torch.sigmoid(out["emit"]).cpu().numpy()[0])
        supp_p = float(torch.sigmoid(out["suppress"]).cpu().numpy()[0])
        abstain_p = float(torch.sigmoid(out["abstain"]).cpu().numpy()[0])
        primary_p = float(torch.sigmoid(out["primary"]).cpu().numpy()[0])

        # Full gate matching offline evaluator:
        # emit >= tau_emit AND suppress <= tau_suppress AND abstain < tau_abstain AND primary >= tau_primary
        emitted = (
            emit_p >= self.tau_emit
            and supp_p <= self.tau_suppress
            and abstain_p < self.tau_abstain
            and primary_p >= self.tau_primary
        )
        return {"emit_p": emit_p, "suppress_p": supp_p, "abstain_p": abstain_p,
                "primary_p": primary_p, "emitted": emitted, "ready": True}

    def reset(self):
        """Clear per-episode buffers."""
        self._buffer_25d = []

    @property
    def is_ready(self) -> bool:
        return len(self._buffer_25d) >= self.window
