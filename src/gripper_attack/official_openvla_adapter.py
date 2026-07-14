"""Execution and score adapters for the pinned OpenVLA model."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .official_libero_protocol import (
    generated_action_tokens,
    official_predict_action,
    postprocess_official_action,
    prepare_official_inputs,
    score_official_action,
)
from .official_detector_features import (
    CLEAN_POLICY_FEATURE_NAMES,
    derive_gripper_token_semantics,
    policy_intent_9d,
    top_token_evidence,
)


class OfficialOpenVLAActionAdapter:
    """Keep execution on ``predict_action`` and expose a score-only path."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: torch.device | str,
        unnorm_key: str,
        *,
        center_crop: bool = True,
        base_vla_name: str = "",
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.unnorm_key = unnorm_key
        self.center_crop = bool(center_crop)
        self.base_vla_name = base_vla_name
        semantics = derive_gripper_token_semantics(model, unnorm_key)
        self.open_token_ids = tuple(semantics["open_token_ids"])
        self.close_token_ids = tuple(semantics["close_token_ids"])
        self.token_action_map = dict(semantics["token_action_map"])

    def predict_action(self, image_np: np.ndarray, task_label: str, *, capture: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        """Official execution path. ``capture`` is parity instrumentation only."""
        if not capture:
            return official_predict_action(
                self.model,
                self.processor,
                image_np,
                task_label,
                self.unnorm_key,
                self.device,
                center_crop=self.center_crop,
                base_vla_name=self.base_vla_name,
            )

        inputs, prompt, processed_image = prepare_official_inputs(
            self.processor,
            image_np,
            task_label,
            self.device,
            center_crop=self.center_crop,
            base_vla_name=self.base_vla_name,
        )
        captured: dict[str, Any] = {}
        original_generate = self.model.generate

        def capture_generate(*args: Any, **kwargs: Any) -> Any:
            result = original_generate(*args, **kwargs)
            captured["generation"] = result
            return result

        self.model.generate = capture_generate
        try:
            action = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        finally:
            self.model.generate = original_generate

        generation = captured.get("generation")
        return np.asarray(action, dtype=np.float32), {
            "inputs": inputs,
            "prompt": prompt,
            "processed_image": processed_image,
            "generation": generation,
            "tokens": generated_action_tokens(self.model, generation, self.unnorm_key) if generation is not None else [],
        }

    def score_action(self, image_np: np.ndarray, task_label: str) -> tuple[np.ndarray, Any, dict[str, Any]]:
        """Same official inputs/de-tokenization, with generation scores exposed."""
        return score_official_action(
            self.model,
            self.processor,
            image_np,
            task_label,
            self.unnorm_key,
            self.device,
            center_crop=self.center_crop,
            base_vla_name=self.base_vla_name,
        )

    def postprocess(self, action: np.ndarray) -> np.ndarray:
        return postprocess_official_action(action)

    def detector_policy_features(self, generation: Any) -> tuple[list[float], list[int], list[float]]:
        """Return frozen 9D intent plus compact token evidence for one step."""
        scores = getattr(generation, "scores", None) or []
        if not scores:
            raise RuntimeError("official score adapter returned no generation scores")
        logits = scores[-1][0].detach()
        intent = policy_intent_9d(
            logits,
            open_token_ids=self.open_token_ids,
            close_token_ids=self.close_token_ids,
        )
        top_ids, top_logits = top_token_evidence(logits)
        return intent, [int(x) for x in top_ids], [float(x) for x in top_logits]

    def forward_action_logits(self, score_meta: dict[str, Any], generation: Any) -> torch.Tensor:
        """Return teacher-forced action-token logits for later PGD objectives."""
        inputs = dict(score_meta["inputs"])
        input_ids = inputs["input_ids"]
        prompt_len = int(input_ids.shape[1])
        full_ids = generation.sequences
        inputs["input_ids"] = full_ids
        if "attention_mask" in inputs and int(inputs["attention_mask"].shape[1]) != int(full_ids.shape[1]):
            # The pinned upstream processor normally omits this field. If a
            # future processor emits it, extend it only for teacher forcing;
            # execution remains exactly the upstream predict_action call.
            inputs["attention_mask"] = torch.ones_like(full_ids, dtype=inputs["attention_mask"].dtype)
        outputs = self.model(**inputs, use_cache=False)
        action_dim = int(self.model.get_action_dim(self.unnorm_key))
        start = max(0, prompt_len - 1)
        return outputs.logits[:, start : start + action_dim, :]
