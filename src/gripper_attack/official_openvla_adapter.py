"""Execution and score adapters for the pinned OpenVLA model."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .official_libero_protocol import (
    decode_official_generated_action,
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
        """Official execution path; optional capture observes without changing kwargs."""
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

        return self.predict_action_observed(image_np, task_label)

    def predict_action_observed(
        self, image_np: np.ndarray, task_label: str
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Observe the uninstrumented official call without changing kwargs."""
        inputs, prompt, processed_image = prepare_official_inputs(
            self.processor,
            image_np,
            task_label,
            self.device,
            center_crop=self.center_crop,
            base_vla_name=self.base_vla_name,
        )
        captured: dict[str, Any] = {}
        generation_passes = 0
        original_generate = self.model.generate

        def observe_generate(*args: Any, **kwargs: Any) -> Any:
            nonlocal generation_passes
            generation_passes += 1
            result = original_generate(*args, **kwargs)
            captured["generation"] = result
            return result

        self.model.generate = observe_generate
        try:
            action = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        finally:
            self.model.generate = original_generate

        if generation_passes != 1:
            raise RuntimeError(f"OFFICIAL_GENERATION_PASS_COUNT_FAIL:{generation_passes}")
        generation = captured.get("generation")
        if generation is None:
            raise RuntimeError("OFFICIAL_UNINSTRUMENTED_CAPTURE_MISSING")
        tokens = generated_action_tokens(self.model, generation, self.unnorm_key)
        return np.asarray(action, dtype=np.float32), {
            "inputs": inputs,
            "prompt": prompt,
            "processed_image": processed_image,
            "generation": generation,
            "tokens": tokens,
            "observed_official_kwargs": True,
            "generation_passes_per_step": generation_passes,
        }

    def predict_action_with_scores(
        self, image_np: np.ndarray, task_label: str
    ) -> tuple[np.ndarray, Any, dict[str, Any]]:
        """Run official ``predict_action`` once while capturing its generation.

        The wrapper returns ``generation.sequences`` to the upstream method, so
        execution still follows the official action path.  Scores and tokens
        are taken from that same generation; a second decode is not performed.
        """
        inputs, prompt, processed_image = prepare_official_inputs(
            self.processor,
            image_np,
            task_label,
            self.device,
            center_crop=self.center_crop,
            base_vla_name=self.base_vla_name,
        )
        captured: dict[str, Any] = {}
        generation_passes = 0
        original_generate = self.model.generate

        def capture_generate(*args: Any, **kwargs: Any) -> Any:
            nonlocal generation_passes
            generation_passes += 1
            kwargs["return_dict_in_generate"] = True
            kwargs["output_scores"] = True
            result = original_generate(*args, **kwargs)
            captured["generation"] = result
            return result.sequences

        self.model.generate = capture_generate
        try:
            action = self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=False)
        finally:
            self.model.generate = original_generate

        if generation_passes != 1:
            raise RuntimeError(f"OFFICIAL_GENERATION_PASS_COUNT_FAIL:{generation_passes}")
        generation = captured.get("generation")
        if generation is None or not hasattr(generation, "sequences"):
            raise RuntimeError("OFFICIAL_SINGLE_GENERATION_CAPTURE_MISSING")
        score_action = decode_official_generated_action(self.model, generation.sequences, self.unnorm_key)
        action = np.asarray(action, dtype=np.float32)
        action_error = float(np.max(np.abs(action - score_action)))
        if action_error > 1e-6:
            raise RuntimeError(f"SINGLE_GENERATION_ACTION_PARITY_FAIL:{action_error:.9g}")
        tokens = generated_action_tokens(self.model, generation, self.unnorm_key)
        scores = getattr(generation, "scores", None) or []
        return action, generation, {
            "inputs": inputs,
            "prompt": prompt,
            "processed_image": processed_image,
            "generation": generation,
            "score_action": np.asarray(score_action, dtype=np.float32),
            "captured_action_token_ids": tokens,
            "tokens": tokens,
            "captured_score_count": len(scores),
            "generation_passes_per_step": generation_passes,
            "single_generation_parity_pass": True,
            "score_adapter_action_max_abs_error": action_error,
            "single_generation": True,
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
