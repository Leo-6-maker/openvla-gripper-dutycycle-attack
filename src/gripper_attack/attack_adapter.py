from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .types import AttackResult
from .execution_target import (
    target_token_cw_loss_and_stats,
    target_token_logratio_loss_and_stats,
    validate_execution_target,
)
from .m3_controls import (
    project_and_cast_processor_values,
    shuffled_grad_direction,
    tensor_sha256,
)
from .gripper_semantics import (
    raw_gripper_is_open,
    raw_gripper_is_close,
    raw_gripper_is_boundary,
    env_gripper_is_open,
    env_gripper_is_close,
    CANONICAL_OPEN_SEMANTICS_VERSION,
)
from .openvla_libero_exec_spec import validate_open_close_token_sets
from .route_contract import (
    RouteContractError,
    attach_route_debug,
    resolve_adapter_class_name,
    route_config_from_attack_config,
    validate_attack_request,
    validate_true_pgd_attack_result,
)


def _prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {str(instruction).lower()}?\nOut:"


def _infer_model_dtype(model) -> torch.dtype:
    """Return the dtype used by model parameters, falling back to fp32 for mocks."""
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def get_adv_inputs_from_attack_result(result: AttackResult) -> Dict[str, Any]:
    """Return TokenPrefixPGD adversarial inputs or raise a clear error.

    TokenPrefixPGDAttacker intentionally returns ``action_adv=None`` because the
    final OpenVLA action must be decoded from adversarial processor inputs. This
    helper prevents callers from silently treating ``None`` as a zero action.
    """
    if not isinstance(result, AttackResult):
        raise TypeError("expected AttackResult")
    adv_inputs = (result.debug or {}).get("adv_inputs")
    if not isinstance(adv_inputs, dict):
        raise ValueError("AttackResult is missing debug['adv_inputs']; re-decode cannot proceed")
    missing = [key for key in ("input_ids", "pixel_values") if key not in adv_inputs]
    if missing:
        raise ValueError(f"AttackResult debug['adv_inputs'] missing required keys: {missing}")
    return adv_inputs


def _pil_center_crop_resize(image: Image.Image, crop_scale: float = 0.9, size: int = 224) -> Image.Image:
    if crop_scale is None or float(crop_scale) >= 0.999:
        return image.resize((size, size), Image.Resampling.LANCZOS)
    w, h = image.size
    scale = float(crop_scale) ** 0.5
    cw, ch = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    left, top = (w - cw) // 2, (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch)).resize((size, size), Image.Resampling.LANCZOS)


def prepare_openvla_image_for_attack(image_np, *, libero_official_preprocess: bool = False, center_crop: bool = False, resize_size: int = 224, libero_preprocess_backend: str = "official_pil_lanczos", **kwargs) -> Image.Image:
    from gripper_attack.openvla_preprocess import prepare_openvla_image
    return prepare_openvla_image(image_np, libero_official_preprocess=libero_official_preprocess,
                                 center_crop=center_crop, resize_size=resize_size,
                                 libero_preprocess_backend=libero_preprocess_backend)


def action_token_logit_row_index(dim: int, action_dim: int) -> int:
    """Return the logit row index (from end) that predicts action token *dim*.

    In a causal LM, the logit at position ``t`` predicts the token at position
    ``t+1``.  With *action_dim* action tokens appended to the input sequence,
    the logit predicting action token *dim* (0-indexed from the start of the
    action prefix) is located at::

        row_index = -(action_dim - dim + 1)

    Examples for action_dim=7:
        dim=0  →  -8   (first arm dim)
        dim=5  →  -3   (last arm dim)
        dim=6  →  -2   (gripper — the final action token)

    ``logits[0, -1, :]`` predicts whatever token follows the action prefix
    (usually an EOS or continuation token) — it is NOT the gripper row.
    """
    return -(int(action_dim) - int(dim) + 1)


class ExistingDenseAttackAdapter:
    # Visual-only fallback adapter. It never edits actuator commands.
    def __init__(self, epsilon: float = 0.03, step_size: float = 0.006, num_steps: int = 5, seed: int = 0):
        self.epsilon = float(epsilon); self.step_size = float(step_size); self.num_steps = int(num_steps); self.seed = int(seed)

    def attack(self, observation: Any, instruction=None, clean_action=None, target_action=None, clean_model_output=None) -> AttackResult:
        x = np.asarray(observation).copy()
        orig_dtype = x.dtype
        rng = np.random.RandomState(self.seed)
        if np.issubdtype(x.dtype, np.integer):
            scale = 255.0; xf = x.astype(np.float32) / 255.0
        else:
            scale = 1.0; xf = x.astype(np.float32)
        pattern = rng.choice([-1.0, 1.0], size=xf.shape).astype(np.float32)
        xadv = np.clip(xf + self.epsilon * pattern, 0.0, 1.0)
        diff = xadv - xf
        out = (xadv * scale).round().astype(orig_dtype) if scale == 255.0 else xadv.astype(orig_dtype)
        return AttackResult(
            x_adv=out,
            action_adv=None,
            attack_method="visual_linf_noise_adapter",
            directional_loss_available=False,
            num_attack_steps=self.num_steps,
            epsilon=self.epsilon,
            step_size=self.step_size,
            observation_perturb_linf=float(np.max(np.abs(diff))) if diff.size else 0.0,
            observation_perturb_l2=float(np.linalg.norm(diff.reshape(-1))) if diff.size else 0.0,
            debug={"fallback_reason": "gradient directional/token loss not available; visual perturbation only"},
        )


class TokenPrefixPGDAttacker:
    """White-box visual PGD on OpenVLA action-token prefix CE.

    It never edits actuator commands.  It optimizes the processor pixel_values so
    the autoregressive action-token prefix is more likely to match the tokenized
    directional target action, then the caller performs adversarial re-decode.

    ``attack()`` returns ``action_adv=None`` by design. Callers must re-decode
    OpenVLA from ``result.debug["adv_inputs"]``, which contains ``input_ids`` and
    adversarial ``pixel_values``. Use ``get_adv_inputs_from_attack_result`` to
    validate that interface; never fallback from ``action_adv=None`` to zeros.
    """
    def __init__(self, model, processor, config: dict, seed: int = 0, preprocess_kwargs: Optional[Dict[str, Any]] = None, device: Optional[str] = None):
        cfg = (config or {}).get("attack_optimizer", config or {})
        self.model = model
        self.processor = processor
        self.epsilon = float(cfg.get("epsilon", 0.03))
        self.step_size = float(cfg.get("step_size", max(self.epsilon / max(int(cfg.get("num_steps", 5)), 1), 1e-4)))
        self.num_steps = int(cfg.get("num_steps", 5))
        self.random_start = bool(cfg.get("random_start", False))
        self.temporal_init = str(cfg.get("temporal_init", "none") or "none").strip().lower()
        self.temporal_smooth_lambda = float(cfg.get("temporal_smooth_lambda", 0.0) or 0.0)
        self._prev_delta = None
        self.objective = str(cfg.get("objective", cfg.get("loss_objective", "targeted_directional_ce")))
        self.loss_weights = None
        _raw_weights = cfg.get("loss_weights", None)
        if _raw_weights is not None and isinstance(_raw_weights, dict):
            self.loss_weights = {int(k) if k.lstrip('-').isdigit() else k: float(v) for k, v in _raw_weights.items()}
        self.arm_preserve_weight = float(cfg.get("arm_preserve_weight", 0.1))
        self.gripper_margin = float(cfg.get("gripper_margin", 5.0))
        self.target_token_id = None if cfg.get("target_token_id", cfg.get("target_token")) is None else int(cfg.get("target_token_id", cfg.get("target_token")))
        self.target_execution_class = None if cfg.get("target_execution_class") is None else str(cfg.get("target_execution_class"))
        self.prefix_refresh_interval = max(1, int(cfg.get("prefix_refresh_interval", 1) or 1))
        self.surrogate_score_path = str(cfg.get("surrogate_score_path", "uncached_full_context_v1") or "uncached_full_context_v1").strip().lower()
        self.gradient_transform = str(cfg.get("gradient_transform", "none") or "none").strip().lower()
        self.gradient_transform_seed = int(cfg.get("gradient_transform_seed", seed))
        self.arm_isolation_candidate_policy = str(cfg.get("arm_isolation_candidate_policy", "FINAL_ONLY") or "FINAL_ONLY").strip()
        self.best_restart_metric = str(cfg.get("best_restart_metric", "target_ce_final"))
        self.seed = int(seed)
        self.preprocess_kwargs = dict(preprocess_kwargs or {})
        self.postprocess_gripper = bool(self.preprocess_kwargs.pop("postprocess_gripper", False))
        self.device = device or "cuda:0"
        self.config = cfg
        self._frozen = False
        self.last_attack_diagnostics: dict[str, Any] = {}

    def reset_temporal_state(self):
        self._prev_delta = None

    def _freeze_model(self):
        if self._frozen or self.model is None:
            return
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._frozen = True

    def _resolve_unnorm_key(self, unnorm_key: str) -> str:
        try:
            keys = list(getattr(self.model, "norm_stats", {}).keys())
            if unnorm_key in keys:
                return unnorm_key
            if keys:
                return str(keys[0])
        except Exception:
            pass
        return unnorm_key

    def _action_stats(self, unnorm_key: str):
        key = self._resolve_unnorm_key(unnorm_key)
        try:
            return self.model.get_action_stats(key), key
        except AssertionError:
            keys = list(getattr(self.model, "norm_stats", {}).keys())
            if keys:
                return self.model.get_action_stats(str(keys[0])), str(keys[0])
            raise

    def action_to_token_ids(self, action, unnorm_key: str) -> torch.LongTensor:
        action = np.asarray(action, dtype=np.float32)
        stats, unnorm_key = self._action_stats(unnorm_key)
        mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)), dtype=bool)
        low = np.asarray(stats["q01"], dtype=np.float32)
        high = np.asarray(stats["q99"], dtype=np.float32)
        denom = np.maximum(high - low, 1e-6)
        norm = np.where(mask, 2.0 * (action - low) / denom - 1.0, action)
        norm = np.clip(norm, -1.0, 1.0)
        centers = np.asarray(self.model.bin_centers, dtype=np.float32)
        disc = np.abs(norm[:, None] - centers[None, :]).argmin(axis=1)
        vocab_size = int(self.model.config.text_config.vocab_size - self.model.config.pad_to_multiple_of)
        token_ids = vocab_size - disc - 1
        return torch.tensor(token_ids, dtype=torch.long, device=self.device)

    def _build_inputs_and_labels(self, observation, instruction: str, target_token_ids: torch.LongTensor):
        image = prepare_openvla_image_for_attack(observation, **self.preprocess_kwargs)
        inputs = self.processor(_prompt(instruction), image, return_tensors="pt")
        inputs.pop("attention_mask", None)  # match working OpenVLA generation path
        input_ids = inputs["input_ids"].to(self.device)
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat([input_ids, torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)], dim=1)
        target = target_token_ids.view(1, -1).to(input_ids.device)
        full_input_ids = torch.cat([input_ids, target], dim=1)
        labels = torch.full_like(full_input_ids, -100)
        labels[:, -target.shape[1]:] = target
        model_dtype = _infer_model_dtype(self.model)
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=model_dtype)
        return input_ids, full_input_ids, labels, pixel_values

    def _project_pixel_master(self, adv_float: torch.Tensor, x_orig_float: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(adv_float, x_orig_float + self.epsilon), x_orig_float - self.epsilon)

    def _cast_projected_pixel_values(self, adv_float: torch.Tensor, x_orig_model: torch.Tensor) -> torch.Tensor:
        """Project in fp32, then cast to model dtype without violating Linf budget.

        OpenVLA often runs in bf16/fp16, but the perturbation budget is checked
        in processor pixel-value space.  Casting a value at the fp32 boundary to
        bf16 can round outside ``x_orig +/- epsilon``.  Any such rounded element
        is reset to the original model-dtype pixel value so the actual
        ``debug["adv_inputs"]["pixel_values"]`` tensor remains budget-valid.
        """

        casted, _ = project_and_cast_processor_values(
            x_orig_model,
            adv_float,
            epsilon=float(self.epsilon),
            candidate_is_delta=False,
        )
        return casted

    def _count_quantized_budget_corrections(self, adv_model: torch.Tensor, adv_float: torch.Tensor, x_orig_model: torch.Tensor) -> int:
        _casted, correction_count = project_and_cast_processor_values(
            x_orig_model,
            adv_float.detach(),
            epsilon=float(self.epsilon),
            candidate_is_delta=False,
        )
        return int(correction_count)

    def action_bins_for_env_sign(self, dim: int, target_env_sign: str, unnorm_key: str, *, postprocess_gripper: bool = False) -> torch.LongTensor:
        """DEPRECATED: sign-string semantics are inverted for postprocessed gripper.

        Use get_gripper_region_by_decoded_action() for any new code.
        Kept for backward compatibility with existing objective code paths.
        """
        stats, unnorm_key = self._action_stats(unnorm_key)
        mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)), dtype=bool)
        low = np.asarray(stats["q01"], dtype=np.float32)
        high = np.asarray(stats["q99"], dtype=np.float32)
        centers = np.asarray(self.model.bin_centers, dtype=np.float32)
        if bool(mask[int(dim)]):
            raw_values = 0.5 * (centers + 1.0) * (high[int(dim)] - low[int(dim)]) + low[int(dim)]
        else:
            raw_values = centers.copy()
        if postprocess_gripper and int(dim) == len(low) - 1:
            env_values = 2.0 * raw_values - 1.0
            env_values = np.sign(env_values)
            env_values[env_values == 0] = 1.0
            env_values = -1.0 * env_values
        else:
            env_values = raw_values
        sign = str(target_env_sign or "negative").strip().lower()
        threshold = 0.5
        if sign in {"positive", "+", "+1", "1", "pos"}:
            disc = np.where(env_values > threshold)[0]
        else:
            disc = np.where(env_values < -threshold)[0]
        vocab_size = int(self.model.config.text_config.vocab_size - self.model.config.pad_to_multiple_of)
        token_ids = vocab_size - disc - 1
        return torch.tensor(token_ids, dtype=torch.long, device=self.device)

    def get_gripper_region_by_decoded_action(self, unnorm_key: str, *, postprocess_gripper: bool = True, open_threshold: float = 0.5) -> dict:
        """Return OPEN/CLOSE/BOUNDARY token sets using canonical decoded-action semantics.

        Canonical exec spec: raw > 0.5 → env -1 → physical OPEN; raw < 0.5 → env +1 → physical CLOSE.
        This means ``decoded_action > open_threshold`` → OPEN (execspec v2 fix).

        Returns dict with keys:
          - open_token_ids, close_token_ids, boundary_token_ids
          - token_action_map: {token_id: decoded_action}
          - open_count, close_count
          - is_corrected: True
          - canonical_semantics_version: str
        """
        stats, unnorm_key = self._action_stats(unnorm_key)
        low = np.asarray(stats["q01"], dtype=np.float32)
        high = np.asarray(stats["q99"], dtype=np.float32)
        centers = np.asarray(self.model.bin_centers, dtype=np.float32)
        vocab_size = int(self.model.config.text_config.vocab_size - self.model.config.pad_to_multiple_of)
        action_dim = int(self.model.get_action_dim(unnorm_key))
        gripper_dim = action_dim - 1

        # Decode every possible bin through the production pipeline
        n_bins = len(centers)
        open_tokens = []
        close_tokens = []
        boundary_tokens = []
        token_action_map = {}

        for disc in range(n_bins):
            # Step 1: unnormalize (same as rollout decode)
            norm = centers[disc]
            raw_action = 0.5 * (norm + 1.0) * (high[gripper_dim] - low[gripper_dim]) + low[gripper_dim]

            # Step 2: normalize_gripper_action (binarize=True)
            env_val = 2.0 * raw_action - 1.0
            env_val = np.sign(env_val)
            env_val = 1.0 if env_val == 0 else env_val

            # Step 3: invert_gripper_action
            env_val = -1.0 * env_val

            # Step 4: canonical classification via exec spec helpers (NOT manual sign checks).
            decoded_action = float(0.5 * (norm + 1.0) * (high[gripper_dim] - low[gripper_dim]) + low[gripper_dim])

            tid = int(vocab_size - disc - 1)
            token_action_map[tid] = decoded_action

            # Canonical: raw > 0.5 → OPEN, raw < 0.5 → CLOSE
            is_open = raw_gripper_is_open(decoded_action, threshold=float(open_threshold))
            is_close = raw_gripper_is_close(decoded_action, threshold=float(open_threshold))

            # Sanity: env-level and raw-level classification must agree
            # Exception: boundary tokens (raw ≈ 0.5) — np.sign fixup maps 0→+1→-1
            # which disagrees with raw threshold classification.
            is_boundary = raw_gripper_is_boundary(decoded_action, threshold=float(open_threshold))
            if not is_boundary:
                assert is_open == env_gripper_is_open(env_val), \
                    f"OPEN classification mismatch at disc={disc}: env={int(env_val)} action={decoded_action:.6f}"
                assert is_close == env_gripper_is_close(env_val), \
                    f"CLOSE classification mismatch at disc={disc}: env={int(env_val)} action={decoded_action:.6f}"

            if is_open:
                open_tokens.append(tid)
            elif is_close:
                close_tokens.append(tid)
            # boundary: neither open nor close → excluded from both sets

            # Boundary detection: adjacent discs with opposite signs
            if disc > 0:
                prev_env_val = 2.0 * (0.5 * (centers[disc-1] + 1.0) * (high[gripper_dim] - low[gripper_dim]) + low[gripper_dim]) - 1.0
                prev_env_val = np.sign(prev_env_val)
                prev_env_val = 1.0 if prev_env_val == 0 else prev_env_val
                prev_env_val = -1.0 * prev_env_val
                if int(env_val) != int(prev_env_val):
                    boundary_tokens.append(int(vocab_size - disc - 1))
                    boundary_tokens.append(int(vocab_size - (disc - 1) - 1))

        open_token_ids = torch.tensor(sorted(set(open_tokens)), dtype=torch.long, device=self.device)
        close_token_ids = torch.tensor(sorted(set(close_tokens)), dtype=torch.long, device=self.device)

        # ── Canonical validation via exec spec helpers (NOT manual assertions) ──
        validate_open_close_token_sets(
            [int(t) for t in open_token_ids],
            [int(t) for t in close_token_ids],
            token_action_map,
            threshold=float(open_threshold),
        )

        return {
            "open_token_ids": open_token_ids,
            "close_token_ids": close_token_ids,
            "boundary_token_ids": sorted(set(boundary_tokens)),
            "token_action_map": token_action_map,
            "open_count": int(open_token_ids.numel()),
            "close_count": int(close_token_ids.numel()),
            "is_corrected": True,
            "canonical_semantics_version": CANONICAL_OPEN_SEMANTICS_VERSION,
        }

    def _active_label_rows(self, logits, labels, action_dim: int):
        action_start = int(labels.shape[1]) - int(action_dim)
        active = (labels != -100).nonzero(as_tuple=False)
        rows = []
        for b, label_pos in active:
            dim = int(label_pos.item()) - action_start
            if dim < 0 or dim >= int(action_dim):
                continue
            row_index = action_token_logit_row_index(dim, action_dim)
            if abs(row_index) > int(logits.shape[1]):
                continue
            rows.append((int(b.item()), int(label_pos.item()), dim, row_index))
        return rows

    def _loss(self, full_input_ids, labels, pixel_values, *, objective: str = "targeted_directional_ce", region_token_ids=None, close_token_ids=None, margin: float = 5.0, num_action_tokens: int = 7, loss_weights: dict = None, arm_preserve_weight: float = 0.1):
        obj = str(objective)
        _PREFIX_LOCKED_OBJS = {"prefix_locked_gripper_open_region_ce", "prefix_locked_gripper_open_margin", "gripper_open_expected_action", "prefix_locked_gripper_top1_open_vs_close", "prefix_locked_gripper_top1_open_vs_close_execspec_v2"}
        _SPECIAL_OBJS = {"gripper_open_region_ce", "gripper_logit_margin_cw", "force_open_region_z_down_ce"} | _PREFIX_LOCKED_OBJS

        if obj not in _SPECIAL_OBJS:
            if loss_weights is None or obj != "force_open_z_down_token_ce":
                out = self.model(input_ids=full_input_ids, pixel_values=pixel_values, labels=labels, use_cache=False, return_dict=True)
                if out.loss is not None:
                    return out.loss
                logits = out.logits[:, :-1, :].contiguous()
                shifted = labels[:, 1:].contiguous()
                return F.cross_entropy(logits.view(-1, logits.shape[-1]), shifted.view(-1), ignore_index=-100)
            out = self.model(input_ids=full_input_ids, pixel_values=pixel_values, use_cache=False, return_dict=True)
            logits = out.logits.float().contiguous()
            action_dim = max(int(num_action_tokens), 1)
            rows = self._active_label_rows(logits, labels, action_dim)
            if not rows:
                return logits.sum() * 0.0
            weighted_losses = []
            for _b, _label_pos, dim, row_index in rows:
                row = logits[_b, row_index, :]
                target = labels[_b, _label_pos]
                ce = F.cross_entropy(row.view(1, -1), target.view(1))
                w = loss_weights.get(str(dim), loss_weights.get(int(dim), 1.0))
                weighted_losses.append(float(w) * ce)
            return torch.stack(weighted_losses).mean() if weighted_losses else logits.sum() * 0.0

        out = self.model(input_ids=full_input_ids, pixel_values=pixel_values, use_cache=False, return_dict=True)
        logits = out.logits.float().contiguous()
        action_dim = max(int(num_action_tokens), 1)

        # ── Prefix-locked objectives: gripper loss computed DIRECTLY from logit row,
        #     NOT dependent on labels (which are masked to -100 for gripper). ──
        if obj in _PREFIX_LOCKED_OBJS:
            # Gripper logit row: logits[:, -2, :] predicts the final action token
            # (dim=6, gripper) because the gripper token is at position -1 in the
            # input and the causal logit at -2 predicts token at -1.
            gripper_row_index = action_token_logit_row_index(action_dim - 1, action_dim)
            gripper_row = logits[0, gripper_row_index, :]  # [vocab_size]

            _gripper_loss_present = False
            _gripper_loss_value = 0.0
            _arm_loss_present = False
            _arm_ce_list = []

            if region_token_ids is not None and int(region_token_ids.numel()) > 0:
                _gripper_loss_present = True
                open_count = int(region_token_ids.numel())
                if obj == "prefix_locked_gripper_open_margin":
                    log_open = torch.logsumexp(gripper_row[region_token_ids], dim=0)
                    non_open_mask = torch.ones_like(gripper_row, dtype=torch.bool)
                    non_open_mask[region_token_ids] = False
                    max_non_open = gripper_row[non_open_mask].max()
                    gripper_loss = F.relu(max_non_open - log_open + float(margin))
                elif obj == "prefix_locked_gripper_open_region_ce":
                    log_region = torch.logsumexp(gripper_row[region_token_ids], dim=0)
                    log_all = torch.logsumexp(gripper_row, dim=0)
                    gripper_loss = -(log_region - log_all)
                elif obj == "gripper_open_expected_action":
                    probs = torch.softmax(gripper_row, dim=-1)
                    open_prob_mass = probs[region_token_ids].sum()
                    gripper_loss = -open_prob_mass
                elif obj == "prefix_locked_gripper_top1_open_vs_close":
                    max_open = gripper_row[region_token_ids].max()
                    max_close = gripper_row[close_token_ids].max()
                    gripper_loss = F.relu(max_close - max_open + float(margin))
                elif obj == "prefix_locked_gripper_top1_open_vs_close_execspec_v2":
                    # SAME formula: relu(max_close - max_open + margin)
                    # DIFFERENCE: token regions are from corrected exec spec (raw>0.5=OPEN)
                    max_open = gripper_row[region_token_ids].max()
                    max_close = gripper_row[close_token_ids].max()
                    gripper_loss = F.relu(max_close - max_open + float(margin))
                else:
                    gripper_loss = logits.sum() * 0.0
                _gripper_loss_value = float(gripper_loss.detach().cpu())
            else:
                open_count = 0
                gripper_loss = logits.sum() * 0.0

            # Arm CE: compute from label rows (arm dims only — gripper label is -100).
            apw = float(arm_preserve_weight)
            rows = self._active_label_rows(logits, labels, action_dim)
            for _b, _label_pos, dim, row_index in rows:
                if dim == action_dim - 1:
                    continue  # skip gripper dim (should be absent, but defensive)
                _arm_loss_present = True
                row = logits[_b, row_index, :]
                target = labels[_b, _label_pos]
                _arm_ce_list.append(apw * F.cross_entropy(row.view(1, -1), target.view(1)))

            arm_term = torch.stack(_arm_ce_list).mean() if _arm_ce_list else 0.0
            total = gripper_loss + arm_term

            # Store debug fields on the returned tensor for inspection in attack().
            # (non-standard but the cleanest way to thread this through the PGD loop)
            total._prefix_debug = {
                "prefix_locked_gripper_loss_present": _gripper_loss_present,
                "prefix_locked_arm_loss_present": _arm_loss_present,
                "gripper_loss_value": _gripper_loss_value,
                "arm_loss_value": float(arm_term.detach().cpu()) if isinstance(arm_term, torch.Tensor) else float(arm_term),
                "gripper_open_region_token_count": open_count,
                "gripper_row_index": int(gripper_row_index),
                "canonical_open_semantics_version": CANONICAL_OPEN_SEMANTICS_VERSION,
            }
            return total

        rows = self._active_label_rows(logits, labels, action_dim)

        # Corrected hybrid: gripper OPEN-region loss + Z weighted CE
        if obj == "force_open_region_z_down_ce" and region_token_ids is not None and int(region_token_ids.numel()) > 0:
            losses = []
            gripper_row_index = action_token_logit_row_index(action_dim - 1, action_dim)
            gripper_row = logits[0, gripper_row_index, :]
            if int(region_token_ids.numel()) > 0:
                log_region = torch.logsumexp(gripper_row[region_token_ids], dim=0)
                log_all = torch.logsumexp(gripper_row, dim=0)
                losses.append(-(log_region - log_all))
            for b, label_pos, dim, row_index in rows:
                if dim == 2:
                    row = logits[b, row_index, :]
                    target = labels[b, label_pos]
                    z_w = loss_weights.get('2', loss_weights.get(2, 0.5)) if loss_weights else 0.5
                    ce = F.cross_entropy(row.view(1, -1), target.view(1))
                    losses.append(float(z_w) * ce)
            return torch.stack(losses).mean() if losses else logits.sum() * 0.0

        if not rows:
            return logits.sum() * 0.0
        losses = []
        for b, label_pos, dim, row_index in rows:
            row = logits[b, row_index, :]
            target = labels[b, label_pos]
            is_gripper_dim = (dim == action_dim - 1)
            if obj == "gripper_open_region_ce":
                if is_gripper_dim and region_token_ids is not None and int(region_token_ids.numel()) > 0:
                    log_region = torch.logsumexp(row[region_token_ids], dim=0)
                    log_all = torch.logsumexp(row, dim=0)
                    losses.append(-(log_region - log_all))
                elif target.item() != -100:
                    losses.append(F.cross_entropy(row.view(1, -1), target.view(1)))
            elif obj == "gripper_open_expected_action":
                if is_gripper_dim and region_token_ids is not None and int(region_token_ids.numel()) > 0:
                    probs = torch.softmax(row, dim=-1)
                    open_prob_mass = probs[region_token_ids].sum()
                    losses.append(-open_prob_mass)
                elif target.item() != -100:
                    apw = float(arm_preserve_weight)
                    losses.append(apw * F.cross_entropy(row.view(1, -1), target.view(1)))
            else:
                target_logit = row[target]
                other = row.clone()
                other[target] = torch.finfo(other.dtype).min
                losses.append(F.relu(torch.max(other) - target_logit + float(margin)))
        return torch.stack(losses).mean() if losses else logits.sum() * 0.0

    def _audit_logits(self, full_input_ids, labels, pixel_values, target_ids, unnorm_key: str, *, postprocess_gripper: bool = False, region_token_ids=None) -> dict:
        with torch.no_grad():
            out = self.model(input_ids=full_input_ids, pixel_values=pixel_values, use_cache=False, return_dict=True)
            logits = out.logits.float().contiguous()
            action_dim = int(target_ids.numel())
            rows = []
            vocab_size = int(self.model.config.text_config.vocab_size - self.model.config.pad_to_multiple_of)
            _region_info = self.get_gripper_region_by_decoded_action(unnorm_key, postprocess_gripper=postprocess_gripper)
            open_tokens = _region_info["open_token_ids"]
            close_tokens = _region_info["close_token_ids"]
            action_start = int(labels.shape[1]) - action_dim
            active = (labels != -100).nonzero(as_tuple=False)
            for b, label_pos in active:
                dim = int(label_pos.item()) - action_start
                if dim < 0 or dim >= action_dim:
                    continue
                row_index = action_token_logit_row_index(dim, action_dim)
                if abs(row_index) > int(logits.shape[1]):
                    continue
                row = logits[int(b.item()), row_index, :]
                probs = torch.softmax(row, dim=-1)
                target = int(labels[int(b.item()), int(label_pos.item())].item())
                top_val, top_idx = torch.max(row, dim=-1)
                rank = int(torch.sum(row > row[target]).item()) + 1
                item = {
                    "dim": dim,
                    "target_token_id": target,
                    "top_token_id": int(top_idx.item()),
                    "target_rank": rank,
                    "target_logit": float(row[target].detach().cpu()),
                    "top_logit": float(top_val.detach().cpu()),
                    "top_minus_target_logit": float((top_val - row[target]).detach().cpu()),
                    "target_prob": float(probs[target].detach().cpu()),
                    "logit_suffix_index": int(row_index),
                }
                if dim == action_dim - 1:
                    item.update({
                        "open_bin_token_count": int(open_tokens.numel()),
                        "close_bin_token_count": int(close_tokens.numel()),
                        "open_bin_prob_mass": float(torch.sum(probs[open_tokens]).detach().cpu()) if int(open_tokens.numel()) else 0.0,
                        "close_bin_prob_mass": float(torch.sum(probs[close_tokens]).detach().cpu()) if int(close_tokens.numel()) else 0.0,
                        "open_bin_token_min": int(torch.min(open_tokens).detach().cpu()) if int(open_tokens.numel()) else None,
                        "open_bin_token_max": int(torch.max(open_tokens).detach().cpu()) if int(open_tokens.numel()) else None,
                        "close_bin_token_min": int(torch.min(close_tokens).detach().cpu()) if int(close_tokens.numel()) else None,
                        "close_bin_token_max": int(torch.max(close_tokens).detach().cpu()) if int(close_tokens.numel()) else None,
                        "bin_mapping": "corrected_decoded_action_semantics_20260602" if postprocess_gripper else "raw_negative_open_positive_closed",
                        "region_is_corrected": True,
                        "open_region_logsumexp": float(torch.logsumexp(row[open_tokens], dim=0).detach().cpu()) if int(open_tokens.numel()) else None,
                        "non_open_max_logit": float(row[close_tokens].max().detach().cpu()) if int(close_tokens.numel()) else float(row.max().detach().cpu()),
                    })
                rows.append(item)
            out = {"action_token_logit_audit": rows}
            for item in rows:
                if item.get("dim") == action_dim - 1:
                    for k, v in item.items():
                        out[f"gripper_{k}"] = v
                if item.get("dim") == 2:
                    for k, v in item.items():
                        out[f"z_{k}"] = v
            return out

    def _tokens_from_generation(self, gen, action_dim: int) -> Optional[torch.LongTensor]:
        if gen is None or not hasattr(gen, "sequences"):
            return None
        try:
            ids = gen.sequences[0, -int(action_dim):].detach().to(device=self.device, dtype=torch.long)
            return ids
        except Exception:
            return None

    def _exact_tokens_from_generation(
        self,
        gen,
        *,
        prompt_len: int,
        action_dim: int,
        context: str,
    ) -> torch.LongTensor:
        if gen is None or not hasattr(gen, "sequences"):
            raise RouteContractError(f"{context} requires clean_model_output with generated sequences")
        seq = gen.sequences
        if int(seq.ndim) != 2 or int(seq.shape[0]) < 1:
            raise RouteContractError(f"{context} clean_model_output.sequences must be [batch, seq]")
        new_count = int(seq.shape[1]) - int(prompt_len)
        if new_count != int(action_dim):
            raise RouteContractError(
                f"{context} requires exact {int(action_dim)} clean generated tokens; got {new_count}"
            )
        return seq[0, int(prompt_len):].detach().to(device=self.device, dtype=torch.long)

    def _generate_action_prefix_tokens(self, prompt_input_ids: torch.LongTensor, pixel_values: torch.Tensor, *, prefix_len: int) -> torch.LongTensor:
        """Greedily generate action-prefix token ids as stop-gradient context."""
        if int(prefix_len) <= 0:
            return torch.empty((0,), dtype=torch.long, device=prompt_input_ids.device)
        with torch.no_grad():
            gen = self.model.generate(
                input_ids=prompt_input_ids,
                pixel_values=pixel_values.detach(),
                max_new_tokens=int(prefix_len),
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=False,
            )
        new_token_count = int(gen.sequences.shape[1]) - int(prompt_input_ids.shape[1])
        if new_token_count != int(prefix_len):
            raise RuntimeError(
                f"V3 prefix generation produced {new_token_count} new tokens, "
                f"expected {int(prefix_len)}. Early EOS or truncation detected."
            )
        return gen.sequences[0, prompt_input_ids.shape[1]:].detach().to(
            device=prompt_input_ids.device, dtype=torch.long)

    def _select_strict_arm_candidate(
        self,
        prompt_input_ids: torch.LongTensor,
        trajectory_candidate_inputs: list[dict[str, Any]],
        clean_generated_action_token_ids: torch.LongTensor,
        open_token_ids: torch.LongTensor,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Select only a predeclared candidate with exact direct-token isolation."""
        if open_token_ids is None or int(open_token_ids.numel()) == 0:
            raise RouteContractError("STRICT_CANDIDATE_AUDIT_OPEN_TOKEN_SET_MISSING")
        clean_tokens = [int(x) for x in clean_generated_action_token_ids.detach().cpu().tolist()]
        if len(clean_tokens) != 7:
            raise RouteContractError(f"ACTION_TOKEN_COUNT_FOR_ARM_AUDIT:{len(clean_tokens)}")
        clean_arm = clean_tokens[:-1]
        clean_gripper_token_id = int(clean_tokens[-1])
        open_ids = {int(x) for x in open_token_ids.detach().cpu().tolist()}
        clean_gripper_is_native_open = clean_gripper_token_id in open_ids
        audit: list[dict[str, Any]] = []
        for candidate in trajectory_candidate_inputs:
            candidate_tokens = [
                int(x)
                for x in self._generate_action_prefix_tokens(
                    prompt_input_ids,
                    candidate["pixel_values"],
                    prefix_len=len(clean_tokens),
                ).detach().cpu().tolist()
            ]
            arm_mismatch = [
                int(dim)
                for dim, (clean_id, candidate_id) in enumerate(zip(clean_arm, candidate_tokens[:-1]))
                if int(clean_id) != int(candidate_id)
            ]
            gripper_token_id = int(candidate_tokens[-1])
            item = {
                "candidate_index": int(candidate["candidate_index"]),
                "candidate_source": str(candidate["candidate_source"]),
                "direct_generated_token_ids": candidate_tokens,
                "clean_arm_token_ids": clean_arm,
                "direct_generated_arm_token_ids": candidate_tokens[:-1],
                "arm_token_ids_equal": not arm_mismatch,
                "arm_mismatch_dimensions": arm_mismatch,
                "direct_generated_gripper_token_id": gripper_token_id,
                "direct_generated_gripper_is_native_open": gripper_token_id in open_ids,
                "clean_gripper_token_id": clean_gripper_token_id,
                "clean_gripper_is_native_open": clean_gripper_is_native_open,
                "gripper_token_changed": gripper_token_id != clean_gripper_token_id,
                "processor_input_sha256": str(candidate.get("processor_input_sha256", "")),
            }
            audit.append(item)
            if not arm_mismatch and not clean_gripper_is_native_open and gripper_token_id in open_ids and gripper_token_id != clean_gripper_token_id:
                self.last_attack_diagnostics = {
                    "candidate_policy": "STRICT_CANDIDATE_AUDIT_V1",
                    "candidate_audit": audit,
                    "selected_candidate_index": int(candidate["candidate_index"]),
                    "selected_candidate_source": str(candidate["candidate_source"]),
                }
                return candidate, audit
        diagnostics = {
            "candidate_policy": "STRICT_CANDIDATE_AUDIT_V1",
            "candidate_audit": audit,
            "selected_candidate_index": None,
            "selected_candidate_source": None,
        }
        self.last_attack_diagnostics = diagnostics
        error = RouteContractError("STRUCTURAL_INVALID_NO_SELECTIVE_CANDIDATE")
        error.diagnostics = diagnostics
        raise error

    def _gripper_row_stats(self, row: torch.Tensor, open_token_ids: torch.LongTensor, close_token_ids: torch.LongTensor) -> dict:
        probs = torch.softmax(row, dim=-1)
        top_val, top_idx = torch.max(row, dim=-1)
        max_open_val, max_open_rel = torch.max(row[open_token_ids], dim=0)
        max_close_val, max_close_rel = torch.max(row[close_token_ids], dim=0)
        strongest_open = int(open_token_ids[int(max_open_rel.detach().cpu())].detach().cpu())
        strongest_close = int(close_token_ids[int(max_close_rel.detach().cpu())].detach().cpu())
        return {
            "top_token_id": int(top_idx.detach().cpu()),
            "top_logit": float(top_val.detach().cpu()),
            "strongest_native_open_token": strongest_open,
            "strongest_native_close_token": strongest_close,
            "open_score": float(max_open_val.detach().cpu()),
            "close_score": float(max_close_val.detach().cpu()),
            "open_minus_close_margin": float((max_open_val - max_close_val).detach().cpu()),
            "open_prob_mass": float(torch.sum(probs[open_token_ids]).detach().cpu()),
            "close_prob_mass": float(torch.sum(probs[close_token_ids]).detach().cpu()),
        }

    def _teacher_forced_gripper_margin_stats(self, full_input_ids, pixel_values, action_dim: int, open_token_ids, close_token_ids) -> dict:
        with torch.no_grad():
            out = self.model(input_ids=full_input_ids, pixel_values=pixel_values, use_cache=False, return_dict=True)
            logits = out.logits.float().contiguous()
            row_index = action_token_logit_row_index(int(action_dim) - 1, int(action_dim))
            stats = self._gripper_row_stats(logits[0, row_index, :], open_token_ids, close_token_ids)
            stats["gripper_row_index"] = int(row_index)
            stats["conditioning"] = "teacher_forced_target_arm_prefix"
            return stats

    def _generated_prefix_gripper_loss_and_stats(
        self,
        prompt_input_ids: torch.LongTensor,
        generated_arm_prefix_token_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        open_token_ids: torch.LongTensor,
        close_token_ids: torch.LongTensor,
        *,
        margin: float,
    ):
        prefix = generated_arm_prefix_token_ids.view(1, -1).to(device=prompt_input_ids.device, dtype=torch.long)
        context_ids = torch.cat([prompt_input_ids, prefix], dim=1)
        out = self.model(input_ids=context_ids, pixel_values=pixel_values, use_cache=False, return_dict=True)
        logits = out.logits.float().contiguous()
        gripper_row = logits[0, -1, :]
        max_open = gripper_row[open_token_ids].max()
        max_close = gripper_row[close_token_ids].max()
        loss = F.relu(max_close - max_open + float(margin))
        stats = self._gripper_row_stats(gripper_row, open_token_ids, close_token_ids)
        stats.update({
            "gripper_row_index": -1,
            "conditioning": "generated_arm_prefix_stop_gradient",
            "generated_arm_prefix_token_ids": [int(x) for x in generated_arm_prefix_token_ids.detach().cpu().tolist()],
        })
        return loss, stats

    def _generated_prefix_target_token_loss_and_stats(
        self,
        prompt_input_ids: torch.LongTensor,
        generated_arm_prefix_token_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        *,
        target_token_id: int,
        margin: float,
    ):
        if self.surrogate_score_path in {"cached_autoregressive_generate_v1", "cached_generate_v1"}:
            return self._generated_prefix_target_token_loss_and_stats_cached(
                prompt_input_ids,
                generated_arm_prefix_token_ids,
                pixel_values,
                target_token_id=int(target_token_id),
                margin=float(margin),
            )
        prefix = generated_arm_prefix_token_ids.view(1, -1).to(device=prompt_input_ids.device, dtype=torch.long)
        context_ids = torch.cat([prompt_input_ids, prefix], dim=1)
        out = self.model(input_ids=context_ids, pixel_values=pixel_values, use_cache=False, return_dict=True)
        logits = out.logits.float().contiguous()
        gripper_row = logits[0, -1, :]
        loss, stats = self._target_token_loss_and_stats_from_row(
            gripper_row,
            target_token_id=int(target_token_id),
            margin=float(margin),
        )
        stats.update({
            "gripper_row_index": -1,
            "conditioning": "generated_arm_prefix_stop_gradient",
            "generated_arm_prefix_token_ids": [int(x) for x in generated_arm_prefix_token_ids.detach().cpu().tolist()],
        })
        return loss, stats

    def _target_token_loss_and_stats_from_row(
        self,
        gripper_row: torch.Tensor,
        *,
        target_token_id: int,
        margin: float,
    ):
        if str(getattr(self, "objective", "")) in {
            "autoregressive_prefix_gripper_target_token_logratio_v2",
            "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
        }:
            return target_token_logratio_loss_and_stats(
                gripper_row,
                target_token_id=int(target_token_id),
                allowed_token_ids=range(int(gripper_row.numel())),
            )
        return target_token_cw_loss_and_stats(
            gripper_row,
            target_token_id=int(target_token_id),
            allowed_token_ids=range(int(gripper_row.numel())),
            margin=float(margin),
        )

    def _generated_prefix_target_token_loss_and_stats_cached(
        self,
        prompt_input_ids: torch.LongTensor,
        generated_arm_prefix_token_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        *,
        target_token_id: int,
        margin: float,
    ):
        """Target-token row using the same cached AR path as default generate().

        OpenVLA's cached generation path can produce a different seventh-token
        score row than a single no-cache forward over ``prompt + arm_prefix``.
        The M3 fixed-frame objective is only valid if its surrogate row follows
        the official cached autoregressive execution path.
        """

        out = self.model(
            input_ids=prompt_input_ids,
            pixel_values=pixel_values,
            use_cache=True,
            return_dict=True,
        )
        past = getattr(out, "past_key_values", None)
        if past is None:
            raise RouteContractError("cached_autoregressive_generate_v1 requires model past_key_values")
        final_out = out
        for token in generated_arm_prefix_token_ids.detach().to(device=prompt_input_ids.device, dtype=torch.long).view(-1):
            step_ids = token.view(1, 1)
            final_out = self.model(
                input_ids=step_ids,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = getattr(final_out, "past_key_values", None)
            if past is None:
                raise RouteContractError("cached autoregressive step did not return past_key_values")
        logits = final_out.logits.float().contiguous()
        gripper_row = logits[0, -1, :]
        loss, stats = self._target_token_loss_and_stats_from_row(
            gripper_row,
            target_token_id=int(target_token_id),
            margin=float(margin),
        )
        stats.update({
            "gripper_row_index": -1,
            "conditioning": "generated_arm_prefix_stop_gradient",
            "surrogate_score_path": "cached_autoregressive_generate_v1",
            "generated_arm_prefix_token_ids": [int(x) for x in generated_arm_prefix_token_ids.detach().cpu().tolist()],
        })
        return loss, stats

    def _arm_preservation_loss_and_stats(self, full_input_ids, labels, pixel_values, action_dim: int, *, arm_preserve_weight: float):
        out = self.model(input_ids=full_input_ids, pixel_values=pixel_values, use_cache=False, return_dict=True)
        logits = out.logits.float().contiguous()
        losses = []
        rows_used = []
        for b, label_pos, dim, row_index in self._active_label_rows(logits, labels, int(action_dim)):
            if dim == int(action_dim) - 1:
                continue
            row = logits[b, row_index, :]
            target = labels[b, label_pos]
            losses.append(float(arm_preserve_weight) * F.cross_entropy(row.view(1, -1), target.view(1)))
            rows_used.append(int(dim))
        loss = torch.stack(losses).mean() if losses else logits.sum() * 0.0
        return loss, {
            "arm_preservation_dims": rows_used,
            "arm_preservation_dim_count": int(len(rows_used)),
            "arm_preservation_loss": float(loss.detach().cpu()),
            "arm_preservation_weight": float(arm_preserve_weight),
        }

    def _clean_generated_arm_preservation_loss_and_stats(
        self,
        prompt_input_ids: torch.LongTensor,
        clean_generated_action_token_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        action_dim: int,
        *,
        arm_preserve_weight: float,
    ):
        target = clean_generated_action_token_ids.view(1, -1).to(prompt_input_ids.device)
        full_input_ids = torch.cat([prompt_input_ids, target], dim=1)
        labels = torch.full_like(full_input_ids, -100)
        action_start = labels.shape[1] - int(action_dim)
        rows_used = []
        for dim in range(max(int(action_dim) - 1, 0)):
            label_pos = action_start + dim
            labels[:, label_pos] = target[:, dim]
            rows_used.append(int(dim))
        out = self.model(input_ids=full_input_ids, pixel_values=pixel_values, use_cache=False, return_dict=True)
        logits = out.logits.float().contiguous()
        losses = []
        active_rows = []
        for b, label_pos, dim, row_index in self._active_label_rows(logits, labels, int(action_dim)):
            if dim == int(action_dim) - 1:
                continue
            row = logits[b, row_index, :]
            label = labels[b, label_pos]
            losses.append(float(arm_preserve_weight) * F.cross_entropy(row.view(1, -1), label.view(1)))
            active_rows.append(int(dim))
        loss = torch.stack(losses).mean() if losses else logits.sum() * 0.0
        return loss, {
            "arm_preservation_dims": active_rows or rows_used,
            "arm_preservation_dim_count": int(len(active_rows or rows_used)),
            "arm_preservation_loss": float(loss.detach().cpu()),
            "arm_preservation_weight": float(arm_preserve_weight),
            "arm_preservation_label_source": "clean_actual_generation",
            "clean_generated_arm_prefix_token_ids": [
                int(x)
                for x in target[0, : max(int(action_dim) - 1, 0)].detach().cpu().tolist()
            ],
        }

    def attack(self, observation: Any, instruction=None, clean_action=None, target_action=None, clean_model_output=None, *, unnorm_key: str = "libero_goal") -> AttackResult:
        self.last_attack_diagnostics = {}
        objective = str(getattr(self, "objective", "targeted_directional_ce"))
        is_untargeted = objective in {"untargeted_clean_token_ce", "untargeted_clean_ce", "maximize_clean_ce", "untargeted_arm_clean_token_ce", "ctrl_random_direction_arm_only"}
        is_arm_only_untargeted = objective in {"untargeted_arm_clean_token_ce", "ctrl_random_direction_arm_only"}
        is_force_gripper_open = objective in {"force_gripper_open_token_ce", "force_gripper_open", "targeted_gripper_open_ce", "adaptive_anti_gripper_token_ce"}
        is_force_open_z_down = objective in {"force_open_z_down_token_ce"}
        is_force_open_region_z_down = objective in {"force_open_region_z_down_ce"}  # corrected hybrid
        is_gripper_margin = objective in {"gripper_logit_margin_cw"}
        is_gripper_region = objective in {"gripper_open_region_ce"}
        # New gripper-specific objectives (2026-06-02)
        is_prefix_locked_open_region = objective in {"prefix_locked_gripper_open_region_ce"}
        is_prefix_locked_open_margin = objective in {"prefix_locked_gripper_open_margin"}
        is_prefix_locked_top1 = objective in {"prefix_locked_gripper_top1_open_vs_close", "prefix_locked_gripper_top1_open_vs_close_execspec_v2"}
        is_generated_prefix_v3 = objective in {"autoregressive_prefix_gripper_open_execspec_v3"}
        is_target_token_cw_v1 = objective in {"autoregressive_prefix_gripper_target_token_cw_v1"}
        is_target_token_logratio_v2 = objective in {"autoregressive_prefix_gripper_target_token_logratio_v2"}
        is_target_token_logratio_arm_v3 = objective in {"autoregressive_prefix_gripper_target_token_logratio_arm_v3"}
        is_target_token_objective = is_target_token_cw_v1 or is_target_token_logratio_v2 or is_target_token_logratio_arm_v3
        if self.arm_isolation_candidate_policy not in {"FINAL_ONLY", "STRICT_CANDIDATE_AUDIT_V1"}:
            raise RouteContractError(f"UNKNOWN_ARM_ISOLATION_CANDIDATE_POLICY:{self.arm_isolation_candidate_policy}")
        if self.arm_isolation_candidate_policy == "STRICT_CANDIDATE_AUDIT_V1" and not is_target_token_objective:
            raise RouteContractError("STRICT_CANDIDATE_AUDIT_REQUIRES_TARGET_TOKEN_OBJECTIVE")
        is_gripper_expected_action = objective in {"gripper_open_expected_action"}
        is_prefix_locked = is_prefix_locked_open_region or is_prefix_locked_open_margin or is_gripper_expected_action or is_prefix_locked_top1
        is_corrected_hybrid = is_force_open_region_z_down  # uses corrected OPEN region + Z CE
        if self.model is None or self.processor is None or ((not is_untargeted) and target_action is None):
            return ExistingDenseAttackAdapter(self.epsilon, self.step_size, self.num_steps, self.seed).attack(observation, instruction, clean_action, target_action, clean_model_output)
        self._freeze_model()
        if is_untargeted:
            unnorm_key = self._resolve_unnorm_key(unnorm_key)
            action_dim = int(self.model.get_action_dim(unnorm_key))
            target_ids = self._tokens_from_generation(clean_model_output, action_dim)
            token_label_source = "clean_model_output_sequences"
            if target_ids is None:
                if clean_action is None:
                    return ExistingDenseAttackAdapter(self.epsilon, self.step_size, self.num_steps, self.seed).attack(observation, instruction, clean_action, target_action, clean_model_output)
                target_ids = self.action_to_token_ids(clean_action, unnorm_key)
                token_label_source = "retokenized_clean_action_fallback"
        else:
            target_ids = self.action_to_token_ids(target_action, unnorm_key)
            token_label_source = "directional_target_action"
        clean_ids, full_ids, labels, x0 = self._build_inputs_and_labels(observation, str(instruction), target_ids)
        retokenized_target_ids = target_ids.detach().clone()
        clean_generated_action_token_ids = None
        clean_generated_arm_prefix_token_ids = None
        retokenized_clean_action_token_ids = [int(x) for x in retokenized_target_ids.detach().cpu().tolist()]
        retokenized_clean_action_arm_token_ids = [
            int(x)
            for x in retokenized_target_ids[: max(int(retokenized_target_ids.numel()) - 1, 0)].detach().cpu().tolist()
        ]
        if is_target_token_objective:
            if (is_target_token_logratio_v2 or is_target_token_logratio_arm_v3) and self.surrogate_score_path not in {"cached_autoregressive_generate_v1", "cached_generate_v1"}:
                raise RouteContractError(f"{objective} requires cached_autoregressive_generate_v1")
            clean_generated_action_token_ids = self._exact_tokens_from_generation(
                clean_model_output,
                prompt_len=int(clean_ids.shape[1]),
                action_dim=int(target_ids.numel()),
                context="strict target-token objective",
            )
            clean_generated_arm_prefix_token_ids = [
                int(x)
                for x in clean_generated_action_token_ids[: max(int(clean_generated_action_token_ids.numel()) - 1, 0)]
                .detach()
                .cpu()
                .tolist()
            ]
        # P0 FIX: prefix_locked must be checked FIRST so arm-preserve branch is reachable.
        # Previously `or is_prefix_locked` in the gripper-only branch made this unreachable.
        if is_prefix_locked or is_generated_prefix_v3 or is_target_token_objective:
            # Prefix-locked objectives: preserve clean arm tokens (dims 0-5), attack gripper only.
            # target_ids encodes the clean action (target_action = clean_action in rollout script).
            # Mask gripper dim to -100 so it's handled by corrected OPEN-region loss.
            action_dim = int(target_ids.numel())
            gripper_dim = action_dim - 1
            masked = labels.clone()  # keep all arm token labels for CE preservation
            gripper_label_pos = labels.shape[1] - action_dim + gripper_dim
            masked[:, gripper_label_pos] = -100
            labels = masked
            token_label_source = (
                "generated_prefix_arm_gate_gripper_target_token"
                if is_target_token_objective
                else
                "generated_prefix_arm_preserve_gripper_autoregressive_prefix_gripper_open_execspec_v3"
                if is_generated_prefix_v3
                else f"prefix_locked_arm_preserve_gripper_{objective}"
            )
        elif is_corrected_hybrid:
            # Corrected hybrid: gripper uses corrected OPEN-region loss, Z uses CE toward Z-down.
            # Unlike old force_open_z_down, gripper target is NOT target_action[-1]=1.0.
            action_dim = int(target_ids.numel())
            gripper_dim = action_dim - 1
            z_dim = 2
            masked = torch.full_like(labels, -100)
            # Keep only Z label for CE; gripper handled by corrected OPEN region
            z_label_pos = labels.shape[1] - action_dim + z_dim
            masked[:, z_label_pos] = labels[:, z_label_pos]
            labels = masked
            token_label_source = "force_open_region_z_down_ce_corrected_hybrid"
        elif is_force_gripper_open or is_force_open_z_down or is_gripper_margin or is_gripper_region:
            action_dim = int(target_ids.numel())
            gripper_dim = action_dim - 1
            label_positions = [labels.shape[1] - action_dim + gripper_dim]
            if is_force_open_z_down:
                z_dim = 2
                label_positions.append(labels.shape[1] - action_dim + z_dim)
            masked = torch.full_like(labels, -100)
            for label_pos in label_positions:
                masked[:, label_pos] = labels[:, label_pos]
            labels = masked
            if is_gripper_margin:
                token_label_source = "gripper_logit_margin_cw_target_action_gripper_only"
            elif is_gripper_region:
                token_label_source = "gripper_open_region_ce_target_action_gripper_only"
            else:
                token_label_source = "force_open_z_down_target_action_z_and_gripper" if is_force_open_z_down else "force_gripper_open_target_action_gripper_only"
        elif is_arm_only_untargeted:
            action_dim = int(target_ids.numel())
            masked = torch.full_like(labels, -100)
            action_start = labels.shape[1] - action_dim
            for dim in range(max(action_dim - 1, 0)):
                label_pos = action_start + dim
                masked[:, label_pos] = labels[:, label_pos]
            labels = masked
            token_label_source = "untargeted_clean_action_arm_only_gripper_masked"
        x_orig_model = x0.detach()
        x_orig = x_orig_model.detach().float()
        gen = torch.Generator(device=x_orig.device); gen.manual_seed(self.seed)
        temporal_prev_delta_used = False
        if self.temporal_init in {"prev_delta", "previous_delta", "carry", "carryover"} and self._prev_delta is not None and tuple(self._prev_delta.shape) == tuple(x_orig.shape):
            delta = torch.clamp(self._prev_delta.detach().to(device=x_orig.device, dtype=torch.float32), -self.epsilon, self.epsilon)
            temporal_prev_delta_used = True
        elif self.random_start:
            delta = torch.empty_like(x_orig).uniform_(-self.epsilon, self.epsilon, generator=gen)
        else:
            delta = torch.zeros_like(x_orig)
        adv = self._project_pixel_master(x_orig + delta, x_orig).detach()
        delta0_adv_model = self._cast_projected_pixel_values(adv.detach(), x_orig_model)
        delta0_diff = (delta0_adv_model.detach().float() - x_orig_model.detach().float()).detach()
        trajectory_candidate_inputs = [
            {
                "candidate_index": 0,
                "candidate_source": "delta0",
                "input_ids": clean_ids.detach(),
                "pixel_values": delta0_adv_model.detach(),
                "delta_sha256": tensor_sha256(delta0_diff),
                "processor_input_sha256": tensor_sha256(delta0_adv_model.detach()),
                "pixel_budget_adv_inputs_linf": float(delta0_diff.abs().max().cpu()) if delta0_diff.numel() else 0.0,
                "pixel_budget_quantized_correction_count": self._count_quantized_budget_corrections(
                    delta0_adv_model, self._project_pixel_master(x_orig + delta0_diff, x_orig), x_orig_model
                ),
            }
        ]
        loss_kwargs = {"objective": objective, "num_action_tokens": int(target_ids.numel())}
        region_token_ids = None
        corrected_region_info = None
        _needs_region = is_gripper_region or is_prefix_locked_open_region or is_prefix_locked_open_margin or is_gripper_expected_action or is_corrected_hybrid or is_prefix_locked_top1 or is_generated_prefix_v3
        if _needs_region:
            # P0 BUG FIX: use decoded-action semantics instead of sign-string heuristic.
            corrected_region_info = self.get_gripper_region_by_decoded_action(
                unnorm_key, postprocess_gripper=bool(self.postprocess_gripper))
            region_token_ids = corrected_region_info["open_token_ids"]
            loss_kwargs["region_token_ids"] = region_token_ids
            loss_kwargs["close_token_ids"] = corrected_region_info.get("close_token_ids")
        if is_gripper_margin:
            loss_kwargs["margin"] = float((getattr(self, "config", {}) or {}).get("cw_margin", 5.0)) if hasattr(self, "config") else 5.0
        if is_prefix_locked:
            loss_kwargs["arm_preserve_weight"] = float(self.arm_preserve_weight)
            loss_kwargs["margin"] = float(self.gripper_margin)
        if is_generated_prefix_v3:
            loss_kwargs["arm_preserve_weight"] = float(self.arm_preserve_weight)
            loss_kwargs["margin"] = float(self.gripper_margin)
        if is_target_token_objective:
            if self.target_token_id is None:
                raise RouteContractError(f"{objective} requires target_token_id")
            if self.target_execution_class:
                stats, resolved_unnorm = self._action_stats(unnorm_key)
                validate_execution_target(
                    token_id=int(self.target_token_id),
                    expected_execution_class=str(self.target_execution_class),
                    vocab_eff=int(self.model.config.text_config.vocab_size - self.model.config.pad_to_multiple_of),
                    n_bins=len(self.model.bin_centers),
                    bin_centers=self.model.bin_centers,
                    action_stats=stats,
                )
                unnorm_key = resolved_unnorm
        arm_audit_open_token_ids = region_token_ids
        if self.arm_isolation_candidate_policy == "STRICT_CANDIDATE_AUDIT_V1" and arm_audit_open_token_ids is None:
            # Target-token loss uses one secondary token, but arm isolation is
            # defined over the checkpoint-local native OPEN execution class.
            arm_audit_open_token_ids = self.get_gripper_region_by_decoded_action(
                unnorm_key, postprocess_gripper=bool(self.postprocess_gripper)
            )["open_token_ids"]
        if is_force_open_z_down and self.loss_weights is not None:
            loss_kwargs["loss_weights"] = self.loss_weights
        initial_loss = None; final_loss = None; _prefix_debug_final = None
        generated_prefix_debug = {}
        selected_candidate_index = None
        arm_isolation_candidate_audit = None
        selected_candidate_adv_model = None
        if is_generated_prefix_v3 or is_target_token_objective:
            prefix_refresh_interval = int(self.prefix_refresh_interval)
            prefix_refresh_count = 0
            num_generation_forwards = 0
            generated_arm_prefix_token_ids = None
            clean_arm_prefix_token_ids = (
                clean_generated_arm_prefix_token_ids
                if is_target_token_objective
                else retokenized_clean_action_arm_token_ids
            )
            teacher_initial = (
                None
                if is_target_token_objective
                else self._teacher_forced_gripper_margin_stats(
                    full_ids, x_orig_model, int(target_ids.numel()), region_token_ids, corrected_region_info.get("close_token_ids"))
            )
            initial_generated_stats = None
            initial_arm_stats = None
            final_generated_stats = None
            target_token_objective_loss_trajectory = []
            target_token_objective_margin_trajectory = []
            target_token_best_margin_trajectory = []
            target_token_logratio_margin_trajectory = []
            target_token_arm_loss_trajectory = []
            gradient_norm_trajectory = []
            generated_arm_prefix_trajectory = []
            for i in range(max(self.num_steps, 1)):
                adv = adv.detach().requires_grad_(True)
                # --- Prefix generation (if needed) ---
                if generated_arm_prefix_token_ids is None or (i % prefix_refresh_interval) == 0:
                    adv_for_gen = self._cast_projected_pixel_values(adv, x_orig_model)
                    generated_arm_prefix_token_ids = self._generate_action_prefix_tokens(
                        clean_ids, adv_for_gen,
                        prefix_len=max(int(target_ids.numel()) - 1, 0))
                    prefix_refresh_count += 1
                    num_generation_forwards += 1

                # --- Generated-prefix gripper loss.  The target-token objective
                # keeps arm preservation as an acceptance gate rather than a
                # competing loss term.
                adv_g = self._cast_projected_pixel_values(adv, x_orig_model)
                if is_target_token_objective:
                    gripper_loss, gripper_stats = self._generated_prefix_target_token_loss_and_stats(
                        clean_ids,
                        generated_arm_prefix_token_ids,
                        adv_g,
                        target_token_id=int(self.target_token_id),
                        margin=float(self.gripper_margin),
                    )
                else:
                    gripper_loss, gripper_stats = self._generated_prefix_gripper_loss_and_stats(
                        clean_ids, generated_arm_prefix_token_ids, adv_g,
                        region_token_ids, corrected_region_info.get("close_token_ids"),
                        margin=float(self.gripper_margin))
                grad_g = torch.autograd.grad(gripper_loss, adv,
                    retain_graph=False, create_graph=False)[0]
                gv = float(gripper_loss.detach().cpu())
                del gripper_loss, adv_g

                if is_target_token_logratio_arm_v3:
                    adv_a = self._cast_projected_pixel_values(adv, x_orig_model)
                    arm_loss, arm_stats = self._clean_generated_arm_preservation_loss_and_stats(
                        clean_ids,
                        clean_generated_action_token_ids,
                        adv_a,
                        int(target_ids.numel()),
                        arm_preserve_weight=float(self.arm_preserve_weight),
                    )
                    grad_a = torch.autograd.grad(arm_loss, adv, retain_graph=False, create_graph=False)[0]
                    av = float(arm_loss.detach().cpu())
                    grad = grad_g + grad_a
                    loss_value = gv + av
                    del grad_g, grad_a, arm_loss, adv_a
                elif is_target_token_objective:
                    arm_stats = None
                    grad = grad_g
                    loss_value = gv
                    del grad_g
                else:
                    # Forward 2: arm preservation loss (rebuild tensor after freeing graph)
                    adv_a = self._cast_projected_pixel_values(adv, x_orig_model)
                    arm_loss, arm_stats = self._arm_preservation_loss_and_stats(
                        full_ids, labels, adv_a,
                        int(target_ids.numel()), arm_preserve_weight=float(self.arm_preserve_weight))
                    grad_a = torch.autograd.grad(arm_loss, adv,
                        retain_graph=False, create_graph=False)[0]
                    av = float(arm_loss.detach().cpu())
                    del arm_loss, adv_a

                    # Combined gradient: ∇(L_gripper + L_arm) = ∇L_gripper + ∇L_arm
                    grad = grad_g + grad_a
                    loss_value = gv + av
                    del grad_g, grad_a

                if i == 0:
                    initial_loss = loss_value
                    initial_generated_stats = dict(gripper_stats)
                    initial_arm_stats = None if arm_stats is None else dict(arm_stats)
                if is_target_token_objective:
                    if self.gradient_transform not in {"", "none"}:
                        grad = shuffled_grad_direction(
                            grad,
                            seed=int(self.gradient_transform_seed) + int(i),
                            mode=self.gradient_transform,
                        )
                    target_token_objective_loss_trajectory.append(float(loss_value))
                    target_token_objective_margin_trajectory.append(
                        float(gripper_stats.get("target_objective_margin", float("nan")))
                    )
                    target_token_best_margin_trajectory.append(
                        float(gripper_stats.get("target_minus_best_competitor_margin", float("nan")))
                    )
                    target_token_logratio_margin_trajectory.append(
                        float(gripper_stats.get("target_minus_competitor_logsumexp_margin", float("nan")))
                    )
                    target_token_arm_loss_trajectory.append(
                        "" if arm_stats is None else float(arm_stats.get("arm_preservation_loss", float("nan")))
                    )
                    gradient_norm_trajectory.append(
                        {
                            "step": int(i),
                            "l1": float(grad.detach().abs().sum().cpu()),
                            "l2": float(torch.linalg.vector_norm(grad.detach().reshape(-1)).cpu()),
                            "linf": float(grad.detach().abs().max().cpu()) if grad.numel() else 0.0,
                        }
                    )
                    generated_arm_prefix_trajectory.append(
                        [int(x) for x in generated_arm_prefix_token_ids.detach().cpu().tolist()]
                    )

                adv = adv.detach() - self.step_size * grad.detach().sign()
                adv = self._project_pixel_master(adv, x_orig)
                if self.temporal_smooth_lambda > 0.0 and self._prev_delta is not None and tuple(self._prev_delta.shape) == tuple(adv.shape):
                    lam = min(max(float(self.temporal_smooth_lambda), 0.0), 1.0)
                    smoothed_delta = (1.0 - lam) * (adv.detach() - x_orig) + lam * self._prev_delta.detach().to(device=x_orig.device, dtype=torch.float32)
                    smoothed_delta = torch.clamp(smoothed_delta, -self.epsilon, self.epsilon)
                    adv = self._project_pixel_master(x_orig + smoothed_delta, x_orig).detach()
                if is_target_token_objective:
                    cand_model = self._cast_projected_pixel_values(adv.detach(), x_orig_model)
                    cand_diff = (cand_model.detach().float() - x_orig_model.detach().float()).detach()
                    trajectory_candidate_inputs.append(
                        {
                            "candidate_index": int(i + 1),
                            "candidate_source": "pgd_iteration",
                            "input_ids": clean_ids.detach(),
                            "pixel_values": cand_model.detach(),
                            "delta_sha256": tensor_sha256(cand_diff),
                            "processor_input_sha256": tensor_sha256(cand_model.detach()),
                            "pixel_budget_adv_inputs_linf": float(cand_diff.abs().max().cpu()) if cand_diff.numel() else 0.0,
                            "pixel_budget_quantized_correction_count": self._count_quantized_budget_corrections(
                                cand_model, adv.detach(), x_orig_model
                            ),
                        }
                    )
                    del cand_model, cand_diff
                del grad
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            selected_candidate_index = None
            arm_isolation_candidate_audit = None
            if self.arm_isolation_candidate_policy == "STRICT_CANDIDATE_AUDIT_V1":
                selected_candidate, arm_isolation_candidate_audit = self._select_strict_arm_candidate(
                    clean_ids,
                    trajectory_candidate_inputs,
                    clean_generated_action_token_ids,
                    arm_audit_open_token_ids,
                )
                selected_candidate_index = int(selected_candidate["candidate_index"])
                selected_candidate_adv_model = selected_candidate["pixel_values"].detach()
                adv = selected_candidate_adv_model.float()
            adv_model_for_final = (
                selected_candidate_adv_model
                if selected_candidate_adv_model is not None
                else self._cast_projected_pixel_values(adv.detach(), x_orig_model)
            )
            final_prefix = self._generate_action_prefix_tokens(
                clean_ids,
                adv_model_for_final,
                prefix_len=max(int(target_ids.numel()) - 1, 0),
            )
            num_generation_forwards += 1
            with torch.no_grad():
                if is_target_token_objective:
                    final_gripper_loss, final_generated_stats = self._generated_prefix_target_token_loss_and_stats(
                        clean_ids,
                        final_prefix,
                        adv_model_for_final,
                        target_token_id=int(self.target_token_id),
                        margin=float(self.gripper_margin),
                    )
                else:
                    final_gripper_loss, final_generated_stats = self._generated_prefix_gripper_loss_and_stats(
                        clean_ids,
                        final_prefix,
                        adv_model_for_final,
                        region_token_ids,
                        corrected_region_info.get("close_token_ids"),
                        margin=float(self.gripper_margin),
                    )
                if is_target_token_logratio_arm_v3:
                    final_arm_loss, final_arm_stats = self._clean_generated_arm_preservation_loss_and_stats(
                        clean_ids,
                        clean_generated_action_token_ids,
                        adv_model_for_final,
                        int(target_ids.numel()),
                        arm_preserve_weight=float(self.arm_preserve_weight),
                    )
                else:
                    final_arm_loss, final_arm_stats = self._arm_preservation_loss_and_stats(
                        full_ids,
                        labels,
                        adv_model_for_final,
                        int(target_ids.numel()),
                        arm_preserve_weight=float(self.arm_preserve_weight),
                    )
                final_loss = float(
                    (final_gripper_loss + final_arm_loss).detach().cpu()
                    if is_target_token_logratio_arm_v3
                    else final_gripper_loss.detach().cpu()
                    if is_target_token_objective
                    else (final_gripper_loss + final_arm_loss).detach().cpu()
                )
            teacher_final = (
                None
                if is_target_token_objective
                else self._teacher_forced_gripper_margin_stats(
                    full_ids, adv_model_for_final, int(target_ids.numel()), region_token_ids, corrected_region_info.get("close_token_ids"))
            )
            generated_arm_prefix_final = [int(x) for x in final_prefix.detach().cpu().tolist()]
            n_arm = min(len(clean_arm_prefix_token_ids), len(generated_arm_prefix_final))
            arm_match_rate = (
                float(sum(int(clean_arm_prefix_token_ids[j] == generated_arm_prefix_final[j]) for j in range(n_arm)) / max(n_arm, 1))
                if n_arm else 0.0
            )
            retok_n_arm = min(len(retokenized_clean_action_arm_token_ids), len(generated_arm_prefix_final))
            retokenized_arm_match_rate = (
                float(
                    sum(
                        int(retokenized_clean_action_arm_token_ids[j] == generated_arm_prefix_final[j])
                        for j in range(retok_n_arm)
                    )
                    / max(retok_n_arm, 1)
                )
                if retok_n_arm else 0.0
            )
            generated_prefix_debug = {
                "objective_name": objective,
                "method_version": "generated_prefix_stop_gradient_v1",
                "prefix_refresh_strategy": "every_k_pgd_steps",
                "prefix_refresh_interval": int(prefix_refresh_interval),
                "prefix_refresh_count": int(prefix_refresh_count),
                "retokenized_clean_action_token_ids": retokenized_clean_action_token_ids,
                "retokenized_clean_action_arm_token_ids": retokenized_clean_action_arm_token_ids,
                "clean_generated_action_token_ids": None
                if clean_generated_action_token_ids is None
                else [int(x) for x in clean_generated_action_token_ids.detach().cpu().tolist()],
                "clean_generated_arm_prefix_token_ids": clean_generated_arm_prefix_token_ids,
                "generated_arm_prefix_token_ids": generated_arm_prefix_final,
                "generated_adv_arm_prefix_token_ids": generated_arm_prefix_final,
                "generated_vs_clean_generated_arm_match_rate": arm_match_rate if is_target_token_objective else None,
                "generated_vs_retokenized_arm_match_rate": retokenized_arm_match_rate,
                "teacher_forced_margin_clean_x0": None if teacher_initial is None else teacher_initial.get("open_minus_close_margin"),
                "teacher_forced_gripper_margin_final": None if teacher_final is None else teacher_final.get("open_minus_close_margin"),
                "generated_prefix_gripper_margin_initial": (
                    (initial_generated_stats or {}).get("target_objective_margin")
                    if is_target_token_objective
                    else (initial_generated_stats or {}).get("open_minus_close_margin")
                ),
                "generated_prefix_gripper_margin_final": (
                    (final_generated_stats or {}).get("target_objective_margin")
                    if is_target_token_objective
                    else (final_generated_stats or {}).get("open_minus_close_margin")
                ),
                "selected_loss_initial": initial_loss,
                "selected_loss_final": final_loss,
                "arm_preservation_loss_initial": (initial_arm_stats or {}).get("arm_preservation_loss"),
                "arm_preservation_loss_final": final_arm_stats.get("arm_preservation_loss"),
                "num_generation_forwards": int(num_generation_forwards),
                "teacher_forced_gripper_stats_initial": teacher_initial,
                "teacher_forced_gripper_stats_final": teacher_final,
                "generated_prefix_gripper_stats_initial": initial_generated_stats,
                "generated_prefix_gripper_stats_final": final_generated_stats,
                "arm_preservation_debug_final": final_arm_stats,
                "generated_prefix_stop_gradient": True,
                "gradient_through_generated_token_ids": False,
            }
            if is_target_token_objective:
                generated_prefix_debug.update({
                    "target_token_id": int(self.target_token_id),
                    "target_execution_class": self.target_execution_class,
                    "surrogate_score_path": self.surrogate_score_path,
                    "target_token_objective": objective,
                    "target_token_objective_margin_name": (final_generated_stats or {}).get("target_objective_margin_name"),
                    "target_token_objective_margin_initial": (initial_generated_stats or {}).get("target_objective_margin"),
                    "target_token_objective_margin_final": (final_generated_stats or {}).get("target_objective_margin"),
                    "arm_preservation_role": "combined_gradient_penalty" if is_target_token_logratio_arm_v3 else "acceptance_gate_not_primary_loss",
                    "arm_preserve_weight": float(self.arm_preserve_weight),
                    "arm_gate_reference": "clean_actual_generation",
                    "arm_prefix_match_count": int(sum(int(clean_arm_prefix_token_ids[j] == generated_arm_prefix_final[j]) for j in range(n_arm))) if n_arm else 0,
                    "arm_prefix_match_denominator": int(n_arm),
                    "retokenized_arm_prefix_match_count": int(sum(int(retokenized_clean_action_arm_token_ids[j] == generated_arm_prefix_final[j]) for j in range(retok_n_arm))) if retok_n_arm else 0,
                    "retokenized_arm_prefix_match_denominator": int(retok_n_arm),
                    "target_token_objective_loss_initial": initial_loss,
                    "target_token_objective_loss_final": final_loss,
                    "target_token_objective_loss_trajectory": target_token_objective_loss_trajectory,
                    "target_token_objective_margin_trajectory": target_token_objective_margin_trajectory,
                    "target_token_arm_preservation_loss_trajectory": target_token_arm_loss_trajectory,
                    "target_token_arm_preservation_loss_initial": None if initial_arm_stats is None else initial_arm_stats.get("arm_preservation_loss"),
                    "target_token_arm_preservation_loss_final": final_arm_stats.get("arm_preservation_loss"),
                    "gradient_norm_trajectory": gradient_norm_trajectory,
                    "generated_arm_prefix_trajectory": generated_arm_prefix_trajectory,
                    "gradient_transform": self.gradient_transform,
                    "gradient_transform_seed": int(self.gradient_transform_seed),
                })
                if is_target_token_cw_v1:
                    generated_prefix_debug.update({
                        "target_token_cw_margin_initial": (initial_generated_stats or {}).get("target_minus_best_competitor_margin"),
                        "target_token_cw_margin_final": (final_generated_stats or {}).get("target_minus_best_competitor_margin"),
                        "target_token_cw_loss_initial": initial_loss,
                        "target_token_cw_loss_final": final_loss,
                        "target_token_cw_loss_trajectory": target_token_objective_loss_trajectory,
                        "target_token_cw_margin_trajectory": target_token_best_margin_trajectory,
                    })
                if is_target_token_logratio_v2 or is_target_token_logratio_arm_v3:
                    generated_prefix_debug.update({
                        "target_token_logratio_margin_initial": (initial_generated_stats or {}).get("target_minus_competitor_logsumexp_margin"),
                        "target_token_logratio_margin_final": (final_generated_stats or {}).get("target_minus_competitor_logsumexp_margin"),
                        "target_token_logratio_loss_initial": initial_loss,
                        "target_token_logratio_loss_final": final_loss,
                        "target_token_logratio_loss_trajectory": target_token_objective_loss_trajectory,
                        "target_token_logratio_margin_trajectory": target_token_logratio_margin_trajectory,
                    })
                if is_target_token_logratio_arm_v3:
                    generated_prefix_debug.update({
                        "target_token_logratio_arm_v3": True,
                        "target_token_logratio_arm_loss_initial": initial_loss,
                        "target_token_logratio_arm_loss_final": final_loss,
                    })
        else:
            for i in range(max(self.num_steps, 1)):
                adv = adv.detach().requires_grad_(True)
                adv_for_loss = self._cast_projected_pixel_values(adv, x_orig_model)
                loss = self._loss(full_ids, labels, adv_for_loss, **loss_kwargs)
                if i == 0:
                    initial_loss = float(loss.detach().cpu())
                if is_prefix_locked and hasattr(loss, "_prefix_debug"):
                    _prefix_debug_final = loss._prefix_debug
                grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
                if is_untargeted:
                    # Maximize CE of the clean action-token prefix.
                    adv = adv.detach() + self.step_size * grad.detach().sign()
                else:
                    # Minimize target CE: signed gradient descent.
                    adv = adv.detach() - self.step_size * grad.detach().sign()
                adv = self._project_pixel_master(adv, x_orig)
                # ``pixel_values`` are processor-normalized OpenVLA inputs, not raw
                # RGB values.  Clamping them to [0, 1] can create a perturbation far
                # larger than epsilon whenever normalized pixels are negative.  The
                # budget enforced here is therefore Linf in processor pixel space.
                if self.temporal_smooth_lambda > 0.0 and self._prev_delta is not None and tuple(self._prev_delta.shape) == tuple(adv.shape):
                    lam = min(max(float(self.temporal_smooth_lambda), 0.0), 1.0)
                    smoothed_delta = (1.0 - lam) * (adv.detach() - x_orig) + lam * self._prev_delta.detach().to(device=x_orig.device, dtype=torch.float32)
                    smoothed_delta = torch.clamp(smoothed_delta, -self.epsilon, self.epsilon)
                    adv = self._project_pixel_master(x_orig + smoothed_delta, x_orig).detach()
                del grad, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        adv_model = (
            selected_candidate_adv_model
            if selected_candidate_adv_model is not None
            else self._cast_projected_pixel_values(adv.detach(), x_orig_model)
        )
        if not (is_generated_prefix_v3 or is_target_token_objective):
            with torch.no_grad():
                final_loss = float(self._loss(full_ids, labels, adv_model, **loss_kwargs).detach().cpu())
        postprocess_gripper = bool(self.postprocess_gripper)
        clean_audit = self._audit_logits(full_ids, labels, x_orig_model, target_ids, unnorm_key, postprocess_gripper=postprocess_gripper, region_token_ids=region_token_ids)
        adv_audit = self._audit_logits(full_ids, labels, adv_model, target_ids, unnorm_key, postprocess_gripper=postprocess_gripper, region_token_ids=region_token_ids)
        diff = (adv_model.detach().float() - x_orig_model.detach().float()).detach()
        master_diff = (self._project_pixel_master(adv.detach(), x_orig) - x_orig).detach().float()
        self._prev_delta = diff.detach()
        adv_inputs = {"input_ids": clean_ids.detach(), "pixel_values": adv_model.detach()}
        delta0_adv_inputs = {"input_ids": clean_ids.detach(), "pixel_values": delta0_adv_model.detach()}
        token_list = [int(x) for x in target_ids.detach().cpu().tolist()]
        quantized_correction_count = self._count_quantized_budget_corrections(adv_model, adv.detach(), x_orig_model)
        debug={
            "adv_inputs": adv_inputs,
            "delta0_adv_inputs": delta0_adv_inputs,
            "attack_objective": objective,
            "loss_direction": "maximize" if is_untargeted else "minimize",
            "token_label_source": token_label_source,
            "pixel_space": "processor_pixel_values",
            "pixel_epsilon_space": "processor_pixel_values_linf",
            "pixel_value_clamp": "project_to_x_orig_plusminus_epsilon_only",
            "pixel_master_dtype": "torch.float32",
            "pixel_model_dtype": str(x_orig_model.dtype),
            "pixel_budget_master_linf": float(master_diff.abs().max().cpu()) if master_diff.numel() else 0.0,
            "pixel_budget_adv_inputs_linf": float(diff.abs().max().cpu()) if diff.numel() else 0.0,
            "pixel_budget_delta0_adv_inputs_linf": float(delta0_diff.abs().max().cpu()) if delta0_diff.numel() else 0.0,
            "pixel_budget_quantized_correction_count": int(quantized_correction_count),
            "pixel_budget_quantized_correction_rate": float(quantized_correction_count / max(int(diff.numel()), 1)),
            "num_loss_forwards": int(max(self.num_steps, 1) + 1),
            "num_backwards": int(max(self.num_steps, 1)),
            "num_adv_decodes": 1,
            "temporal_init": self.temporal_init,
            "temporal_prev_delta_used": bool(temporal_prev_delta_used),
            "temporal_smooth_lambda": float(self.temporal_smooth_lambda),
            "temporal_prev_delta_linf": float(delta.detach().abs().max().cpu()) if delta.numel() else 0.0,
            "clean_logit_audit": clean_audit,
            "adv_logit_audit": adv_audit,
            "delta0_sha256": tensor_sha256(delta0_diff),
            "delta_final_sha256": tensor_sha256(diff),
            "delta0_processor_input_sha256": tensor_sha256(delta0_adv_model.detach()),
            "processor_input_sha256": tensor_sha256(adv_model.detach()),
            "arm_isolation_candidate_policy": self.arm_isolation_candidate_policy,
            "arm_isolation_candidate_audit": arm_isolation_candidate_audit,
            "selected_candidate_index": selected_candidate_index,
        }
        if is_generated_prefix_v3 or is_target_token_objective:
            debug.update(generated_prefix_debug)
            if is_target_token_objective:
                debug["trajectory_candidate_inputs"] = trajectory_candidate_inputs
                debug["trajectory_candidate_count"] = int(len(trajectory_candidate_inputs))
            debug["num_generation_forwards"] = int(generated_prefix_debug.get("num_generation_forwards", 0) or 0)
            debug["num_loss_forwards"] = (
                int(max(self.num_steps, 1) + 1)
                if is_target_token_objective
                else int(2 * max(self.num_steps, 1) + 2)
            )
        if is_untargeted:
            debug.update({
                "clean_token_label_ids": token_list,
                "clean_ce_initial": initial_loss,
                "clean_ce_final": final_loss,
                "arm_only_untargeted": bool(is_arm_only_untargeted),
                "gripper_dim_masked_from_loss": bool(is_arm_only_untargeted),
            })
        else:
            debug.update({
                "target_token_ids": token_list,
                "target_ce_initial": None if is_target_token_objective else initial_loss,
                "target_ce_final": None if is_target_token_objective else final_loss,
            })
            if is_target_token_cw_v1:
                debug["target_token_cw_loss_initial"] = initial_loss
                debug["target_token_cw_loss_final"] = final_loss
            if is_target_token_logratio_v2 or is_target_token_logratio_arm_v3:
                debug["target_token_logratio_loss_initial"] = initial_loss
                debug["target_token_logratio_loss_final"] = final_loss
            # Gripper-specific restart-selection metrics (now using corrected region)
            if corrected_region_info is not None:
                debug["corrected_open_token_count"] = corrected_region_info["open_count"]
                debug["corrected_close_token_count"] = corrected_region_info["close_count"]
                debug["corrected_boundary_tokens"] = corrected_region_info["boundary_token_ids"]
                debug["region_mapping_status"] = "corrected_decoded_action_20260602"
            gripper_adv_audit = adv_audit.get("action_token_logit_audit", [])
            if gripper_adv_audit:
                _gadv = gripper_adv_audit[-1]  # last dim = gripper
                debug["open_region_prob_mass_after"] = _gadv.get("open_bin_prob_mass", None)
                debug["close_bin_prob_mass_after"] = _gadv.get("close_bin_prob_mass", None)
                _open_mass = float(_gadv.get("open_bin_prob_mass", 0.0) or 0.0)
                _close_mass = float(_gadv.get("close_bin_prob_mass", 0.0) or 0.0)
                # Probability mass margin (not logit margin — see gripper_logit_margin_after)
                debug["gripper_prob_mass_margin_after"] = _open_mass - _close_mass
                debug["gripper_open_prob_mass"] = _open_mass
                # True logit margin: logsumexp(open) - max(non-open)
                _gripper_open_lse = _gadv.get("open_region_logsumexp")
                _gripper_non_open_max = _gadv.get("non_open_max_logit")
                if _gripper_open_lse is not None and _gripper_non_open_max is not None:
                    debug["gripper_logit_margin_after"] = float(_gripper_open_lse) - float(_gripper_non_open_max)
            if is_prefix_locked:
                debug["arm_preserve_weight"] = float(self.arm_preserve_weight)
                debug["gripper_margin_param"] = float(self.gripper_margin)
                debug["best_restart_metric"] = self.best_restart_metric
                if _prefix_debug_final is not None:
                    debug.update(_prefix_debug_final)
            if is_force_gripper_open or is_force_open_z_down or is_gripper_margin or is_gripper_region or is_prefix_locked or is_corrected_hybrid or is_generated_prefix_v3 or is_target_token_objective:
                debug.update({
                    "target_gripper_token_id": None if is_target_token_objective else (int(token_list[-1]) if token_list else None),
                    "attack_target_gripper_token_id": int(self.target_token_id) if is_target_token_objective else (int(token_list[-1]) if token_list else None),
                    "arm_reference_retokenized_gripper_token_id": int(token_list[-1]) if token_list else None,
                    "gripper_only_loss": bool(is_force_gripper_open or is_gripper_margin or is_gripper_region or is_prefix_locked or is_generated_prefix_v3 or is_target_token_objective),
                    "z_and_gripper_loss": bool(is_force_open_z_down),
                    "corrected_open_region_z_down": bool(is_corrected_hybrid),
                    "gripper_logit_margin_loss": bool(is_gripper_margin or is_prefix_locked_open_margin),
                    "gripper_open_region_loss": bool(is_gripper_region or is_prefix_locked_open_region),
                    "gripper_top1_open_vs_close_loss": bool(is_prefix_locked_top1),
                    "autoregressive_prefix_gripper_open_loss": bool(is_generated_prefix_v3),
                    "autoregressive_prefix_target_token_cw_loss": bool(is_target_token_cw_v1),
                    "autoregressive_prefix_target_token_logratio_loss": bool(is_target_token_logratio_v2 or is_target_token_logratio_arm_v3),
                    "autoregressive_prefix_target_token_logratio_arm_loss": bool(is_target_token_logratio_arm_v3),
                    "gripper_expected_action_loss": bool(is_gripper_expected_action),
                    "prefix_locked_arm_preserve": bool(is_prefix_locked or is_generated_prefix_v3),
                    "arm_preservation_as_acceptance_gate": bool(is_target_token_objective),
                })
                if _needs_region and region_token_ids is not None:
                    vals = [int(x) for x in region_token_ids.detach().cpu().tolist()]
                    debug["gripper_open_region_token_ids"] = vals
                    debug["gripper_open_region_token_count"] = int(len(vals))
                if is_force_open_z_down and len(token_list) > 2:
                    debug["target_z_token_id"] = int(token_list[2])
                    # P0: old force_open_z_down uses target_action[-1]=1.0 which may target CLOSE.
                    # Mark as deprecated; use force_open_region_z_down_ce for corrected hybrid.
                    debug["force_open_z_down_gripper_target_status"] = "deprecated_unsafe_old_target_semantics"
                if is_corrected_hybrid:
                    debug["force_open_z_down_gripper_target_status"] = "corrected_open_region_z_down_20260602"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # TokenPrefixPGD produces adversarial processor inputs, not an already
        # decoded actuator action. Downstream evaluation must re-decode from
        # debug["adv_inputs"]; action_adv intentionally remains None.
        return AttackResult(
            x_adv=None,
            action_adv=None,
            attack_method=(
                ("token_prefix_pgd_pixel_values_untargeted_arm_only_clean_ce" if is_arm_only_untargeted else "token_prefix_pgd_pixel_values_untargeted_clean_ce")
                if is_untargeted
                else (
                    "token_prefix_pgd_pixel_values_autoregressive_prefix_v3"
                    if is_generated_prefix_v3
                    else (
                        (
                            "token_prefix_pgd_pixel_values_target_token_logratio_arm_v3"
                            if is_target_token_logratio_arm_v3
                            else "token_prefix_pgd_pixel_values_target_token_logratio_v2"
                            if is_target_token_logratio_v2
                            else "token_prefix_pgd_pixel_values_target_token_cw_v1"
                        )
                        if is_target_token_objective
                        else ("token_prefix_pgd_pixel_values_gripper_only" if (is_force_gripper_open or is_gripper_margin or is_gripper_region) else "token_prefix_pgd_pixel_values")
                    )
                )
            ),
            directional_loss_available=False if is_untargeted else True,
            num_attack_steps=self.num_steps,
            epsilon=self.epsilon,
            step_size=self.step_size,
            observation_perturb_linf=float(diff.abs().max().cpu()) if diff.numel() else 0.0,
            observation_perturb_l2=float(torch.linalg.vector_norm(diff.reshape(-1)).cpu()) if diff.numel() else 0.0,
            debug=debug,
        )


class OpenVLAVisualAttacker:
    def __init__(self, model=None, processor=None, config: dict | None = None, direction_spec=None, seed: int = 0, preprocess_kwargs: Optional[Dict[str, Any]] = None, device: Optional[str] = None):
        cfg = (config or {}).get("attack_optimizer", config or {})
        self.route = route_config_from_attack_config(config or {})
        resolved = resolve_adapter_class_name(self.route)
        validate_attack_request(self.route, target_action_present=True)
        method = self.route.requested_method or "visual_linf_noise_adapter"
        self.method = method
        self.resolved_adapter_class = resolved
        if resolved == "TokenPrefixPGDAttacker":
            self.adapter = TokenPrefixPGDAttacker(model, processor, config or {}, seed=seed, preprocess_kwargs=preprocess_kwargs, device=device)
        else:
            self.adapter = ExistingDenseAttackAdapter(
                epsilon=cfg.get("epsilon", 0.03),
                step_size=cfg.get("step_size", 0.006),
                num_steps=cfg.get("num_steps", 5),
                seed=seed,
            )

    def reset_temporal_state(self):
        reset = getattr(self.adapter, "reset_temporal_state", None)
        if callable(reset):
            reset()

    def attack(
        self,
        observation,
        instruction,
        clean_action,
        target_action,
        clean_model_output=None,
        *,
        unnorm_key: str = "libero_goal",
        execution_trace: Optional[Dict[str, Any]] = None,
    ) -> AttackResult:
        validate_attack_request(self.route, target_action_present=target_action is not None)
        if execution_trace is not None:
            execution_trace["attack_invocation_started"] = True
        try:
            if self.route.strict_route:
                result = self.adapter.attack(
                    observation,
                    instruction,
                    clean_action,
                    target_action,
                    clean_model_output,
                    unnorm_key=unnorm_key,
                )
            else:
                try:
                    result = self.adapter.attack(observation, instruction, clean_action, target_action, clean_model_output, unnorm_key=unnorm_key)
                except TypeError:
                    result = self.adapter.attack(observation, instruction, clean_action, target_action, clean_model_output)
        except Exception as exc:
            if execution_trace is not None:
                execution_trace["attack_error"] = f"{type(exc).__name__}:{exc}"
                diagnostics = getattr(self.adapter, "last_attack_diagnostics", None)
                if diagnostics:
                    execution_trace["attack_contract_diagnostics"] = diagnostics
            raise
        if execution_trace is not None:
            execution_trace["attack_result_returned"] = True
            debug_returned = getattr(result, "debug", {}) or {}
            for source, target in (
                ("num_backwards", "backward_invocation_count"),
                ("num_loss_forwards", "loss_forward_count"),
            ):
                if source in debug_returned:
                    execution_trace[target] = int(debug_returned[source])
        fallback_reason = (result.debug or {}).get("fallback_reason")
        fallback_used = bool(self.resolved_adapter_class != "TokenPrefixPGDAttacker" or fallback_reason)
        result.debug = attach_route_debug(
            result.debug or {},
            self.route,
            resolved_adapter_class=self.resolved_adapter_class,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            target_action_present=target_action is not None,
        )
        result.debug["x_adv_is_none"] = result.x_adv is None
        result.debug["action_adv_is_none"] = result.action_adv is None
        if self.route.strict_route:
            try:
                validate_true_pgd_attack_result(result, self.route)
            except Exception:
                if execution_trace is not None:
                    execution_trace["attack_result_accepted"] = False
                raise
        if execution_trace is not None:
            execution_trace["attack_result_accepted"] = True
        return result
