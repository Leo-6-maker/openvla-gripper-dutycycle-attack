from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .types import AttackResult


def _prompt(instruction: str) -> str:
    return f"In: What action should the robot take to {str(instruction).lower()}?\nOut:"


def _pil_center_crop_resize(image: Image.Image, crop_scale: float = 0.9, size: int = 224) -> Image.Image:
    if crop_scale is None or float(crop_scale) >= 0.999:
        return image.resize((size, size), Image.Resampling.LANCZOS)
    w, h = image.size
    scale = float(crop_scale) ** 0.5
    cw, ch = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    left, top = (w - cw) // 2, (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch)).resize((size, size), Image.Resampling.LANCZOS)


def prepare_openvla_image_for_attack(image_np, *, libero_official_preprocess: bool = False, center_crop: bool = False, resize_size: int = 224) -> Image.Image:
    arr = np.asarray(image_np)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if libero_official_preprocess:
        arr = arr[::-1, ::-1]
    image = Image.fromarray(arr).convert("RGB")
    if libero_official_preprocess:
        image = image.resize((int(resize_size), int(resize_size)), Image.Resampling.LANCZOS)
    if center_crop:
        image = _pil_center_crop_resize(image, crop_scale=0.9, size=int(resize_size))
    return image


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
        self.seed = int(seed)
        self.preprocess_kwargs = dict(preprocess_kwargs or {})
        self.postprocess_gripper = bool(self.preprocess_kwargs.pop("postprocess_gripper", False))
        self.device = device or "cuda:0"
        self.config = cfg
        self._frozen = False

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
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=torch.float16)
        return input_ids, full_input_ids, labels, pixel_values

    def action_bins_for_env_sign(self, dim: int, target_env_sign: str, unnorm_key: str, *, postprocess_gripper: bool = False) -> torch.LongTensor:
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

    def _active_label_rows(self, logits, labels, action_dim: int):
        action_start = int(labels.shape[1]) - int(action_dim)
        active = (labels != -100).nonzero(as_tuple=False)
        rows = []
        for b, label_pos in active:
            dim = int(label_pos.item()) - action_start
            if dim < 0 or dim >= int(action_dim):
                continue
            # The logit that predicts action token dim is emitted by the previous
            # text/action token. OpenVLA's forward logits include visual tokens,
            # so address from the suffix rather than aligning against text labels.
            row_index = -(int(action_dim) - dim + 1)
            if abs(row_index) > int(logits.shape[1]):
                continue
            rows.append((int(b.item()), int(label_pos.item()), dim, row_index))
        return rows

    def _loss(self, full_input_ids, labels, pixel_values, *, objective: str = "targeted_directional_ce", region_token_ids=None, margin: float = 5.0, num_action_tokens: int = 7):
        obj = str(objective)
        if obj not in {"gripper_open_region_ce", "gripper_logit_margin_cw"}:
            # Keep the proven OpenVLA/HF label path for ordinary targeted CE;
            # hand-aligning visual-token logits against text labels is brittle.
            out = self.model(input_ids=full_input_ids, pixel_values=pixel_values, labels=labels, use_cache=False, return_dict=True)
            if out.loss is not None:
                return out.loss
            logits = out.logits[:, :-1, :].contiguous()
            shifted = labels[:, 1:].contiguous()
            return F.cross_entropy(logits.view(-1, logits.shape[-1]), shifted.view(-1), ignore_index=-100)
        out = self.model(input_ids=full_input_ids, pixel_values=pixel_values, use_cache=False, return_dict=True)
        logits = out.logits.float().contiguous()
        action_dim = max(int(num_action_tokens), 1)
        rows = self._active_label_rows(logits, labels, max(action_dim, 1))
        if not rows:
            return logits.sum() * 0.0
        losses = []
        for b, label_pos, dim, row_index in rows:
            row = logits[b, row_index, :]
            target = labels[b, label_pos]
            if obj == "gripper_open_region_ce":
                if region_token_ids is None or int(region_token_ids.numel()) == 0:
                    losses.append(F.cross_entropy(row.view(1, -1), target.view(1)))
                else:
                    log_region = torch.logsumexp(row[region_token_ids], dim=0)
                    log_all = torch.logsumexp(row, dim=0)
                    losses.append(-(log_region - log_all))
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
            open_tokens = self.action_bins_for_env_sign(action_dim - 1, "negative", unnorm_key, postprocess_gripper=postprocess_gripper)
            close_tokens = self.action_bins_for_env_sign(action_dim - 1, "positive", unnorm_key, postprocess_gripper=postprocess_gripper)
            action_start = int(labels.shape[1]) - action_dim
            active = (labels != -100).nonzero(as_tuple=False)
            for b, label_pos in active:
                dim = int(label_pos.item()) - action_start
                if dim < 0 or dim >= action_dim:
                    continue
                row_index = -(action_dim - dim + 1)
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
                        "bin_mapping": "env_negative_open_positive_closed_after_postprocess" if postprocess_gripper else "raw_negative_open_positive_closed",
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

    def attack(self, observation: Any, instruction=None, clean_action=None, target_action=None, clean_model_output=None, *, unnorm_key: str = "libero_goal") -> AttackResult:
        objective = str(getattr(self, "objective", "targeted_directional_ce"))
        is_untargeted = objective in {"untargeted_clean_token_ce", "untargeted_clean_ce", "maximize_clean_ce", "untargeted_arm_clean_token_ce", "ctrl_random_direction_arm_only"}
        is_arm_only_untargeted = objective in {"untargeted_arm_clean_token_ce", "ctrl_random_direction_arm_only"}
        is_force_gripper_open = objective in {"force_gripper_open_token_ce", "force_gripper_open", "targeted_gripper_open_ce", "adaptive_anti_gripper_token_ce"}
        is_force_open_z_down = objective in {"force_open_z_down_token_ce"}
        is_gripper_margin = objective in {"gripper_logit_margin_cw"}
        is_gripper_region = objective in {"gripper_open_region_ce"}
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
        if is_force_gripper_open or is_force_open_z_down or is_gripper_margin or is_gripper_region:
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
        x_orig = x0.detach()
        gen = torch.Generator(device=x_orig.device); gen.manual_seed(self.seed)
        temporal_prev_delta_used = False
        if self.temporal_init in {"prev_delta", "previous_delta", "carry", "carryover"} and self._prev_delta is not None and tuple(self._prev_delta.shape) == tuple(x_orig.shape):
            delta = torch.clamp(self._prev_delta.detach().to(device=x_orig.device, dtype=x_orig.dtype), -self.epsilon, self.epsilon)
            temporal_prev_delta_used = True
        elif self.random_start:
            delta = torch.empty_like(x_orig).uniform_(-self.epsilon, self.epsilon, generator=gen)
        else:
            delta = torch.zeros_like(x_orig)
        adv = (x_orig + delta).detach()
        loss_kwargs = {"objective": objective, "num_action_tokens": int(target_ids.numel())}
        region_token_ids = None
        if is_gripper_region:
            region_token_ids = self.action_bins_for_env_sign(
                int(target_ids.numel()) - 1,
                "negative",
                unnorm_key,
                postprocess_gripper=bool(self.postprocess_gripper),
            )
            loss_kwargs["region_token_ids"] = region_token_ids
        if is_gripper_margin:
            loss_kwargs["margin"] = float((getattr(self, "config", {}) or {}).get("cw_margin", 5.0)) if hasattr(self, "config") else 5.0
        initial_loss = None; final_loss = None
        for i in range(max(self.num_steps, 1)):
            adv = adv.detach().requires_grad_(True)
            loss = self._loss(full_ids, labels, adv, **loss_kwargs)
            if i == 0:
                initial_loss = float(loss.detach().cpu())
            grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False)[0]
            if is_untargeted:
                # Maximize CE of the clean action-token prefix.
                adv = adv.detach() + self.step_size * grad.detach().sign()
            else:
                # Minimize target CE: signed gradient descent.
                adv = adv.detach() - self.step_size * grad.detach().sign()
            adv = torch.max(torch.min(adv, x_orig + self.epsilon), x_orig - self.epsilon)
            adv = torch.clamp(adv, 0.0, 1.0)  # valid pixel range for processor inputs
            if self.temporal_smooth_lambda > 0.0 and self._prev_delta is not None and tuple(self._prev_delta.shape) == tuple(adv.shape):
                lam = min(max(float(self.temporal_smooth_lambda), 0.0), 1.0)
                smoothed_delta = (1.0 - lam) * (adv.detach() - x_orig) + lam * self._prev_delta.detach().to(device=x_orig.device, dtype=x_orig.dtype)
                smoothed_delta = torch.clamp(smoothed_delta, -self.epsilon, self.epsilon)
                adv = (x_orig + smoothed_delta).detach()
            del grad, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        with torch.no_grad():
            final_loss = float(self._loss(full_ids, labels, adv, **loss_kwargs).detach().cpu())
        postprocess_gripper = bool(self.postprocess_gripper)
        clean_audit = self._audit_logits(full_ids, labels, x_orig, target_ids, unnorm_key, postprocess_gripper=postprocess_gripper, region_token_ids=region_token_ids)
        adv_audit = self._audit_logits(full_ids, labels, adv, target_ids, unnorm_key, postprocess_gripper=postprocess_gripper, region_token_ids=region_token_ids)
        diff = (adv - x_orig).detach().float()
        self._prev_delta = (adv.detach() - x_orig).detach()
        adv_inputs = {"input_ids": clean_ids.detach(), "pixel_values": adv.detach()}
        token_list = [int(x) for x in target_ids.detach().cpu().tolist()]
        debug={
            "adv_inputs": adv_inputs,
            "attack_objective": objective,
            "loss_direction": "maximize" if is_untargeted else "minimize",
            "token_label_source": token_label_source,
            "pixel_space": "processor_pixel_values",
            "num_loss_forwards": int(max(self.num_steps, 1) + 1),
            "num_backwards": int(max(self.num_steps, 1)),
            "num_adv_decodes": 1,
            "temporal_init": self.temporal_init,
            "temporal_prev_delta_used": bool(temporal_prev_delta_used),
            "temporal_smooth_lambda": float(self.temporal_smooth_lambda),
            "temporal_prev_delta_linf": float(delta.detach().abs().max().cpu()) if delta.numel() else 0.0,
            "clean_logit_audit": clean_audit,
            "adv_logit_audit": adv_audit,
        }
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
                "target_ce_initial": initial_loss,
                "target_ce_final": final_loss,
            })
            if is_force_gripper_open or is_force_open_z_down or is_gripper_margin or is_gripper_region:
                debug.update({
                    "target_gripper_token_id": int(token_list[-1]) if token_list else None,
                    "gripper_only_loss": bool(is_force_gripper_open or is_gripper_margin or is_gripper_region),
                    "z_and_gripper_loss": bool(is_force_open_z_down),
                    "gripper_logit_margin_loss": bool(is_gripper_margin),
                    "gripper_open_region_loss": bool(is_gripper_region),
                })
                if is_gripper_region and region_token_ids is not None:
                    vals = [int(x) for x in region_token_ids.detach().cpu().tolist()]
                    debug["gripper_open_region_token_ids"] = vals
                    debug["gripper_open_region_token_count"] = int(len(vals))
                if is_force_open_z_down and len(token_list) > 2:
                    debug["target_z_token_id"] = int(token_list[2])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return AttackResult(
            x_adv=None,
            action_adv=None,
            attack_method=(
                ("token_prefix_pgd_pixel_values_untargeted_arm_only_clean_ce" if is_arm_only_untargeted else "token_prefix_pgd_pixel_values_untargeted_clean_ce")
                if is_untargeted
                else ("token_prefix_pgd_pixel_values_gripper_only" if (is_force_gripper_open or is_gripper_margin or is_gripper_region) else "token_prefix_pgd_pixel_values")
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
        method = str(cfg.get("method", "visual_linf_noise_adapter"))
        self.method = method
        if method in {"token_prefix_pgd", "openvla_token_prefix_pgd", "visual_token_prefix_pgd", "untargeted_token_prefix_pgd"}:
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

    def attack(self, observation, instruction, clean_action, target_action, clean_model_output=None, *, unnorm_key: str = "libero_goal") -> AttackResult:
        try:
            return self.adapter.attack(observation, instruction, clean_action, target_action, clean_model_output, unnorm_key=unnorm_key)
        except TypeError:
            return self.adapter.attack(observation, instruction, clean_action, target_action, clean_model_output)
