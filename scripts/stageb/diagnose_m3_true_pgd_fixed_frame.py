from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from gripper_attack.attack_adapter import get_adv_inputs_from_attack_result
from gripper_attack.route_contract import RouteContractError


def extract_exact_new_tokens(sequences: torch.Tensor, prompt_len: int, *, expected_len: int = 7) -> list[int]:
    if not torch.is_tensor(sequences) or sequences.ndim != 2:
        raise ValueError("generation sequences must be a [batch, tokens] tensor")
    new_count = int(sequences.shape[1]) - int(prompt_len)
    if new_count != int(expected_len):
        raise ValueError(f"expected {expected_len} new tokens, got {new_count}")
    return [int(x) for x in sequences[0, int(prompt_len):].detach().cpu().tolist()]


def validate_processed_argmax_matches_emitted(score_row: torch.Tensor, emitted_token: int, *, tolerance: float = 0.0) -> dict[str, Any]:
    row = score_row.detach().float().cpu()
    emitted = int(emitted_token)
    if emitted < 0 or emitted >= int(row.numel()):
        raise ValueError("emitted token outside score row")
    max_score = float(row.max().item())
    emitted_score = float(row[emitted].item())
    argmax_token = int(torch.argmax(row).item())
    tie_aware_ok = emitted_score >= max_score - float(tolerance)
    strict_ok = argmax_token == emitted
    if not strict_ok and not tie_aware_ok:
        raise ValueError(
            f"processed-score argmax {argmax_token} does not match emitted {emitted}"
        )
    return {
        "strict_argmax_match": bool(strict_ok),
        "tie_aware_pass": bool(tie_aware_ok),
        "argmax_token": argmax_token,
        "emitted_token": emitted,
        "max_score": max_score,
        "emitted_score": emitted_score,
    }


def actual_generated_arm_prefix(action_tokens: Sequence[int]) -> list[int]:
    toks = [int(x) for x in action_tokens]
    if len(toks) != 7:
        raise ValueError("action token list must contain exactly 7 tokens")
    return toks[:6]


def require_runner_uses_adv_inputs(result: Any) -> Mapping[str, Any]:
    adv_inputs = get_adv_inputs_from_attack_result(result)
    if getattr(result, "x_adv", None) is not None:
        raise RouteContractError("fixed-frame harness must decode from adv_inputs, not x_adv")
    return adv_inputs
