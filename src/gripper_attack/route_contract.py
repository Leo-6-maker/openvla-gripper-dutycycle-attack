from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


TOKEN_PREFIX_METHODS = {
    "token_prefix_pgd",
    "openvla_token_prefix_pgd",
    "visual_token_prefix_pgd",
    "untargeted_token_prefix_pgd",
}

UNTARGETED_OBJECTIVES = {
    "untargeted_clean_token_ce",
    "untargeted_clean_ce",
    "maximize_clean_ce",
    "untargeted_arm_clean_token_ce",
    "ctrl_random_direction_arm_only",
}

TARGET_TOKEN_OBJECTIVES = {
    "autoregressive_prefix_gripper_target_token_cw_v1",
    "autoregressive_prefix_gripper_target_token_logratio_v2",
    "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
    # F1-B keeps the secondary 31745 binding for route validation, but the
    # optimization target is the checkpoint-local native OPEN class.
    "autoregressive_prefix_gripper_native_open_logratio_v4",
    "autoregressive_prefix_gripper_native_open_logratio_arm_v5",
}

EXPECTED_M3_TARGET_TOKEN_ID = 31745  # FIXED
EXPECTED_M3_TARGET_EXECUTION_CLASS = "NATIVE_OPEN"  # FIXED


class RouteContractError(RuntimeError):
    """Raised when a scientific attack route cannot prove its execution path."""


def _bool_cfg(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def optimizer_cfg(config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = dict(config or {})
    opt = cfg.get("attack_optimizer")
    if isinstance(opt, Mapping):
        return dict(opt)
    return cfg


@dataclass(frozen=True)
class RouteConfig:
    requested_method: str | None
    requested_objective: str
    strict_route: bool
    allow_fallback: bool
    target_token_id: int | None
    target_execution_class: str | None
    expected_num_backwards: int | None


def route_config_from_attack_config(config: Mapping[str, Any] | None) -> RouteConfig:
    root = dict(config or {})
    cfg = optimizer_cfg(root)
    requested_method = cfg.get("method")
    strict_route = _bool_cfg(cfg.get("strict_route", root.get("strict_route")), False)
    allow_fallback = _bool_cfg(cfg.get("allow_fallback", root.get("allow_fallback")), True)
    objective = str(cfg.get("objective", cfg.get("loss_objective", "targeted_directional_ce")))
    target_token = cfg.get("target_token_id", cfg.get("target_token"))
    expected = cfg.get("expected_num_backwards", cfg.get("num_steps"))
    return RouteConfig(
        requested_method=None if requested_method is None else str(requested_method),
        requested_objective=objective,
        strict_route=bool(strict_route),
        allow_fallback=bool(allow_fallback),
        target_token_id=None if target_token is None else int(target_token),
        target_execution_class=None
        if cfg.get("target_execution_class") is None
        else str(cfg.get("target_execution_class")),
        expected_num_backwards=None if expected is None else int(expected),
    )


def resolve_adapter_class_name(route: RouteConfig) -> str:
    if route.requested_method is None:
        if route.strict_route:
            raise RouteContractError("strict_route requires explicit method")
        return "ExistingDenseAttackAdapter"
    if route.requested_method in TOKEN_PREFIX_METHODS:
        return "TokenPrefixPGDAttacker"
    if route.strict_route or not route.allow_fallback:
        raise RouteContractError(f"unknown attack method in strict route: {route.requested_method}")
    return "ExistingDenseAttackAdapter"


def validate_attack_request(route: RouteConfig, *, target_action_present: bool) -> None:
    if route.strict_route and route.allow_fallback:
        raise RouteContractError("strict_route requires allow_fallback=False")
    resolved = resolve_adapter_class_name(route)
    if route.strict_route and resolved != "TokenPrefixPGDAttacker":
        raise RouteContractError(f"strict route resolved {resolved}, expected TokenPrefixPGDAttacker")
    if route.requested_objective in TARGET_TOKEN_OBJECTIVES:
        if route.target_token_id is None:
            raise RouteContractError("target-token objective requires target_token_id")
        if route.strict_route and not route.target_execution_class:
            raise RouteContractError("strict target-token objective requires target_execution_class")
        if route.strict_route and not target_action_present:
            raise RouteContractError("strict target-token objective requires target_action as clean arm reference")
    elif route.requested_objective not in UNTARGETED_OBJECTIVES and not target_action_present:
        raise RouteContractError("targeted objective requires target_action")


def attach_route_debug(
    debug: dict[str, Any],
    route: RouteConfig,
    *,
    resolved_adapter_class: str,
    fallback_used: bool,
    fallback_reason: str | None = None,
    target_action_present: bool,
) -> dict[str, Any]:
    debug = dict(debug or {})
    debug.update(
        {
            "requested_method": route.requested_method,
            "resolved_adapter_class": str(resolved_adapter_class),
            "strict_route": bool(route.strict_route),
            "allow_fallback": bool(route.allow_fallback),
            "fallback_used": bool(fallback_used),
            "fallback_reason": fallback_reason,
            "requested_objective": route.requested_objective,
            "resolved_objective": route.requested_objective,
            "target_action_present": bool(target_action_present),
            "target_token_id": route.target_token_id,
            "target_execution_class": route.target_execution_class,
        }
    )
    debug["adv_inputs_present"] = isinstance(debug.get("adv_inputs"), Mapping)
    return debug


def validate_true_pgd_attack_result(result: Any, route: RouteConfig) -> None:
    debug = getattr(result, "debug", {}) or {}
    if debug.get("strict_route") is not True:
        raise RouteContractError("strict route result missing strict_route=True")
    if debug.get("allow_fallback") is not False:
        raise RouteContractError("strict route result must record allow_fallback=False")
    if debug.get("fallback_used") is not False:
        raise RouteContractError(f"strict route used fallback: {debug.get('fallback_reason')}")
    if "fallback_reason" in debug and debug.get("fallback_reason"):
        raise RouteContractError(f"fallback_reason present: {debug.get('fallback_reason')}")
    if debug.get("resolved_adapter_class") != "TokenPrefixPGDAttacker":
        raise RouteContractError("strict route did not resolve TokenPrefixPGDAttacker")
    attack_method = str(getattr(result, "attack_method", "") or "")
    if not attack_method.startswith("token_prefix_pgd"):
        raise RouteContractError(f"unexpected attack_method for true PGD: {attack_method}")
    if getattr(result, "directional_loss_available", None) is not True:
        raise RouteContractError("true target-token PGD must report directional_loss_available=True")
    if debug.get("requested_objective") != debug.get("resolved_objective"):
        raise RouteContractError("requested_objective and resolved_objective differ")
    if route.requested_objective in TARGET_TOKEN_OBJECTIVES:
        if debug.get("resolved_objective") != route.requested_objective:
            raise RouteContractError(f"wrong resolved objective: {debug.get('resolved_objective')}")
        if int(debug.get("target_token_id", -1)) != EXPECTED_M3_TARGET_TOKEN_ID:
            raise RouteContractError(f"wrong target_token_id: {debug.get('target_token_id')}")
        if str(debug.get("target_execution_class")) != EXPECTED_M3_TARGET_EXECUTION_CLASS:
            raise RouteContractError(f"wrong target_execution_class: {debug.get('target_execution_class')}")
    adv_inputs = debug.get("adv_inputs")
    if not isinstance(adv_inputs, Mapping):
        raise RouteContractError("strict route result missing debug['adv_inputs']")
    missing = [key for key in ("input_ids", "pixel_values") if key not in adv_inputs]
    if missing:
        raise RouteContractError(f"strict route adv_inputs missing keys: {missing}")
    if getattr(result, "x_adv", None) is not None:
        raise RouteContractError("true token-prefix PGD must return x_adv=None")
    if debug.get("x_adv_is_none") is False:
        raise RouteContractError("debug claims x_adv is not None")
    if getattr(result, "action_adv", None) is not None:
        raise RouteContractError("true token-prefix PGD must return action_adv=None")
    if debug.get("action_adv_is_none") is False:
        raise RouteContractError("debug claims action_adv is not None")
    if route.expected_num_backwards is not None and int(debug.get("num_backwards", -1)) != int(route.expected_num_backwards):
        raise RouteContractError(
            f"num_backwards={debug.get('num_backwards')} expected {route.expected_num_backwards}"
        )
    num_backwards = int(debug.get("num_backwards", -1))
    num_loss_forwards = int(debug.get("num_loss_forwards", -1))
    if num_loss_forwards < num_backwards + 1:
        raise RouteContractError(
            f"num_loss_forwards={num_loss_forwards} must be >= num_backwards+1 ({num_backwards + 1})"
        )
    if debug.get("pixel_space") != "processor_pixel_values":
        raise RouteContractError(f"unexpected pixel_space: {debug.get('pixel_space')}")
    epsilon = float(getattr(result, "epsilon", 0.0) or 0.0)
    linf = float(debug.get("pixel_budget_adv_inputs_linf", getattr(result, "observation_perturb_linf", 0.0)) or 0.0)
    if epsilon >= 0.0 and linf > epsilon + 1e-6:
        raise RouteContractError(f"processor-space Linf budget violation: {linf} > {epsilon}")
