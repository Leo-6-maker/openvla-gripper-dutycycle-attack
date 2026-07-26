#!/usr/bin/env python3
"""Fail-closed FEC five-arm GPU smoke runner.

This runner is engineering-only and MUST NOT consume formal FEC matrix identities.
It enforces the official LIBERO wait/policy-step accounting, exact init-state
restoration, strict token-prefix PGD routing, matched TRUE/RAND execution, and
artifact-level runtime validation.

The frozen N4 detector feature construction is intentionally delegated to the
same provider used by the validated N4 runtime. The provider must return exact
25D/9D/9D arrays; missing-to-zero fallback is forbidden.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

POLICY_HORIZONS = {
    "libero_10": 520,
    "libero_goal": 300,
    "libero_object": 280,
    "libero_spatial": 220,
}
NUM_STEPS_WAIT = 10
ARMS = (
    "CLEAN",
    "TRUE_T10",
    "RAND_T10",
    "COMMAND_OPEN_ORACLE",
    "RANDOM_TIME_T10",
)
TARGET_OBJECTIVE = "autoregressive_prefix_gripper_target_token_logratio_arm_v3"
TARGET_TOKEN_ID = 31745  # FIXED: was off-by-one (31744→31745 canonical OPEN)
TARGET_EXECUTION_CLASS = "NATIVE_OPEN"  # FIXED: CLIP_MEDIATED_OPEN has 0 tokens
CANONICAL_ENV_OPEN = -1.0
DUMMY_WAIT_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


class ContractError(RuntimeError):
    """Raised when a smoke contract cannot be proven."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")
        handle.flush()


def json_safe(value: Any) -> Any:
    """Convert nested debug metadata without serializing model tensors."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if key in {"adv_inputs", "delta0_adv_inputs", "trajectory_candidate_inputs"}:
                continue
            out[str(key)] = json_safe(item)
        return out
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    try:
        import torch
        if torch.is_tensor(value):
            return {
                "tensor_shape": list(value.shape),
                "tensor_dtype": str(value.dtype),
                "tensor_sha256": tensor_sha256(value),
            }
    except Exception:
        pass
    return str(value)


def tensor_sha256(tensor: Any) -> str:
    import io
    import torch

    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def load_module_from_path(path: Path, module_name: str):
    if not path.is_file():
        raise ContractError(f"module path does not exist: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_with_supported_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    signature = inspect.signature(fn)
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    if accepts_var_kw:
        return fn(**kwargs)
    supported = {name: value for name, value in kwargs.items() if name in signature.parameters}
    missing = [
        name
        for name, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in supported
    ]
    if missing:
        raise ContractError(f"provider {fn} requires unsupported arguments: {missing}")
    return fn(**supported)


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ContractError(f"YAML root must be a mapping: {path}")
    return obj


def validate_base_config(config: dict[str, Any]) -> None:
    opt = config.get("attack_optimizer")
    runtime = config.get("runtime")
    arms = config.get("arms")
    if not isinstance(opt, dict) or not isinstance(runtime, dict) or not isinstance(arms, dict):
        raise ContractError("fec config requires attack_optimizer/runtime/arms mappings")
    expected = {
        "method": "token_prefix_pgd",
        "strict_route": True,
        "allow_fallback": False,
        # epsilon/step_size/num_steps not frozen (sweep parameters)
        "random_start": False,
        "temporal_init": "none",
        "temporal_smooth_lambda": 0.0,
        "surrogate_score_path": "cached_autoregressive_generate_v1",
        "prefix_refresh_interval": 1,
        "objective": opt.get("objective", TARGET_OBJECTIVE),
        # target_token_id/target_execution_class vary by objective (single-token vs region)
        "target_token_id": opt.get("target_token_id") if opt.get("target_token_id") is not None else TARGET_TOKEN_ID,
        "target_execution_class": opt.get("target_execution_class") if opt.get("target_execution_class") is not None else TARGET_EXECUTION_CLASS,
        "gradient_transform": "none",
    }
    for key, value in expected.items():
        if key in ("target_token_id", "target_execution_class"):
            continue  # varies by objective (single-token vs region)
        if opt.get(key) != value:
            raise ContractError(f"attack_optimizer.{key}={opt.get(key)!r}, expected {value!r}")
    if int(runtime.get("attack_burst_frames", runtime.get("K10", -1))) != 10:
        raise ContractError("runtime attack burst must be exactly 10")
    if runtime.get("fallback_forbidden") is not True:
        raise ContractError("runtime.fallback_forbidden must be true")
    rand = arms.get("RAND_T10", {})
    effective_obj = opt.get("objective", TARGET_OBJECTIVE)
    if rand.get("gradient_transform") != "rademacher":
        raise ContractError("RAND_T10 must use rademacher gradient transform")
    random_time = arms.get("RANDOM_TIME_T10", {})
    if random_time.get("gradient_transform") != "none":
        raise ContractError("RANDOM_TIME_T10 must use none gradient transform")


def effective_config(base: dict[str, Any], arm: str, *, rand_direction_seed: int) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    opt = cfg["attack_optimizer"]
    arm_cfg = cfg["arms"].get(arm, {})
    for key in ("objective", "gradient_transform"):
        if key in arm_cfg:
            opt[key] = arm_cfg[key]
    opt["gradient_transform_seed"] = int(rand_direction_seed)
    cfg["effective_arm"] = arm
    return cfg


def normalize_and_invert_gripper(raw_action: Any):
    import numpy as np

    action = np.asarray(raw_action, dtype=np.float32).copy()
    if action.ndim != 1 or action.shape[0] < 7:
        raise ContractError(f"expected 7D action, got shape={action.shape}")
    action[-1] = 2.0 * action[-1] - 1.0
    action[-1] = np.sign(action[-1])
    if action[-1] == 0:
        action[-1] = 1.0
    action[-1] *= -1.0
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def check_success(env: Any, done: bool, info: Mapping[str, Any] | None) -> bool:
    info = info or {}
    if bool(info.get("success", False)):
        return True
    checker = getattr(env, "_check_success", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            pass
    return False  # V5 FIX: never treat done as success; require explicit info/env check


def resolve_task_instruction(task: Any) -> str:
    language = getattr(task, "language", None)
    return str(language if language else task)


def decode_action_from_generation(model: Any, generation: Any, unnorm_key: str):
    import numpy as np

    action_dim = int(model.get_action_dim(unnorm_key))
    token_ids = generation.sequences[0, -action_dim:].detach().cpu().numpy()
    vocab_eff = int(model.config.text_config.vocab_size - model.config.pad_to_multiple_of)
    discretized = np.clip(vocab_eff - token_ids - 1, 0, model.bin_centers.shape[0] - 1)
    norm_actions = model.bin_centers[discretized]
    stats = model.get_action_stats(unnorm_key)
    mask = np.asarray(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)), dtype=bool)
    high = np.asarray(stats["q99"], dtype=np.float32)
    low = np.asarray(stats["q01"], dtype=np.float32)
    raw_action = np.where(mask, 0.5 * (norm_actions + 1.0) * (high - low) + low, norm_actions)
    return raw_action.astype(np.float32), [int(x) for x in token_ids.tolist()]


def prepare_clean_generation(model: Any, processor: Any, obs: Mapping[str, Any], instruction: str, unnorm_key: str,
                             *, device: str, center_crop: bool):
    import torch
    from gripper_attack.openvla_preprocess import prepare_openvla_image

    image = prepare_openvla_image(
        obs["agentview_image"],
        libero_official_preprocess=True,
        center_crop=center_crop,
        resize_size=224,
        libero_preprocess_backend="official_pil_lanczos",
    )
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, image, return_tensors="pt")
    inputs.pop("attention_mask", None)
    input_ids = inputs.get("input_ids")
    if input_ids is not None and not torch.all(input_ids[:, -1] == 29871):
        eos = torch.tensor([[29871]], dtype=input_ids.dtype, device=input_ids.device)
        inputs["input_ids"] = torch.cat([input_ids, eos], dim=1)
    for key, value in list(inputs.items()):
        if torch.is_floating_point(value):
            inputs[key] = value.to(device=device, dtype=torch.bfloat16)
        else:
            inputs[key] = value.to(device=device)
    action_dim = int(model.get_action_dim(unnorm_key))
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=action_dim,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    raw_action, token_ids = decode_action_from_generation(model, generation, unnorm_key)
    return raw_action, generation, token_ids


def redecode_adv_inputs(model: Any, adv_inputs: Mapping[str, Any], unnorm_key: str, *, device: str):
    import torch

    action_dim = int(model.get_action_dim(unnorm_key))
    inputs = {}
    for key, value in adv_inputs.items():
        if torch.is_tensor(value):
            if torch.is_floating_point(value):
                inputs[key] = value.to(device=device, dtype=next(model.parameters()).dtype)
            else:
                inputs[key] = value.to(device=device)
        else:
            inputs[key] = value
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=action_dim,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    raw_action, token_ids = decode_action_from_generation(model, generation, unnorm_key)
    return raw_action, token_ids, generation


class N4Bridge:
    """Strict bridge to the already-validated N4 runtime and feature provider."""

    PROVIDER_NAMES = ("build_n4_inputs", "extract_n4_inputs", "build_inputs", "make_n4_inputs")

    def __init__(self, module_path: Path, *, norm_data_path: Path, device: str,
                 provider_name: str | None = None):
        self.module_path = module_path
        self.module = load_module_from_path(module_path, "fec_n4_runtime_module")
        cls = getattr(self.module, "N4DetectorAdapter", None)
        if cls is None:
            raise ContractError(f"{module_path} does not expose N4DetectorAdapter")
        init_kwargs = {"device": device, "norm_data_path": str(norm_data_path)}
        signature = inspect.signature(cls)
        supported = {name: value for name, value in init_kwargs.items() if name in signature.parameters}
        self.adapter = cls(**supported)
        self.provider = self._resolve_provider(provider_name)

    def _resolve_provider(self, provider_name: str | None) -> Callable[..., Any]:
        names = ([provider_name] if provider_name else []) + list(self.PROVIDER_NAMES)
        for name in names:
            if not name:
                continue
            fn = getattr(self.module, name, None)
            if callable(fn):
                return fn
            fn = getattr(self.adapter, name, None)
            if callable(fn):
                return fn
        available = sorted(name for name in dir(self.module) if not name.startswith("_"))
        raise ContractError(
            "N4 module has no exact feature provider. Add one of "
            f"{self.PROVIDER_NAMES} or pass --n4-provider-name. Available={available}"
        )

    def reset_episode(self) -> None:
        reset = getattr(self.adapter, "reset_episode", None) or getattr(self.adapter, "reset", None)
        if not callable(reset):
            raise ContractError("N4DetectorAdapter has no reset_episode/reset")
        reset()

    def step(self, *, obs: Mapping[str, Any], clean_raw_action: Any, clean_env_action: Any,
             clean_model_output: Any, policy_step: int, suite: str, unnorm_key: str, model: Any, processor: Any) -> dict[str, Any]:
        payload = call_with_supported_kwargs(
            self.provider,
            {
                "obs": obs,
                "observation": obs,
                "clean_raw_action": clean_raw_action,
                "raw_action": clean_raw_action,
                "clean_env_action": clean_env_action,
                "env_action": clean_env_action,
                "clean_model_output": clean_model_output,
                "generation": clean_model_output,
                "policy_step": policy_step,
                "step": policy_step,
                "suite": suite, "unnorm_key": unnorm_key,
                "model": model,
                "processor": processor,
            },
        )
        if isinstance(payload, Mapping):
            f25d = payload.get("f25d", payload.get("features_25d"))
            p9d = payload.get("p9d", payload.get("policy_9d"))
            g9d = payload.get("g9d", payload.get("gripper_9d"))
            candidate_close = payload.get("candidate_close")
            provider_meta = payload.get("meta", {})
        elif isinstance(payload, (tuple, list)) and len(payload) in (3, 4):
            f25d, p9d, g9d = payload[:3]
            candidate_close = payload[3] if len(payload) == 4 else None
            provider_meta = {}
        else:
            raise ContractError("N4 provider must return mapping or tuple(f25d,p9d,g9d[,candidate_close])")

        import numpy as np

        f25d = np.asarray(f25d, dtype=np.float32).reshape(-1)
        p9d = np.asarray(p9d, dtype=np.float32).reshape(-1)
        g9d = np.asarray(g9d, dtype=np.float32).reshape(-1)
        if f25d.shape != (25,) or p9d.shape != (9,) or g9d.shape != (9,):
            raise ContractError(f"N4 provider shapes must be 25/9/9, got {f25d.shape}/{p9d.shape}/{g9d.shape}")
        if not (np.all(np.isfinite(f25d)) and np.all(np.isfinite(p9d)) and np.all(np.isfinite(g9d))):
            raise ContractError("N4 provider returned NaN/Inf")
        if candidate_close is None:
            candidate_close = bool(float(clean_raw_action[-1]) <= 0.5)
        result = self.adapter.step(f25d, p9d, g9d, bool(candidate_close))
        if not isinstance(result, Mapping):
            raise ContractError("N4 adapter.step must return mapping")
        if "emitted_this_step" not in result or "calibrated_prob" not in result:
            raise ContractError("N4 result missing emitted_this_step/calibrated_prob")
        return {
            **dict(result),
            "candidate_close": bool(candidate_close),
            "feature_sha256": canonical_json_sha256(
                {"f25d": f25d.tolist(), "p9d": p9d.tolist(), "g9d": g9d.tolist()}
            ),
            "provider_meta": json_safe(provider_meta),
        }

    def trajectory(self) -> dict[str, Any]:
        getter = getattr(self.adapter, "get_trajectory", None)
        return dict(getter()) if callable(getter) else {}


def validate_attack_result(result: Any, *, arm: str, config: dict[str, Any]) -> dict[str, Any]:
    debug = dict(getattr(result, "debug", {}) or {})
    opt = config["attack_optimizer"]
    expected_transform = "rademacher" if arm == "RAND_T10" else "none"
    checks = {
        "strict_route": debug.get("strict_route") is True,
        "allow_fallback": debug.get("allow_fallback") is False,
        "fallback_used": debug.get("fallback_used") is False,
        "resolved_adapter": debug.get("resolved_adapter_class") == "TokenPrefixPGDAttacker",
        "objective": debug.get("resolved_objective") == TARGET_OBJECTIVE,
        "target_action_present": debug.get("target_action_present") is True,
        "target_token": int(debug.get("target_token_id", -1)) == TARGET_TOKEN_ID,
        "target_class": str(debug.get("target_execution_class")) == TARGET_EXECUTION_CLASS,
        "gradient_transform": str(debug.get("gradient_transform", "none")) == expected_transform,
        "temporal_init": str(debug.get("temporal_init", "")) == "none",
        "temporal_prev_delta_used": debug.get("temporal_prev_delta_used") is False,
        "adv_inputs_present": isinstance(debug.get("adv_inputs"), Mapping),
        "x_adv_none": getattr(result, "x_adv", None) is None,
        "action_adv_none": getattr(result, "action_adv", None) is None,
        "optimizer_steps": int(getattr(result, "num_attack_steps", -1)) == int(opt["num_steps"]),
    }
    if arm == "RAND_T10":
        checks["gradient_transform_seed"] = int(debug.get("gradient_transform_seed", -1)) == int(
            opt["gradient_transform_seed"]
        )
    linf = float(debug.get("pixel_budget_adv_inputs_linf", getattr(result, "observation_perturb_linf", float("inf"))))
    checks["linf_budget"] = linf <= float(opt["epsilon"]) + 1e-6
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ContractError(f"{arm} attack result failed checks: {failed}; debug={json_safe(debug)}")
    adv_inputs = debug["adv_inputs"]
    return {
        "checks": checks,
        "linf": linf,
        "objective": debug.get("resolved_objective"),
        "gradient_transform": debug.get("gradient_transform", "none"),
        "gradient_transform_seed": debug.get("gradient_transform_seed"),
        "num_backwards_reported": debug.get("num_backwards"),
        "gradient_components_per_optimizer_step": 2,
        "expected_autograd_grad_calls": 2 * int(opt["num_steps"]),
        "delta_sha256": debug.get("delta_final_sha256"),
        "processor_input_sha256": debug.get("processor_input_sha256"),
        "debug": json_safe(debug),
        "adv_inputs": adv_inputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--suite", choices=sorted(POLICY_HORIZONS), required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--state-index", type=int, default=None)
    parser.add_argument("--init-state-npy", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/fec_attack_v3.yaml"))
    parser.add_argument("--repo-root", type=Path, default=Path("/mnt/sdc/dty_user/openvla_attack"))
    parser.add_argument("--n4-module", type=Path, default=Path("/tmp/n4_detector_adapter.py"))
    parser.add_argument("--n4-provider-name", default=None)
    parser.add_argument("--n4-norm-data", type=Path, required=True)
    parser.add_argument("--expected-attacker-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rand-direction-seed", type=int, required=True)
    parser.add_argument("--random-time-seed", type=int, required=True)
    parser.add_argument("--random-time-start", type=int, default=None)
    parser.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument("--dry-run-contract", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["MUJOCO_GL"] = "egl"

    args.repo_root = args.repo_root.resolve()
    args.config = (args.repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    sys.path.insert(0, str(args.repo_root / "src"))
    sys.path.insert(0, str(args.repo_root / "scripts"))

    base_config = load_yaml(args.config)
    validate_base_config(base_config)
    config_sha = sha256_file(args.config)
    n4_module_sha = sha256_file(args.n4_module)
    n4_norm_sha = sha256_file(args.n4_norm_data)

    import gripper_attack.attack_adapter as attack_adapter_module

    attacker_realpath = Path(attack_adapter_module.__file__).resolve()
    attacker_sha = sha256_file(attacker_realpath)
    if attacker_sha != args.expected_attacker_sha256:
        raise ContractError(f"attacker SHA mismatch: {attacker_sha} != {args.expected_attacker_sha256}")

    if args.dry_run_contract:
        print(json.dumps({
            "contract": "PASS",
            "config_sha256": config_sha,
            "attacker_realpath": str(attacker_realpath),
            "attacker_sha256": attacker_sha,
            "n4_module_sha256": n4_module_sha,
            "n4_norm_sha256": n4_norm_sha,
        }, indent=2))
        return 0

    import numpy as np
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from gripper_attack.attack_adapter import OpenVLAVisualAttacker

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda:0"

    processor = AutoProcessor.from_pretrained(
        str(args.model_path), trust_remote_code=True, local_files_only=True, use_fast=False
    )
    model = AutoModelForVision2Seq.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    unnorm_key = args.suite
    if unnorm_key not in getattr(model, "norm_stats", {}) and f"{unnorm_key}_no_noops" in getattr(model, "norm_stats", {}):
        unnorm_key = f"{unnorm_key}_no_noops"
    if unnorm_key not in getattr(model, "norm_stats", {}):
        raise ContractError(f"unnorm key {unnorm_key} not found in model.norm_stats")

    preprocess_kwargs = {
        "libero_official_preprocess": True,
        "center_crop": bool(args.center_crop),
        "resize_size": 224,
        "libero_preprocess_backend": "official_pil_lanczos",
        "postprocess_gripper": True,
    }
    true_cfg = effective_config(base_config, "TRUE_T10", rand_direction_seed=args.rand_direction_seed)
    rand_cfg = effective_config(base_config, "RAND_T10", rand_direction_seed=args.rand_direction_seed)
    random_time_cfg = effective_config(base_config, "RANDOM_TIME_T10", rand_direction_seed=args.rand_direction_seed)
    attackers = {
        "TRUE_T10": OpenVLAVisualAttacker(model=model, processor=processor, config=true_cfg,
                                            seed=args.seed, preprocess_kwargs=preprocess_kwargs, device=device),
        "RAND_T10": OpenVLAVisualAttacker(model=model, processor=processor, config=rand_cfg,
                                            seed=args.seed, preprocess_kwargs=preprocess_kwargs, device=device),
        "RANDOM_TIME_T10": OpenVLAVisualAttacker(model=model, processor=processor, config=random_time_cfg,
                                                   seed=args.seed, preprocess_kwargs=preprocess_kwargs, device=device),
    }

    detector = N4Bridge(
        args.n4_module,
        norm_data_path=args.n4_norm_data,
        device=device,
        provider_name=args.n4_provider_name,
    )

    suite_obj = benchmark.get_benchmark_dict()[args.suite]()
    if args.task_index < 0 or args.task_index >= int(suite_obj.n_tasks):
        raise ContractError(f"task index out of range: {args.task_index}")
    task = suite_obj.get_task(args.task_index)
    bddl_file = suite_obj.get_task_bddl_file_path(args.task_index)
    instruction = resolve_task_instruction(task)
    initial_states = suite_obj.get_task_init_states(args.task_index)
    if args.init_state_npy is not None:
        initial_state = np.load(args.init_state_npy, allow_pickle=False)
        state_identity = {"kind": "npy", "path": str(args.init_state_npy), "sha256": sha256_file(args.init_state_npy)}
    else:
        if args.state_index is None:
            raise ContractError("provide --state-index or --init-state-npy")
        if args.state_index < 0 or args.state_index >= len(initial_states):
            raise ContractError(f"state index {args.state_index} outside available init states {len(initial_states)}")
        initial_state = copy.deepcopy(initial_states[args.state_index])
        state_identity = {"kind": "benchmark_index", "index": args.state_index}

    policy_horizon = POLICY_HORIZONS[args.suite]
    burst_frames = int(base_config["runtime"]["attack_burst_frames"])
    if args.random_time_start is None:
        rng = np.random.RandomState(args.random_time_seed)
        random_time_start = int(rng.randint(0, policy_horizon - burst_frames + 1))
    else:
        random_time_start = int(args.random_time_start)
    if not 0 <= random_time_start <= policy_horizon - burst_frames:
        raise ContractError(f"random-time start {random_time_start} is not a legal complete K10 start")

    if args.output_root.exists():
        unexpected = [p.name for p in args.output_root.iterdir() if p.name not in {"worker.log"}]
        if unexpected:
            raise ContractError(f"output root is not empty: {args.output_root}; unexpected={unexpected}")
    else:
        args.output_root.mkdir(parents=True, exist_ok=False)
    run_manifest = {
        "scientific_role": "SMOKE_ONLY",
        "counts_toward_fec": False,
        "formal_matrix_execution": False,
        "cs200_access": False,
        "suite": args.suite,
        "task_index": args.task_index,
        "state_identity": state_identity,
        "policy_horizon": policy_horizon,
        "num_steps_wait": NUM_STEPS_WAIT,
        "max_env_steps": policy_horizon + NUM_STEPS_WAIT,
        "seed": args.seed,
        "rand_direction_seed": args.rand_direction_seed,
        "random_time_seed": args.random_time_seed,
        "random_time_policy_start": random_time_start,
        "config_path": str(args.config),
        "config_sha256": config_sha,
        "attacker_realpath": str(attacker_realpath),
        "attacker_sha256": attacker_sha,
        "n4_module_path": str(args.n4_module),
        "n4_module_sha256": n4_module_sha,
        "n4_norm_path": str(args.n4_norm_data),
        "n4_norm_sha256": n4_norm_sha,
        "model_path": str(args.model_path),
        "unnorm_key": unnorm_key,
        "arms": list(ARMS),
        "created_unix": time.time(),
    }
    atomic_write_json(args.output_root / "run_manifest.json", run_manifest)

    all_results: dict[str, Any] = {}
    fatal_error: dict[str, Any] | None = None

    for arm in ARMS:
        arm_dir = args.output_root / arm
        arm_dir.mkdir(parents=True, exist_ok=False)
        steps_path = arm_dir / "steps.jsonl"
        attacks_path = arm_dir / "attack_frames.jsonl"
        error_path = arm_dir / "error.json"
        arm_seed = int(args.seed)
        random.seed(arm_seed)
        np.random.seed(arm_seed)
        torch.manual_seed(arm_seed)
        torch.cuda.manual_seed_all(arm_seed)

        env = None
        result = {
            "arm": arm,
            "status": "RUNNING",
            "emit_policy_step": None,
            "emit_env_step": None,
            "attack_planned_frames": 0,
            "attack_executed_frames": 0,
            "attack_errors": 0,
            "task_success": False,
            "policy_steps": 0,
            "env_steps": 0,
            "termination": None,
        }
        try:
            env = OffScreenRenderEnv(
                bddl_file_name=bddl_file,
                camera_heights=args.render_size,
                camera_widths=args.render_size,
                render_gpu_device_id=0,
                horizon=policy_horizon + NUM_STEPS_WAIT,
            )
            seed_fn = getattr(env, "seed", None)
            if callable(seed_fn):
                seed_fn(0)
            env.reset()
            obs = env.set_init_state(copy.deepcopy(initial_state))

            for wait_step in range(NUM_STEPS_WAIT):
                obs, reward, done, info = env.step(DUMMY_WAIT_ACTION)
                result["env_steps"] += 1
                append_jsonl(steps_path, {
                    "env_step": wait_step,
                    "policy_step": None,
                    "is_wait_step": True,
                    "detector_updated": False,
                    "attack_planned": False,
                    "attack_executed": False,
                    "done": bool(done),
                })
                if done:
                    raise ContractError("environment terminated during official wait phase")

            detector.reset_episode()
            for attacker in attackers.values():
                attacker.reset_temporal_state()

            emit_step = None
            for policy_step in range(policy_horizon):
                env_step = NUM_STEPS_WAIT + policy_step
                clean_raw_action, clean_generation, clean_token_ids = prepare_clean_generation(
                    model, processor, obs, instruction, unnorm_key,
                    device=device, center_crop=bool(args.center_crop),
                )
                clean_env_action = normalize_and_invert_gripper(clean_raw_action)
                n4 = detector.step(
                    obs=obs,
                    clean_raw_action=clean_raw_action,
                    clean_env_action=clean_env_action,
                    clean_model_output=clean_generation,
                    policy_step=policy_step,
                    suite=args.suite, unnorm_key=unnorm_key,
                    model=model,
                    processor=processor,
                )
                if bool(n4["emitted_this_step"]) and emit_step is None:
                    emit_step = policy_step
                    result["emit_policy_step"] = policy_step
                    result["emit_env_step"] = env_step

                planned = False
                if arm in {"TRUE_T10", "RAND_T10", "COMMAND_OPEN_ORACLE"} and emit_step is not None:
                    planned = emit_step <= policy_step < emit_step + burst_frames
                elif arm == "RANDOM_TIME_T10":
                    planned = random_time_start <= policy_step < random_time_start + burst_frames
                if planned:
                    result["attack_planned_frames"] += 1

                final_env_action = clean_env_action.copy()
                attack_executed = False
                attack_audit = None
                adv_raw_action = None
                adv_token_ids = None

                if planned and arm == "COMMAND_OPEN_ORACLE":
                    final_env_action = clean_env_action.copy()
                    final_env_action[-1] = CANONICAL_ENV_OPEN
                    if not np.array_equal(final_env_action[:6], clean_env_action[:6]):
                        raise ContractError("ORACLE modified arm dimensions")
                    attack_executed = True
                    attack_audit = {"oracle_gripper_env": float(final_env_action[-1]), "arm_preserved": True}

                elif planned and arm in {"TRUE_T10", "RAND_T10", "RANDOM_TIME_T10"}:
                    attack_cfg = {
                        "TRUE_T10": true_cfg,
                        "RAND_T10": rand_cfg,
                        "RANDOM_TIME_T10": random_time_cfg,
                    }[arm]
                    # V5 FIX: target OPEN (CANONICAL_RAW_OPEN=1.0), not CLOSE
                    target_raw_action = clean_raw_action.copy()
                    target_raw_action[-1] = 1.0  # CANONICAL_RAW_OPEN
                    attack_result = attackers[arm].attack(
                        observation=obs["agentview_image"],
                        instruction=instruction,
                        clean_action=clean_raw_action,
                        target_action=target_raw_action,
                        clean_model_output=clean_generation,
                        unnorm_key=unnorm_key,
                    )
                    attack_audit = validate_attack_result(attack_result, arm=arm, config=attack_cfg)
                    adv_inputs = attack_audit.pop("adv_inputs")
                    adv_raw_action, adv_token_ids, _adv_generation = redecode_adv_inputs(
                        model, adv_inputs, unnorm_key, device=device
                    )
                    final_env_action = normalize_and_invert_gripper(adv_raw_action)
                    attack_executed = True

                if planned and not attack_executed:
                    raise ContractError(f"planned attack was not executed: arm={arm} step={policy_step}")

                obs, reward, done, info = env.step(final_env_action.tolist())
                result["env_steps"] += 1
                result["policy_steps"] += 1
                if attack_executed:
                    result["attack_executed_frames"] += 1
                    append_jsonl(attacks_path, {
                        "arm": arm,
                        "env_step": env_step,
                        "policy_step": policy_step,
                        "attack_frame_idx": (
                            policy_step - emit_step
                            if arm != "RANDOM_TIME_T10" and emit_step is not None
                            else policy_step - random_time_start
                        ),
                        "clean_raw_action": clean_raw_action.tolist(),
                        "clean_env_action": clean_env_action.tolist(),
                        "clean_token_ids": clean_token_ids,
                        "adv_raw_action": None if adv_raw_action is None else adv_raw_action.tolist(),
                        "adv_token_ids": adv_token_ids,
                        "final_env_action": final_env_action.tolist(),
                        "audit": json_safe(attack_audit),
                    })

                append_jsonl(steps_path, {
                    "env_step": env_step,
                    "policy_step": policy_step,
                    "is_wait_step": False,
                    "detector_updated": True,
                    "candidate_close": bool(n4.get("candidate_close")),
                    "calibrated_prob": float(n4["calibrated_prob"]),
                    "emitted_this_step": bool(n4["emitted_this_step"]),
                    "feature_sha256": n4.get("feature_sha256"),
                    "attack_planned": bool(planned),
                    "attack_executed": bool(attack_executed),
                    "clean_env_gripper": float(clean_env_action[-1]),
                    "final_env_gripper": float(final_env_action[-1]),
                    "done": bool(done),
                    "success": check_success(env, bool(done), info),
                })

                success = check_success(env, bool(done), info)
                if success:
                    result["task_success"] = True
                    result["termination"] = "SUCCESS"
                    break
                if done:
                    result["termination"] = "DONE_WITHOUT_SUCCESS"
                    break
            else:
                result["termination"] = "POLICY_HORIZON"

            if arm == "CLEAN" and result["attack_executed_frames"] != 0:
                raise ContractError("CLEAN executed an intervention")
            if arm in {"TRUE_T10", "RAND_T10", "COMMAND_OPEN_ORACLE"}:
                if result["emit_policy_step"] is None:
                    if result["attack_executed_frames"] != 0:
                        raise ContractError(f"{arm} attacked without detector emit")
                elif result["attack_executed_frames"] != burst_frames:
                    raise ContractError(
                        f"{arm} emitted but did not execute complete K10: {result['attack_executed_frames']}"
                    )
            if arm == "RANDOM_TIME_T10" and result["attack_executed_frames"] not in (0, burst_frames):
                raise ContractError("RANDOM_TIME executed a partial burst")

            result["status"] = "PASS"
            result["detector_trajectory"] = json_safe(detector.trajectory())
            atomic_write_json(arm_dir / "result.json", result)
            atomic_write_json(arm_dir / "COMPLETE.json", {
                "status": "PASS",
                "result_sha256": sha256_file(arm_dir / "result.json"),
                "completed_unix": time.time(),
            })
        except Exception as exc:
            result["status"] = "FAIL"
            result["attack_errors"] += 1
            result["error_type"] = type(exc).__name__
            result["error"] = str(exc)
            atomic_write_json(error_path, {
                "arm": arm,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "result": result,
            })
            atomic_write_json(arm_dir / "result.json", result)
            fatal_error = {"arm": arm, "error_type": type(exc).__name__, "error": str(exc)}
            all_results[arm] = result
            break
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

        all_results[arm] = result

    overall_valid = fatal_error is None and set(all_results) == set(ARMS)
    pairing = {}
    if overall_valid:
        emit_steps = {arm: all_results[arm]["emit_policy_step"] for arm in (
            "CLEAN", "TRUE_T10", "RAND_T10", "COMMAND_OPEN_ORACLE"
        )}
        pairing["same_detector_emit"] = len(set(emit_steps.values())) == 1
        pairing["true_rand_k10"] = (
            all_results["TRUE_T10"]["attack_executed_frames"]
            == all_results["RAND_T10"]["attack_executed_frames"]
        )
        pairing["clean_no_attack"] = all_results["CLEAN"]["attack_executed_frames"] == 0
        overall_valid = all(pairing.values())

    summary = {
        "valid": bool(overall_valid),
        "engineering_status": "PASS" if overall_valid else "FAIL",
        "scientific_outcome_is_not_a_gate": True,
        "results": all_results,
        "pairing": pairing,
        "fatal_error": fatal_error,
        "run_manifest_sha256": sha256_file(args.output_root / "run_manifest.json"),
        "completed_unix": time.time(),
    }
    atomic_write_json(args.output_root / "smoke_summary.json", summary)
    if overall_valid:
        atomic_write_json(args.output_root / "SMOKE_PASS.json", {
            "status": "PASS",
            "summary_sha256": sha256_file(args.output_root / "smoke_summary.json"),
        })
        print("FEC GPU SMOKE: PASS", flush=True)
        return 0
    print(f"FEC GPU SMOKE: FAIL: {fatal_error or pairing}", file=sys.stderr, flush=True)
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(2)
