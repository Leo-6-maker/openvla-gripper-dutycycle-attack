
#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, getpass, json, os, socket, sys, time
from pathlib import Path
import numpy as np, torch, yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from patch_openvla_compat import patch_openvla
from gripper_attack.attack_adapter import OpenVLAVisualAttacker
from gripper_attack.budget import OnlineBudgetController
from gripper_attack.directional import build_target_action, compute_alignment, compute_delta_action, load_direction_spec
from gripper_attack.grasp import (
    GraspPhaseTracker,
    MokaTwoPotStageTracker,
    compute_grasp_metadata,
    compute_moka_two_pot_stage_metadata,
    eef_pos,
    infer_failure_phase,
    object_pos,
    proxy_grasp_metadata,
)
from gripper_attack.io import make_run_id, read_json, sha256_jsonable, write_csv, write_json, write_jsonl
from gripper_attack.logging_schema import validate_episode_record, validate_run_manifest, validate_step_record
from gripper_attack.metrics import aggregate_episode_from_steps, aggregate_run, normalized_action_discrepancy_cleanref
from gripper_attack.triggers import make_trigger
from gripper_attack.types import AttackResult, TriggerContext
from gripper_attack.uncertainty import extract_prefix_logits

FORCE_OPEN_OBJECTIVES = {"force_gripper_open_token_ce", "force_gripper_open", "targeted_gripper_open_ce"}
FORCE_OPEN_Z_DOWN_OBJECTIVES = {"force_open_z_down_token_ce"}
GRIPPER_DIAGNOSTIC_OBJECTIVES = {"gripper_logit_margin_cw", "gripper_open_region_ce"}
ADAPTIVE_ANTI_GRIPPER_OBJECTIVES = {"adaptive_anti_gripper_token_ce"}
ORACLE_ENV_GRIPPER_OPEN_OBJECTIVES = {"oracle_env_gripper_open", "oracle_force_env_gripper_open"}
ARM_ONLY_UNTARGETED_OBJECTIVES = {"untargeted_arm_clean_token_ce", "ctrl_random_direction_arm_only"}
CONSTANT_DELTA_OBJECTIVES = {"constant_delta_pregrasp"}
NOISE_BASELINE_OBJECTIVES = {"noise_baseline"}

def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f: return yaml.safe_load(f)

def rho_key(rho: float) -> str:
    return f"rho_{float(rho):.2f}"

def validate_thresholds_for_trigger(trigger_name: str, thresholds_path: str, thresholds: dict, task_id: str, rho: float, *, require_rollout_source: bool = False, min_steps: int = 0) -> None:
    score_by_trigger = {
        "entropy_threshold": "entropy",
        "entropy_threshold_cooldown": "entropy",
        "xyz_entropy_threshold": "xyz_entropy",
        "xyz_entropy_cooldown": "xyz_entropy",
        "arm_entropy_threshold": "arm_entropy",
        "arm_entropy_cooldown": "arm_entropy",
        "motion_weighted_xyz_entropy_threshold": "motion_weighted_xyz_entropy",
        "motion_weighted_xyz_entropy_cooldown": "motion_weighted_xyz_entropy",
        "motion_weighted_arm_entropy_threshold": "motion_weighted_arm_entropy",
        "motion_weighted_arm_entropy_cooldown": "motion_weighted_arm_entropy",
        "gripper_entropy_threshold": "gripper_entropy",
        "gripper_entropy_cooldown": "gripper_entropy",
        "grasp_composite_entropy_threshold": "grasp_composite_entropy",
        "grasp_composite_entropy_cooldown": "grasp_composite_entropy",
        "priv_grasp_xyz_entropy_cooldown": "xyz_entropy",
        "priv_grasp_gripper_entropy_cooldown": "gripper_entropy",
        "priv_grasp_composite_entropy_cooldown": "grasp_composite_entropy",
        "proxy_grasp_xyz_entropy_cooldown": "xyz_entropy",
        "proxy_grasp_gripper_entropy_cooldown": "gripper_entropy",
        "proxy_grasp_composite_entropy_cooldown": "grasp_composite_entropy",
        "margin_threshold": "margin",
    }
    if trigger_name not in score_by_trigger:
        return
    if not thresholds_path:
        raise SystemExit(f"--thresholds is required for trigger={trigger_name}")
    if not Path(thresholds_path).exists():
        raise SystemExit(f"thresholds file does not exist for trigger={trigger_name}: {thresholds_path}")
    score_name = score_by_trigger[trigger_name]
    key = rho_key(rho)
    try:
        entry = thresholds["tasks"][task_id][key]
        _ = entry[score_name]
    except Exception as e:
        raise SystemExit(f"missing threshold entry tasks[{task_id!r}][{key!r}][{score_name!r}] in {thresholds_path}: {e}")
    n = int(entry.get("num_steps", thresholds.get("num_steps_total", 0)) or 0)
    if min_steps and n < int(min_steps):
        raise SystemExit(f"threshold calibration has num_steps={n} < required {min_steps} for task={task_id} rho={key}")
    if require_rollout_source:
        if thresholds.get("dry_run"):
            raise SystemExit("dry_run thresholds are not allowed when --require_rollout_calibration is set")
        source = thresholds.get("calibration_source")
        mode = thresholds.get("calibration_mode", "")
        if source != "rollout_passive_clean_observer" and mode != "proxy_local":
            raise SystemExit(
                f"thresholds calibration_source={source!r} calibration_mode={mode!r}; "
                "expected rollout passive calibration or proxy_local thresholds"
            )
def prompt(instruction): return f"In: What action should the robot take to {instruction}?\nOut:"
def norm_name(x): return "".join(ch.lower() for ch in str(x) if ch.isalnum() or ch == "_")
def resolve_task_index(bench, task_name):
    names=list(bench.get_task_names())
    if task_name in names: return names.index(task_name)
    n=norm_name(task_name)
    for i,nm in enumerate(names):
        if norm_name(nm)==n: return i
    raise ValueError(f"task_name not found: {task_name}; available={names}")
def get_instruction(bench, idx, fallback):
    try:
        task=bench.get_task(idx); lang=getattr(task, "language", None)
        if lang: return str(lang)
    except Exception: pass
    return fallback.replace("_", " ")

def resolve_instruction_for_run(args, base_instruction: str) -> tuple[str, dict]:
    override = str(getattr(args, "instruction_override", "") or "")
    prompt_id = str(getattr(args, "prompt_id", "") or "base")
    prompt_type = str(getattr(args, "prompt_type", "") or ("override" if override.strip() else "clean"))
    instruction = override if override.strip() else str(base_instruction)
    suffix = str(getattr(args, "instruction_suffix", "") or "")
    if suffix:
        instruction = f"{instruction} {suffix}".strip()
    return instruction, {
        "base_instruction": str(base_instruction),
        "effective_instruction": instruction,
        "instruction_override": override,
        "instruction_suffix": suffix,
        "prompt_variants_path": str(getattr(args, "prompt_variants_path", "") or ""),
        "offline_prompt_audit": bool(getattr(args, "offline_prompt_audit", False)),
        "prompt_id": prompt_id,
        "prompt_type": prompt_type,
        "prompt_attack_config": str(getattr(args, "prompt_attack_config", "") or ""),
        "target_primitive": str(getattr(args, "target_primitive", "") or "none"),
    }

def load_prompt_variants(path: str, *, base_instruction: str, args=None) -> list[dict]:
    """Load prompt variants from JSON/JSONL/CSV.

    Each row may define prompt_id, prompt_type, instruction, instruction_suffix,
    target_primitive, and prompt_attack_config. Missing fields inherit from args.
    """
    def normalize(row: dict, idx: int) -> dict:
        suffix = str(row.get("instruction_suffix", row.get("suffix", "")) or "")
        instruction = str(row.get("instruction", row.get("prompt", "")) or "").strip()
        if not instruction:
            instruction = str(base_instruction)
        if suffix:
            instruction = f"{instruction} {suffix}".strip()
        return {
            "prompt_id": str(row.get("prompt_id", row.get("id", f"prompt_{idx:03d}")) or f"prompt_{idx:03d}"),
            "prompt_type": str(row.get("prompt_type", row.get("type", "variant")) or "variant"),
            "instruction": instruction,
            "instruction_override": str(row.get("instruction", row.get("prompt", "")) or ""),
            "instruction_suffix": suffix,
            "prompt_attack_config": str(row.get("prompt_attack_config", getattr(args, "prompt_attack_config", "") if args else "") or ""),
            "target_primitive": str(row.get("target_primitive", getattr(args, "target_primitive", "none") if args else "none") or "none"),
        }
    if not str(path or "").strip():
        instruction, meta = resolve_instruction_for_run(args, base_instruction) if args is not None else (base_instruction, {})
        return [normalize({**meta, "instruction": instruction}, 0)]
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--prompt_variants_path does not exist: {path}")
    if p.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif p.suffix.lower() == ".csv":
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
    else:
        obj = json.loads(p.read_text(encoding="utf-8"))
        rows = obj.get("prompts", obj) if isinstance(obj, dict) else obj
    if not isinstance(rows, list):
        raise SystemExit(f"prompt variants must be a list or dict with prompts list: {path}")
    return [normalize(dict(row), i) for i, row in enumerate(rows)]

def _pil_center_crop_resize(image: Image.Image, crop_scale: float = 0.9, size: int = 224) -> Image.Image:
    """Fallback center crop used only when official preprocessing is disabled."""
    if crop_scale is None or float(crop_scale) >= 0.999:
        return image.resize((size, size), Image.Resampling.LANCZOS)
    w, h = image.size
    scale = float(crop_scale) ** 0.5
    cw, ch = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    left, top = (w - cw) // 2, (h - ch) // 2
    image = image.crop((left, top, left + cw, top + ch))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def _official_pil_libero_image(image_np, *, center_crop: bool = False, resize_size: int = 224) -> Image.Image:
    """Official OpenVLA LIBERO image path (PIL): rotate 180, Lanczos resize, Lanczos center crop.

    Matches the corrected official OpenVLA eval script exactly:
    - Rotate agentview 180 degrees
    - PIL LANCZOS resize to resize_size
    - Optional center crop (0.9 scale) → PIL LANCZOS resize back
    - NO JPEG round-trip, NO TensorFlow dependency
    """
    arr = np.asarray(image_np)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = arr[::-1, ::-1]  # rotate 180 degrees
    image = Image.fromarray(arr).convert("RGB")
    image = image.resize((int(resize_size), int(resize_size)), Image.LANCZOS)
    if center_crop:
        crop_scale = 0.9 ** 0.5
        w, h = image.size
        cw, ch = max(1, int(w * crop_scale)), max(1, int(h * crop_scale))
        left, top = (w - cw) // 2, (h - ch) // 2
        image = image.crop((left, top, left + cw, top + ch))
        image = image.resize((int(resize_size), int(resize_size)), Image.LANCZOS)
    return image


def _official_tf_libero_image(image_np, *, center_crop: bool = False, resize_size: int = 224) -> Image.Image:
    """Official OpenVLA LIBERO image path: rotate, JPEG round-trip, TF Lanczos3.

    This intentionally requires TensorFlow instead of silently using the older
    PIL approximation when ``--libero_official_preprocess`` is requested.
    """
    try:
        import tensorflow as tf
    except Exception as exc:
        raise SystemExit(
            "--libero_official_preprocess now requires TensorFlow for official "
            f"LIBERO preprocessing; import failed: {exc}"
        )
    arr = np.asarray(image_np)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = arr[::-1, ::-1]
    tensor = tf.convert_to_tensor(arr)
    tensor = tf.io.decode_image(tf.io.encode_jpeg(tensor), expand_animations=False, dtype=tf.uint8)
    tensor = tf.image.resize(tensor, [int(resize_size), int(resize_size)], method="lanczos3", antialias=True)
    if center_crop:
        crop_scale = 0.9 ** 0.5
        box = [[
            (1.0 - crop_scale) / 2.0,
            (1.0 - crop_scale) / 2.0,
            (1.0 + crop_scale) / 2.0,
            (1.0 + crop_scale) / 2.0,
        ]]
        tensor = tf.image.crop_and_resize(
            tf.expand_dims(tensor, axis=0),
            boxes=tf.convert_to_tensor(box, dtype=tf.float32),
            box_indices=tf.convert_to_tensor([0], dtype=tf.int32),
            crop_size=[int(resize_size), int(resize_size)],
            method="bilinear",
        )[0]
    tensor = tf.cast(tf.clip_by_value(tf.round(tensor), 0, 255), tf.uint8)
    return Image.fromarray(tensor.numpy()).convert("RGB")


def prepare_openvla_image(image_np, *, libero_official_preprocess: bool = False, center_crop: bool = False, resize_size: int = 224, libero_preprocess_backend: str = "official_pil_lanczos") -> Image.Image:
    """Prepare a LIBERO observation image for OpenVLA inference.

    Backends:
      - official_pil_lanczos (default): rotate 180, PIL Lanczos resize, Lanczos center crop.
        Matches the corrected official OpenVLA eval script. No TensorFlow dependency.
      - tf_jpeg_legacy: rotate 180, JPEG round-trip, TF Lanczos3 resize, TF bilinear crop.
        Legacy backend requiring TensorFlow. Known to produce +9 lower Object SR.

    The ``--libero_official_preprocess`` flag is a deprecated compatibility alias;
    it no longer switches the backend. Use --libero_preprocess_backend to select.
    """
    backend = str(libero_preprocess_backend or "official_pil_lanczos")
    # --libero_official_preprocess is a deprecated compatibility alias.
    # It does NOT switch to tf_jpeg_legacy. Use --libero_preprocess_backend explicitly.

    if backend == "official_pil_lanczos":
        return _official_pil_libero_image(image_np, center_crop=center_crop, resize_size=int(resize_size))
    elif backend == "tf_jpeg_legacy":
        return _official_tf_libero_image(image_np, center_crop=center_crop, resize_size=int(resize_size))
    else:
        arr = np.asarray(image_np)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr).convert("RGB")
        if center_crop:
            image = _pil_center_crop_resize(image, crop_scale=0.9, size=int(resize_size))
        return image


def normalize_gripper_action(action, binarize: bool = True):
    """Official OpenVLA gripper postprocess: map last dim [0,1] -> [-1,+1]."""
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = 2.0 * action[..., -1] - 1.0
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
        action[..., -1] = 1.0 if action[..., -1] == 0 else action[..., -1]
    return action


def invert_gripper_action(action):
    """Official LIBERO/OpenVLA gripper postprocess sign flip."""
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = -1.0 * action[..., -1]
    return action


def lift_env_gripper_closed(value: float) -> bool:
    sign = os.environ.get("V4_LIFT_CLOSED_GRIPPER_SIGN", "positive").strip().lower()
    threshold = abs(float(os.environ.get("V4_LIFT_CLOSED_GRIPPER_THRESHOLD", "0.5")))
    v = float(value)
    if sign in {"positive", "+", "+1", "1", "pos"}:
        return v > threshold
    return v < -threshold

def force_open_env_target_ok(value: float) -> bool:
    sign = os.environ.get("V4_FORCE_OPEN_ENV_SIGN", "negative").strip().lower()
    threshold = abs(float(os.environ.get("V4_FORCE_OPEN_ENV_THRESHOLD", "0.5")))
    v = float(value)
    if sign in {"positive", "+", "+1", "1", "pos"}:
        return v > threshold
    return v < -threshold

def raw_gripper_for_target_env(target_env_sign: str, *, postprocess_enabled: bool) -> float:
    sign = str(target_env_sign or "negative").strip().lower()
    want_negative = sign in {"negative", "-", "-1", "neg"}
    if postprocess_enabled:
        return 1.0 if want_negative else 0.0
    return -1.0 if want_negative else 1.0

def postprocess_openvla_action_for_libero(action, enabled: bool = False):
    env_action = np.asarray(action, dtype=np.float32).copy()
    if enabled:
        env_action = normalize_gripper_action(env_action, binarize=True)
        env_action = invert_gripper_action(env_action)
    return np.clip(env_action, -1.0, 1.0).astype(np.float32)

def apply_action_clamp(env_action, clean_env_action, mode: str = "none"):
    mode = str(mode or "none").strip()
    before = np.asarray(env_action, dtype=np.float32).copy()
    clean = np.asarray(clean_env_action, dtype=np.float32).copy()
    after = before.copy()
    n = min(len(after), len(clean))
    if mode == "none" or n == 0:
        pass
    elif mode == "gripper_clean":
        after[n - 1] = clean[n - 1]
    elif mode == "arm_clean":
        arm_n = min(6, max(0, n - 1))
        if arm_n:
            after[:arm_n] = clean[:arm_n]
    else:
        raise ValueError(f"unknown action_clamp_mode: {mode}")
    return np.clip(after, -1.0, 1.0).astype(np.float32), before

def action_clamp_audit(mode: str, env_action_before, env_action_after, clean_env_action) -> dict:
    before = np.asarray(env_action_before, dtype=np.float32).reshape(-1)
    after = np.asarray(env_action_after, dtype=np.float32).reshape(-1)
    clean = np.asarray(clean_env_action, dtype=np.float32).reshape(-1)
    n = min(len(after), len(clean))
    arm_n = min(6, max(0, n - 1))
    delta = after[:n] - clean[:n] if n else np.asarray([], dtype=np.float32)
    arm_delta = delta[:arm_n] if arm_n else np.asarray([], dtype=np.float32)
    rot_delta = delta[3:arm_n] if arm_n > 3 else np.asarray([], dtype=np.float32)
    return {
        "action_clamp_mode": str(mode or "none"),
        "env_action_before_clamp": before.tolist(),
        "env_action_after_clamp": after.tolist(),
        "arm_delta_l2": float(np.linalg.norm(arm_delta)) if arm_delta.size else 0.0,
        "gripper_delta_env": float(delta[n - 1]) if n else 0.0,
        "dx_delta": float(delta[0]) if arm_n >= 1 else 0.0,
        "dy_delta": float(delta[1]) if arm_n >= 2 else 0.0,
        "dz_delta": float(delta[2]) if arm_n >= 3 else 0.0,
        "rot_delta_l2": float(np.linalg.norm(rot_delta)) if rot_delta.size else 0.0,
    }

def physical_gripper_state(env, obs=None) -> dict:
    """Best-effort physical gripper opening readout from obs or MuJoCo qpos."""
    keys = ("robot0_gripper_qpos", "gripper_qpos")
    if isinstance(obs, dict):
        for key in keys:
            if key in obs:
                vals = np.asarray(obs[key], dtype=np.float32).reshape(-1)
                return {
                    "source": f"obs.{key}",
                    "qpos": vals.tolist(),
                    "qpos_sum": float(np.sum(vals)),
                    "qpos_abs_sum": float(np.sum(np.abs(vals))),
                    "joint_names": [],
                }
    try:
        model = env.sim.model
        data = env.sim.data
        names = list(getattr(model, "joint_names", []) or [])
        if not names and hasattr(model, "joint_id2name"):
            names = [model.joint_id2name(i) for i in range(int(getattr(model, "njnt", 0)))]
        pairs = []
        for name in names:
            lname = str(name).lower()
            if "finger" not in lname and "gripper" not in lname:
                continue
            vals = []
            try:
                addr = model.get_joint_qpos_addr(name)
            except Exception:
                try:
                    jid = model.joint_name2id(name)
                    addr = int(model.jnt_qposadr[jid])
                except Exception:
                    continue
            try:
                if isinstance(addr, tuple):
                    vals = np.asarray(data.qpos[int(addr[0]):int(addr[1])], dtype=np.float32).reshape(-1).tolist()
                elif isinstance(addr, slice):
                    vals = np.asarray(data.qpos[addr], dtype=np.float32).reshape(-1).tolist()
                else:
                    vals = [float(data.qpos[int(addr)])]
            except Exception:
                vals = []
            if vals:
                pairs.append((str(name), vals))
        flat = [float(v) for _, vals in pairs for v in vals]
        if flat:
            vals = np.asarray(flat, dtype=np.float32)
            return {
                "source": "sim.qpos.finger_or_gripper_joints",
                "qpos": vals.tolist(),
                "qpos_sum": float(np.sum(vals)),
                "qpos_abs_sum": float(np.sum(np.abs(vals))),
                "joint_names": [name for name, _ in pairs],
            }
    except Exception:
        pass
    return {"source": "", "qpos": [], "qpos_sum": None, "qpos_abs_sum": None, "joint_names": []}

def effective_attack_objective(args, cfg) -> str:
    override = str(getattr(args, "attack_objective", "") or "").strip()
    if override:
        return override
    env_override = str(os.environ.get("V4_ATTACK_OBJECTIVE", "") or "").strip()
    if env_override:
        return env_override
    return str((cfg.get("attack_optimizer") or {}).get("objective", (cfg.get("attack_optimizer") or {}).get("loss_objective", "targeted_directional_ce")))

def apply_attack_objective_override(args, cfg) -> str:
    objective = effective_attack_objective(args, cfg)
    opt = cfg.setdefault("attack_optimizer", {})
    opt["objective"] = objective
    epsilon = getattr(args, "epsilon", None)
    if epsilon is None and os.environ.get("V4_PGD_EPSILON"):
        epsilon = float(os.environ["V4_PGD_EPSILON"])
    if epsilon is not None:
        opt["epsilon"] = float(epsilon)
    step_size = getattr(args, "step_size", None)
    if step_size is None and os.environ.get("V4_PGD_STEP_SIZE"):
        step_size = float(os.environ["V4_PGD_STEP_SIZE"])
    if step_size is not None:
        opt["step_size"] = float(step_size)
    attack_steps = getattr(args, "attack_steps", None)
    if attack_steps is None and os.environ.get("V4_PGD_STEPS"):
        attack_steps = int(os.environ["V4_PGD_STEPS"])
    if attack_steps is not None:
        opt["num_steps"] = int(attack_steps)
    temporal_init = str(getattr(args, "temporal_init", "") or os.environ.get("V4_TEMPORAL_INIT", "") or "").strip()
    if temporal_init:
        opt["temporal_init"] = temporal_init
    cw_margin = getattr(args, "cw_margin", None)
    if cw_margin is None and os.environ.get("V4_CW_MARGIN"):
        cw_margin = float(os.environ["V4_CW_MARGIN"])
    if cw_margin is not None:
        opt["cw_margin"] = float(cw_margin)
    return objective

def build_target_action_for_objective(clean, direction, cfg, args):
    objective = effective_attack_objective(args, cfg)
    target = np.asarray(clean, dtype=np.float32).copy()
    if objective in FORCE_OPEN_OBJECTIVES or objective in FORCE_OPEN_Z_DOWN_OBJECTIVES or objective in GRIPPER_DIAGNOSTIC_OBJECTIVES or objective in ORACLE_ENV_GRIPPER_OPEN_OBJECTIVES:
        raw_value = float(getattr(args, "force_open_raw_gripper", 0.0))
        target[-1] = raw_value
        if objective in FORCE_OPEN_Z_DOWN_OBJECTIVES and len(target) > 2:
            target[2] = -1.0
        return np.clip(target, -1.0, 1.0).astype(np.float32)
    if objective in CONSTANT_DELTA_OBJECTIVES:
        gripper_delta = float(os.environ.get("V4_CONSTANT_DELTA_GRIPPER", "-1.0"))
        target[-1] = np.clip(target[-1] + gripper_delta, -1.0, 1.0)
        return np.clip(target, -1.0, 1.0).astype(np.float32)
    if objective in ADAPTIVE_ANTI_GRIPPER_OBJECTIVES:
        clean_env = postprocess_openvla_action_for_libero(clean, enabled=args.postprocess_gripper)
        clean_env_gripper = float(np.asarray(clean_env, dtype=np.float32)[-1])
        if lift_env_gripper_closed(clean_env_gripper):
            target[-1] = raw_gripper_for_target_env(os.environ.get("V4_FORCE_OPEN_ENV_SIGN", "negative"), postprocess_enabled=args.postprocess_gripper)
        else:
            target[-1] = raw_gripper_for_target_env(os.environ.get("V4_LIFT_CLOSED_GRIPPER_SIGN", "positive"), postprocess_enabled=args.postprocess_gripper)
        return np.clip(target, -1.0, 1.0).astype(np.float32)
    return build_target_action(clean, direction.g_hat, direction.dims, cfg["directional_target"].get("alpha", 1.0))

def target_metadata_for_objective(clean, executed, target, args, cfg):
    objective = effective_attack_objective(args, cfg)
    clean_env = postprocess_openvla_action_for_libero(clean, enabled=args.postprocess_gripper)
    target_env = postprocess_openvla_action_for_libero(target, enabled=args.postprocess_gripper)
    executed_env = postprocess_openvla_action_for_libero(executed, enabled=args.postprocess_gripper)
    meta = {
        "attack_objective": objective,
        "target_action": np.asarray(target, dtype=np.float32).tolist(),
        "target_gripper_raw": float(np.asarray(target, dtype=np.float32)[-1]),
        "target_gripper_env": float(np.asarray(target_env, dtype=np.float32)[-1]),
        "clean_gripper_raw": float(np.asarray(clean, dtype=np.float32)[-1]),
        "clean_gripper_env": float(np.asarray(clean_env, dtype=np.float32)[-1]),
        "executed_gripper_raw": float(np.asarray(executed, dtype=np.float32)[-1]),
        "executed_gripper_env": float(np.asarray(executed_env, dtype=np.float32)[-1]),
    }
    clean_arr = np.asarray(clean, dtype=np.float32)
    executed_arr = np.asarray(executed, dtype=np.float32)
    if len(clean_arr) and len(executed_arr):
        delta_arr = executed_arr - clean_arr
        arm_delta = delta_arr[:-1] if len(delta_arr) > 1 else delta_arr[:0]
        meta["full_action_l2"] = float(np.linalg.norm(delta_arr.reshape(-1)))
        meta["arm_only_l2"] = float(np.linalg.norm(arm_delta.reshape(-1))) if len(arm_delta) else 0.0
        meta["gripper_action_delta"] = float(delta_arr[-1])
        meta["gripper_dim_masked_from_loss"] = bool(objective in ARM_ONLY_UNTARGETED_OBJECTIVES)
    if objective in FORCE_OPEN_OBJECTIVES or objective in FORCE_OPEN_Z_DOWN_OBJECTIVES or objective in GRIPPER_DIAGNOSTIC_OBJECTIVES or objective in ORACLE_ENV_GRIPPER_OPEN_OBJECTIVES:
        meta["force_open_sign_target_ok"] = bool(force_open_env_target_ok(meta["target_gripper_env"]))
    if objective in FORCE_OPEN_Z_DOWN_OBJECTIVES:
        clean_arr = np.asarray(clean, dtype=np.float32)
        target_arr = np.asarray(target, dtype=np.float32)
        executed_arr = np.asarray(executed, dtype=np.float32)
        meta["clean_z_action_raw"] = float(clean_arr[2]) if len(clean_arr) > 2 else None
        meta["target_z_action_raw"] = float(target_arr[2]) if len(target_arr) > 2 else None
        meta["executed_z_action_raw"] = float(executed_arr[2]) if len(executed_arr) > 2 else None
        meta["z_down_target_ok"] = bool(meta["target_z_action_raw"] is not None and meta["target_z_action_raw"] < -0.5)
    if objective in ORACLE_ENV_GRIPPER_OPEN_OBJECTIVES:
        meta["oracle_env_override_objective"] = True
        meta["oracle_gripper_env_forced"] = float(os.environ.get("V4_ORACLE_FORCE_GRIPPER_ENV_VALUE", "-1.0"))
    if objective in ADAPTIVE_ANTI_GRIPPER_OBJECTIVES:
        clean_closed = lift_env_gripper_closed(meta["clean_gripper_env"])
        target_open = force_open_env_target_ok(meta["target_gripper_env"])
        target_closed = lift_env_gripper_closed(meta["target_gripper_env"])
        meta["adaptive_clean_gripper_env"] = meta["clean_gripper_env"]
        meta["adaptive_target_gripper_env"] = meta["target_gripper_env"]
        meta["adaptive_flip_direction"] = "closed_to_open" if clean_closed else "open_to_closed"
        meta["adaptive_target_sign_ok"] = bool((clean_closed and target_open) or ((not clean_closed) and target_closed))
    return meta

def guard_should_block_open(step_meta: dict, env_action, *, close_threshold: float = -0.5, lift_z_min: float = 0.02, release_proximity: float = 0.05, mode: str = "conservative") -> tuple[bool, str]:
    """Online command-layer guard: block open-gripper commands during grasp→pre-release.

    Returns (block, reason).  Block is True only when ALL of these hold:
      (a) env_action gripper is an open command (gripper < close_threshold)
      (b) a close/grasp event has been detected
      (c) bowl has been lifted (bowl_z_delta >= lift_z_min)
      (d) bowl is NOT yet near the plate (release not started)

    This naturally permits normal release-phase gripper opening at the end of
    the task, and should never trigger on clean rollouts (no spurios open
    commands during grasp→carry).
    """
    gripper_cmd = float(np.asarray(env_action, dtype=np.float32).reshape(-1)[-1])
    if gripper_cmd >= close_threshold:
        return False, "gripper_not_open"
    guard_mode = str(mode or "conservative").strip().lower()
    close_seen = bool(step_meta.get("grasp_close_intent", False) or step_meta.get("grasp_first_close_step") is not None or step_meta.get("grasp_first_gate_step") is not None)
    if not close_seen:
        return False, "no_close_intent"
    bowl_plate_dxy = float(step_meta.get("grasp_bowl_plate_dxy", 999.0) or 999.0)
    bowl_plate_dz = float(step_meta.get("grasp_bowl_plate_dz", 999.0) or 999.0)
    if bowl_plate_dxy < release_proximity and bowl_plate_dz < release_proximity:
        return False, "near_release_target"
    if guard_mode in {"strict_after_close", "strict", "after_close"}:
        return True, "guard_blocked_strict_after_close"
    bowl_z_delta = float(step_meta.get("grasp_bowl_z_delta", 0.0) or 0.0)
    if bowl_z_delta < lift_z_min:
        return False, "bowl_not_lifted"
    return True, "guard_blocked"


def primitive_target_ok(value: float, target_primitive: str) -> bool:
    primitive = str(target_primitive or "none").strip().lower()
    if primitive in {"open_gripper", "open", "open_at_lift", "persistent_open"}:
        return force_open_env_target_ok(float(value))
    if primitive in {"close_gripper", "close", "persistent_close"}:
        return lift_env_gripper_closed(float(value))
    return False

def gripper_token_id(token_ids: list[int]) -> int | None:
    return int(token_ids[-1]) if token_ids else None

def resolve_model_gpu_device_id(requested_id: int, render_gpu_device_id: int) -> int:
    visible = int(torch.cuda.device_count())
    if visible <= 0:
        raise SystemExit("CUDA unavailable")
    if requested_id >= 0:
        if requested_id >= visible:
            raise SystemExit(f"requested --model_gpu_device_id={requested_id} but only {visible} CUDA device(s) are visible")
        return int(requested_id)
    if visible == 1:
        return 0
    for idx in range(visible - 1, -1, -1):
        if idx != int(render_gpu_device_id):
            return idx
    return 0


def validate_local_gpu_device_id(requested_id: int, flag_name: str) -> int:
    visible = int(torch.cuda.device_count())
    if visible <= 0:
        raise SystemExit("CUDA unavailable")
    if requested_id < 0 or requested_id >= visible:
        raise SystemExit(f"{flag_name} expects a CUDA-visible local index in [0, {visible - 1}], got {requested_id}")
    return int(requested_id)


def load_model(model_path, model_gpu_device_id: int = -1):
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as AutoModelCls
    processor=AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, use_fast=False)
    visible = int(torch.cuda.device_count())
    mm = os.environ.get("OPENVLA_CUDA_MAX_MEMORY", "").strip()
    if not mm:
        mm = "10000MiB"
    if int(model_gpu_device_id) < 0:
        max_memory = {idx: mm for idx in range(max(visible, 1))}
        max_memory["cpu"] = "128GiB"
        extra_kw = {"device_map": "auto", "max_memory": max_memory}
        mode = "controlled_auto_visible_slot"
    else:
        extra_kw = {
            "device_map": {"": int(model_gpu_device_id)},
            "max_memory": {int(model_gpu_device_id): mm, "cpu": "128GiB"},
        }
        mode = "single_visible_gpu"
    attn_impl = os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "flash_attention_2").strip() or "flash_attention_2"
    model=AutoModelCls.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
        **extra_kw,
    )
    dev = "cuda:0"
    if hasattr(model, "hf_device_map"):
        for v in model.hf_device_map.values():
            if isinstance(v, str) and v.startswith("cuda"):
                dev = v
                break
            if isinstance(v, int):
                dev = f"cuda:{v}"
                break
    print(
        f"[model] loaded path={model_path} mode={mode} primary_device={dev} render_device=cuda:{os.environ.get('OPENVLA_RENDER_LOCAL_DEVICE','0')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')} OPENVLA_CUDA_MAX_MEMORY={mm} attn_implementation={attn_impl} "
        f"hf_device_map={getattr(model, 'hf_device_map', {})}",
        flush=True,
    )
    return model, processor, dev


def resolve_unnorm_key(args, task, model) -> str:
    """Official LIBERO eval uses the suite name as the unnormalization key."""
    keys = list(getattr(model, "norm_stats", {}).keys())
    expected = str(task.get("suite", "") or getattr(args, "unnorm_key", ""))
    if expected not in keys and f"{expected}_no_noops" in keys:
        expected = f"{expected}_no_noops"
    if expected not in keys:
        raise SystemExit(
            f"Official LIBERO eval requires unnorm_key={task.get('suite')} in model.norm_stats; "
            f"available keys={keys}"
        )
    requested = str(getattr(args, "unnorm_key", expected) or expected)
    if requested not in (expected, str(task.get("suite", ""))):
        raise SystemExit(
            f"Refusing silent unnorm fallback: requested {requested!r}, official key is {expected!r}"
        )
    return expected

def _model_float_dtype(model):
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        return dtype
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def decode_with_scores(model, processor, device, image_np, instruction, unnorm_key, k, *, libero_official_preprocess=False, center_crop=False, resize_size=224, drop_attention_mask=True, libero_preprocess_backend="official_pil_lanczos"):
    image=prepare_openvla_image(image_np, libero_official_preprocess=libero_official_preprocess, center_crop=center_crop, resize_size=resize_size, libero_preprocess_backend=libero_preprocess_backend)
    inputs=processor(prompt(str(instruction).lower()), image, return_tensors="pt")
    if drop_attention_mask:
        # OpenVLA's Prismatic generation code inserts visual tokens internally.
        # With transformers 4.40, passing the text-only attention_mask causes
        # a 279 vs 278 causal-mask mismatch; with transformers 5 it can silently
        # degenerate to constant action tokens. Official predict_action works
        # without attention_mask, so V4 uses that path by default.
        inputs.pop("attention_mask", None)
    for key,val in list(inputs.items()):
        if torch.is_floating_point(val): inputs[key]=val.to(device=device, dtype=_model_float_dtype(model))
        else: inputs[key]=val.to(device=device)
    input_ids=inputs.get("input_ids")
    if input_ids is not None and not torch.all(input_ids[:, -1] == 29871):
        inputs["input_ids"]=torch.cat((input_ids, torch.unsqueeze(torch.tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1)
    action_dim=int(model.get_action_dim(unnorm_key))
    with torch.inference_mode():
        t0=time.time()
        gen=model.generate(**inputs, max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
        dt=time.time()-t0
    token_ids=gen.sequences[0, -action_dim:].detach().cpu().numpy()
    vocab_size=model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    discretized=np.clip(vocab_size-token_ids-1, a_min=0, a_max=model.bin_centers.shape[0]-1)
    norm_actions=model.bin_centers[discretized]
    stats=model.get_action_stats(unnorm_key); mask=stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    high, low=np.array(stats["q99"]), np.array(stats["q01"])
    action=np.where(mask, 0.5*(norm_actions+1)*(high-low)+low, norm_actions).astype(np.float32)
    return action, extract_prefix_logits(gen, k), dt, gen



def decode_prepared_inputs_with_scores(model, device, prepared_inputs, unnorm_key, k):
    inputs = {kk: vv for kk, vv in prepared_inputs.items()}
    for key, val in list(inputs.items()):
        if torch.is_tensor(val):
            if torch.is_floating_point(val):
                inputs[key] = val.to(device=device, dtype=_model_float_dtype(model))
            else:
                inputs[key] = val.to(device=device)
    input_ids = inputs.get("input_ids")
    if input_ids is not None and not torch.all(input_ids[:, -1] == 29871):
        inputs["input_ids"] = torch.cat((input_ids, torch.unsqueeze(torch.tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1)
    action_dim = int(model.get_action_dim(unnorm_key))
    with torch.inference_mode():
        t0=time.time()
        gen=model.generate(**inputs, max_new_tokens=action_dim, do_sample=False, return_dict_in_generate=True, output_scores=True)
        dt=time.time()-t0
    token_ids=gen.sequences[0, -action_dim:].detach().cpu().numpy()
    vocab_size=model.config.text_config.vocab_size - model.config.pad_to_multiple_of
    discretized=np.clip(vocab_size-token_ids-1, a_min=0, a_max=model.bin_centers.shape[0]-1)
    norm_actions=model.bin_centers[discretized]
    stats=model.get_action_stats(unnorm_key); mask=stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    high, low=np.array(stats["q99"]), np.array(stats["q01"])
    action=np.where(mask, 0.5*(norm_actions+1)*(high-low)+low, norm_actions).astype(np.float32)
    return action, extract_prefix_logits(gen, k), dt, gen



def action_token_ids_from_gen(gen, action_dim):
    if gen is None or not hasattr(gen, "sequences"):
        return []
    return [int(x) for x in gen.sequences[0, -int(action_dim):].detach().cpu().tolist()]

def fake_logits(rng,k=8,v=32): return rng.randn(k,v).astype(np.float32)
def fake_action(rng): return rng.uniform(-1,1,size=7).astype(np.float32)

def build_record(args,cfg,task,run_id,ep,t,max_steps,dec,bd,clean,executed,al,attack_res,Tclean,Ttrig,Tattack,action_low=None,action_high=None,nad_dims=None,step_metadata=None):
    delta=compute_delta_action(executed, clean)
    if nad_dims is None:
        nad_dims=cfg.get("directional_target",{}).get("dims", list(range(len(clean))))
    nad_clean=normalized_action_discrepancy_cleanref(clean, executed, action_low, action_high, nad_dims)
    low_list=[] if action_low is None else np.asarray(action_low,dtype=np.float32).tolist()
    high_list=[] if action_high is None else np.asarray(action_high,dtype=np.float32).tolist()
    dbg = attack_res.debug if (attack_res and isinstance(attack_res.debug, dict)) else {}
    n_loss_fwd = int(dbg.get("num_loss_forwards", 0)) if bd.attack_active else 0
    n_back = int(dbg.get("num_backwards", 0)) if bd.attack_active else 0
    n_adv_dec = int(dbg.get("num_adv_decodes", 1)) if bd.attack_active else 0
    threshold_active = dec.raw_active if getattr(dec, "threshold_active", None) is None else bool(dec.threshold_active)
    rec = {"version":"v4","run_id":run_id,"experiment_id":cfg["experiment_id"],"task_id":task["task_id"],"suite":task["suite"],"episode_id":ep,"seed":args.seed,"step_idx":t,"max_steps":max_steps,"trigger_name":args.trigger,"rho":args.rho,"trigger_timing":cfg.get("trigger_timing","current_step_observer_decode"),"raw_trigger_score":dec.score,"trigger_active_raw":threshold_active,"trigger_request_active":dec.raw_active,"attack_active":bd.attack_active,"budget_max_steps":bd.budget_used_before+bd.budget_remaining_after+(1 if bd.attack_active else 0),"budget_used_before":bd.budget_used_before,"budget_remaining_after":bd.budget_remaining_after,"budget_blocked":bd.budget_blocked,"signal_available":dec.signal_available,"fallback":dec.fallback,"fallback_reason":dec.reason,"oracle":dec.oracle,"privileged":dec.privileged,"Ntrig_fwd":0,"Ntrig_decode":0,"Nattack_fwd":n_loss_fwd,"Nattack_decode":n_adv_dec,"Tclean_decode":Tclean,"Ttrig":Ttrig,"Tattack":Tattack,"Ttotal":Tclean+Ttrig+Tattack,"Mcache":0,"Asignal":1 if dec.signal_available else 0,"clean_action":clean.tolist(),"executed_action":executed.tolist(),"action_delta":delta.tolist(),"action_bound_low":low_list,"action_bound_high":high_list,"nad_cleanref_dims":[int(x) for x in nad_dims],"nad_cleanref_step":nad_clean,"directional_alignment":al["alignment"],"directional_alignment_cos":al["alignment_cos"],"delta_l2":al["delta_l2"],"delta_linf":al["delta_linf"],"attack_method":attack_res.attack_method if attack_res else "none","directional_loss_available":attack_res.directional_loss_available if attack_res else False,"epsilon":cfg["attack_optimizer"]["epsilon"],"attack_steps":attack_res.num_attack_steps if attack_res else 0,"observation_perturb_linf":attack_res.observation_perturb_linf if attack_res else 0.0,"observation_perturb_l2":attack_res.observation_perturb_l2 if attack_res else 0.0,"target_token_ids":[],"clean_token_ids":[],"adv_token_ids":[],"target_ce_initial":None,"target_ce_final":None,"token_match_rate":None,"token_changed_count":0,"pixel_space":"","num_loss_forwards":n_loss_fwd,"num_backwards":n_back,"success_so_far":False,"moka_stage_id":"","moka_attack_enabled_by_stage":True,"moka_anchor_step":None,"moka_relative_step":None,"moka_first_pot_on_stove":False,"moka_second_pot_on_stove":False,"moka_first_pot_stove_dxy":None,"moka_first_pot_stove_dz":None,"moka_second_pot_stove_dxy":None,"moka_second_pot_stove_dz":None,"moka_first_on_stove_streak":0,"moka_second_on_stove_streak":0,"moka_anchor_reason":""}
    if step_metadata:
        rec.update(step_metadata)
    return rec


def matched_provenance(args) -> dict:
    return {
        "matched_to_run_id": args.matched_to_run_id or "",
        "matched_to_trigger": args.matched_to_trigger or "",
        "matched_to_attacked_ratio": (None if args.matched_to_attacked_ratio < 0 else float(args.matched_to_attacked_ratio)),
    }


def write_progress_stub(args, task, run_id, out, max_steps, model_path, status="starting", error=""):
    attack_objective = effective_attack_objective(args, load_yaml(args.attack_config) if getattr(args, "attack_config", "") and Path(args.attack_config).exists() else {})
    write_json(str(out/"progress.json"), {
        "version":"v4",
        "status":status,
        "error":error,
        "run_id":run_id,
        "task_id":task["task_id"],
        "trigger_name":args.trigger,
        "rho":args.rho,
        "seed":args.seed,
        "episodes_requested":args.episodes,
        "episodes_completed":0,
        "steps_written":0,
        "max_steps":max_steps,
        "model_checkpoint_path":model_path,
        "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "render_gpu_device_id":int(args.render_gpu_device_id),
        "model_gpu_device_id":int(args.model_gpu_device_id),
        "attack_objective": attack_objective,
        "matched_to_run_id":args.matched_to_run_id or "",
        "matched_to_trigger":args.matched_to_trigger or "",
        "matched_to_attacked_ratio":(None if args.matched_to_attacked_ratio < 0 else float(args.matched_to_attacked_ratio)),
        "updated_at":time.strftime("%Y-%m-%dT%H:%M:%S"),
    })


def build_run_manifest(args,cfg,task,run_id,step_path,ep_path,summary_path,model_path,max_steps,status="done",error=""):
    manifest={"version":"v4","run_id":run_id,"created_at":time.strftime("%Y-%m-%dT%H:%M:%S"),"host":socket.gethostname(),"user":getpass.getuser(),"cwd":os.getcwd(),"command":" ".join(sys.argv),"code_git_commit":"unknown","code_dirty":"unknown","config_hash":sha256_jsonable(cfg),"attack_config_path":args.attack_config,"tasks_config_path":args.tasks_config,"directions_config_path":args.directions_config,"thresholds_path":args.thresholds or "","model_id":cfg.get("victim","openvla_7b"),"model_checkpoint_path":model_path,"dataset_manifest_hash":"","task_id":task["task_id"],"suite":task["suite"],"seed":args.seed,"trigger_name":args.trigger,"rho":args.rho,"episodes":args.episodes,"max_steps":max_steps,"output_files":{"steps":str(step_path),"episodes":str(ep_path),"summary":str(summary_path)},"status":status,"error":error,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES", ""),"render_gpu_device_id":int(args.render_gpu_device_id),"model_gpu_device_id":int(args.model_gpu_device_id),"attack_objective":effective_attack_objective(args,cfg),"force_open_raw_gripper":float(getattr(args,"force_open_raw_gripper",0.0)),"grasp_gate_dist_threshold":float(getattr(args,"grasp_gate_dist",0.10)),
        "preprocess_backend": str(getattr(args, "libero_preprocess_backend", "official_pil_lanczos")),
        "resize_interpolation": "LANCZOS" if str(getattr(args, "libero_preprocess_backend", "official_pil_lanczos")) == "official_pil_lanczos" else "lanczos3" if str(getattr(args, "libero_preprocess_backend", "official_pil_lanczos")) == "tf_jpeg_legacy" else "LANCZOS",
        "uses_jpeg_roundtrip": str(getattr(args, "libero_preprocess_backend", "official_pil_lanczos")) == "tf_jpeg_legacy",
        "center_crop": bool(args.center_crop),
        "rotate_180": True,
        "prompt_format": "In: What action should the robot take to {task}?\nOut:",
        "eos_token_handling": "add_if_missing_29871",
        "model_inference_path": "model.generate()",
        "python_executable": sys.executable,
        "transformers_version": str(transformers.__version__) if hasattr(transformers, "__version__") else "unknown",
        "tokenizers_version": str(tokenizers.__version__) if hasattr(tokenizers, "__version__") else "unknown",
        "torch_version": str(torch.__version__) if hasattr(torch, "__version__") else "unknown",
        "mujoco_version": "",
        "robosuite_version": "",
        **matched_provenance(args)}
    manifest.update({
        "instruction_override": str(getattr(args, "instruction_override", "") or ""),
        "instruction_suffix": str(getattr(args, "instruction_suffix", "") or ""),
        "prompt_variants_path": str(getattr(args, "prompt_variants_path", "") or ""),
        "offline_prompt_audit": bool(getattr(args, "offline_prompt_audit", False)),
        "prompt_id": str(getattr(args, "prompt_id", "") or ""),
        "prompt_type": str(getattr(args, "prompt_type", "") or ""),
        "prompt_attack_config": str(getattr(args, "prompt_attack_config", "") or ""),
        "target_primitive": str(getattr(args, "target_primitive", "") or "none"),
    })
    validate_run_manifest(manifest)
    return manifest


def write_failed_artifacts(args,cfg,task,run_id,out,step_path,ep_path,model_path,max_steps,error):
    summary_path = out/"summary.csv"
    if not step_path.exists():
        write_jsonl(str(step_path), [])
    if not ep_path.exists():
        write_jsonl(str(ep_path), [])
    if not summary_path.exists():
        write_csv(str(summary_path), [])
    progress_path = out/"progress.json"
    if progress_path.exists():
        progress = read_json(str(progress_path))
        progress.update({"status":"failed","error":error,"updated_at":time.strftime("%Y-%m-%dT%H:%M:%S")})
        write_json(str(progress_path), progress)
    else:
        write_progress_stub(args, task, run_id, out, max_steps, model_path, status="failed", error=error)
    manifest=build_run_manifest(args,cfg,task,run_id,step_path,ep_path,out/"summary.csv",model_path,max_steps,status="failed",error=error)
    write_json(str(out/"run_manifest.json"), manifest)

def persist_progress(args,cfg,task,run_id,out,step_path,ep_path,episodes,steps,max_steps,model_path,status="running",error=""):
    """Crash-safe progress writer used after each episode.

    The real LIBERO loop is slow and runs in parallel workers.  Writing partial
    artifacts after every episode makes progress observable and allows the
    aggregate script to consume completed episodes before the full run exits.
    """
    write_jsonl(str(step_path), steps)
    write_jsonl(str(ep_path), episodes)
    summary=aggregate_run(episodes); summary.update({"run_id":run_id,"task_id":task["task_id"],"trigger_name":args.trigger,"rho":args.rho,"seed":args.seed})
    write_csv(str(out/"summary.csv"), [summary])
    write_json(str(out/"progress.json"), {
        "version":"v4", "status":status, "error":error,
        "run_id":run_id, "task_id":task["task_id"], "trigger_name":args.trigger,
        "rho":args.rho, "seed":args.seed, "episodes_requested":args.episodes,
        "episodes_completed":len(episodes), "steps_written":len(steps),
        "max_steps":max_steps, "model_checkpoint_path":model_path,
        "render_gpu_device_id":int(args.render_gpu_device_id),
        "model_gpu_device_id":int(args.model_gpu_device_id),
        "attack_objective":effective_attack_objective(args,cfg),
        "matched_to_run_id":args.matched_to_run_id or "",
        "matched_to_trigger":args.matched_to_trigger or "",
        "matched_to_attacked_ratio":(None if args.matched_to_attacked_ratio < 0 else float(args.matched_to_attacked_ratio)),
        "updated_at":time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    })

def finalize(args,cfg,task,run_id,out,step_path,ep_path,episodes,steps,model_path,max_steps,status="done",error=""):
    write_jsonl(str(step_path), steps); write_jsonl(str(ep_path), episodes)
    summary=aggregate_run(episodes); summary.update({"run_id":run_id,"task_id":task["task_id"],"trigger_name":args.trigger,"rho":args.rho,"seed":args.seed})
    write_csv(str(out/"summary.csv"), [summary])
    manifest=build_run_manifest(args,cfg,task,run_id,step_path,ep_path,out/"summary.csv",model_path,max_steps,status=status,error=error)
    write_json(str(out/"run_manifest.json"), manifest)

def run_dry(args, task, cfg, direction, thresholds):
    apply_attack_objective_override(args, cfg)
    rng=np.random.RandomState(args.seed); trigger=make_trigger(args.trigger,args.seed,thresholds); attacker=OpenVLAVisualAttacker(config=cfg,direction_spec=direction,seed=args.seed)
    run_id=args.run_id or make_run_id({"task_id":task["task_id"],"trigger_name":args.trigger,"rho":args.rho,"seed":args.seed}); out=Path(args.output_root)/run_id; step_path=out/"step_records.jsonl"; ep_path=out/"episode_records.jsonl"
    max_steps=int(args.max_steps_override or task.get("max_steps",30)); all_steps=[]; episodes=[]
    write_progress_stub(args, task, run_id, out, max_steps, "dry_run", status="starting")
    for ep in range(args.episodes):
        trigger.reset(ep,max_steps); budget=OnlineBudgetController(args.rho,max_steps,cfg.get("budget",{}).get("budget_rounding","floor"),cfg.get("budget",{}).get("min_budget_steps",1)); ep_steps=[]
        for t in range(max_steps):
            clean=fake_action(rng); t0=time.time(); dec=trigger.evaluate(TriggerContext(task["task_id"],ep,t,args.rho,prefix_logits=fake_logits(rng, int(cfg["uncertainty"]["K_trigger"])),clean_action=clean)); Ttrig=time.time()-t0; bd=budget.decide(dec.raw_active); executed=clean.copy(); attack_res=None; Tattack=0; adv_gen=None
            target=build_target_action_for_objective(clean,direction,cfg,args)
            if bd.attack_active:
                objective = effective_attack_objective(args, cfg)
                at=time.time()
                if objective in ORACLE_ENV_GRIPPER_OPEN_OBJECTIVES:
                    executed=clean.copy()
                elif objective in CONSTANT_DELTA_OBJECTIVES:
                    executed=target.copy()
                else:
                    attack_res=attacker.attack(np.zeros((16,16,3),dtype=np.uint8),"",clean,target,None)
                    executed=np.clip(target if objective in FORCE_OPEN_OBJECTIVES else clean+0.02*direction.g_hat,-1,1)
                Tattack=time.time()-at
            else:
                attacker.reset_temporal_state()
            al=compute_alignment(compute_delta_action(executed,clean),direction.g_hat,direction.dims); rec=build_record(args,cfg,task,run_id,ep,t,max_steps,dec,bd,clean,executed,al,attack_res,0.001,Ttrig,Tattack,action_low=np.full_like(clean,-1.0),action_high=np.full_like(clean,1.0),nad_dims=cfg.get("directional_target",{}).get("dims", list(range(len(clean))))); rec.update(resolve_instruction_for_run(args, task["task_name"])[1])
            rec.update(target_metadata_for_objective(clean, executed, target, args, cfg))
            rec["guard_blocked"] = False
            rec["guard_reason"] = ""
            rec["guard_mode"] = str(args.guard_mode) if getattr(args, "guard_enabled", False) else ""
            rec.setdefault("clean_gripper_token", "")
            rec.setdefault("attack_gripper_token", "")
            rec.setdefault("gripper_token_flip", False)
            rec["target_primitive_ok"]=bool(len(executed) and primitive_target_ok(float(executed[-1]), rec.get("target_primitive", "none")))
            rec["moka_stage_id"] = "dry_run"
            rec["moka_attack_enabled_by_stage"] = True
            rec["moka_anchor_step"] = None
            rec["moka_relative_step"] = None
            rec["moka_first_pot_on_stove"] = False
            rec["moka_second_pot_on_stove"] = False
            validate_step_record(rec); ep_steps.append(rec); all_steps.append(rec)
        success=True if args.trigger=="clean" else bool(rng.rand()>0.25); epagg=aggregate_episode_from_steps(ep_steps,success,False,False); epagg.update({"version":"v4","run_id":run_id,"experiment_id":cfg["experiment_id"],"task_id":task["task_id"],"suite":task["suite"],"seed":args.seed,"episode_id":ep,"trigger_name":args.trigger,"rho":args.rho,"invalid_reason":"","feasibility_pass":True,"artifact_step_jsonl":str(step_path)}); validate_episode_record(epagg); episodes.append(epagg); persist_progress(args,cfg,task,run_id,out,step_path,ep_path,episodes,all_steps,max_steps,"dry_run",status="running"); print(f"[progress] run={run_id} episode={ep+1}/{args.episodes} steps={len(ep_steps)} success={success}", flush=True)
    finalize(args,cfg,task,run_id,out,step_path,ep_path,episodes,all_steps,"dry_run",max_steps); persist_progress(args,cfg,task,run_id,out,step_path,ep_path,episodes,all_steps,max_steps,"dry_run",status="done"); print("[ok] v4 dry run ->", out, flush=True)

def run_offline_prompt_audit(args, task, cfg, direction, thresholds):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    model_path=args.model_path or cfg.get("model_paths",{}).get(task["suite"]) or cfg.get("model_paths",{}).get("libero_goal")
    run_id=args.run_id or make_run_id({"task_id":task["task_id"],"mode":"offline_prompt_audit","seed":args.seed})
    out=Path(args.output_root)/run_id
    out.mkdir(parents=True, exist_ok=True)
    step_path=out/"step_records.jsonl"
    ep_path=out/"episode_records.jsonl"
    summary_path=out/"summary.csv"
    audit_path=out/"offline_prompt_audit_records.jsonl"
    audit_summary_path=out/"offline_prompt_audit_summary.csv"
    audit_steps=int(getattr(args, "offline_prompt_audit_steps", 1) or 1)
    audit_start_step=int(getattr(args, "offline_prompt_audit_start_step", 0) or 0)
    audit_stride=max(1, int(getattr(args, "offline_prompt_audit_stride", 1) or 1))
    max_steps=int(audit_start_step + max(audit_steps, 1) * audit_stride)
    write_jsonl(str(step_path), [])
    write_jsonl(str(ep_path), [])
    write_csv(str(summary_path), [])
    write_progress_stub(args, task, run_id, out, max_steps, model_path, status="starting")
    if args.auto_patch_compat:
        patch_openvla(Path(args.base_model_code_dir), Path(model_path))
    model,processor,device=load_model(model_path, model_gpu_device_id=int(args.model_gpu_device_id))
    unnorm=resolve_unnorm_key(args, task, model)
    bench=get_benchmark(task["suite"])()
    idx=resolve_task_index(bench,task["task_name"])
    base_instruction=get_instruction(bench,idx,task["task_name"])
    variants=load_prompt_variants(args.prompt_variants_path, base_instruction=base_instruction, args=args)
    init_states=bench.get_task_init_states(idx)
    env=OffScreenRenderEnv(bddl_file_name=bench.get_task_bddl_file_path(idx),camera_heights=int(args.image_size),camera_widths=int(args.image_size),render_gpu_device_id=args.render_gpu_device_id,horizon=int(args.max_steps_override or task["max_steps"])+int(args.num_steps_wait))
    try:
        env.seed(0)
    except Exception:
        pass
    if str(getattr(args, "state_ids", "") or "").strip():
        state_ids=np.asarray([int(x.strip()) for x in str(args.state_ids).split(",") if x.strip() != ""], dtype=np.int64)
        args.episodes=int(len(state_ids))
    else:
        rng=np.random.RandomState(args.seed)
        state_ids=np.arange(args.episodes) % len(init_states) if args.deterministic_init_states else rng.choice(len(init_states), size=args.episodes, replace=args.episodes>len(init_states))
    action_dim_for_tokens=int(model.get_action_dim(unnorm))
    rows=[]
    print(f"[offline-audit] run={run_id} variants={len(variants)} states={','.join(str(int(x)) for x in state_ids)} start={audit_start_step} collect_steps={audit_steps} stride={audit_stride} horizon={max_steps}", flush=True)
    for ep,sid in enumerate(state_ids):
        obs=env.reset()
        obs=env.set_init_state(init_states[int(sid)])
        collected = 0
        for t in range(max_steps):
            if args.camera_obs_key not in obs:
                break
            base_action,_,base_dt,base_gen=decode_with_scores(model,processor,device,obs[args.camera_obs_key],base_instruction,unnorm,int(cfg["uncertainty"]["K_trigger"]),libero_official_preprocess=args.libero_official_preprocess,libero_preprocess_backend=args.libero_preprocess_backend,center_crop=args.center_crop,resize_size=args.openvla_resize_size,drop_attention_mask=(not args.keep_attention_mask))
            base_env=postprocess_openvla_action_for_libero(base_action, enabled=args.postprocess_gripper)
            base_tokens=action_token_ids_from_gen(base_gen, action_dim_for_tokens)
            base_gripper_token=gripper_token_id(base_tokens)
            should_collect = bool(t >= audit_start_step and ((t - audit_start_step) % audit_stride == 0) and collected < audit_steps)
            if should_collect:
                for variant in variants:
                    action,_,dt,gen=decode_with_scores(model,processor,device,obs[args.camera_obs_key],variant["instruction"],unnorm,int(cfg["uncertainty"]["K_trigger"]),libero_official_preprocess=args.libero_official_preprocess,libero_preprocess_backend=args.libero_preprocess_backend,center_crop=args.center_crop,resize_size=args.openvla_resize_size,drop_attention_mask=(not args.keep_attention_mask))
                    env_action=postprocess_openvla_action_for_libero(action, enabled=args.postprocess_gripper)
                    tokens=action_token_ids_from_gen(gen, action_dim_for_tokens)
                    gtok=gripper_token_id(tokens)
                    arm_n=min(len(base_action), len(action), 6)
                    arm_l2=float(np.linalg.norm(np.asarray(action[:arm_n], dtype=np.float32)-np.asarray(base_action[:arm_n], dtype=np.float32))) if arm_n else 0.0
                    rows.append({
                        "run_id":run_id,
                        "task_id":task["task_id"],
                        "seed":int(args.seed),
                        "episode_id":int(ep),
                        "state_id":int(sid),
                        "frame_step":int(t),
                        "audit_start_step":int(audit_start_step),
                        "audit_stride":int(audit_stride),
                        "prompt_id":variant["prompt_id"],
                        "prompt_type":variant["prompt_type"],
                        "target_primitive":variant["target_primitive"],
                        "base_instruction":base_instruction,
                        "effective_instruction":variant["instruction"],
                        "base_gripper_token":base_gripper_token,
                        "attack_gripper_token":gtok,
                        "gripper_token_flip":bool(gtok is not None and base_gripper_token is not None and gtok != base_gripper_token),
                        "base_gripper_env":float(base_env[-1]) if len(base_env) else None,
                        "executed_gripper_env":float(env_action[-1]) if len(env_action) else None,
                        "target_primitive_ok":bool(len(env_action) and primitive_target_ok(float(env_action[-1]), variant["target_primitive"])),
                        "force_open_sign_target_ok":bool(len(env_action) and force_open_env_target_ok(float(env_action[-1]))),
                        "non_gripper_action_drift_l2":arm_l2,
                        "decode_seconds":float(dt),
                        "base_decode_seconds":float(base_dt),
                        "clean_token_ids":base_tokens,
                        "attack_token_ids":tokens,
                    })
                collected += 1
            obs,_,done,_=env.step(base_env)
            if collected >= audit_steps:
                break
            if bool(done) or bool(env.check_success()):
                break
    write_jsonl(str(audit_path), rows)
    grouped={}
    for row in rows:
        key=(row["prompt_id"],row["prompt_type"],row["target_primitive"])
        grouped.setdefault(key,[]).append(row)
    summary_rows=[]
    for (pid,ptype,target),grp in grouped.items():
        n=max(len(grp),1)
        summary_rows.append({
            "condition":pid,
            "prompt_type":ptype,
            "target_primitive":target,
            "frames":len(grp),
            "token_flip_rate":sum(1 for r in grp if r["gripper_token_flip"])/n,
            "target_primitive_command_rate":sum(1 for r in grp if r["target_primitive_ok"])/n,
            "exec_open_rate":sum(1 for r in grp if r["force_open_sign_target_ok"])/n,
            "non_gripper_action_drift_l2_mean":float(np.mean([float(r["non_gripper_action_drift_l2"]) for r in grp])) if grp else "",
        })
    write_csv(str(audit_summary_path), summary_rows)
    manifest=build_run_manifest(args,cfg,task,run_id,step_path,ep_path,summary_path,model_path,max_steps,status="done")
    manifest["output_files"].update({"offline_prompt_audit_records":str(audit_path),"offline_prompt_audit_summary":str(audit_summary_path)})
    write_json(str(out/"run_manifest.json"), manifest)
    write_progress_stub(args, task, run_id, out, max_steps, model_path, status="done")
    try:
        env.close()
    except Exception as e:
        print(f"[warn] env.close failed after offline audit: {e}", flush=True)
    print("[ok] offline prompt audit ->", out, flush=True)

def run_real(args, task, cfg, direction, thresholds):
    if not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    model_path=args.model_path or cfg.get("model_paths",{}).get(task["suite"]) or cfg.get("model_paths",{}).get("libero_goal")
    run_id=args.run_id or make_run_id({"task_id":task["task_id"],"trigger_name":args.trigger,"rho":args.rho,"seed":args.seed})
    out=Path(args.output_root)/run_id; step_path=out/"step_records.jsonl"; ep_path=out/"episode_records.jsonl"; max_steps=int(args.max_steps_override or task["max_steps"]); all_steps=[]; episodes=[]; rng=np.random.RandomState(args.seed)
    write_progress_stub(args, task, run_id, out, max_steps, model_path, status="starting")
    if args.auto_patch_compat: patch_openvla(Path(args.base_model_code_dir), Path(model_path))
    model,processor,device=load_model(model_path, model_gpu_device_id=int(args.model_gpu_device_id)); unnorm=resolve_unnorm_key(args, task, model)
    action_stats=model.get_action_stats(unnorm); action_low=np.asarray(action_stats["q01"],dtype=np.float32); action_high=np.asarray(action_stats["q99"],dtype=np.float32); nad_dims=cfg.get("directional_target",{}).get("dims", list(range(len(action_low))))
    bench=get_benchmark(task["suite"])(); idx=resolve_task_index(bench,task["task_name"]); base_instruction=get_instruction(bench,idx,task["task_name"]); instruction,prompt_meta=resolve_instruction_for_run(args, base_instruction); init_states=bench.get_task_init_states(idx)
    env=OffScreenRenderEnv(bddl_file_name=bench.get_task_bddl_file_path(idx),camera_heights=int(args.image_size),camera_widths=int(args.image_size),render_gpu_device_id=args.render_gpu_device_id,horizon=int(args.max_steps_override or task["max_steps"])+int(args.num_steps_wait))
    try:
        env.seed(0)
    except Exception:
        pass
    apply_attack_objective_override(args, cfg)
    trigger=make_trigger(args.trigger,args.seed,thresholds); attacker=OpenVLAVisualAttacker(model,processor,cfg,direction,args.seed,preprocess_kwargs={"libero_official_preprocess": args.libero_official_preprocess, "libero_preprocess_backend": args.libero_preprocess_backend, "center_crop": args.center_crop, "resize_size": args.openvla_resize_size, "postprocess_gripper": args.postprocess_gripper}, device=device)
    print(f"[start] run={run_id} task={task['task_id']} trigger={args.trigger} rho={args.rho} seed={args.seed} episodes={args.episodes} max_steps={max_steps} prompt_id={prompt_meta['prompt_id']} target_primitive={prompt_meta['target_primitive']} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}", flush=True)
    if str(getattr(args, "state_ids", "") or "").strip():
        state_ids=np.asarray([int(x.strip()) for x in str(args.state_ids).split(",") if x.strip() != ""], dtype=np.int64)
        args.episodes=int(len(state_ids))
    else:
        state_ids=np.arange(args.episodes) % len(init_states) if args.deterministic_init_states else rng.choice(len(init_states), size=args.episodes, replace=args.episodes>len(init_states))
    for ep,sid in enumerate(state_ids):
        obs=env.reset(); obs=env.set_init_state(init_states[int(sid)]);
        if args.num_steps_wait > 0:
            dummy_action=np.array([0,0,0,0,0,0,-1], dtype=np.float32)
            for _ in range(int(args.num_steps_wait)):
                obs,_,_,_=env.step(dummy_action)
        target_object_name=str(os.environ.get("V4_TARGET_OBJECT_NAME", "akita_black_bowl_1"))
        target_receptacle_name=str(os.environ.get("V4_TARGET_RECEPTACLE_NAME", "plate_1"))
        bowl0=object_pos(env, target_object_name)
        eef0=eef_pos(env)
        eef_z_episode_min=float(eef0[2]) if eef0 is not None else None
        grasp_tracker=GraspPhaseTracker()
        grasp_tracker.reset(float(bowl0[2]) if bowl0 is not None else 0.0)
        moka_stage_tracker = MokaTwoPotStageTracker(stable_steps=int(max(1, int(args.moka_stage_stable_steps))))
        moka_stage_tracker.reset()
        proxy_gripper_history=[]; proxy_lift_carry_z_up_streak=0; consecutive_open_streak=0; lift_proxy_streak_min=int(os.environ.get("V4_LIFT_PROXY_Z_UP_STREAK_MIN", "4")); lift_proxy_eef_delta_min=float(os.environ.get("V4_LIFT_PROXY_EEF_Z_DELTA_MIN", "0.04"))
        trigger.reset(ep,max_steps); attacker.reset_temporal_state(); budget=OnlineBudgetController(args.rho,max_steps,cfg.get("budget",{}).get("budget_rounding","floor"),cfg.get("budget",{}).get("min_budget_steps",1)); ep_steps=[]; success=False; timeout=False; invalid=False; invalid_reason=""
        for t in range(max_steps):
            if args.camera_obs_key not in obs: invalid=True; invalid_reason=f"missing camera {args.camera_obs_key}; keys={list(obs.keys())}"; break
            clean,prefix_logits,Tclean,out_clean=decode_with_scores(model,processor,device,obs[args.camera_obs_key],instruction,unnorm,int(cfg["uncertainty"]["K_trigger"]),libero_official_preprocess=args.libero_official_preprocess,libero_preprocess_backend=args.libero_preprocess_backend,center_crop=args.center_crop,resize_size=args.openvla_resize_size,drop_attention_mask=(not args.keep_attention_mask))
            clean_env_action=postprocess_openvla_action_for_libero(clean, enabled=args.postprocess_gripper)
            grasp_meta=compute_grasp_metadata(env,t,clean,clean_env_action,grasp_tracker,bowl_name=target_object_name,plate_name=target_receptacle_name,gate_dist_threshold=float(args.grasp_gate_dist))
            proxy_meta=proxy_grasp_metadata(t,clean,clean_env_action,proxy_gripper_history)
            eef_for_proxy=eef_pos(env)
            if eef_for_proxy is not None:
                eef_z_now=float(eef_for_proxy[2])
                eef_z_episode_min=eef_z_now if eef_z_episode_min is None else min(float(eef_z_episode_min), eef_z_now)
            else:
                eef_z_now=None
            proxy_lift_eef_delta=0.0 if (eef_z_now is None or eef_z_episode_min is None) else float(eef_z_now)-float(eef_z_episode_min)
            proxy_lift_closed=bool(len(clean_env_action) and lift_env_gripper_closed(float(clean_env_action[-1])))
            proxy_lift_z_up=bool(len(clean) >= 3 and float(clean[2]) > 0.0)
            proxy_lift_carry_z_up_streak = (proxy_lift_carry_z_up_streak + 1) if (proxy_lift_closed and proxy_lift_z_up) else 0
            proxy_meta.update({
                "proxy_lift_carry_closed": proxy_lift_closed,
                "proxy_lift_carry_z_up": proxy_lift_z_up,
                "proxy_lift_carry_z_up_streak": int(proxy_lift_carry_z_up_streak),
                "proxy_lift_carry_z_up_streak_min": int(lift_proxy_streak_min),
                "eef_z_episode_min": None if eef_z_episode_min is None else float(eef_z_episode_min),
                "proxy_lift_carry_eef_z": None if eef_z_now is None else float(eef_z_now),
                "proxy_lift_carry_eef_z_delta_from_min": float(proxy_lift_eef_delta),
                "proxy_lift_carry_eef_z_delta_min": float(lift_proxy_eef_delta_min),
                "proxy_lift_carry_gate_active": bool(proxy_lift_carry_z_up_streak >= int(lift_proxy_streak_min)),
                "proxy_lift_carry_eefrise_gate_active": bool(proxy_lift_carry_z_up_streak >= int(lift_proxy_streak_min) and float(proxy_lift_eef_delta) >= float(lift_proxy_eef_delta_min)),
            })
            moka_meta = compute_moka_two_pot_stage_metadata(
                env,
                t,
                moka_stage_tracker,
                enabled=bool(args.moka_two_pot_mode),
                stage_anchor=str(args.moka_stage_anchor),
                first_pot_name=str(args.moka_first_pot_name),
                second_pot_name=str(args.moka_second_pot_name),
                stove_name=str(args.moka_stove_name),
                on_stove_dxy_threshold=float(args.moka_on_stove_dxy_threshold),
                on_stove_dz_threshold=float(args.moka_on_stove_dz_threshold),
            )
            step_meta={**grasp_meta, **proxy_meta, **moka_meta}
            phase_attack_enabled = True
            if bool(args.moka_two_pot_mode):
                phase_attack_enabled = str(moka_meta.get("moka_stage_id", "")) == "second_pot_phase"
            tt=time.time(); dec=trigger.evaluate(TriggerContext(task["task_id"],ep,t,args.rho,prefix_logits=prefix_logits,clean_action=clean,metadata=step_meta,phase_attack_enabled=phase_attack_enabled)); Ttrig=time.time()-tt; bd=budget.decide(dec.raw_active if phase_attack_enabled else False); executed=clean.copy(); attack_res=None; Tattack=0; adv_gen=None; objective=effective_attack_objective(args,cfg)
            if bd.attack_active:
                if step_meta.get("grasp_first_gate_step") is not None:
                    step_meta["first_attack_step_relative_to_grasp"]=int(t)-int(step_meta["grasp_first_gate_step"])
                at=time.time()
                target=build_target_action_for_objective(clean,direction,cfg,args)
                if objective in ORACLE_ENV_GRIPPER_OPEN_OBJECTIVES:
                    executed=clean.copy()
                elif objective in CONSTANT_DELTA_OBJECTIVES:
                    executed=target.copy()
                elif objective in NOISE_BASELINE_OBJECTIVES:
                    noise_eps=float(cfg.get("attack_optimizer",{}).get("epsilon",0.10))
                    rng_noise=np.random.RandomState(args.seed*1000+t*7+ep*13)
                    obs_img=np.asarray(obs[args.camera_obs_key],dtype=np.float32)
                    if bool(obs_img.max()>1.0):
                        obs_img=obs_img/255.0
                    noise=rng_noise.uniform(-noise_eps,noise_eps,size=obs_img.shape).astype(np.float32)
                    noisy_img=np.clip(obs_img+noise,0.0,1.0)
                    noisy_img_uint8=(noisy_img*255.0).astype(np.uint8)
                    executed,_,adv_decode_sec,adv_gen=decode_with_scores(model,processor,device,noisy_img_uint8,instruction,unnorm,int(cfg["uncertainty"]["K_trigger"]),libero_official_preprocess=args.libero_official_preprocess,libero_preprocess_backend=args.libero_preprocess_backend,center_crop=args.center_crop,resize_size=args.openvla_resize_size,drop_attention_mask=(not args.keep_attention_mask))
                    attack_res=AttackResult(x_adv=noisy_img_uint8,action_adv=executed,attack_method="random_uniform_noise",directional_loss_available=False,num_attack_steps=0,epsilon=noise_eps,step_size=0.0,observation_perturb_linf=float(np.max(np.abs(noise))),observation_perturb_l2=float(np.linalg.norm(noise.reshape(-1))))
                else:
                    attack_res=attacker.attack(obs[args.camera_obs_key],instruction,clean,target,out_clean,unnorm_key=unnorm)
                    if attack_res and isinstance(attack_res.debug, dict) and "adv_inputs" in attack_res.debug:
                        executed,_,adv_decode_sec,adv_gen=decode_prepared_inputs_with_scores(model,device,attack_res.debug["adv_inputs"],unnorm,int(cfg["uncertainty"]["K_trigger"]))
                    else:
                        adv_img=attack_res.x_adv
                        executed,_,adv_decode_sec,adv_gen=decode_with_scores(model,processor,device,adv_img,instruction,unnorm,int(cfg["uncertainty"]["K_trigger"]),libero_official_preprocess=args.libero_official_preprocess,libero_preprocess_backend=args.libero_preprocess_backend,center_crop=args.center_crop,resize_size=args.openvla_resize_size,drop_attention_mask=(not args.keep_attention_mask))
                Tattack=time.time()-at
            else:
                attacker.reset_temporal_state()
            if not bd.attack_active:
                target=build_target_action_for_objective(clean,direction,cfg,args)
            al=compute_alignment(compute_delta_action(executed,clean),direction.g_hat,direction.dims)
            target_meta=target_metadata_for_objective(clean,executed,target,args,cfg)
            rec=build_record(args,cfg,task,run_id,ep,t,max_steps,dec,bd,clean,executed,al,attack_res,Tclean,Ttrig,Tattack,action_low=action_low,action_high=action_high,nad_dims=nad_dims,step_metadata={**step_meta, **target_meta})
            rec.update(prompt_meta)
            rec["moka_attack_enabled_by_stage"] = bool(phase_attack_enabled)
            rec["moka_anchor_step"] = step_meta.get("moka_stage_anchor_step")
            rec["moka_relative_step"] = (None if step_meta.get("moka_stage_anchor_step") is None else int(t) - int(step_meta.get("moka_stage_anchor_step")))
            rec["moka_first_pot_stove_dxy"] = step_meta.get("moka_first_pot_stove_dxy")
            rec["moka_first_pot_stove_dz"] = step_meta.get("moka_first_pot_stove_dz")
            rec["moka_second_pot_stove_dxy"] = step_meta.get("moka_second_pot_stove_dxy")
            rec["moka_second_pot_stove_dz"] = step_meta.get("moka_second_pot_stove_dz")
            rec["moka_first_on_stove_streak"] = step_meta.get("moka_first_on_stove_streak", 0)
            rec["moka_second_on_stove_streak"] = step_meta.get("moka_second_on_stove_streak", 0)
            rec["moka_anchor_reason"] = step_meta.get("moka_anchor_reason", "")
            action_dim_for_tokens=int(model.get_action_dim(unnorm))
            clean_token_ids=action_token_ids_from_gen(out_clean, action_dim_for_tokens)
            adv_token_ids=action_token_ids_from_gen(adv_gen, action_dim_for_tokens) if adv_gen is not None else []
            rec["clean_token_ids"]=clean_token_ids
            rec["adv_token_ids"]=adv_token_ids
            rec["clean_gripper_token"]=gripper_token_id(clean_token_ids)
            rec["attack_gripper_token"]=gripper_token_id(adv_token_ids) if adv_token_ids else rec["clean_gripper_token"]
            rec["gripper_token_flip"]=bool(adv_token_ids and rec["attack_gripper_token"] != rec["clean_gripper_token"])
            if attack_res and isinstance(attack_res.debug, dict):
                for _k in ("target_token_ids","target_ce_initial","target_ce_final","pixel_space","num_loss_forwards","num_backwards","temporal_init","temporal_prev_delta_used","temporal_smooth_lambda","temporal_prev_delta_linf"):
                    if _k in attack_res.debug:
                        rec[_k]=attack_res.debug[_k]
                for _prefix in ("clean", "adv"):
                    audit = attack_res.debug.get(f"{_prefix}_logit_audit")
                    if isinstance(audit, dict):
                        rec[f"{_prefix}_logit_audit"] = audit.get("action_token_logit_audit", [])
                        for _k, _v in audit.items():
                            if _k == "action_token_logit_audit":
                                continue
                            rec[f"{_prefix}_{_k}"] = _v
                if "gripper_open_region_token_count" in attack_res.debug:
                    rec["gripper_open_region_token_count"] = attack_res.debug.get("gripper_open_region_token_count")
                target_ids=rec.get("target_token_ids") or []
                if adv_token_ids and target_ids:
                    n=min(len(adv_token_ids), len(target_ids))
                    rec["token_match_rate"]=float(sum(int(adv_token_ids[i]==target_ids[i]) for i in range(n))/max(n,1))
                if adv_token_ids and clean_token_ids:
                    n=min(len(adv_token_ids), len(clean_token_ids))
                    rec["token_changed_count"]=int(sum(int(adv_token_ids[i]!=clean_token_ids[i]) for i in range(n)))
            env_action=postprocess_openvla_action_for_libero(executed, enabled=args.postprocess_gripper)
            oracle_override_active=bool(bd.attack_active and objective in ORACLE_ENV_GRIPPER_OPEN_OBJECTIVES)
            constant_delta_override_active=bool(bd.attack_active and objective in CONSTANT_DELTA_OBJECTIVES)
            rec["oracle_env_override_active"]=oracle_override_active
            rec["constant_delta_override_active"]=constant_delta_override_active
            rec["oracle_env_action_before_override"]=env_action.tolist()
            rec["oracle_env_action_after_override"]=env_action.tolist()
            rec["oracle_gripper_env_forced"]=None
            if oracle_override_active:
                forced=float(os.environ.get("V4_ORACLE_FORCE_GRIPPER_ENV_VALUE", "-1.0"))
                pattern=str(os.environ.get("V4_ORACLE_GRIPPER_PATTERN", "continuous_open")).strip().lower()
                burst_index=int(getattr(bd, "budget_used_before", 0))
                apply_open=True
                if pattern in {"alternating", "alternating_0101", "0101"}:
                    apply_open = (burst_index % 2 == 0)
                elif pattern in {"density30", "30", "30pct", "0.30"}:
                    apply_open = (burst_index % 10) in {0, 3, 6}
                elif pattern in {"density60", "60", "60pct", "0.60"}:
                    apply_open = (burst_index % 10) in {0, 1, 3, 5, 6, 8}
                elif pattern in {"continuous_open", "open", "100", "1.0"}:
                    apply_open = True
                else:
                    apply_open = True
                env_action=np.asarray(env_action, dtype=np.float32).copy()
                if apply_open:
                    env_action[-1]=np.clip(forced, -1.0, 1.0)
                rec["oracle_gripper_pattern"] = pattern
                rec["oracle_gripper_pattern_index"] = burst_index
                rec["oracle_gripper_pattern_open_active"] = bool(apply_open)
                rec["oracle_gripper_env_forced"]=float(env_action[-1]) if apply_open else None
                rec["oracle_env_action_after_override"]=env_action.tolist()
                rec["executed_gripper_env"]=float(env_action[-1])
                rec["target_gripper_env"]=float(env_action[-1]) if apply_open else rec.get("target_gripper_env")
                rec["force_open_sign_target_ok"]=bool(force_open_env_target_ok(float(env_action[-1]))) if apply_open else False
            if len(env_action) and force_open_env_target_ok(float(env_action[-1])):
                consecutive_open_streak += 1
            else:
                consecutive_open_streak = 0
            rec["consecutive_open_streak"] = int(consecutive_open_streak)

            # ── Online command-layer guard ──
            guard_blocked = False; guard_reason = ""
            if args.guard_enabled:
                guard_blocked, guard_reason = guard_should_block_open(step_meta, env_action, mode=args.guard_mode)
                if guard_blocked:
                    env_action = np.asarray(env_action, dtype=np.float32).copy()
                    env_action[-1] = np.clip(1.0, -1.0, 1.0)  # closed in LIBERO postprocessed space
            rec["guard_blocked"] = bool(guard_blocked)
            rec["guard_reason"] = guard_reason
            rec["guard_mode"] = str(args.guard_mode) if args.guard_enabled else ""
            env_action, env_action_before_clamp = apply_action_clamp(env_action, clean_env_action, args.action_clamp_mode)
            rec.update(action_clamp_audit(args.action_clamp_mode, env_action_before_clamp, env_action, clean_env_action))
            rec["env_action_after_clamp"] = env_action.tolist()
            rec["executed_gripper_env"] = float(env_action[-1]) if len(env_action) else rec.get("executed_gripper_env")
            rec["force_open_sign_target_ok"] = bool(force_open_env_target_ok(float(env_action[-1]))) if len(env_action) else False
            rec["target_primitive_ok"]=bool(len(env_action) and primitive_target_ok(float(env_action[-1]), prompt_meta["target_primitive"]))
            eef_before=eef_pos(env); bowl_before=object_pos(env, target_object_name)
            rec["clean_z_action"]=float(np.asarray(clean, dtype=np.float32)[2]) if len(clean) >= 3 else 0.0
            rec["eef_z_before"]=None if eef_before is None else float(eef_before[2])
            rec["bowl_z_before"]=None if bowl_before is None else float(bowl_before[2])
            rec["bowl_z_delta_before"]=float(rec.get("grasp_bowl_z_delta", 0.0) or 0.0)
            gripper_phys_before=physical_gripper_state(env, obs)
            obs,reward,done,info=env.step(env_action)
            eef_after=eef_pos(env); bowl_after=object_pos(env, target_object_name)
            gripper_phys_after=physical_gripper_state(env, obs)
            rec["eef_z_after"]=None if eef_after is None else float(eef_after[2])
            rec["bowl_z_after"]=None if bowl_after is None else float(bowl_after[2])
            rec["gripper_qpos_source"]=gripper_phys_after.get("source") or gripper_phys_before.get("source", "")
            rec["gripper_qpos_joint_names"]=gripper_phys_after.get("joint_names") or gripper_phys_before.get("joint_names", [])
            rec["gripper_qpos_before"]=gripper_phys_before.get("qpos", [])
            rec["gripper_qpos_after"]=gripper_phys_after.get("qpos", [])
            rec["gripper_qpos_sum_before"]=gripper_phys_before.get("qpos_sum")
            rec["gripper_qpos_sum_after"]=gripper_phys_after.get("qpos_sum")
            rec["gripper_qpos_abs_sum_before"]=gripper_phys_before.get("qpos_abs_sum")
            rec["gripper_qpos_abs_sum_after"]=gripper_phys_after.get("qpos_abs_sum")
            rec["physical_gripper_opening_delta"]=None
            if rec["gripper_qpos_abs_sum_before"] is not None and rec["gripper_qpos_abs_sum_after"] is not None:
                rec["physical_gripper_opening_delta"]=float(rec["gripper_qpos_abs_sum_after"])-float(rec["gripper_qpos_abs_sum_before"])
            dz_eef=0.0 if (rec["eef_z_before"] is None or rec["eef_z_after"] is None) else float(rec["eef_z_after"])-float(rec["eef_z_before"])
            dz_bowl=0.0 if (rec["bowl_z_before"] is None or rec["bowl_z_after"] is None) else float(rec["bowl_z_after"])-float(rec["bowl_z_before"])
            rec["z_action_sign_sanity_up"]=bool(rec["clean_z_action"] > 0.0 and (dz_eef > 1e-5 or dz_bowl > 1e-5))
            proxy_gripper_history.append(float(clean_env_action[-1]) if len(clean_env_action) else 0.0)
            proxy_gripper_history=proxy_gripper_history[-5:]
            success_check=bool(env.check_success()); success_done=bool(done); success=success_done if args.success_metric=="done" else success_check
            rec["env_action"]=env_action.tolist(); rec["success_done"]=success_done; rec["success_check"]=success_check; rec["success_so_far"]=success; validate_step_record(rec); ep_steps.append(rec); all_steps.append(rec)
            if success or bool(done): break
        timeout=(not success) and (len(ep_steps)>=max_steps); epagg=aggregate_episode_from_steps(ep_steps,success,timeout,invalid); epagg.update({"version":"v4","run_id":run_id,"experiment_id":cfg["experiment_id"],"task_id":task["task_id"],"suite":task["suite"],"seed":args.seed,"episode_id":ep,"trigger_name":args.trigger,"rho":args.rho,"invalid_reason":invalid_reason,"feasibility_pass":True,"artifact_step_jsonl":str(step_path),"failure_phase":infer_failure_phase(ep_steps,success)}); validate_episode_record(epagg); episodes.append(epagg); persist_progress(args,cfg,task,run_id,out,step_path,ep_path,episodes,all_steps,max_steps,model_path,status="running"); print(f"[progress] run={run_id} episode={ep+1}/{args.episodes} sid={int(sid)} steps={len(ep_steps)} success={success} timeout={timeout} invalid={invalid} phase={epagg.get('failure_phase','')} attacks={epagg.get('num_attack_active_steps',0)} ratio={epagg.get('attacked_step_ratio',0):.3f}", flush=True)
    finalize(args,cfg,task,run_id,out,step_path,ep_path,episodes,all_steps,model_path,max_steps); persist_progress(args,cfg,task,run_id,out,step_path,ep_path,episodes,all_steps,max_steps,model_path,status="done")
    try:
        env.close()
    except Exception as e:
        print(f"[warn] env.close failed after artifacts finalized: {e}", flush=True)
    print("[ok] v4 real run ->", out, flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--tasks_config",default="configs/v4_tasks_libero.yaml"); ap.add_argument("--attack_config",default="configs/v4_attack.yaml"); ap.add_argument("--directions_config",default="configs/v4_directions.yaml"); ap.add_argument("--thresholds",default=""); ap.add_argument("--task_id",required=True); ap.add_argument("--trigger",required=True); ap.add_argument("--rho",type=float,default=0.0); ap.add_argument("--seed",type=int,default=0); ap.add_argument("--episodes",type=int,default=1); ap.add_argument("--max_steps_override",type=int,default=0); ap.add_argument("--output_root",default="outputs/v4/smoke"); ap.add_argument("--run_id",default=""); ap.add_argument("--dry_run",action="store_true"); ap.add_argument("--model_path",default=""); ap.add_argument("--base_model_code_dir",default="${OPENVLA_BASE_MODEL_DIR}"); ap.add_argument("--unnorm_key",default="libero_goal"); ap.add_argument("--camera_obs_key",default="agentview_image"); ap.add_argument("--render_gpu_device_id",type=int,default=0); ap.add_argument("--model_gpu_device_id",type=int,default=-1,help="CUDA-visible local GPU index for model inference; default chooses a non-render GPU when available"); ap.add_argument("--grasp_gate_dist",type=float,default=0.10,help="Privileged grasp gate eef-bowl distance threshold in meters"); ap.add_argument("--attack_objective",default="",help="Override attack_optimizer.objective for this run, e.g. force_gripper_open_token_ce"); ap.add_argument("--epsilon",type=float,default=None,help="Override attack_optimizer.epsilon for reproducibility entrypoints"); ap.add_argument("--step_size",type=float,default=None,help="Override attack_optimizer.step_size for reproducibility entrypoints"); ap.add_argument("--attack_steps",type=int,default=None,help="Override attack_optimizer.num_steps for reproducibility entrypoints"); ap.add_argument("--temporal_init",default="",help="Override attack_optimizer.temporal_init, e.g. prev_delta or none"); ap.add_argument("--cw_margin",type=float,default=None,help="Override attack_optimizer.cw_margin for gripper-logit margin objectives"); ap.add_argument("--force_open_raw_gripper",type=float,default=0.0,help="Raw OpenVLA gripper target used by force_gripper_open objectives"); ap.add_argument("--action_clamp_mode",choices=["none","gripper_clean","arm_clean"],default="none",help="Clamp selected postprocessed env-action dimensions back to clean before env.step."); ap.add_argument("--matched_to_run_id",default=""); ap.add_argument("--matched_to_trigger",default=""); ap.add_argument("--matched_to_attacked_ratio",type=float,default=-1.0); ap.add_argument("--auto_patch_compat",action="store_true"); ap.add_argument("--libero_official_preprocess",action="store_true",help="(Deprecated) Compatibility alias. Does NOT switch preprocessing backend. Use --libero_preprocess_backend to select. Default is official_pil_lanczos.")
    ap.add_argument("--libero_preprocess_backend",choices=["official_pil_lanczos","tf_jpeg_legacy","none"],default="official_pil_lanczos",help="Image preprocessing backend. official_pil_lanczos matches corrected official eval (no JPG, PIL Lanczos). tf_jpeg_legacy is old TF path with JPEG round-trip."); ap.add_argument("--image_size",type=int,default=256); ap.add_argument("--openvla_resize_size",type=int,default=224); ap.add_argument("--center_crop",action="store_true"); ap.add_argument("--num_steps_wait",type=int,default=10); ap.add_argument("--postprocess_gripper",action="store_true",help="Apply official normalize_gripper_action + invert_gripper_action before env.step"); ap.add_argument("--success_metric",choices=["done","check_success"],default="done"); ap.add_argument("--deterministic_init_states",action="store_true",help="Use official-style episode index init states instead of random sampling"); ap.add_argument("--state_ids",default="",help="Comma-separated deterministic LIBERO init-state ids, e.g. 5,7,8; overrides --episodes count for rollout order"); ap.add_argument("--keep_attention_mask",action="store_true",help="Debug only: keep processor attention_mask during OpenVLA generation"); ap.add_argument("--require_rollout_calibration", action="store_true", help="Require rollout-passive calibration provenance for entropy/margin thresholds"); ap.add_argument("--min_calibration_steps", type=int, default=0, help="Minimum calibration num_steps for entropy/margin thresholds"); ap.add_argument("--instruction_override",default="",help="Replace the benchmark instruction for prompt-channel experiments"); ap.add_argument("--instruction_suffix",default="",help="Append a suffix to the effective instruction"); ap.add_argument("--prompt_variants_path",default="",help="JSON/JSONL/CSV prompt set for offline prompt audit"); ap.add_argument("--offline_prompt_audit",action="store_true",help="Run fixed-observation prompt-to-gripper audit without prompt attacks or visual PGD"); ap.add_argument("--offline_prompt_audit_steps",type=int,default=1,help="Number of collected clean-trajectory frames per state for offline prompt audit"); ap.add_argument("--offline_prompt_audit_start_step",type=int,default=0,help="Clean rollout step at which offline prompt audit starts collecting frames"); ap.add_argument("--offline_prompt_audit_stride",type=int,default=1,help="Stride between collected offline prompt audit frames"); ap.add_argument("--prompt_id",default="",help="Stable prompt identifier for prompt-hijack diagnostics"); ap.add_argument("--prompt_type",default="",help="Prompt category, e.g. clean/paraphrase/conflict/suffix"); ap.add_argument("--prompt_attack_config",default="",help="Optional prompt attack manifest/config path"); ap.add_argument("--target_primitive",choices=["none","open_gripper","close_gripper","freeze","open_at_lift","persistent_open"],default="none",help="Primitive targeted by inference-time prompt attack"); ap.add_argument("--guard_enabled",action="store_true",help="Enable online command-layer gripper guard during grasp-to-pre-release"); ap.add_argument("--guard_mode",choices=["conservative","strict_after_close"],default="conservative",help="Guard policy. strict_after_close blocks any open command after close/gate detection until near release."); ap.add_argument("--moka_two_pot_mode",action="store_true",help="Enable two-pot stage tracking for Moka and phase-gated attack activation."); ap.add_argument("--moka_stage_anchor",choices=["first_pot_on_stove_stable"],default="first_pot_on_stove_stable",help="Anchor policy for second-pot relative attack window."); ap.add_argument("--moka_second_window_start",type=int,default=0,help="Relative start step (from anchor) for second-pot window trigger."); ap.add_argument("--moka_second_window_end",type=int,default=30,help="Relative end step (from anchor) for second-pot window trigger."); ap.add_argument("--moka_stage_stable_steps",type=int,default=10,help="Required consecutive stable steps before setting first-pot anchor."); ap.add_argument("--moka_first_pot_name",default="moka_pot_1"); ap.add_argument("--moka_second_pot_name",default="moka_pot_2"); ap.add_argument("--moka_stove_name",default="flat_stove_1"); ap.add_argument("--moka_on_stove_dxy_threshold",type=float,default=0.10); ap.add_argument("--moka_on_stove_dz_threshold",type=float,default=0.08); args=ap.parse_args()
    os.environ["V4_MOKA_SECOND_WINDOW_START"] = str(int(args.moka_second_window_start))
    os.environ["V4_MOKA_SECOND_WINDOW_END"] = str(int(args.moka_second_window_end))
    tasks=load_yaml(args.tasks_config)["tasks"]; task=next(t for t in tasks if t["task_id"]==args.task_id); cfg=load_yaml(args.attack_config); direction=load_direction_spec(args.directions_config,cfg["directional_target"]["direction_id"]); thresholds=read_json(args.thresholds) if args.thresholds and Path(args.thresholds).exists() else {}; validate_thresholds_for_trigger(args.trigger, args.thresholds, thresholds, task["task_id"], args.rho, require_rollout_source=getattr(args, "require_rollout_calibration", False), min_steps=getattr(args, "min_calibration_steps", 0))
    if args.dry_run:
        args.model_gpu_device_id = int(args.model_gpu_device_id)
    else:
        if os.environ.get("OPENVLA_STRICT_LOCAL_RENDER_GPU", "0") == "1":
            args.render_gpu_device_id = validate_local_gpu_device_id(int(args.render_gpu_device_id), "--render_gpu_device_id")
        else:
            args.render_gpu_device_id = int(args.render_gpu_device_id)
            if args.render_gpu_device_id < 0:
                raise SystemExit(f"--render_gpu_device_id must be non-negative, got {args.render_gpu_device_id}")
        if int(args.model_gpu_device_id) >= 0:
            args.model_gpu_device_id = resolve_model_gpu_device_id(int(args.model_gpu_device_id), int(args.render_gpu_device_id))
        else:
            args.model_gpu_device_id = -1
    os.environ["OPENVLA_RENDER_LOCAL_DEVICE"] = str(int(args.render_gpu_device_id))
    if not args.run_id:
        args.run_id = make_run_id({"task_id":task["task_id"],"trigger_name":args.trigger,"rho":args.rho,"seed":args.seed})
    model_path=args.model_path or cfg.get("model_paths",{}).get(task["suite"]) or cfg.get("model_paths",{}).get("libero_goal") or "dry_run"
    max_steps=int(args.max_steps_override or task.get("max_steps",30))
    try:
        if args.dry_run: run_dry(args,task,cfg,direction,thresholds)
        elif args.offline_prompt_audit: run_offline_prompt_audit(args,task,cfg,direction,thresholds)
        else: run_real(args,task,cfg,direction,thresholds)
    except BaseException as exc:
        error=f"{type(exc).__name__}: {exc}"
        try:
            out=Path(args.output_root)/args.run_id; step_path=out/"step_records.jsonl"; ep_path=out/"episode_records.jsonl"
            write_failed_artifacts(args,cfg,task,args.run_id,out,step_path,ep_path,model_path,max_steps,error)
        except Exception as artifact_exc:
            print(f"[warn] failed to write failed artifacts for run={args.run_id}: {artifact_exc}", flush=True)
        raise
if __name__=="__main__": main()
