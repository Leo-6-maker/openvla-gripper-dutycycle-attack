#!/usr/bin/env python3
"""C2f LIBERO/OpenVLA clean rollout adapter.

Implements RuntimeAdapter from collect_c2f_observation_clean_rollouts.py.
Runs CLEAN-only OpenVLA episodes, yielding StepRecord per step with:
  - RGB frame (numpy array)
  - 25D canonical features
  - task language
  - teacher labels (hazard, primary_attackable, release_safe, event_role, phase)

Teacher labels use privileged simulator state for label generation only.
Privileged values are NEVER exposed in student-accessible fields.
"""

from __future__ import annotations

import os, re, sys, time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))

try:
    from scripts.stageb.collect_c2f_observation_clean_rollouts import StepRecord, RuntimeAdapter
except ImportError:
    from collect_c2f_observation_clean_rollouts import StepRecord, RuntimeAdapter

CANONICAL_25D_FEATURES = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close", "eef_speed", "eef_z_delta_since_close",
    "qpos_delta_1", "qpos_delta_3", "opening_proxy_delta_3",
    "opening_proxy_variance_5", "eef_speed_variance_5",
]

SUITE_MODELS = {
    "libero_10": "/mnt/sdc/dty_user/openvla_attack/models/libero-10/openvla-7b-finetuned-libero-10",
    "libero_goal": "/mnt/sdc/dty_user/openvla_attack/models/libero-goal",
    "libero_object": "/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object",
    "libero_spatial": "/mnt/sdc/dty_user/openvla_attack/models/libero-spatial/spatial_c8f03f4_20260620",
}

MAX_STEPS = 300


def _visible_gpu_id() -> int:
    """Return the first visible GPU id for LIBERO env construction.

    D7 workers usually run with a single CUDA_VISIBLE_DEVICES value.  This
    helper keeps the adapter robust to values like "5,1" while the model still
    uses process-local cuda:0.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    try:
        return int(raw)
    except Exception:
        return 0


def _clean_language_candidate(value: Any) -> str:
    """Normalize a LIBERO task-language candidate.

    LIBERO task objects differ across versions.  Some expose a natural-language
    `.language`, while others leave it blank and keep the usable instruction in
    `.name`, the BDDL stem, or manifest fields.  This helper converts those
    fallbacks into OpenVLA-compatible natural text.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return ""
    text = Path(text).name if "/" in text or "\\" in text else text
    text = re.sub(r"\.bddl$", "", text, flags=re.IGNORECASE)
    # Common LIBERO BDDL/name stems: KITCHEN_SCENE3_put_the_black_bowl_...
    text = re.sub(r"^(?:[A-Z]+_)*SCENE\d+_", "", text)
    text = re.sub(r"^(?:LIBERO[_-]?)?(?:10|GOAL|OBJECT|SPATIAL)[_-]", "", text, flags=re.IGNORECASE)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _resolve_task_language(task: Any, episode_cfg: Dict[str, Any]) -> Tuple[str, str]:
    """Resolve the natural-language instruction robustly.

    Preference order:
      1. explicit manifest fields, if present and non-empty;
      2. natural-language fields on the LIBERO task object;
      3. `task.name` / `problem_name` / `bddl_file` fallback.

    The function raises rather than returning an empty string because an empty
    instruction makes both OpenVLA decoding and C2f teacher grounding invalid.
    """
    manifest_keys = ["task_language", "language", "instruction", "task_instruction", "task_name", "name"]
    for key in manifest_keys:
        text = _clean_language_candidate(episode_cfg.get(key, ""))
        if text:
            return text, f"manifest.{key}"

    attr_keys = [
        "language", "task_language", "instruction", "task_instruction",
        "natural_language", "description", "name", "problem_name", "bddl_file",
    ]
    for key in attr_keys:
        if hasattr(task, key):
            text = _clean_language_candidate(getattr(task, key))
            if text:
                return text, f"task.{key}"

    raise RuntimeError(
        "Could not resolve non-empty LIBERO task language from manifest or task object; "
        f"task_attrs={sorted([a for a in dir(task) if not a.startswith('_')])[:80]}"
    )


def _rgb_from_obs(obs: Dict[str, Any], step: int) -> np.ndarray:
    """Return the victim-aligned RGB frame used by OpenVLA.

    D7 reads obs["agentview_image"] for OpenVLA decoding.  C2f should save the
    same camera stream instead of a different frontview fallback; otherwise the
    visual detector would be trained on a view that is not the attacked input.
    """
    if "agentview_image" not in obs:
        raise RuntimeError(f"C2f RGB capture failed at step {step}: obs lacks agentview_image")
    rgb = np.asarray(obs["agentview_image"])
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[-1] not in (3, 4):
        rgb = np.moveaxis(rgb, 0, -1)
    if rgb.dtype != np.uint8:
        if np.nanmax(rgb) <= 1.0:
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] < 3 or rgb.size == 0 or np.max(rgb[..., :3]) < 5:
        raise RuntimeError(f"C2f RGB capture failed at step {step}: image is blank or malformed, shape={rgb.shape}")
    return rgb[..., :3].copy()


def _extract_gripper_qpos(gs: Any, env: Any) -> Tuple[float, float]:
    """Extract physical gripper qpos from D7-compatible physical_gripper_state.

    D7 calls physical_gripper_state(env, obs) and expects a mapping with a
    qpos field.  Older call signatures or fallback paths may return arrays, so
    keep a conservative fallback to sim.data.qpos[7:9].
    """
    q = None
    if isinstance(gs, dict):
        q = gs.get("qpos", None)
    elif hasattr(gs, "get"):
        q = gs.get("qpos", None)
    elif isinstance(gs, (list, tuple, np.ndarray)):
        q = gs
    try:
        if q is not None and len(q) >= 2:
            return float(q[0]), float(q[1])
    except Exception:
        pass
    try:
        sim_q = env.sim.data.qpos
        return float(sim_q[7]), float(sim_q[8])
    except Exception:
        return float("nan"), float("nan")


class C2fLiberoOpenVLAAdapter(RuntimeAdapter):
    """Runs clean OpenVLA episodes and yields per-step observation records."""

    def __init__(self, args):
        super().__init__()
        self._model_cache: Dict[str, Any] = {}
        self._args = args
        self._last_episode_info: Dict[str, Any] = {}

    def _load_model(self, suite: str):
        if suite in self._model_cache:
            return self._model_cache[suite]
        import torch
        model_path = SUITE_MODELS[suite]
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoModelCls
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoModelCls
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        vla_model = AutoModelCls.from_pretrained(model_path, trust_remote_code=True, local_files_only=True,
                                                  torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda:0")
        model_dtype = next(vla_model.parameters()).dtype
        action_dim = int(vla_model.get_action_dim(suite))
        cache = {"processor": processor, "vla_model": vla_model, "model_dtype": model_dtype,
                 "action_dim": action_dim, "unnorm_key": suite}
        self._model_cache[suite] = cache
        return cache

    def run_clean_episode(self, episode_cfg: Dict[str, Any]) -> Iterable[StepRecord]:
        import torch
        from v4_run_eval_openvla import decode_with_scores, postprocess_openvla_action_for_libero, physical_gripper_state
        from gripper_attack.libero_v4_env_factory import build_v4_exact_env, apply_dummy_wait
        from gripper_attack.sc5_streaming_features_v2 import SC5StreamingFeatureAdapterV2
        from libero.libero import benchmark, get_libero_path

        suite = str(episode_cfg.get("suite", "libero_10"))
        task_idx = int(episode_cfg.get("task_index", 0))
        state_id = int(episode_cfg.get("state_id", 0) or 0)
        seed = int(episode_cfg.get("seed", 42))

        cache = self._load_model(suite)
        processor = cache["processor"]
        vla_model = cache["vla_model"]
        model_dtype = cache["model_dtype"]
        action_dim = cache["action_dim"]
        unnorm_key = cache["unnorm_key"]
        device = "cuda:0"

        bm = benchmark.get_benchmark_dict()
        task_suite = bm[suite]()
        task = task_suite.get_task(task_idx)
        init_states = task_suite.get_task_init_states(task_idx)
        if state_id < 0 or state_id >= len(init_states):
            raise IndexError(f"state_id={state_id} out of range for {suite} task_idx={task_idx}; n_init_states={len(init_states)}")
        task_language, task_language_source = _resolve_task_language(task, episode_cfg)
        task_bddl = str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
        self._last_episode_info = {
            "task_language": task_language,
            "task_language_source": task_language_source,
            "task_name_resolved": _clean_language_candidate(getattr(task, "name", "")),
            "task_bddl": task_bddl,
            "clean_success_observed": False,
        }

        # Match D7 persistent worker order exactly:
        #   build env -> set_init_state(init_states[state_id]) -> dummy wait.
        env, obs = build_v4_exact_env(str(task_bddl), _visible_gpu_id(), MAX_STEPS, 10)
        obs = env.set_init_state(init_states[state_id])
        env, obs = apply_dummy_wait(env, obs, 10)

        teacher = _TeacherLabeler(env, task_language)
        _streamer = SC5StreamingFeatureAdapterV2()
        eef_sid = env.sim.model.site_name2id("gripper0_grip_site")
        _eef_init = env.sim.data.site_xpos[eef_sid]
        _prev_eef = (float(_eef_init[0]), float(_eef_init[1]), float(_eef_init[2]))
        last_info: Dict[str, Any] = {}
        n_yielded = 0

        try:
            for step in range(MAX_STEPS):
                raw = _rgb_from_obs(obs, step)
                gs = physical_gripper_state(env, obs)
                q7, q8 = _extract_gripper_qpos(gs, env)
                qpos_sum = q7 + q8 if np.isfinite(q7) and np.isfinite(q8) else float("nan")

                action, _, _, _ = decode_with_scores(
                    vla_model, processor, device, raw, task_language, unnorm_key, 8,
                    libero_preprocess_backend="upstream_tf_jpeg", center_crop=True,
                    resize_size=224, drop_attention_mask=True,
                )
                raw_grip = float(action[-1])
                env_action = postprocess_openvla_action_for_libero(
                    np.asarray(action, dtype=np.float32), enabled=True,
                )

                # 25D features via the same streaming adapter inputs used by D7.
                eef_pos = env.sim.data.site_xpos[eef_sid]
                eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])
                eef_valid = np.all(np.isfinite([eef_x, eef_y, eef_z]))
                _vx = eef_x - _prev_eef[0] if _prev_eef and eef_valid else float("nan")
                _vy = eef_y - _prev_eef[1] if _prev_eef and eef_valid else float("nan")
                _vz = eef_z - _prev_eef[2] if _prev_eef and eef_valid else float("nan")
                if eef_valid:
                    _prev_eef = (eef_x, eef_y, eef_z)

                gw = abs(q7) + abs(q8) if np.isfinite(q7) and np.isfinite(q8) else float("nan")
                gq = float(qpos_sum) if np.isfinite(qpos_sum) else float("nan")

                try:
                    _res = _streamer.update(
                        step_id=step,
                        raw_gripper=raw_grip,
                        env_gripper=-1.0 if raw_grip > 0.5 else 1.0,
                        gripper_qpos=gq,
                        gripper_opening_proxy=gw,
                        eef_x=eef_x, eef_y=eef_y, eef_z=eef_z,
                        eef_vx=_vx, eef_vy=_vy, eef_vz=_vz,
                        action_dx=float(action[0]), action_dy=float(action[1]),
                        action_dz=float(action[2]), action_gripper=raw_grip,
                    )
                except Exception as e:
                    raise RuntimeError(f"C2f 25D feature construction failed at step {step}: {e}") from e

                if not _res.get("valid"):
                    raise RuntimeError(f"C2f 25D feature construction invalid at step {step}")

                fv = {f: float(_res["features"].get(f, 0.0) or 0.0) for f in CANONICAL_25D_FEATURES}
                features_25d = [fv[f] for f in CANONICAL_25D_FEATURES]

                t = teacher.label(step)

                rec = StepRecord(
                    step=step,
                    rgb_array=raw,
                    rgb_path=None,
                    features_25d=features_25d,
                    task_language=task_language,
                    teacher_hazard=t["hazard"],
                    teacher_primary_attackable=t["primary_attackable"],
                    teacher_release_safe=t["release_safe"],
                    teacher_event_role=t["event_role"],
                    teacher_phase=t["phase"],
                )
                n_yielded += 1
                yield rec

                obs, reward, done, info = env.step(env_action)
                last_info = dict(info or {})
                if done:
                    break
        finally:
            success = bool(
                last_info.get("success", False)
                or last_info.get("task_success", False)
                or last_info.get("is_success", False)
            )
            self._last_episode_info.update({
                "clean_success_observed": success,
                "last_info_keys": sorted(list(last_info.keys())),
                "n_steps_observed": n_yielded,
            })
            env.close()

    def close(self) -> None:
        self._model_cache.clear()


class _TeacherLabeler:
    """Produces teacher labels from clean privileged simulator state.

    Conservative: defaults to unsupported_or_abstain when object grounding fails.
    Does NOT mark all stable_carry as primary — requires object identity match.

    Privileged state NEVER exposed in student features.
    """

    def __init__(self, env, task_language: str):
        self._env = env
        self._task_language = task_language
        self._prev_grasp = False
        self._phase = "approach"
        self._grasped_object_name = ""

    def label(self, step: int) -> Dict[str, Any]:
        env = self._env
        result = {"hazard": 0, "primary_attackable": 0, "release_safe": 0,
                  "event_role": "unsupported_or_abstain", "phase": "unknown"}

        try:
            gripper_qpos = env.sim.data.qpos[7:9]
            gripper_closed = float(gripper_qpos[0] + gripper_qpos[1]) < 0.04
            eef_sid = env.sim.model.site_name2id("gripper0_grip_site")
            eef_z = float(env.sim.data.site_xpos[eef_sid][2])
        except Exception:
            return result

        is_grasping = gripper_closed

        # Phase detection
        if is_grasping and not self._prev_grasp:
            self._phase = "grasp_close"
        elif is_grasping and self._prev_grasp:
            self._phase = "stable_carry" if eef_z > 0.85 else "stable_grasp"
        elif not is_grasping and self._prev_grasp:
            self._phase = "release_safe"
        else:
            self._phase = "approach"

        self._prev_grasp = is_grasping
        result["phase"] = self._phase

        # Release safe
        if self._phase == "release_safe":
            result["release_safe"] = 1
            result["event_role"] = "unsupported_or_abstain"
            return result

        # Hazard window: only during stable_carry (potential attackable phase)
        is_hazard_phase = self._phase == "stable_carry"

        # Try to identify grasped object from simulator state
        grasped_obj = self._identify_grasped_object()
        self._grasped_object_name = grasped_obj if grasped_obj else self._grasped_object_name

        # Primary event: stable_carry AND grasped object matches task language target
        matches_target = self._object_matches_task_target(grasped_obj)

        if is_hazard_phase and matches_target:
            result["hazard"] = 1
            result["primary_attackable"] = 1
            result["event_role"] = "primary_attackable"
        elif is_hazard_phase and grasped_obj and not matches_target:
            # Carrying something, but NOT the task target → distractor/auxiliary
            result["hazard"] = 0
            result["primary_attackable"] = 0
            result["event_role"] = "distractor_or_setup"
        elif is_hazard_phase and not grasped_obj:
            # In stable_carry phase but can't identify object → abstain
            result["hazard"] = 0
            result["primary_attackable"] = 0
            result["event_role"] = "unsupported_or_abstain"
        elif is_grasping and not is_hazard_phase:
            result["event_role"] = "auxiliary_manipulation"
        elif self._phase == "approach":
            result["event_role"] = "distractor_or_setup"
        else:
            result["event_role"] = "unsupported_or_abstain"

        return result

    def _identify_grasped_object(self) -> str:
        """Try to identify which object is grasped using simulator contact/distance.

        Returns object name if identifiable, empty string otherwise.
        Conservative: returns "" when uncertain.
        """
        env = self._env
        try:
            eef_sid = env.sim.model.site_name2id("gripper0_grip_site")
            eef_pos = env.sim.data.site_xpos[eef_sid]

            # Check distance from gripper to each object body
            closest_obj = ""
            closest_dist = float("inf")
            for body_name in env.sim.model.body_names:
                # Filter for manipulable objects (exclude robot, floor, etc.)
                if any(skip in body_name for skip in ["robot", "floor", "world", "gripper", "link", "collision", "visual"]):
                    continue
                try:
                    bid = env.sim.model.body_name2id(body_name)
                    body_pos = env.sim.data.body_xpos[bid]
                    dist = float(np.linalg.norm(eef_pos - body_pos))
                    if dist < 0.15 and dist < closest_dist:
                        closest_dist = dist
                        closest_obj = body_name
                except Exception:
                    continue
            return closest_obj
        except Exception:
            return ""

    def _object_matches_task_target(self, grasped_obj: str) -> bool:
        """Check if grasped object name appears in task language instruction.

        Simple substring matching against LIBERO task language.
        Example: "pick up the black bowl" → "black_bowl" matches "bowl"
        """
        if not grasped_obj:
            return False
        lang_lower = self._task_language.lower()
        # Extract object name parts from simulator body name
        obj_parts = grasped_obj.lower().replace("_", " ").split()
        # Check if any meaningful object part appears in task language
        for part in obj_parts:
            if len(part) > 2 and part in lang_lower:
                return True
        return False


def make_adapter(args):
    """Factory for --adapter-module scripts.stageb.c2f_libero_openvla_adapter:make_adapter"""
    return C2fLiberoOpenVLAAdapter(args)
