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

import os, sys, time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "src"))

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


class C2fLiberoOpenVLAAdapter(RuntimeAdapter):
    """Runs clean OpenVLA episodes and yields per-step observation records."""

    def __init__(self, args):
        super().__init__()
        self._model_cache: Dict[str, Any] = {}
        self._args = args

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
        from v4_run_eval_openvla import decode_with_scores, prompt, postprocess_openvla_action_for_libero, physical_gripper_state
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
        task_language = task.language
        task_bddl = get_libero_path("bddl_files") / task.problem_folder / task.bddl_file

        env_args = {
            "task_name": task.name, "task_bddl_file": str(task_bddl),
            "robots": "Panda", "controller": "OSC_POSE",
            "has_renderer": False, "has_offscreen_renderer": True,
            "use_camera_obs": True, "camera_names": ["frontview"],
            "camera_heights": 224, "camera_widths": 224,
            "control_freq": 20, "env_state_init_seed": state_id,
        }
        env = build_v4_exact_env(**env_args)
        env.seed(seed)
        obs = env.reset()
        apply_dummy_wait(env)

        teacher = _TeacherLabeler(env, task_language)
        _prev_eef = None
        _streamer = SC5StreamingFeatureAdapterV2()

        for step in range(MAX_STEPS):
            raw = obs
            raw_grip = physical_gripper_state(env)
            action, _, _, _ = decode_with_scores(
                vla_model, processor, device, raw, task_language, unnorm_key, 8,
                libero_preprocess_backend="upstream_tf_jpeg", center_crop=True,
                resize_size=224, drop_attention_mask=True,
            )
            env_action = postprocess_openvla_action_for_libero(
                np.asarray(action, dtype=np.float32), enabled=True,
            )

            # RGB from frontview camera
            rgb = obs.get("frontview_image", None)
            if rgb is not None:
                rgb = np.asarray(rgb, dtype=np.uint8)
                if rgb.ndim == 2:
                    rgb = np.stack([rgb]*3, axis=-1)
            else:
                img = env.sim.render(224, 224, camera_name="frontview")
                rgb = np.asarray(img, dtype=np.uint8)
            if rgb is None or rgb.size == 0 or np.max(rgb) < 5:
                raise RuntimeError(f"C2f RGB capture failed at step {step}: image is blank or missing")

            # 25D features via SC5StreamingFeatureAdapterV2
            eef_pos = np.zeros(3)
            try:
                eef_sid = env.sim.model.site_name2id("gripper0_grip_site")
                eef_pos = env.sim.data.site_xpos[eef_sid]
            except Exception:
                pass
            eef_x, eef_y, eef_z = float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])

            if _prev_eef is None:
                _prev_eef = (eef_x, eef_y, eef_z)
            _vx = eef_x - _prev_eef[0]; _vy = eef_y - _prev_eef[1]; _vz = eef_z - _prev_eef[2]
            _prev_eef = (eef_x, eef_y, eef_z)

            qpos = env.sim.data.qpos
            gw = abs(float(qpos[7])) + abs(float(qpos[8]))
            gq = float(qpos[7] + qpos[8])

            try:
                _res = _streamer.update(
                    step_id=step, raw_gripper=raw_grip,
                    env_gripper=-1.0 if raw_grip > 0.5 else 1.0,
                    gripper_qpos=gq, gripper_opening_proxy=gw,
                    eef_x=eef_x, eef_y=eef_y, eef_z=eef_z,
                    eef_vx=_vx, eef_vy=_vy, eef_vz=_vz,
                    action_dx=float(action[0]), action_dy=float(action[1]),
                    action_dz=float(action[2]), action_gripper=raw_grip,
                )
            except Exception:
                _res = {"valid": False}

            fv = {}
            if _res.get("valid"):
                fv = {f: float(_res["features"].get(f, 0.0) or 0.0) for f in CANONICAL_25D_FEATURES}
            else:
                fv = {f: 0.0 for f in CANONICAL_25D_FEATURES}
            features_25d = [fv[f] for f in CANONICAL_25D_FEATURES]

            t = teacher.label(step)

            rec = StepRecord(
                step=step,
                rgb_array=rgb,
                rgb_path=None,
                features_25d=features_25d,
                task_language=task_language,
                teacher_hazard=t["hazard"],
                teacher_primary_attackable=t["primary_attackable"],
                teacher_release_safe=t["release_safe"],
                teacher_event_role=t["event_role"],
                teacher_phase=t["phase"],
            )
            yield rec

            obs, reward, done, info = env.step(env_action)
            if done:
                break

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
