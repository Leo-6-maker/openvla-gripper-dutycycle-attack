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
                try:
                    img = env.sim.render(224, 224, camera_name="frontview")
                    rgb = np.asarray(img, dtype=np.uint8)
                except Exception:
                    rgb = np.zeros((224, 224, 3), dtype=np.uint8)

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
            gq = float(qpos.sum())

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

    Labels are conservative: defaults to unsupported_or_abstain when uncertain.
    Privileged state NEVER exposed in student features.
    """

    def __init__(self, env, task_language: str):
        self._env = env
        self._task_language = task_language
        self._prev_grasp = False
        self._phase = "approach"

    def label(self, step: int) -> Dict[str, Any]:
        env = self._env
        try:
            gripper_qpos = env.sim.data.qpos[7:9]
            gripper_closed = float(gripper_qpos[0] + gripper_qpos[1]) < 0.04
            eef_sid = env.sim.model.site_name2id("gripper0_grip_site")
            eef_z = float(env.sim.data.site_xpos[eef_sid][2])
        except Exception:
            return self._fallback()

        is_grasping = gripper_closed

        if is_grasping and not self._prev_grasp:
            self._phase = "grasp_close"
        elif is_grasping and self._prev_grasp:
            self._phase = "stable_carry" if eef_z > 0.85 else "stable_grasp"
        elif not is_grasping and self._prev_grasp:
            self._phase = "release_safe"
        else:
            self._phase = "approach"

        self._prev_grasp = is_grasping

        hazard = 1 if self._phase == "stable_carry" else 0
        primary = 1 if hazard == 1 else 0
        release_safe = 1 if self._phase == "release_safe" else 0

        if primary and hazard:
            role = "primary_attackable"
        elif is_grasping and not hazard:
            role = "auxiliary_manipulation"
        elif self._phase == "approach":
            role = "distractor_or_setup"
        else:
            role = "unsupported_or_abstain"

        return {"hazard": hazard, "primary_attackable": primary,
                "release_safe": release_safe, "event_role": role, "phase": self._phase}

    def _fallback(self) -> Dict[str, Any]:
        return {"hazard": 0, "primary_attackable": 0, "release_safe": 0,
                "event_role": "unsupported_or_abstain", "phase": "unknown"}


def make_adapter(args):
    """Factory for --adapter-module scripts.stageb.c2f_libero_openvla_adapter:make_adapter"""
    return C2fLiberoOpenVLAAdapter(args)
