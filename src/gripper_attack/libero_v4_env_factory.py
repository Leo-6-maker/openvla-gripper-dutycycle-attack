#!/usr/bin/env python3
"""Unified V4-aligned LIBERO environment factory.

Single source of truth for env construction used by:
- V4 clean scan (run_s20d_v4_fixed_window_l3_runner)
- V5 window smoke (run_s20d_v5_token_pgd_window_smoke)
- V5 full rollout (run_s20d_v5_token_pgd_fixed_window_l3_runner)
- All Day 2+ diagnostic scripts

Ensures identical: camera resolution, camera_names, horizon, env.seed,
dummy wait, render GPU indexing.
"""
import os
import numpy as np
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path

# ── Canonical constants (from V4 runner) ──
CAMERA_HEIGHT = 256
CAMERA_WIDTH = 256
CAMERA_NAMES = ["agentview"]
CONTROL_FREQ = 20
DUMMY_OPEN_ACTION = [0, 0, 0, 0, 0, 0, -1]


def build_v4_exact_env(
    bddl_file: str,
    render_gpu_device_id: int,
    max_steps: int = 280,
    num_steps_wait: int = 10,
):
    """Create an OffScreenRenderEnv exactly matching V4 clean-scan protocol.

    Args:
        bddl_file: path to BDDL task file
        render_gpu_device_id: EGL GPU index (actual, not CUDA_VISIBLE_DEVICES remapped)
        max_steps: maximum episode steps
        num_steps_wait: number of dummy OPEN actions before policy loop

    Returns:
        (env, obs): initialized environment and first observation after dummy wait
    """
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=CAMERA_HEIGHT,
        camera_widths=CAMERA_WIDTH,
        camera_names=CAMERA_NAMES,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        control_freq=CONTROL_FREQ,
        render_gpu_device_id=int(render_gpu_device_id),
        horizon=max_steps + num_steps_wait,
    )
    # V4: env.seed(0) hardcoded
    env.seed(0)

    obs = env.reset()
    # dummy wait is applied AFTER set_init_state by the caller
    return env, obs


def apply_dummy_wait(env, obs, num_steps_wait: int = 10):
    """Apply V4-aligned dummy OPEN actions."""
    for _ in range(num_steps_wait):
        obs, _, _, _ = env.step(DUMMY_OPEN_ACTION)
    return env, obs


def set_init_state(env, obs, init_state):
    """Set initial state and apply dummy wait. Returns (env, obs)."""
    obs = env.set_init_state(init_state)
    return env, obs


def seed_everything(seed: int = 0):
    """Apply determinism controls for reproducibility auditing."""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: cudnn deterministic can slow inference; use only for determinism audit
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# ── TARGET_OBJECT_GUESS (V4 runner names, used in postprocess) ──
TARGET_OBJECT_GUESS_V4 = {
    'ketchup': 'ketchup_green_bottle_1',
    'tomato_sauce': 'tomato_sauce_bottle_1',
    'milk': 'milk_carton_1',
    'butter': 'butter_box_1',
    'cream_cheese': 'cream_cheese_box_1',
    'salad_dressing': 'salad_dressing_bottle_1',
    'bbq_sauce': 'bbq_sauce_bottle_1',
    'alphabet_soup': 'alphabet_soup_can_1',
    'orange_juice': 'orange_juice_carton_1',
    'chocolate_pudding': 'chocolate_pudding_box_1',
}
