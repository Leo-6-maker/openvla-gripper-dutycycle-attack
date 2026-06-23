#!/usr/bin/env python3
"""SC5-v2 perturbation generator — clean object-pose offsets only.

Produces deterministic initial-state perturbations for LIBERO Object tasks.
No action noise, no observation patch, no adversarial modification.
"""
import hashlib, json, numpy as np
from pathlib import Path

PERTURBATION_SPEC = {
    "P0": {"dx": 0.0, "dy": 0.0, "dyaw": 0.0},
    "P1": {"dx": 0.005, "dy": 0.0, "dyaw": 0.0},
    "P2": {"dx": -0.005, "dy": 0.0, "dyaw": 0.0},
    "P3": {"dx": 0.0, "dy": 0.005, "dyaw": 0.0},
    "P4": {"dx": 0.0, "dy": -0.005, "dyaw": 0.0},
    "P5": {"dx": 0.0, "dy": 0.0, "dyaw": np.deg2rad(5)},
    "P6": {"dx": 0.0, "dy": 0.0, "dyaw": np.deg2rad(-5)},
    "P7": {"dx": "random_10mm", "dy": "random_10mm", "dyaw": 0.0},
}


def get_perturbation(template_id, base_seed=42):
    """Return (dx, dy, dyaw) in SI units for a given perturbation template."""
    if template_id not in PERTURBATION_SPEC:
        raise ValueError(f"Unknown perturbation template: {template_id}")
    spec = PERTURBATION_SPEC[template_id]
    dx, dy, dyaw = spec["dx"], spec["dy"], spec["dyaw"]
    if template_id == "P7":
        rng = np.random.RandomState(base_seed * 100 + 7)
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.010)
        dx = float(radius * np.cos(angle))
        dy = float(radius * np.sin(angle))
    return dx, dy, dyaw


def apply_perturbation(env, obs, template_id, base_seed=42, task_obj=None, verify=True):
    """Apply perturbation to initial state. Returns (env, obs, spec_dict)."""
    dx, dy, dyaw = get_perturbation(template_id, base_seed)

    # Get object body name from BDDL
    from libero.libero import get_libero_path

    obj_body = None
    bddl_path = Path(get_libero_path("bddl_files")) / task_obj.problem_folder / task_obj.bddl_file
    for line in open(bddl_path).read().split('\n'):
        line = line.strip()
        if line and not line.startswith('(:') and ' - ' in line:
            parts = line.split(' - ')
            main_name = parts[0].strip() + "_main"
            if parts[1].strip() not in ['basket', 'bin']:
                if main_name in set(env.sim.model.body_names):
                    obj_body = main_name
                    break

    if obj_body is None:
        raise ValueError("Could not identify object body for perturbation")

    # Capture initial object pose
    bid = env.sim.model.body_name2id(obj_body)
    orig_pos = env.sim.data.body_xpos[bid].copy()
    orig_quat = env.sim.data.body_xquat[bid].copy()

    # Compute perturbed pose
    new_pos = orig_pos.copy()
    new_pos[0] += dx
    new_pos[1] += dy
    if dyaw != 0:
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_euler('z', dyaw)
        new_quat = (Rotation.from_quat([orig_quat[1], orig_quat[2], orig_quat[3], orig_quat[0]]) * rot).as_quat()
        new_quat = np.array([new_quat[3], new_quat[0], new_quat[1], new_quat[2]])
    else:
        new_quat = orig_quat.copy()

    # Apply perturbation
    env.sim.data.body_xpos[bid] = new_pos
    env.sim.data.body_xquat[bid] = new_quat
    env.sim.forward()

    # Verify collision-free if requested
    if verify:
        env.sim.forward()
        # Basic check: object not below table
        if new_pos[2] < 0.01:
            raise ValueError(f"Perturbation pushed object below table: z={new_pos[2]:.4f}")

    spec = {
        "template_id": template_id,
        "base_seed": base_seed,
        "dx_m": float(dx), "dy_m": float(dy), "dyaw_rad": float(dyaw),
        "object_body": obj_body,
        "original_pos": orig_pos.tolist(),
        "perturbed_pos": new_pos.tolist(),
        "perturbation_generator_sha256": hashlib.sha256(
            json.dumps(PERTURBATION_SPEC, sort_keys=True).encode()
        ).hexdigest(),
    }

    return env, obs, spec


def compute_initial_state_hash(env):
    """Compute SHA256 of the raw MuJoCo state after perturbation."""
    state_bytes = b""
    state_bytes += env.sim.data.qpos.tobytes()
    state_bytes += env.sim.data.qvel.tobytes()
    state_bytes += env.sim.data.act.tobytes()
    return hashlib.sha256(state_bytes).hexdigest()
