"""[DeepSeek] FIT Collection Core — reusable atomic collection primitives.

Extracted from run_r5f_full40_materialize.py and run_r5f_nd_diagnostic.py.
Shared by: R5-F fresh40 runner, FIT670 atomic worker, ND diagnostic.

Protocol: forward-before-capture (PROTOCOL_AMENDMENT_V5_G_REC_DIRECT_POSE)
Quaternion convention: wxyz (MuJoCo body_xquat)
"""
import hashlib, io, json, math, os, pickle, shutil, subprocess, uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

# ── Constants ──
HORIZONS = {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220}
FOUR_SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
FORBIDDEN_PATH_TOKENS = {"cal", "check", "g10", "t2r", "attack", "teacher", "student"}


class CollectionHold(RuntimeError):
    """Episode-level collection failure (non-retryable)."""


# ── File / hash utilities ──
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_image(img):
    """SHA256 of a PIL Image as PNG bytes (deterministic)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def sha256_tensor(t):
    """SHA256 of a tensor's raw bytes (contiguous, CPU, cast from bfloat16 if needed)."""
    if t is None:
        return "NONE"
    ct = t.detach().cpu().contiguous()
    if ct.dtype == torch.bfloat16:
        ct = ct.float()
    return hashlib.sha256(ct.numpy().tobytes()).hexdigest()


def sha256_numpy(arr):
    """SHA256 of a numpy array's raw bytes."""
    if arr is None:
        return "NONE"
    a = np.asarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()


def tensor_info(t):
    """Return {sha256, dtype, shape, device} for a tensor."""
    if t is None:
        return {"sha256": "NONE", "dtype": "NONE", "shape": "NONE", "device": "NONE"}
    return {
        "sha256": sha256_tensor(t),
        "dtype": str(t.dtype),
        "shape": list(t.shape),
        "device": str(t.device),
    }


def git_value(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def reject_path(path):
    parts = {p.lower() for p in Path(path).resolve().parts}
    if parts & FORBIDDEN_PATH_TOKENS:
        raise CollectionHold(f"forbidden path: {path}")


# ── Math utilities ──
def mat_to_quat(m):
    values = [float(x) for x in m]
    a00, a01, a02, a10, a11, a12, a20, a21, a22 = values
    trace = a00 + a11 + a22
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = (0.25 * s, (a21 - a12) / s, (a02 - a20) / s, (a10 - a01) / s)
    elif a00 > a11 and a00 > a22:
        s = math.sqrt(1 + a00 - a11 - a22) * 2
        q = ((a21 - a12) / s, 0.25 * s, (a01 + a10) / s, (a02 + a20) / s)
    elif a11 > a22:
        s = math.sqrt(1 + a11 - a00 - a22) * 2
        q = ((a02 - a20) / s, (a01 + a10) / s, 0.25 * s, (a12 + a21) / s)
    else:
        s = math.sqrt(1 + a22 - a00 - a11) * 2
        q = ((a10 - a01) / s, (a02 + a20) / s, (a12 + a21) / s, 0.25 * s)
    norm = math.sqrt(sum(x * x for x in q))
    return [x / norm for x in q]


def jsonable(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


# ── MuJoCo utilities ──
def _verify_source_stability(qpos_before, qvel_before, act_before, time_before, data, step, label):
    qpos_after = data.qpos.copy()
    qvel_after = data.qvel.copy()
    act_after = data.act.copy() if hasattr(data, 'act') and data.act is not None else None
    time_after = float(data.time)
    pos_drift = float(np.max(np.abs(qpos_before - qpos_after)))
    vel_drift = float(np.max(np.abs(qvel_before - qvel_after)))
    time_drift = abs(float(time_before) - time_after)
    act_none_transition = (act_before is None) != (act_after is None)
    act_len_change = False
    act_drift = 0.0
    if not act_none_transition and act_before is not None and act_after is not None:
        if len(act_before) != len(act_after):
            act_len_change = True
        elif len(act_before) > 0:
            act_drift = float(np.max(np.abs(act_before - act_after)))
    if pos_drift > 0 or vel_drift > 0 or time_drift > 0 or act_drift > 0 or act_none_transition or act_len_change:
        raise CollectionHold(
            f"source state mutated by {label} at step {step}: "
            f"qpos_drift={pos_drift:.2e} qvel_drift={vel_drift:.2e} "
            f"time_drift={time_drift:.2e} act_drift={act_drift:.2e}"
            f"{' act_none_transition' if act_none_transition else ''}"
            f"{' act_len_change' if act_len_change else ''}")
    return True


def collect_entity(model, data, resolution):
    kind = str(resolution.get("entity_type") or "")
    entity_id = int(resolution.get("entity_id", -1))
    if kind == "body":
        if entity_id < 0 or entity_id >= int(model.nbody):
            raise CollectionHold(f"body id out of range: {entity_id}")
        actual_name = str(model.body(entity_id).name or "")
        pos = data.body_xpos[entity_id].tolist()
        quat = [float(x) for x in data.body_xquat[entity_id]]
        if not all(math.isfinite(x) for x in pos):
            raise CollectionHold(f"non-finite body position: {actual_name}")
        if not all(math.isfinite(x) for x in quat):
            raise CollectionHold(f"non-finite body quaternion: {actual_name}")
        parent = int(model.body_parentid[entity_id])
        return {"entity_type": kind, "entity_id": entity_id, "entity_name": actual_name,
                "parent_body_id": parent, "world_pose": {"position": pos, "quaternion": quat}}
    if kind == "site":
        if entity_id < 0 or entity_id >= int(model.nsite):
            raise CollectionHold(f"site id out of range: {entity_id}")
        actual_name = str(model.site(entity_id).name or "")
        body_id = int(model.site_bodyid[entity_id])
        pos = data.site_xpos[entity_id].tolist()
        quat = mat_to_quat(data.site_xmat[entity_id])
        if not all(math.isfinite(x) for x in pos):
            raise CollectionHold(f"non-finite site position: {actual_name}")
        if not all(math.isfinite(x) for x in quat):
            raise CollectionHold(f"non-finite site quaternion: {actual_name}")
        return {"entity_type": kind, "entity_id": entity_id, "entity_name": actual_name,
                "parent_body_id": body_id, "world_pose": {"position": pos, "quaternion": quat}}
    if kind == "geom":
        if entity_id < 0 or entity_id >= int(model.ngeom):
            raise CollectionHold(f"geom id out of range: {entity_id}")
        actual_name = str(model.geom(entity_id).name or "")
        body_id = int(model.geom_bodyid[entity_id])
        pos = data.geom_xpos[entity_id].tolist()
        quat = mat_to_quat(data.geom_xmat[entity_id])
        if not all(math.isfinite(x) for x in pos):
            raise CollectionHold(f"non-finite geom position: {actual_name}")
        if not all(math.isfinite(x) for x in quat):
            raise CollectionHold(f"non-finite geom quaternion: {actual_name}")
        return {"entity_type": kind, "entity_id": entity_id, "entity_name": actual_name,
                "parent_body_id": body_id, "world_pose": {"position": pos, "quaternion": quat}}
    raise CollectionHold(f"unsupported entity kind: {kind}")


def verify_entity_identity(model, etype, eid, expected_name):
    """Verify the live MuJoCo model entity matches the registry."""
    if etype == "body":
        if eid < 0 or eid >= int(model.nbody):
            raise CollectionHold(f"body {eid} out of range")
        actual = str(model.body(eid).name or "")
    elif etype == "site":
        if eid < 0 or eid >= int(model.nsite):
            raise CollectionHold(f"site {eid} out of range")
        actual = str(model.site(eid).name or "")
    elif etype == "geom":
        if eid < 0 or eid >= int(model.ngeom):
            raise CollectionHold(f"geom {eid} out of range")
        actual = str(model.geom(eid).name or "")
    else:
        raise CollectionHold(f"unknown entity type: {etype}")
    if actual != expected_name:
        raise CollectionHold(f"entity identity mismatch: {etype}[{eid}] expected={expected_name} actual={actual}")


def collect_contact_pairs(model, data, registry_resolutions=None, max_contacts=100):
    """Collect contact pairs with position, normal, force, and object-gripper matching.

    MuJoCo mjContact fields used:
      - dist: signed distance (negative = penetration)
      - pos[3]: contact position in world coordinates
      - frame[9]: contact frame; first 3 components = contact normal
      - efc_address: index into efc_force for constraint force

    Object-gripper matching: marks pairs where one body maps to a C1 registry
    object role and the other body is a gripper finger.
    """
    pairs = []
    n = min(int(data.ncon), max_contacts)

    # Build set of object body names from registry for object-gripper matching
    object_body_names = set()
    if registry_resolutions:
        for (etype, eid), res in registry_resolutions.items():
            if res.get("role") == "object" and etype == "body":
                name = res.get("alias_to", res.get("name", ""))
                if name:
                    object_body_names.add(name)

    # Gripper finger body name patterns
    gripper_patterns = ("gripper", "finger", "robot0_right", "robot0_left",
                        "r_gripper", "l_gripper")

    for i in range(n):
        c = data.contact[i]
        g1 = int(c.geom1); g2 = int(c.geom2)
        geom1_name = str(model.geom(g1).name or "") if 0 <= g1 < model.ngeom else "NONE"
        geom2_name = str(model.geom(g2).name or "") if 0 <= g2 < model.ngeom else "NONE"
        b1_id = int(model.geom_bodyid[g1]) if 0 <= g1 < model.ngeom else -1
        b2_id = int(model.geom_bodyid[g2]) if 0 <= g2 < model.ngeom else -1
        body1_name = str(model.body(b1_id).name or "") if 0 <= b1_id < model.nbody else "NONE"
        body2_name = str(model.body(b2_id).name or "") if 0 <= b2_id < model.nbody else "NONE"

        # Contact position and normal from MuJoCo
        pos = [float(c.pos[0]), float(c.pos[1]), float(c.pos[2])]
        normal = [float(c.frame[0]), float(c.frame[1]), float(c.frame[2])]

        # Constraint force (scalar from efc_force array if available)
        force = None
        efc_addr = int(c.efc_address)
        if efc_addr >= 0 and hasattr(data, 'efc_force') and data.efc_force is not None:
            try:
                force = float(data.efc_force[efc_addr])
            except (IndexError, TypeError):
                pass

        # Object-gripper contact detection
        is_object_gripper = False
        body1_lower = body1_name.lower()
        body2_lower = body2_name.lower()
        b1_is_obj = body1_name in object_body_names
        b2_is_obj = body2_name in object_body_names
        b1_is_gripper = any(p in body1_lower for p in gripper_patterns)
        b2_is_gripper = any(p in body2_lower for p in gripper_patterns)
        if (b1_is_obj and b2_is_gripper) or (b2_is_obj and b1_is_gripper):
            is_object_gripper = True

        pairs.append({
            "geom1": geom1_name, "geom2": geom2_name,
            "body1": body1_name, "body2": body2_name,
            "geom1_id": g1, "geom2_id": g2,
            "body1_id": b1_id, "body2_id": b2_id,
            "dist": float(c.dist),
            "position": pos,
            "normal": normal,
            "force": force,
            "efc_address": efc_addr,
            "is_object_gripper_contact": is_object_gripper,
        })
    return pairs


def compute_gripper_width(obs):
    """Compute gripper width from robot0_gripper_qpos observation."""
    qpos = obs.get("robot0_gripper_qpos", [])
    if isinstance(qpos, (list, np.ndarray)) and len(list(qpos)) >= 2:
        vals = [float(x) for x in list(qpos)[:2]]
        return float(vals[0] + vals[1])
    return None


def compute_eef_velocity(obs_current, obs_previous):
    """Compute EEF velocity from consecutive observations (finite difference)."""
    if obs_previous is None:
        return None
    cp = obs_current.get("robot0_eef_pos", [])
    pp = obs_previous.get("robot0_eef_pos", [])
    if len(cp) == 3 and len(pp) == 3:
        return [float(cp[0]) - float(pp[0]),
                float(cp[1]) - float(pp[1]),
                float(cp[2]) - float(pp[2])]
    return None


def capture_gpu_identity(physical_gpu):
    """Capture GPU UUID and PCI bus ID via nvidia-smi. Returns dict or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name",
             "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip().split("\n")
        for line in out:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4 and parts[0] == str(physical_gpu):
                return {
                    "physical_gpu_index": int(parts[0]),
                    "gpu_uuid": parts[1],
                    "pci_bus_id": parts[2],
                    "gpu_name": parts[3],
                }
    except Exception:
        pass
    return {"physical_gpu_index": physical_gpu, "gpu_uuid": "UNAVAILABLE",
            "pci_bus_id": "UNAVAILABLE", "gpu_name": "UNAVAILABLE"}


def get_geom_extents(model, geom_id):
    """Get MuJoCo geom size/extents as list of floats."""
    if geom_id < 0 or geom_id >= int(model.ngeom):
        return []
    gtype = int(model.geom_type[geom_id])
    gsize = model.geom_size[geom_id]
    try:
        return [float(x) for x in gsize]
    except (TypeError, IndexError):
        return []


# ── Validation ──
def _validate_episode_shapes(episode):
    """Validate shape + finiteness of all actions, states, EEF, gripper, entities."""
    for step_idx, row in enumerate(episode.get("steps", [])):
        for key in ["action_raw_7d", "score_action_7d", "action_env_7d"]:
            arr = row.get(key, [])
            if len(arr) != 7:
                raise CollectionHold(f"step {step_idx}: {key} shape {len(arr)} != 7")
            if not all(math.isfinite(float(x)) for x in arr):
                raise CollectionHold(f"step {step_idx}: non-finite {key}")
    for tel_idx, tel in enumerate(episode.get("telemetry", [])):
        for field in ["qpos", "qvel"]:
            arr = tel.get("sim_state", {}).get(field, [])
            if len(arr) == 0:
                raise CollectionHold(f"telemetry {tel_idx}: {field} empty")
            if not all(math.isfinite(float(x)) for x in arr):
                raise CollectionHold(f"telemetry {tel_idx}: non-finite {field}")
        for field in ["robot0_eef_pos", "robot0_eef_quat"]:
            arr = tel.get(field, [])
            if len(arr) == 0:
                raise CollectionHold(f"telemetry {tel_idx}: {field} empty")
            if not all(math.isfinite(float(x)) for x in arr):
                raise CollectionHold(f"telemetry {tel_idx}: non-finite {field}")
        gripper_qpos = tel.get("robot0_gripper_qpos", [])
        if isinstance(gripper_qpos, (list, np.ndarray)):
            vals = list(gripper_qpos)
            if vals and not all(math.isfinite(float(x)) for x in vals):
                raise CollectionHold(f"telemetry {tel_idx}: non-finite gripper_qpos")
        for ent in tel.get("entities", []):
            pos = ent.get("world_pose", {}).get("position", [])
            quat = ent.get("world_pose", {}).get("quaternion", [])
            if len(pos) != 3 or not all(math.isfinite(float(x)) for x in pos):
                raise CollectionHold(f"telemetry {tel_idx}: bad entity position")
            if len(quat) != 4 or not all(math.isfinite(float(x)) for x in quat):
                raise CollectionHold(f"telemetry {tel_idx}: bad entity quaternion")


# ── Sealing ──
def seal_root(staging):
    payload = sorted(p for p in staging.rglob("*") if p.is_file())
    sums = "\n".join(
        f"{sha256_file(p)}  {p.relative_to(staging).as_posix()}"
        for p in payload) + "\n"
    (staging / "SHA256SUMS").write_text(sums, encoding="utf-8")
    sums_sha = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return {"sha256sums_sha256": sums_sha, "file_count": len(payload)}


# ── Registry ──
def load_resolutions(reg_path, allow_articulated=False):
    """Load C1-V2 entity registry and relations. Returns (resolutions, relations)."""
    reg_data = json.loads(Path(reg_path).read_text(encoding="utf-8"))
    legacy = reg_data.get("legacy", reg_data)
    if legacy.get("task_disposition") == "ARTICULATED_UNSUPPORTED":
        if allow_articulated:
            return {}, []
        raise CollectionHold(f"articulated task unsupported: {reg_path}")
    resolutions = {}
    for binding in legacy.get("bindings", []):
        if binding.get("resolution") == "UNRESOLVED":
            raise CollectionHold(f"unresolved binding in {reg_path}: {binding}")
        etype = binding.get("entity_type", "")
        eid = int(binding.get("entity_id", -1))
        key = (etype, eid)
        if key in resolutions:
            existing = resolutions[key]
            if existing.get("role") != binding.get("role"):
                raise CollectionHold(f"binding conflict at {key}")
            continue
        resolutions[key] = {
            "entity_type": etype, "entity_id": eid,
            "name": binding.get("name", ""), "role": binding.get("role", ""),
            "resolution": binding.get("resolution", ""),
            "alias_to": binding.get("alias_to", ""),
        }
    relations = legacy.get("relations", [])
    return resolutions, relations


# ── Model geometry snapshot ──
def capture_model_geometry_snapshot(model, registry_resolutions=None, bddl_sha=None,
                                    model_config_sha=None, c1_registry_binding=None):
    """Capture full model geometry at episode start (once per episode, not per step)."""
    bodies = []
    for i in range(int(model.nbody)):
        bodies.append({
            "id": int(i),
            "name": str(model.body(i).name or ""),
            "parent_id": int(model.body_parentid[i]),
            "default_pos": [float(x) for x in model.body_pos[i]],
            "default_quat": [float(x) for x in model.body_quat[i]],
        })
    geoms = []
    for i in range(int(model.ngeom)):
        gtype = int(model.geom_type[i])
        gsize = model.geom_size[i]
        geoms.append({
            "id": int(i),
            "name": str(model.geom(i).name or ""),
            "type": gtype,
            "body_id": int(model.geom_bodyid[i]),
            "pos": [float(x) for x in model.geom_pos[i]],
            "quat": [float(x) for x in model.geom_quat[i]],
            "size": [float(x) for x in gsize] if hasattr(gsize, '__len__') and len(gsize) >= 1 else [],
        })
    sites = []
    for i in range(int(model.nsite)):
        ssize = getattr(model, 'site_size', None)
        sites.append({
            "id": int(i),
            "name": str(model.site(i).name or ""),
            "body_id": int(model.site_bodyid[i]),
            "pos": [float(x) for x in model.site_pos[i]],
            "quat": [float(x) for x in model.site_quat[i]],
            "size": [float(x) for x in ssize[i]] if ssize is not None and hasattr(ssize, '__len__') else [],
        })
    snapshot = {
        "quaternion_convention": "wxyz",
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "nsite": int(model.nsite),
        "bodies": bodies,
        "geoms": geoms,
        "sites": sites,
    }
    if bddl_sha:
        snapshot["bddl_sha256"] = bddl_sha
    if model_config_sha:
        snapshot["model_config_sha256"] = model_config_sha
    if c1_registry_binding:
        snapshot["c1_registry_binding"] = c1_registry_binding
    if registry_resolutions is not None:
        snapshot["registry_entity_count"] = len(registry_resolutions)
    return snapshot


# ── Atomic staging / publish ──
def make_episode_staging(episode_id, parent_dir):
    """Create a unique staging directory for one episode."""
    safe_id = episode_id.replace("/", "_")
    staging = Path(parent_dir) / f".{safe_id}.staging.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    if staging.exists():
        raise CollectionHold(f"staging collision: {staging}")
    staging.mkdir(parents=True)
    return staging


def compute_episode_target(output_root, suite, task_id, state_id):
    """Compute canonical target path for one episode."""
    return Path(output_root) / "episodes" / suite / f"task_{task_id:02d}" / f"state_{state_id:02d}"


def publish_episode(staging, target):
    """Seal staging and atomically rename to target. Raises CollectionHold if target exists."""
    if target.exists():
        shutil.rmtree(staging, ignore_errors=True)
        raise CollectionHold(f"target exists (would overwrite): {target}")
    seal = seal_root(staging)
    try:
        staging.rename(target)
    except OSError as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise CollectionHold(f"atomic rename failed: {e}")
    return seal


def stage_cleanup(staging):
    """Remove a staging directory that was never published."""
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
