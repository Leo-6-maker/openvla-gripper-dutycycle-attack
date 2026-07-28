"""[DeepSeek] R5-F-ND: Non-Determinism Root-Cause Diagnostic (v1).

Runs only the 13 divergent tasks from the B/C comparison with full per-step
pipeline instrumentation. Does NOT produce consumable episode data — this is
purely diagnostic.

Instrumented pipeline stages per step:
  0. raw_rgb                — SHA of raw image_np as passed to adapter
  1. pil_rgb                — SHA of PIL Image (after .convert("RGB"))
  2. center_crop            — SHA of center-cropped PIL Image
  3. prompt_text            — full prompt string
  4. pixel_values           — SHA/dtype/shape of processor pixel_values tensor
  5. input_ids              — SHA/dtype/shape of processor input_ids tensor
  6. generation_seq         — SHA of generated token IDs
  7. decoded_action         — the 7-D float array before postprocess
  8. executed_action        — the 7-D float array after postprocess
  9. fallback_flag          — whether any error/timeout/exception occurred
  A. rng_state              — SHA of torch + CUDA RNG states before inference
  B. inference_latency_ms   — wall-clock time for predict_action_with_scores
  C. backend_info           — CUDA device, Flash-Attention, PyTorch determinism flags

Usage (server):
  python n5/phase2_labels/run_r5f_nd_diagnostic.py \
    --model-path /path/to/model \
    --official-worker /path/to/official_clean_worker.py \
    --pilot-manifest /path/to/pilot_manifest.json \
    --transition-receipt /path/to/transition_root \
    --registry-root /path/to/registry/per_task \
    --alias-ledger /path/to/ALIAS_LEDGER.json \
    --output-root /path/to/diagnostic_output \
    --gpu 2 \
    --physical-gpu 2 \
    --tasks "libero_goal:0,1,2,3,4,5,6,7,8,9" \
    --tasks "libero_10:2,8,9"
"""
import argparse, copy, hashlib, importlib, io, json, math, os, pickle, shutil
import platform, random, socket, subprocess, sys, time, uuid
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

HORIZONS = {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220}
FOUR_SUITES = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
FORBIDDEN_PATH_TOKENS = {"cal", "check", "g10", "t2r", "attack", "teacher", "student"}

# ── The 13 divergent tasks from B/C comparison ──
DIVERGENT_TASKS = {
    "libero_goal": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "libero_10": [2, 8, 9],
}


class CollectionHold(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_image(img):
    """SHA256 of a PIL Image as PNG bytes (deterministic)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def sha256_tensor(t):
    """SHA256 of a tensor's raw bytes (contiguous, CPU, cast to float32 if needed)."""
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


def load_pilot_identities(pilot_path):
    with open(pilot_path) as f:
        pilot = json.load(f)
    if pilot.get("protected_payload_read") is not False:
        raise CollectionHold("pilot manifest must have protected_payload_read=false")
    if pilot.get("no_attack") is not True:
        raise CollectionHold("pilot manifest must have no_attack=true")
    records = pilot.get("records", [])
    if len(records) != 40:
        raise CollectionHold(f"pilot must have exactly 40 records, got {len(records)}")
    identities = []
    seen = set()
    suite_counts = {s: 0 for s in FOUR_SUITES}
    task_per_suite = {s: set() for s in FOUR_SUITES}
    for rec in records:
        suite = str(rec["suite"])
        task_id = int(rec["task_id"])
        state_id = int(rec["state_id"])
        ep_id = str(rec["episode_id"])
        if "collection_seed" not in rec:
            raise CollectionHold(f"pilot record {ep_id} missing collection_seed")
        seed_val = int(rec["collection_seed"])
        init_sha = rec.get("initial_state_sha256", "")
        if not init_sha or not isinstance(init_sha, str) or len(init_sha) != 64:
            raise CollectionHold(f"pilot record {ep_id} missing or invalid initial_state_sha256")
        if suite not in FOUR_SUITES:
            raise CollectionHold(f"unknown suite: {suite}")
        expected_ep = f"{suite}/task_{task_id:02d}/state_{state_id}"
        if ep_id != expected_ep:
            raise CollectionHold(f"episode_id mismatch: {ep_id} != {expected_ep}")
        if ep_id in seen:
            raise CollectionHold(f"duplicate episode_id: {ep_id}")
        seen.add(ep_id)
        suite_counts[suite] += 1
        task_per_suite[suite].add(task_id)
        identities.append({
            "episode_id": ep_id, "suite": suite, "task_id": task_id,
            "state_id": state_id, "collection_seed": seed_val,
            "initial_state_sha256": init_sha,
        })
    for suite in FOUR_SUITES:
        if suite_counts[suite] != 10:
            raise CollectionHold(f"{suite}: expected 10 identities, got {suite_counts[suite]}")
        if task_per_suite[suite] != set(range(10)):
            raise CollectionHold(f"{suite}: missing task ids")
    return identities


def load_resolutions(reg_path, allow_articulated=True):
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


def capture_backend_info(device):
    """Capture CUDA, Flash-Attention, PyTorch determinism state."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    if hasattr(torch.backends.cuda, 'matmul'):
        info["allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        info["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        info["current_cuda_device"] = dev
        info["cuda_device_name"] = torch.cuda.get_device_name(dev)
        try:
            cap = torch.cuda.get_device_capability(dev)
            info["cuda_compute_capability"] = f"{cap[0]}.{cap[1]}"
        except Exception:
            pass
        try:
            info["cuda_driver_version"] = torch.cuda._CUDA_DRIVER_VERSION  # type: ignore[attr-defined]
        except Exception:
            pass

    # Flash-Attention detection
    try:
        import flash_attn
        info["flash_attn_version"] = getattr(flash_attn, "__version__", "unknown")
        if hasattr(flash_attn, "flash_attn_func"):
            info["flash_attn_func_available"] = True
    except ImportError:
        info["flash_attn_version"] = "NOT_INSTALLED"

    # Check environment variables
    for env_var in [
        "TORCH_CUDNN_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_LAUNCH_BLOCKING", "NVIDIA_TF32_OVERRIDE",
        "FLASH_ATTENTION_DETERMINISTIC",
    ]:
        info[f"env_{env_var}"] = os.environ.get(env_var, "NOT_SET")

    # Check if SDPA is available and which backend is used
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        info["sdpa_available"] = True
        try:
            info["sdpa_backend_used"] = str(torch.backends.cuda.sdp_kernel) if hasattr(
                torch.backends.cuda, 'sdp_kernel') else "NOT_CONFIGURABLE"
        except Exception:
            info["sdpa_backend_used"] = "ERROR_QUERYING"
    else:
        info["sdpa_available"] = False

    return info


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


def create_instrumented_adapter(real_adapter):
    """Wrap the OfficialOpenVLAActionAdapter to capture per-stage diagnostic SHAs.

    We monkey-patch predict_action_with_scores to insert per-stage captures.
    Returns the extended adapter (the original adapter object, modified in place).
    """
    _original_predict = real_adapter.predict_action_with_scores

    # The adapter imports prepare_official_inputs as a local reference, so we
    # must patch the adapter's OWN module, not the protocol module.
    import gripper_attack.official_openvla_adapter as adapter_module
    import gripper_attack.official_libero_protocol as protocol_module
    _original_prepare = adapter_module.prepare_official_inputs

    diag_store = {}

    def instrumented_prepare(processor, image_np, task_label, device, *,
                             center_crop=True, base_vla_name=""):
        """Instrumented version of prepare_official_inputs."""
        # Stage 0: raw RGB numpy array
        diag_store["raw_rgb_sha256"] = sha256_numpy(image_np)
        diag_store["raw_rgb_shape"] = list(np.asarray(image_np).shape)
        diag_store["raw_rgb_dtype"] = str(np.asarray(image_np).dtype)

        # Stage 1: PIL Image
        image = Image.fromarray(np.asarray(image_np)).convert("RGB")
        diag_store["pil_rgb_sha256"] = sha256_image(image)

        # Stage 2: center crop
        if center_crop:
            image = protocol_module.official_center_crop(image)
        diag_store["center_crop_sha256"] = sha256_image(image)

        # Stage 3: prompt
        prompt = protocol_module.official_prompt(task_label, base_vla_name)
        diag_store["prompt_text"] = prompt
        diag_store["prompt_sha256"] = sha256_bytes(prompt.encode("utf-8"))

        # Stage 4: processor
        inputs = processor(prompt, image)
        if hasattr(inputs, "get"):
            for key in ["pixel_values", "input_ids", "attention_mask"]:
                val = inputs.get(key)
                diag_store[f"processor_{key}"] = tensor_info(val)
        else:
            diag_store["processor_output_type"] = str(type(inputs))

        # Stage 5: device transfer
        inputs = inputs.to(device, dtype=torch.bfloat16)
        if hasattr(inputs, "get"):
            for key in ["pixel_values", "input_ids", "attention_mask"]:
                val = inputs.get(key)
                diag_store[f"device_{key}"] = tensor_info(val)
        else:
            diag_store["device_output_type"] = str(type(inputs))

        return dict(inputs), prompt, image

    def instrumented_predict_with_scores(image_np, task_label):
        """Instrumented predict_action_with_scores."""
        diag_store.clear()

        # RNG state before inference
        diag_store["torch_rng_sha256"] = sha256_bytes(
            torch.random.get_rng_state().numpy().tobytes())
        if torch.cuda.is_available():
            diag_store["cuda_rng_sha256"] = sha256_bytes(
                torch.cuda.get_rng_state().numpy().tobytes())

        fallback_detected = False
        error_info = None

        # Monkey-patch adapter module's local reference to prepare_official_inputs
        adapter_module.prepare_official_inputs = instrumented_prepare

        try:
            t0 = time.perf_counter()
            try:
                action, generation, score_meta = _original_predict(image_np, task_label)
            except Exception as e:
                fallback_detected = True
                error_info = {"type": type(e).__name__, "message": str(e)[:500]}
                raise
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                adapter_module.prepare_official_inputs = _original_prepare

            diag_store["inference_latency_ms"] = round(elapsed_ms, 3)
            diag_store["fallback_detected"] = fallback_detected
            if error_info:
                diag_store["error_info"] = error_info

            # Stage 6: generated sequences
            gen = score_meta.get("generation") if isinstance(score_meta, dict) else None
            if gen is not None and hasattr(gen, "sequences"):
                diag_store["generation_seq"] = tensor_info(gen.sequences)
                diag_store["generation_seq_values"] = gen.sequences[0].detach().cpu().tolist()
            if gen is not None and hasattr(gen, "scores"):
                scores = gen.scores
                if scores:
                    diag_store["generation_scores_last"] = tensor_info(scores[-1][0])
                    diag_store["generation_scores_count"] = len(scores)

            # Stage 7: decoded action tokens
            tokens = score_meta.get("tokens") if isinstance(score_meta, dict) else None
            if tokens is not None:
                diag_store["action_tokens"] = [int(t) for t in tokens]
                diag_store["action_tokens_sha256"] = sha256_bytes(
                    np.array(tokens, dtype=np.int64).tobytes())

            # Stage 8: decoded action (raw)
            raw_action = [float(x) for x in jsonable(action)]
            diag_store["decoded_raw_action"] = raw_action
            diag_store["decoded_raw_action_sha256"] = hashlib.sha256(
                np.array(raw_action, dtype=np.float32).tobytes()).hexdigest()

            # Stage 9: postprocess
            executed = [float(x) for x in jsonable(real_adapter.postprocess(action))]
            diag_store["executed_action"] = executed
            diag_store["executed_action_sha256"] = hashlib.sha256(
                np.array(executed, dtype=np.float32).tobytes()).hexdigest()

            # Detect "stay-still default" action pattern
            defaults = [0.002127, -0.00197, 0.000341, -0.000464, -0.000315, -0.000658, 0.996078]
            is_default = all(abs(a - b) < 1e-5 for a, b in zip(raw_action, defaults))
            diag_store["matches_stay_still_default"] = is_default

            return action, generation, score_meta

        except Exception:
            diag_store["fallback_detected"] = True
            diag_store["inference_latency_ms"] = round(
                (time.perf_counter() - t0) * 1000.0, 3)
            raise

    # Monkey-patch the adapter
    real_adapter.predict_action_with_scores = instrumented_predict_with_scores
    real_adapter._diag_store = diag_store  # expose for collection loop

    return real_adapter


def collect_episode_nd(suite, task_idx, state_id, collection_seed, canonical_state,
                       registry_dir, task, adapter, module):
    """Collect a single episode with full ND diagnostics."""
    from experiments.robot.libero.libero_utils import get_libero_image
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    reg_path = Path(registry_dir) / f"{suite}_task_{task_idx:02d}.json"
    registry_data = json.loads(reg_path.read_text(encoding="utf-8"))
    legacy = registry_data.get("legacy", registry_data)
    is_articulated = legacy.get("task_disposition") == "ARTICULATED_UNSUPPORTED"
    resolutions, relations = load_resolutions(str(reg_path), allow_articulated=True)

    bddl_root = Path(get_libero_path("bddl_files")).resolve()
    task_bddl = (bddl_root / task.problem_folder / task.bddl_file).resolve()
    task_bddl_sha = sha256_file(task_bddl)

    module.set_official_seed(collection_seed)
    env = OffScreenRenderEnv(bddl_file_name=str(task_bddl), camera_heights=256, camera_widths=256)
    try:
        env.seed(collection_seed)
        env.reset()
        obs = env.set_init_state(copy.deepcopy(canonical_state))
        for _ in range(int(module.NUM_STEPS_WAIT)):
            obs = env.step([0, 0, 0, 0, 0, 0, -1])[0]

        model = env.sim.model
        for (etype, eid), res in resolutions.items():
            expected_name = res.get("alias_to", res.get("name", "?"))
            verify_entity_identity(model, etype, eid, expected_name)

        rows = []
        privileged = []
        generation_counts = []
        step_diagnostics = []

        for step_num in range(HORIZONS[suite]):
            # ── forward-before-capture protocol ──
            qpos_pre = env.sim.data.qpos.copy()
            qvel_pre = env.sim.data.qvel.copy()
            act_pre = env.sim.data.act.copy() if (hasattr(env.sim.data, 'act') and
                        env.sim.data.act is not None) else None
            time_pre = float(env.sim.data.time)

            if not all(math.isfinite(float(x)) for x in qpos_pre):
                raise CollectionHold(f"non-finite qpos at step {step_num}")
            if not all(math.isfinite(float(x)) for x in qvel_pre):
                raise CollectionHold(f"non-finite qvel at step {step_num}")

            env.sim.forward()
            _verify_source_stability(qpos_pre, qvel_pre, act_pre, time_pre,
                                     env.sim.data, step_num, "capture_forward")
            model = env.sim.model; data = env.sim.data
            sim_state = env.sim.get_state()
            entities = [collect_entity(model, data, res) for res in resolutions.values()]

            privileged.append({
                "step": step_num, "suite": suite, "task_idx": task_idx, "state_id": state_id,
                "sim_state": {
                    "time": float(data.time),
                    "qpos": sim_state.qpos.tolist(),
                    "qvel": sim_state.qvel.tolist(),
                    "act": getattr(sim_state, "act", None).tolist() if getattr(sim_state, "act", None) is not None else None,
                },
                "robot0_eef_pos": jsonable(obs.get("robot0_eef_pos", [])),
                "robot0_eef_quat": jsonable(obs.get("robot0_eef_quat", [])),
                "robot0_gripper_qpos": jsonable(obs.get("robot0_gripper_qpos", [])),
                "object_state": jsonable(obs.get("object-state", [])),
                "entities": entities,
                "forward_before_capture": True,
                "protocol_amendment": "PROTOCOL_AMENDMENT_V5_G_REC_DIRECT_POSE",
                "contact_count": int(data.ncon),
            })

            image = get_libero_image(obs, 224)

            # ── Instrumented inference ──
            clean_action, generation, score_meta = adapter.predict_action_with_scores(
                image, str(task.language))

            # Capture diagnostic data from the instrumented adapter
            diag = dict(adapter._diag_store) if hasattr(adapter, '_diag_store') else {}
            diag["step"] = step_num
            step_diagnostics.append(diag)

            count = score_meta.get("generation_passes_per_step")
            if isinstance(count, bool) or not isinstance(count, int) or count != 1:
                raise CollectionHold(f"generation pass count: {count}")
            generation_counts.append(count)
            score_action = [float(x) for x in jsonable(score_meta["score_action"])]
            raw_action = [float(x) for x in jsonable(clean_action)]
            if len(raw_action) != 7 or len(score_action) != 7:
                raise CollectionHold(f"action shape failed at step {step_num}")
            if max(abs(a - b) for a, b in zip(raw_action, score_action)) > 1e-6:
                raise CollectionHold(f"action parity failed at step {step_num}")
            executed = [float(x) for x in jsonable(adapter.postprocess(clean_action))]
            if len(executed) != 7:
                raise CollectionHold(f"executed action shape failed at step {step_num}")
            for action_label, action_arr in [("raw", raw_action), ("score", score_action), ("executed", executed)]:
                if not all(math.isfinite(x) for x in action_arr):
                    raise CollectionHold(f"non-finite {action_label}_action at step {step_num}: {action_arr}")
            rows.append({
                "step": step_num, "suite": suite, "task_idx": task_idx, "state_id": state_id,
                "action_raw_7d": raw_action, "score_action_7d": score_action,
                "action_env_7d": executed, "generation_passes_per_step": count,
                "single_generation_parity_pass": True, "action_mutation_by_detector": False,
            })
            obs, _reward, done, _info = env.step(executed)
            if done:
                break
    finally:
        env.close()

    if not generation_counts or any(x != 1 for x in generation_counts):
        raise CollectionHold("generation closure failed")

    return {
        "episode_id": f"{suite}/task_{task_idx:02d}/state_{state_id}",
        "suite": suite, "task_id": task_idx, "state_id": state_id,
        "collection_seed": collection_seed,
        "pilot_identity_bound": True,
        "task_bddl_sha256": task_bddl_sha,
        "registry_task_sha256": sha256_file(str(reg_path)),
        "step_count": len(rows), "official_horizon": HORIZONS[suite],
        "generation_passes_per_step": generation_counts,
        "steps": rows, "telemetry": privileged,
        "relations": relations,
        "source_mode": "ND_DIAGNOSTIC_ONLY",
        "forward_before_capture": True,
        "protocol_amendment": "PROTOCOL_AMENDMENT_V5_G_REC_DIRECT_POSE",
        "geometry_status": "NOT_APPLICABLE" if (is_articulated and not resolutions) else "OK",
        "model_inference": True, "attack_enabled": False,
        "detector_loaded": False, "teacher_labels_generated": False,
        # ND-specific: per-step diagnostics
        "nd_diagnostics": step_diagnostics,
    }


def seal_root(staging):
    payload = sorted(p for p in staging.rglob("*") if p.is_file())
    sums = "\n".join(
        f"{sha256_file(p)}  {p.relative_to(staging).as_posix()}"
        for p in payload) + "\n"
    (staging / "SHA256SUMS").write_text(sums, encoding="utf-8")
    sums_sha = sha256_file(staging / "SHA256SUMS")
    (staging / "SHA256SUMS.sha256").write_text(f"{sums_sha}  SHA256SUMS\n", encoding="utf-8")
    return {"sha256sums_sha256": sums_sha, "file_count": len(payload)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--official-worker", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--transition-receipt", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--alias-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--tasks", action="append", default=[],
                        help="Filter: suite:task_id1,task_id2,... (repeatable)")
    parser.add_argument("--label", default="nd",
                        help="Run label suffix for output dir")
    args = parser.parse_args()

    # Parse task filter
    task_filter = {}
    if args.tasks:
        for spec in args.tasks:
            suite, ids_str = spec.split(":")
            task_filter[suite] = [int(x.strip()) for x in ids_str.split(",")]
    if not task_filter:
        task_filter = DIVERGENT_TASKS

    parent_root = Path(args.output_root).resolve()
    out_root = parent_root / f"nd_diag_{args.label}"
    if out_root.exists():
        raise SystemExit(f"output exists: {out_root}")

    for path in [args.model_path, args.official_worker, args.pilot_manifest,
                 args.transition_receipt, args.registry_root, args.alias_ledger]:
        reject_path(path)

    # ── Verify transition receipt ──
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fit_transition import verify_transition
    t_root = Path(args.transition_receipt).resolve()
    verify_transition(
        t_root,
        execution_source_commit=None,  # ND: no commit binding for diagnostics
        script_sha=None,
        model_path=args.model_path,
        official_worker_path=args.official_worker,
        pilot_manifest_path=args.pilot_manifest,
        registry_root=args.registry_root,
        alias_ledger_path=args.alias_ledger,
        upstream_root=Path("/nonexistent"),  # ND: relaxed
        libero_root=Path("/nonexistent"),    # ND: relaxed
        output_root=str(parent_root),
        gpu=args.gpu,
        physical_gpu=args.physical_gpu,
        repo_root=None,
        nd_diagnostic_mode=True,  # signals relaxed verification
    )

    # ── Load identities ──
    all_identities = load_pilot_identities(args.pilot_manifest)
    selected = []
    for ident in all_identities:
        suite = ident["suite"]
        task_id = ident["task_id"]
        if suite in task_filter and task_id in task_filter[suite]:
            selected.append(ident)

    n_selected = len(selected)
    print(f"ND Diagnostic: {n_selected} tasks selected for diagnosis")
    for ident in selected:
        print(f"  {ident['episode_id']}  seed={ident['collection_seed']}")

    # ── Load model ──
    print(f"\n[*] Loading model (GPU {args.gpu}, physical {args.physical_gpu})...")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    worker = Path(args.official_worker).resolve()

    # Protect sys.argv — worker has module-level parse_args() requiring full CLI
    saved_argv = sys.argv[:]
    dummy = "0" * 64
    sys.argv = [
        str(worker), "--suite", "libero_10", "--gpu", str(args.physical_gpu),
        "--worker-id", "r5f_nd_diag", "--model-path", str(args.model_path),
        "--manifest", str(args.pilot_manifest),
        "--output-root", str(parent_root),
        "--upstream-root", str(parent_root),
        "--worker-start-manifest-dir", str(parent_root),
        "--prelease-gate-dir", str(parent_root),
        "--queue-epoch-id", "ND_DIAGNOSTIC",
        "--queue-manifest-sha256", dummy,
        "--canonical-manifest-sha256", dummy,
        "--runtime-config-sha256", dummy,
        "--protocol-config", str(args.pilot_manifest),
        "--processor-path", str(args.model_path),
        "--supervisor-pid", "0",
        "--supervisor-config-sha256", dummy,
        "--relay-archive-commit", "74e5ad0",
        "--provenance-path", str(args.pilot_manifest),
        "--seed", str(args.seed),
    ]
    try:
        spec = importlib.util.spec_from_file_location("official_clean_worker", str(worker))
        module = importlib.util.module_from_spec(spec)
        sys.modules["official_clean_worker"] = module
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved_argv

    model, processor, device, unnorm_key = module.load_policy()
    adapter = module.OfficialOpenVLAActionAdapter(
        model, processor, device, unnorm_key, center_crop=True,
    )

    # ── Instrument adapter ──
    adapter = create_instrumented_adapter(adapter)

    # ── Capture backend info ──
    backend_info = capture_backend_info(device)
    print(f"  PyTorch: {backend_info['pytorch_version']}")
    print(f"  cuDNN deterministic: {backend_info['cudnn_deterministic']}")
    print(f"  cuDNN benchmark: {backend_info['cudnn_benchmark']}")
    print(f"  Flash-Attention: {backend_info.get('flash_attn_version', '?')}")
    print(f"  SDPA available: {backend_info.get('sdpa_available', '?')}")
    for env_var in ["TORCH_CUDNN_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG",
                     "CUDA_LAUNCH_BLOCKING", "NVIDIA_TF32_OVERRIDE",
                     "FLASH_ATTENTION_DETERMINISTIC"]:
        val = backend_info.get(f"env_{env_var}", "NOT_CHECKED")
        if val != "NOT_SET":
            print(f"  {env_var}={val}")

    # ── Load LIBERO tasks ──
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    benchmark_dict = benchmark.get_benchmark_dict()

    # ── Run episodes ──
    staging = out_root.parent / f".{out_root.name}.nd_staging.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)
    episodes_dir = staging / "episodes"
    episodes_dir.mkdir()
    published = False

    try:
        for ident in selected:
            suite = ident["suite"]
            task_idx = ident["task_id"]
            state_id = ident["state_id"]
            ep_id = ident["episode_id"]
            seed = ident["collection_seed"]
            declared_sha = ident["initial_state_sha256"]

            print(f"\n  {ep_id}...", end=" ", flush=True)
            t0 = time.perf_counter()

            # Load init state from LIBERO (same as R5-F collector)
            suite_obj = benchmark_dict[suite]()
            task = suite_obj.get_task(task_idx)
            states = suite_obj.get_task_init_states(task_idx)
            if state_id >= len(states):
                raise CollectionHold(f"state_id {state_id} >= {len(states)}")
            canonical_state = copy.deepcopy(states[state_id])

            # Verify initial-state SHA
            init_state_sha = sha256_bytes(pickle.dumps(canonical_state, protocol=4))
            if init_state_sha != declared_sha:
                raise CollectionHold(
                    f"initial_state_sha mismatch: computed={init_state_sha[:16]} "
                    f"declared={declared_sha[:16]}")

            episode_data = collect_episode_nd(
                suite, task_idx, state_id, seed,
                canonical_state,
                args.registry_root, task, adapter, module,
            )

            ep_dir = episodes_dir / ep_id.replace("/", "_")
            ep_dir.mkdir()
            (ep_dir / "episode.json").write_text(
                json.dumps(episode_data, indent=2, sort_keys=True), encoding="utf-8")

            elapsed = time.perf_counter() - t0
            nd_steps = len(episode_data.get("nd_diagnostics", []))
            fallbacks = sum(1 for d in episode_data.get("nd_diagnostics", [])
                           if d.get("fallback_detected"))
            n_default = sum(1 for d in episode_data.get("nd_diagnostics", [])
                           if d.get("matches_stay_still_default"))
            print(f"steps={episode_data['step_count']} nd_records={nd_steps} "
                  f"fallbacks={fallbacks} stay_still_default={n_default} "
                  f"elapsed={elapsed:.0f}s OK")

        # ── Write backend info ──
        (staging / "ND_BACKEND_INFO.json").write_text(
            json.dumps(backend_info, indent=2, sort_keys=True), encoding="utf-8")

        # ── Write manifest ──
        manifest = {
            "gate": "R5F_ND_DIAGNOSTIC",
            "purpose": "ROOT_CAUSE_CONFIRMATION",
            "run_label": args.label,
            "physical_gpu": args.physical_gpu,
            "logical_gpu": args.gpu,
            "n_tasks": n_selected,
            "task_filter": task_filter,
            "backend_info_summary": {
                "pytorch": backend_info["pytorch_version"],
                "cudnn_deterministic": backend_info["cudnn_deterministic"],
                "flash_attn": backend_info.get("flash_attn_version", "?"),
            },
        }
        (staging / "ND_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        seal_info = seal_root(staging)
        staging.rename(out_root)
        published = True

        print(f"\nDiagnostic sealed: {out_root}")
        print(f"  SHA256SUMS: {seal_info['sha256sums_sha256']}")
        print(f"  Tasks: {n_selected}")
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
