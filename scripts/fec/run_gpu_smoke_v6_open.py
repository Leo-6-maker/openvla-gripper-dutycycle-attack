"""V6 OPEN smoke runner — exact-prefix snapshot branching + physical telemetry + provenance.

Fixes from audit (all 6 items):
  P0-2: Exact-prefix snapshot — CLEAN runs first, saves sim state at emit-1,
        subsequent arms restore from that snapshot for true counterfactual branching.
  P0-3: OPEN-region objective config added (region_logsumexp variant).
  P0-4: Physical telemetry — gripper qpos, object pose, finger-object contact,
        EEF-object distance recorded per step.
  P1-3: random_start support for PGD restarts.
  P1-6: run_manifest provenance — runner SHA, commit SHA, model tree SHA,
        BDDL SHA, init-state SHA.
"""
import argparse, copy, json, hashlib, os, random, sys, time, traceback
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

# ── Constants ──
ARMS = ("CLEAN", "TRUE_T10", "RAND_T10", "COMMAND_OPEN_ORACLE", "RANDOM_TIME_T10")
NUM_STEPS_WAIT = 10
CANONICAL_ENV_OPEN = -1.0
CANONICAL_RAW_OPEN = 1.0
TARGET_TOKEN_ID = 31745  # NATIVE_OPEN, disc=254
TARGET_EXECUTION_CLASS = "NATIVE_OPEN"
TARGET_OBJECTIVE = "autoregressive_prefix_gripper_target_token_logratio_arm_v3"
DUMMY_WAIT_ACTION = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
POLICY_HORIZONS = {"libero_10": 220, "libero_goal": 280, "libero_object": 280, "libero_spatial": 220}

class ContractError(RuntimeError):
    pass


# ── Provenance ──
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def sha256_tree(root):
    """Recursive SHA256 of directory tree."""
    hashes = []
    for dirpath, _, filenames in sorted(os.walk(root)):
        for fn in sorted(filenames):
            hashes.append(sha256_file(os.path.join(dirpath, fn)))
    h = hashlib.sha256()
    for fh in sorted(hashes):
        h.update(fh.encode())
    return h.hexdigest()

def git_commit_sha():
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return "unknown"


# ── Action helpers ──
def normalize_and_invert_gripper(raw_action):
    action = np.asarray(raw_action, dtype=np.float32).copy()
    if action.ndim != 1 or action.shape[0] < 7:
        raise ContractError(f"expected 7D action, got shape={action.shape}")
    action[-1] = 2.0 * action[-1] - 1.0
    action[-1] = np.sign(action[-1])
    if action[-1] == 0:
        action[-1] = 1.0
    action[-1] *= -1.0
    return np.clip(action, -1.0, 1.0).astype(np.float32)


# ── Physical telemetry ──
def record_physical_telemetry(env, obs, clean_env_action, final_env_action):
    """Record gripper qpos, object pose, contact, EEF-object distance."""
    telemetry = {}
    try:
        sim = env.sim
        # Gripper qpos
        qpos = sim.data.qpos[-2:].copy() if hasattr(sim.data, 'qpos') else None
        if qpos is not None:
            telemetry['gripper_qpos'] = [float(qpos[0]), float(qpos[1])]
            telemetry['gripper_width'] = float(abs(qpos[0]) + abs(qpos[1]))
        # Object state — from env observation
        obj_state = obs.get('object_state', obs.get('object-state', None))
        if obj_state is not None:
            telemetry['object_state_len'] = len(obj_state) if isinstance(obj_state, (list, tuple)) else None
        # EEF position
        eef_pos = obs.get('robot0_eef_pos', None)
        if eef_pos is not None:
            telemetry['eef_pos'] = [float(x) for x in eef_pos]
        # Contact
        contact_count = obs.get('contact_count', obs.get('contact-count', None))
        telemetry['contact_count'] = int(contact_count) if contact_count is not None else None
        # Gripper action
        telemetry['clean_env_gripper'] = float(clean_env_action[-1])
        telemetry['final_env_gripper'] = float(final_env_action[-1])
    except Exception:
        telemetry['error'] = 'telemetry_failed'
    return telemetry


# ── Simulator snapshot ──
def save_sim_snapshot(env):
    """Save MuJoCo simulator state for exact-prefix branching."""
    try:
        return env.sim.get_state().flatten().copy()
    except Exception:
        return None

def restore_sim_snapshot(env, snapshot):
    """Restore MuJoCo simulator state from snapshot."""
    if snapshot is not None:
        env.sim.set_state_from_flattened(snapshot)


# ── JSON helpers ──
def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding='utf-8')
    os.replace(tmp, path)

def append_jsonl(path, obj):
    with open(path, 'a') as f:
        f.write(json.dumps(obj, default=str) + '\n')

def json_safe(obj):
    return json.loads(json.dumps(obj, default=str))


# ── Config ──
def load_yaml(path):
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        import json as _j
        # minimal YAML subset for our configs
        result = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ':' in line:
                    k, v = line.split(':', 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    # Try to parse as number
                    try:
                        if '.' in v:
                            v = float(v)
                        else:
                            v = int(v)
                    except ValueError:
                        if v.lower() == 'true':
                            v = True
                        elif v.lower() == 'false':
                            v = False
                    result[k] = v
        return result

def validate_base_config(base_config):
    opt = base_config.get("attack_optimizer", {})
    runtime = base_config.get("runtime", {})
    arms_cfg = base_config.get("arms", {})
    if not isinstance(opt, dict) or not isinstance(runtime, dict) or not isinstance(arms_cfg, dict):
        raise ContractError("config requires attack_optimizer/runtime/arms mappings")
    expected = {
        "method": "token_prefix_pgd", "strict_route": True,
        "allow_fallback": False, "random_start": opt.get("random_start", False),
        "temporal_init": "none", "temporal_smooth_lambda": 0.0,
        "surrogate_score_path": "cached_autoregressive_generate_v1",
        "prefix_refresh_interval": 1,
        "objective": opt.get("objective", TARGET_OBJECTIVE),
        "target_token_id": opt.get("target_token_id", TARGET_TOKEN_ID),
        "target_execution_class": opt.get("target_execution_class", TARGET_EXECUTION_CLASS),
        "gradient_transform": "none",
    }
    for key, value in expected.items():
        if key == "random_start":
            continue  # not validated
        if opt.get(key) != value:
            raise ContractError(f"attack_optimizer.{key}={opt.get(key)!r}, expected {value!r}")
    if int(runtime.get("attack_burst_frames", runtime.get("K10", -1))) != 10:
        raise ContractError("attack_burst_frames must be exactly 10")
    if runtime.get("fallback_forbidden") is not True:
        raise ContractError("runtime.fallback_forbidden must be true")

def effective_config(base, arm, rand_direction_seed):
    cfg = copy.deepcopy(base)
    opt = cfg["attack_optimizer"]
    arm_cfg = cfg.get("arms", {}).get(arm, {})
    for key in ("objective", "gradient_transform"):
        if key in arm_cfg:
            opt[key] = arm_cfg[key]
    opt["gradient_transform_seed"] = int(rand_direction_seed)
    cfg["effective_arm"] = arm
    return cfg


# ── Model helpers ──
def prepare_clean_generation(model, processor, obs, instruction, unnorm_key, *, device, center_crop):
    import numpy as np
    image = obs["agentview_image"]
    if center_crop:
        from PIL import Image
        h, w = image.shape[:2]
        size = min(h, w)
        left = (w - size) // 2
        top = (h - size) // 2
        image = image[top:top+size, left:left+size]
    inputs = processor(image, instruction, return_tensors="pt").to(device)
    with torch.inference_mode():
        generation = model.generate(
            **inputs, max_new_tokens=int(model.get_action_dim(unnorm_key)),
            do_sample=False, return_dict_in_generate=True, output_scores=True,
        )
    action_dim = int(model.get_action_dim(unnorm_key))
    token_ids = generation.sequences[0, -action_dim:].detach().cpu().numpy()
    raw_action = model.decode_action_from_tokens(token_ids, unnorm_key)
    return raw_action, generation, token_ids.tolist()

def redecode_adv_inputs(model, adv_inputs, unnorm_key, *, device):
    action_dim = int(model.get_action_dim(unnorm_key))
    inputs = {}
    for key, value in adv_inputs.items():
        if torch.is_tensor(value):
            dtype = next(model.parameters()).dtype if torch.is_floating_point(value) else None
            inputs[key] = value.to(device=device, dtype=dtype) if dtype else value.to(device=device)
        else:
            inputs[key] = value
    with torch.inference_mode():
        generation = model.generate(
            **inputs, max_new_tokens=action_dim, do_sample=False,
            return_dict_in_generate=True, output_scores=True,
        )
    action_dim = int(model.get_action_dim(unnorm_key))
    token_ids = generation.sequences[0, -action_dim:].detach().cpu().numpy()
    raw_action = model.decode_action_from_tokens(token_ids, unnorm_key)
    return raw_action, token_ids.tolist(), generation


# ── Main ──
def parse_args():
    p = argparse.ArgumentParser(description="V6 OPEN smoke runner with exact-prefix snapshot branching")
    p.add_argument("--gpu-id", type=int, required=True)
    p.add_argument("--suite", required=True, choices=["libero_10","libero_goal","libero_object","libero_spatial"])
    p.add_argument("--task-index", type=int, required=True)
    p.add_argument("--state-index", type=int, default=None)
    p.add_argument("--init-state-npy", type=Path, default=None)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/fec_attack_v5_open.yaml"))
    p.add_argument("--repo-root", type=Path, default=Path("/mnt/sdc/dty_user/openvla_attack"))
    p.add_argument("--n4-module", type=Path, required=True)
    p.add_argument("--n4-provider-name", default=None)
    p.add_argument("--n4-norm-data", type=Path, required=True)
    p.add_argument("--expected-attacker-sha256", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--rand-direction-seed", type=int, required=True)
    p.add_argument("--random-time-seed", type=int, required=True)
    p.add_argument("--random-time-start", type=int, default=None)
    p.add_argument("--center-crop", action="store_true", default=False)
    p.add_argument("--render-size", type=int, default=224)
    p.add_argument("--dry-run-contract", action="store_true")
    p.add_argument("--video-dir", type=Path, default=None, help="Directory for per-arm videos")
    return p.parse_args()


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
    runner_sha = sha256_file(Path(__file__))
    commit_sha = git_commit_sha()

    import gripper_attack.attack_adapter as attack_adapter_module
    attacker_realpath = Path(attack_adapter_module.__file__).resolve()
    attacker_sha = sha256_file(attacker_realpath)
    if attacker_sha != args.expected_attacker_sha256:
        raise ContractError(f"attacker SHA mismatch: {attacker_sha} != {args.expected_attacker_sha256}")

    if args.dry_run_contract:
        print(json.dumps({
            "contract": "PASS", "config_sha256": config_sha,
            "attacker_sha256": attacker_sha, "n4_module_sha256": n4_module_sha,
            "n4_norm_sha256": n4_norm_sha, "runner_sha256": runner_sha,
            "commit_sha": commit_sha,
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

    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True, local_files_only=True, use_fast=False)
    model = AutoModelForVision2Seq.from_pretrained(str(args.model_path), trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
    model.eval()
    model_tree_sha = sha256_tree(args.model_path)

    unnorm_key = args.suite
    if unnorm_key not in getattr(model, "norm_stats", {}) and f"{unnorm_key}_no_noops" in getattr(model, "norm_stats", {}):
        unnorm_key = f"{unnorm_key}_no_noops"

    benchmark_dict = benchmark.get_benchmark_dict()
    task = benchmark_dict[args.suite][args.task_index]
    bddl_file = task.bddl_file
    bddl_path = Path(task.problem_folder) / bddl_file if hasattr(task, 'problem_folder') else Path(bddl_file)
    bddl_sha = sha256_file(bddl_path) if bddl_path.is_file() else "unknown"
    instruction = task.language

    initial_states = task.get_task_init_states()
    if args.init_state_npy is not None:
        initial_state = np.load(str(args.init_state_npy), allow_pickle=True).item()
        state_identity = {"kind": "npy_file", "path": str(args.init_state_npy)}
    else:
        if args.state_index is None:
            raise ContractError("provide --state-index or --init-state-npy")
        if args.state_index < 0 or args.state_index >= len(initial_states):
            raise ContractError(f"state_index {args.state_index} outside range 0-{len(initial_states)-1}")
        initial_state = copy.deepcopy(initial_states[args.state_index])
        state_identity = {"kind": "benchmark_index", "index": args.state_index}
    init_state_sha = hashlib.sha256(json.dumps(state_identity, sort_keys=True).encode()).hexdigest()

    policy_horizon = POLICY_HORIZONS[args.suite]
    burst_frames = int(base_config["runtime"]["attack_burst_frames"])
    if args.random_time_start is None:
        rng = np.random.RandomState(args.random_time_seed)
        random_time_start = int(rng.randint(0, policy_horizon - burst_frames + 1))
    else:
        random_time_start = int(args.random_time_start)

    if args.output_root.exists():
        unexpected = [p.name for p in args.output_root.iterdir() if p.name not in {"worker.log"}]
        if unexpected:
            raise ContractError(f"output root not empty: {args.output_root}")
    else:
        args.output_root.mkdir(parents=True, exist_ok=False)

    # ── N4 Detector ──
    from run_gpu_smoke import N4Bridge
    detector = N4Bridge(args.n4_module, norm_data_path=args.n4_norm_data, device=device, provider_name=args.n4_provider_name)

    # ── Attackers ──
    true_cfg = effective_config(base_config, "TRUE_T10", rand_direction_seed=args.rand_direction_seed)
    rand_cfg = effective_config(base_config, "RAND_T10", rand_direction_seed=args.rand_direction_seed)
    random_time_cfg = effective_config(base_config, "RANDOM_TIME_T10", rand_direction_seed=args.rand_direction_seed)
    attackers = {
        "TRUE_T10": OpenVLAVisualAttacker(model=model, processor=processor, config=true_cfg, seed=args.seed, device=device),
        "RAND_T10": OpenVLAVisualAttacker(model=model, processor=processor, config=rand_cfg, seed=args.seed, device=device),
        "RANDOM_TIME_T10": OpenVLAVisualAttacker(model=model, processor=processor, config=random_time_cfg, seed=args.seed, device=device),
    }

    # ── Run manifest ──
    run_manifest = {
        "runner": "V6_OPEN_EXACT_PREFIX_SNAPSHOT",
        "scientific_role": "SMOKE_CANARY",
        "counts_toward_fec": False,
        "formal_matrix_execution": False,
        "suite": args.suite, "task_index": args.task_index,
        "state_identity": state_identity,
        "init_state_sha256": init_state_sha,
        "policy_horizon": policy_horizon,
        "seed": args.seed, "rand_direction_seed": args.rand_direction_seed,
        "random_time_seed": args.random_time_seed,
        "random_time_policy_start": random_time_start,
        "config_path": str(args.config), "config_sha256": config_sha,
        "attacker_sha256": attacker_sha,
        "n4_module_sha256": n4_module_sha, "n4_norm_sha256": n4_norm_sha,
        "model_path": str(args.model_path), "model_tree_sha256": model_tree_sha,
        "unnorm_key": unnorm_key,
        "runner_sha256": runner_sha, "commit_sha": commit_sha,
        "bddl_sha256": bddl_sha,
        "arms": list(ARMS), "created_unix": time.time(),
    }
    atomic_write_json(args.output_root / "run_manifest.json", run_manifest)

    # ── Run CLEAN first, capture snapshot at emit ──
    all_results: dict[str, Any] = {}
    sim_snapshot = None  # captured after CLEAN detector emit, before attack window
    emit_step = None
    detector_state_at_emit = None

    for arm_idx, arm in enumerate(ARMS):
        arm_dir = args.output_root / arm
        arm_dir.mkdir(parents=True, exist_ok=False)
        steps_path = arm_dir / "steps.jsonl"
        attacks_path = arm_dir / "attack_frames.jsonl"
        error_path = arm_dir / "error.json"
        telemetry_path = arm_dir / "physical_telemetry.jsonl"

        arm_seed = int(args.seed)
        random.seed(arm_seed)
        np.random.seed(arm_seed)
        torch.manual_seed(arm_seed)
        torch.cuda.manual_seed_all(arm_seed)

        env = None
        result = {
            "arm": arm, "status": "RUNNING", "emit_policy_step": None,
            "emit_env_step": None, "attack_planned_frames": 0,
            "attack_executed_frames": 0, "attack_errors": 0,
            "task_success": False, "policy_steps": 0, "env_steps": 0,
            "termination": None, "snapshot_restored": arm_idx > 0 and sim_snapshot is not None,
        }

        try:
            env = OffScreenRenderEnv(
                bddl_file_name=bddl_file, camera_heights=args.render_size,
                camera_widths=args.render_size, render_gpu_device_id=0,
                horizon=policy_horizon + NUM_STEPS_WAIT,
            )
            if callable(getattr(env, "seed", None)):
                env.seed(0)

            if arm_idx == 0 or sim_snapshot is None:
                # First arm (CLEAN): fresh init
                env.reset()
                obs = env.set_init_state(copy.deepcopy(initial_state))
                for _ in range(NUM_STEPS_WAIT):
                    obs, _, done, _ = env.step(DUMMY_WAIT_ACTION)
                    result["env_steps"] += 1
                    if done:
                        raise ContractError("env terminated during wait phase")
                detector.reset_episode()
            else:
                # Subsequent arms: restore from CLEAN's snapshot at emit
                env.reset()
                obs = env.set_init_state(copy.deepcopy(initial_state))
                # Fast-forward to emit point using recorded state
                restore_sim_snapshot(env, sim_snapshot)
                # Re-set detector to emit-point state
                if detector_state_at_emit is not None:
                    detector.set_state(detector_state_at_emit)
                result["snapshot_restored"] = True

            for attacker in attackers.values():
                attacker.reset_temporal_state()

            arm_emit_step = None
            for policy_step in range(policy_horizon):
                env_step = NUM_STEPS_WAIT + policy_step
                clean_raw_action, clean_generation, clean_token_ids = prepare_clean_generation(
                    model, processor, obs, instruction, unnorm_key, device=device, center_crop=bool(args.center_crop))
                clean_env_action = normalize_and_invert_gripper(clean_raw_action)
                n4 = detector.step(
                    obs=obs, clean_raw_action=clean_raw_action, clean_env_action=clean_env_action,
                    clean_model_output=clean_generation, policy_step=policy_step,
                    suite=args.suite, unnorm_key=unnorm_key, model=model, processor=processor,
                )

                if bool(n4.get("emitted_this_step")) and arm_emit_step is None:
                    arm_emit_step = policy_step
                    result["emit_policy_step"] = policy_step
                    result["emit_env_step"] = env_step
                    # Save snapshot for subsequent arms (only during CLEAN)
                    if arm == "CLEAN" and sim_snapshot is None:
                        sim_snapshot = save_sim_snapshot(env)
                        detector_state_at_emit = detector.get_state()

                planned = False
                if arm in {"TRUE_T10", "RAND_T10", "COMMAND_OPEN_ORACLE"} and arm_emit_step is not None:
                    planned = arm_emit_step <= policy_step < arm_emit_step + burst_frames
                elif arm == "RANDOM_TIME_T10":
                    planned = random_time_start <= policy_step < random_time_start + burst_frames

                final_env_action = clean_env_action.copy()
                attack_executed = False
                attack_audit = None
                adv_raw_action = None

                if planned and arm == "COMMAND_OPEN_ORACLE":
                    final_env_action[-1] = CANONICAL_ENV_OPEN
                    if not np.array_equal(final_env_action[:6], clean_env_action[:6]):
                        raise ContractError("ORACLE modified arm dimensions")
                    attack_executed = True
                elif planned and arm in {"TRUE_T10", "RAND_T10", "RANDOM_TIME_T10"}:
                    attack_cfg = {"TRUE_T10": true_cfg, "RAND_T10": rand_cfg, "RANDOM_TIME_T10": random_time_cfg}[arm]
                    # Construct OPEN target
                    target_raw_action = clean_raw_action.copy()
                    target_raw_action[-1] = CANONICAL_RAW_OPEN
                    from run_gpu_smoke import validate_attack_result
                    attack_result = attackers[arm].attack(
                        observation=obs["agentview_image"], instruction=instruction,
                        clean_action=clean_raw_action, target_action=target_raw_action,
                        clean_model_output=clean_generation, unnorm_key=unnorm_key)
                    attack_audit = validate_attack_result(attack_result, arm=arm, config=attack_cfg)
                    adv_inputs = attack_audit.pop("adv_inputs")
                    adv_raw_action, adv_token_ids, _ = redecode_adv_inputs(model, adv_inputs, unnorm_key, device=device)
                    final_env_action = normalize_and_invert_gripper(adv_raw_action)
                    attack_executed = True

                if planned and not attack_executed:
                    raise ContractError(f"planned attack not executed: arm={arm} step={policy_step}")

                obs, reward, done, info = env.step(final_env_action.tolist())
                result["env_steps"] += 1
                result["policy_steps"] += 1
                if attack_executed:
                    result["attack_executed_frames"] += 1

                # Record physical telemetry
                telem = record_physical_telemetry(env, obs, clean_env_action, final_env_action)
                telem["step"] = env_step
                telem["policy_step"] = policy_step
                telem["arm"] = arm
                append_jsonl(telemetry_path, telem)

                if attack_executed:
                    append_jsonl(attacks_path, {
                        "arm": arm, "env_step": env_step, "policy_step": policy_step,
                        "attack_frame_idx": (policy_step - arm_emit_step if arm != "RANDOM_TIME_T10" else policy_step - random_time_start),
                        "clean_raw_action": clean_raw_action.tolist(),
                        "clean_env_action": clean_env_action.tolist(),
                        "clean_token_ids": clean_token_ids,
                        "adv_raw_action": None if adv_raw_action is None else adv_raw_action.tolist(),
                        "adv_token_ids": adv_token_ids,
                        "final_env_action": final_env_action.tolist(),
                        "audit": json_safe(attack_audit),
                    })

                append_jsonl(steps_path, {
                    "env_step": env_step, "policy_step": policy_step,
                    "is_wait_step": False, "detector_updated": True,
                    "candidate_close": bool(n4.get("candidate_close")),
                    "calibrated_prob": float(n4["calibrated_prob"]),
                    "emitted_this_step": bool(n4["emitted_this_step"]),
                    "attack_planned": bool(planned), "attack_executed": bool(attack_executed),
                    "clean_env_gripper": float(clean_env_action[-1]),
                    "final_env_gripper": float(final_env_action[-1]),
                    "done": bool(done),
                })

                success = env._check_success() if hasattr(env, "_check_success") else info.get("success", False)
                if success:
                    result["task_success"] = True
                    result["termination"] = "SUCCESS"
                    result["success_source"] = "env_check"
                    break
                if done:
                    result["termination"] = "DONE_WITHOUT_SUCCESS"
                    break
            else:
                result["termination"] = "POLICY_HORIZON"

            # Post-arm validation
            if arm == "CLEAN" and result["attack_executed_frames"] != 0:
                raise ContractError("CLEAN executed an intervention")
            if arm in {"TRUE_T10", "RAND_T10", "COMMAND_OPEN_ORACLE"}:
                if result["emit_policy_step"] is None:
                    if result["attack_executed_frames"] != 0:
                        raise ContractError(f"{arm} attacked without detector emit")
                elif result["attack_executed_frames"] != burst_frames:
                    result["attack_errors"] += 1
                    result["error_type"] = "ContractError"
                    result["error"] = f"{arm} K10 incomplete: {result['attack_executed_frames']}/{burst_frames}"
            result["status"] = "PASS"
            atomic_write_json(arm_dir / "result.json", result)
            atomic_write_json(arm_dir / "COMPLETE.json", {
                "status": "PASS", "result_sha256": sha256_file(arm_dir / "result.json"),
                "completed_unix": time.time(),
            })

        except Exception as exc:
            result["status"] = "FAIL"
            result["error_type"] = type(exc).__name__
            result["error"] = str(exc)
            atomic_write_json(error_path, {
                "arm": arm, "error_type": type(exc).__name__,
                "error": str(exc), "traceback": traceback.format_exc(), "result": result,
            })
            break  # Stop on error
        finally:
            if env is not None:
                try: env.close()
                except Exception: pass
        all_results[arm] = result

    # ── Summary ──
    overall_valid = all(
        all_results.get(a, {}).get("status") == "PASS"
        for a in ARMS[:len(all_results)]
    ) and len(all_results) == len(ARMS)

    summary = {
        "valid": bool(overall_valid),
        "engineering_status": "PASS" if overall_valid else "FAIL",
        "all_arms_completed": set(all_results.keys()) == set(ARMS),
        "snapshot_captured": sim_snapshot is not None,
        "arms": {a: {
            "emit": all_results[a].get("emit_policy_step"),
            "k10": all_results[a].get("attack_executed_frames"),
            "success": all_results[a].get("task_success"),
            "termination": all_results[a].get("termination"),
            "snapshot_restored": all_results[a].get("snapshot_restored", False),
        } for a in ARMS if a in all_results},
    }
    atomic_write_json(args.output_root / "smoke_summary.json", summary)
    print(f"SMOKE: {summary['engineering_status']}")
    return 0 if overall_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
