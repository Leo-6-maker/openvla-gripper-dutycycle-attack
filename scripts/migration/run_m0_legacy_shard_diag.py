#!/usr/bin/env python3
"""M0 Step 6: Legacy device-map diagnosis for s2 CLEAN.

Simulates old 2080Ti sharding on A800:
  device_map="auto", max_memory ~10000MiB per GPU, BF16, Eager.

Strategy: Do NOT use CUDA_VISIBLE_DEVICES (breaks EGL). Instead,
use max_memory dict with PHYSICAL GPU indices to constrain model to free GPUs,
keeping the original render GPU (4) dedicated for MuJoCo EGL rendering.

Model GPUs: 3,5,6 (all free)
Render GPU: 4 (known working, free)
"""
import os, sys, subprocess, json, hashlib, time, csv

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BRIDGE = os.path.join(REPO, "scripts", "stageb", "run_v2_vis_sc5_mlp_bridge.py")
CKPT = os.path.join(REPO, "artifacts", "detector", "sc5_mlp_s2.pt")
MODEL = os.environ.get("OPENVLA_MODEL_PATH", os.path.join(REPO, "models", "openvla-7b-finetuned-libero-object"))
PYTHON = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3"

RENDER_GPU = 4          # dedicated EGL rendering GPU (known working)
MODEL_GPUS = [3, 5, 6]  # GPUs for model sharding (all free)

OUT_DIR = os.path.join(REPO, "evidence", "object_checkpoint_migration", "m0_legacy_shard")
os.makedirs(OUT_DIR, exist_ok=True)

# Read original bridge and create patched version
with open(BRIDGE) as f:
    bridge_code = f.read()

patched_code = bridge_code.replace(
    'device_map = "cuda:0" if not IS_ATTACK else "auto"',
    'device_map = "auto"  # LEGACY SHARDING TEST'
)
# Inject max_memory with physical GPU indices
patched_code = patched_code.replace(
    'model = PrismaticVLModel.from_pretrained(MODEL_PATH, torch_dtype=model_dtype, device_map=device_map)',
    'max_memory = {3: "10000MiB", 5: "10000MiB", 6: "10000MiB"}; '
    'print(f"[legacy shard] max_memory={max_memory}"); '
    'model = PrismaticVLModel.from_pretrained(MODEL_PATH, torch_dtype=model_dtype, device_map=device_map, max_memory=max_memory)'
)

PATCHED_BRIDGE = os.path.join(os.path.dirname(BRIDGE), "_legacy_shard_bridge.py")
with open(PATCHED_BRIDGE, "w") as f:
    f.write(patched_code)
print(f"Patched bridge: {PATCHED_BRIDGE}")

# Run legacy-sharded s2 CLEAN
label = "legacy_shard_s2_clean"
cell_dir = os.path.join(OUT_DIR, label)
os.makedirs(cell_dir, exist_ok=True)

env = os.environ.copy()
# Do NOT set CUDA_VISIBLE_DEVICES — we use max_memory to constrain model GPUs
env["MUJOCO_GL"] = "osmesa"  # software rendering avoids EGL multi-GPU context conflicts
env["HF_HUB_OFFLINE"] = "1"
env["TRANSFORMERS_OFFLINE"] = "1"
env["HOME"] = "/mnt/sdc/dty_user/openvla_attack/sandbox_home"
env["TMPDIR"] = "/mnt/sdc/dty_user/openvla_attack/tmp"
env["OPENVLA_ATTN_IMPLEMENTATION"] = "eager"
env["OPENVLA_MODEL_PATH"] = MODEL

cmd = [
    PYTHON, PATCHED_BRIDGE,
    "--condition", "CLEAN",
    "--state_id", "2",
    "--anchor", "96",
    "--seed_id", "42",
    "--output_dir", cell_dir,
    "--render_gpu", str(RENDER_GPU),
    "--mlp_path", CKPT,
]

print(f"\n{'='*60}")
print(f"Step 6: Legacy Shard s2 CLEAN")
print(f"  Model GPUs (max_memory): {MODEL_GPUS}")
print(f"  Render GPU: {RENDER_GPU}")
print(f"  device_map=auto, max_memory=10000MiB per GPU")
print(f"  cmd: {' '.join(cmd)}")

t0 = time.time()
result = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
dt = time.time() - t0

# Save logs
with open(os.path.join(cell_dir, "stdout.log"), "w") as f:
    f.write(result.stdout)
with open(os.path.join(cell_dir, "stderr.log"), "w") as f:
    f.write(result.stderr)

# Parse result
summary_path = os.path.join(cell_dir, "episode_summary.json")
summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
telemetry_path = os.path.join(cell_dir, "step_telemetry.csv")
telemetry_exists = os.path.exists(telemetry_path)
telemetry_sha = hashlib.sha256(open(telemetry_path, "rb").read()).hexdigest() if telemetry_exists else "MISSING"

legacy_result = {
    "exit_code": result.returncode,
    "duration_s": round(dt, 1),
    "telemetry_sha": telemetry_sha,
    "success": summary.get("task_success", False),
    "emit_step": summary.get("mlp_emit_step", -1),
    "steps": summary.get("n_steps", -1),
}

print(f"  exit={result.returncode} time={dt:.0f}s succ={legacy_result['success']} emit={legacy_result['emit_step']} steps={legacy_result['steps']}")
print(f"  telemetry_exists={telemetry_exists} sha={telemetry_sha[:16] if telemetry_exists else 'MISSING'}...")
if result.stderr:
    err_lines = result.stderr.strip().split('\n')
    print(f"  stderr (last 5 of {len(err_lines)}):")
    for line in err_lines[-5:]:
        print(f"    {line[:120]}")

if not telemetry_exists:
    print("\n  Telemetry not produced — bridge crashed before episode start.")
    print(f"  stdout: {result.stdout[-300:]}")
    manifest = {
        "gate": "M0_S2_LEGACY_DEVICE_MAP_DIAGNOSIS",
        "status": "FAILED",
        "error": "Bridge crashed — see stderr.log",
        "classification": "LEGACY_SHARD_NOT_EXECUTABLE",
    }
else:
    # Compare with single-GPU baseline
    baseline_path = os.path.join(REPO, "evidence", "object_checkpoint_migration", "m0_s2_repeatability", "repeat1_s2_clean", "step_telemetry.csv")
    baseline_sha = hashlib.sha256(open(baseline_path, "rb").read()).hexdigest()
    print(f"\n  Baseline (single-GPU cuda:0) SHA: {baseline_sha[:16]}...")
    print(f"  Legacy (multi-GPU auto) SHA:     {telemetry_sha[:16]}...")

    t_base = list(csv.DictReader(open(baseline_path)))
    t_legacy = list(csv.DictReader(open(telemetry_path)))

    behavior_keys = [
        'raw_gripper','env_gripper','eef_x','eef_y','eef_z','obj_x','obj_y','obj_z',
        'eef_obj_dist','corridor_p','release_p','pred_phase','detector_state',
        'f_eef_vx','f_eef_vy','f_eef_vz','f_action_dx','f_action_dy','f_action_dz',
        'f_action_gripper','f_gripper_command','f_eef_speed',
    ]

    print(f"  Row counts: baseline={len(t_base)} legacy={len(t_legacy)}")

    diverged_keys = []
    first_div = None
    for i in range(min(len(t_base), len(t_legacy))):
        for k in behavior_keys:
            v_base = t_base[i].get(k, '')
            v_legacy = t_legacy[i].get(k, '')
            if v_base != v_legacy:
                if k not in diverged_keys:
                    diverged_keys.append(k)
                if first_div is None:
                    first_div = {"step": i, "key": k, "baseline": v_base[:30], "legacy": v_legacy[:30]}

    if summary.get("task_success", False):
        classification = "M0_S2_DEVICE_MAP_NUMERICAL_SENSITIVITY — legacy-sharded SUCCEEDS where single-GPU FAILS"
    elif diverged_keys:
        classification = f"M0_S2_DEVICE_MAP_BEHAVIORAL_DIVERGENCE — {len(diverged_keys)} keys diverged; first: step={first_div['step']} {first_div['key']}"
    else:
        classification = "M0_S2_DEVICE_MAP_NOT_ROOT_CAUSE — identical behavior, device_map not the factor"

    manifest = {
        "gate": "M0_S2_LEGACY_DEVICE_MAP_DIAGNOSIS",
        "status": "COMPLETED",
        "condition": "s2 CLEAN",
        "legacy_config": {
            "render_gpu": RENDER_GPU,
            "model_gpus": MODEL_GPUS,
            "device_map": "auto",
            "max_memory_per_gpu": "10000MiB",
            "dtype": "BF16",
            "attention": "eager",
        },
        "baseline_config": {
            "cuda_visible_devices": "4",
            "device_map": "cuda:0",
            "dtype": "BF16",
            "attention": "eager",
        },
        "legacy_result": legacy_result,
        "baseline_telemetry_sha": baseline_sha,
        "baseline_n_steps": len(t_base),
        "legacy_n_steps": len(t_legacy),
        "diverged_behavioral_keys": diverged_keys,
        "first_divergence": first_div,
        "classification": classification,
    }

    print(f"\nClassification: {classification}")

json.dump(manifest, open(os.path.join(OUT_DIR, "legacy_shard_manifest.json"), "w"), indent=2)
print(f"\nSaved: {OUT_DIR}")
