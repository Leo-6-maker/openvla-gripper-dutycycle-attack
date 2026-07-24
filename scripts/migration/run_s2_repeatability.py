#!/usr/bin/env python3
"""M0 Step 4: Same-GPU repeatability diagnosis for s2 CLEAN.
Runs: s0 CLEAN × 1 (sentinel) + s2 CLEAN × 3 (repeats).
Compares telemetry SHAs to determine determinism.
"""
import os, sys, subprocess, json, hashlib, time, csv

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BRIDGE = os.path.join(REPO, "scripts", "stageb", "run_v2_vis_sc5_mlp_bridge.py")
CKPT = os.path.join(REPO, "artifacts", "detector", "sc5_mlp_s2.pt")
OUT_BASE = os.path.join(REPO, "evidence", "object_checkpoint_migration", "m0_s2_repeatability")
PYTHON = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3"
GPU = 4

os.makedirs(OUT_BASE, exist_ok=True)

ENV = os.environ.copy()
ENV["CUDA_VISIBLE_DEVICES"] = str(GPU)
ENV["MUJOCO_GL"] = "egl"
ENV["HF_HUB_OFFLINE"] = "1"
ENV["TRANSFORMERS_OFFLINE"] = "1"
ENV["HOME"] = "/mnt/sdc/dty_user/openvla_attack/sandbox_home"
ENV["TMPDIR"] = "/mnt/sdc/dty_user/openvla_attack/tmp"
ENV["OPENVLA_ATTN_IMPLEMENTATION"] = "eager"

RUNS = [
    ("sentinel_s0_clean", 0, 94),
    ("repeat1_s2_clean", 2, 96),
    ("repeat2_s2_clean", 2, 96),
    ("repeat3_s2_clean", 2, 96),
]

results = {}
for label, state_id, anchor in RUNS:
    out_dir = os.path.join(OUT_BASE, label)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        PYTHON, BRIDGE,
        "--condition", "CLEAN",
        "--state_id", str(state_id),
        "--anchor", str(anchor),
        "--seed_id", "42",
        "--output_dir", out_dir,
        "--render_gpu", str(GPU),
        "--mlp_path", CKPT,
    ]

    print(f"\n{'='*60}")
    print(f"Run: {label} (state={state_id} anchor={anchor})")
    print(f"  cmd: {' '.join(cmd)}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO, env=ENV, capture_output=True, text=True)
    dt = time.time() - t0

    # Save logs
    with open(os.path.join(out_dir, "stdout.log"), "w") as f:
        f.write(result.stdout)
    with open(os.path.join(out_dir, "stderr.log"), "w") as f:
        f.write(result.stderr)

    # Hash telemetry
    telemetry_path = os.path.join(out_dir, "step_telemetry.csv")
    telemetry_sha = hashlib.sha256(open(telemetry_path, "rb").read()).hexdigest() if os.path.exists(telemetry_path) else "MISSING"

    # Parse summary
    summary_path = os.path.join(out_dir, "episode_summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

    results[label] = {
        "exit_code": result.returncode,
        "duration_s": round(dt, 1),
        "telemetry_sha": telemetry_sha,
        "success": summary.get("task_success", False),
        "emit_step": summary.get("mlp_emit_step", -1),
        "steps": summary.get("n_steps", -1),
    }
    print(f"  exit={result.returncode} time={dt:.0f}s succ={results[label]['success']} emit={results[label]['emit_step']} steps={results[label]['steps']}")
    print(f"  telemetry_sha={telemetry_sha[:16]}...")

# Compare s2 repeats
s2_shas = [results[f"repeat{i}_s2_clean"]["telemetry_sha"] for i in range(1, 4)]
all_same = len(set(s2_shas)) == 1

print(f"\n{'='*60}")
print(f"Repeatability Analysis")
print(f"  s0 sentinel SHA: {results['sentinel_s0_clean']['telemetry_sha'][:16]}...")
for i in range(1, 4):
    print(f"  s2 repeat{i} SHA:  {results[f'repeat{i}_s2_clean']['telemetry_sha'][:16]}...")

# If not all same, find first divergence
first_divergence = None
if not all_same and len(set(s2_shas)) > 1:
    print("\n=== First Divergence Analysis ===")
    # Load all 3 telemetry CSVs
    telemetries = []
    for i in range(1, 4):
        path = os.path.join(OUT_BASE, f"repeat{i}_s2_clean", "step_telemetry.csv")
        with open(path) as f:
            telemetries.append(list(csv.DictReader(f)))

    # Find first diverging step across any pair
    n_steps = min(len(t) for t in telemetries)
    for step_idx in range(n_steps):
        row0 = telemetries[0][step_idx]
        row1 = telemetries[1][step_idx]
        row2 = telemetries[2][step_idx]

        # Check key fields for divergence
        keys_to_check = [
            'eef_x', 'eef_y', 'eef_z', 'obj_x', 'obj_y', 'obj_z',
            'raw_gripper', 'env_gripper',
            'f_eef_vx', 'f_eef_vy', 'f_eef_vz',
            'f_action_dx', 'f_action_dy', 'f_action_dz', 'f_action_gripper',
            'corridor_p', 'release_p', 'pred_phase', 'detector_state'
        ]
        diverged = False
        for key in keys_to_check:
            v0 = row0.get(key, '')
            v1 = row1.get(key, '')
            v2 = row2.get(key, '')
            if v0 != v1 or v0 != v2 or v1 != v2:
                if not first_divergence:
                    first_divergence = {
                        "step": int(row0['step']),
                        "key": key,
                        "values": [v0, v1, v2],
                    }
                diverged = True
        if diverged and not first_divergence:
            break

    if first_divergence:
        print(f"  First divergence at step {first_divergence['step']}: {first_divergence['key']}")
        print(f"    Values: {first_divergence['values']}")
    else:
        print("  No single-field divergence found within step range (may differ in row count or other columns)")

# Write manifest
manifest = {
    "gate": "M0_S2_REPEATABILITY",
    "gpu": GPU,
    "gpu_uuid": "GPU-e85ed586-ba64-a9e3-8fa9-07f16f84dcda",
    "deterministic": all_same,
    "classification": "S2_A800_SINGLE_GPU_DETERMINISTIC" if all_same else "S2_RUNTIME_NONDETERMINISTIC",
    "results": results,
    "s2_telemetry_shas": s2_shas,
    "first_divergence": first_divergence,
}
json.dump(manifest, open(os.path.join(OUT_BASE, "repeatability_manifest.json"), "w"), indent=2)

print(f"\nClassification: {manifest['classification']}")
print(f"All s2 telemetry SHAs match: {all_same}")
sys.exit(0 if all_same else 1)
