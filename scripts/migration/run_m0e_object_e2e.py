#!/usr/bin/env python3
"""M0-E: Run 6-cell Object E2E reproduction on A800 with Seed 2 detector.

Sequential execution on single GPU. Uses original bridge code adapted for A800.
"""
import os, sys, subprocess, json, hashlib, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL = os.path.join(REPO, "models", "openvla-7b-finetuned-libero-object")
CKPT = os.path.join(REPO, "artifacts", "detector", "sc5_mlp_s2.pt")
BRIDGE = os.path.join(REPO, "scripts", "stageb", "run_v2_vis_sc5_mlp_bridge.py")
OUT_BASE = os.path.join(REPO, "evidence", "object_checkpoint_migration", "m0_e2e")
PYTHON = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3"
GPU = 4

# 6 cells
CELLS = [
    ("butter_s0_clean",  6, 0, "CLEAN",     94),
    ("butter_s0_true_t10", 6, 0, "TRUE_T10",  94),
    ("butter_s0_rand_t10", 6, 0, "RAND_T10",  94),
    ("butter_s2_clean",  6, 2, "CLEAN",     96),
    ("butter_s2_true_t10", 6, 2, "TRUE_T10",  96),
    ("butter_s2_rand_t10", 6, 2, "RAND_T10",  96),
]

results = []
for label, task_idx, state_id, condition, anchor in CELLS:
    out_dir = os.path.join(OUT_BASE, label)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        PYTHON, BRIDGE,
        "--condition", condition,
        "--state_id", str(state_id),
        "--anchor", str(anchor),
        "--seed_id", "42",
        "--output_dir", out_dir,
        "--render_gpu", str(GPU),
        "--mlp_path", CKPT,
    ]

    print(f"\n{'='*60}")
    print(f"Cell: {label}")
    print(f"  condition={condition} task={task_idx} state={state_id} anchor={anchor}")
    print(f"  cmd: {' '.join(cmd)}")

    t0 = time.time()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU)
    env["MUJOCO_GL"] = "egl"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HOME"] = "/mnt/sdc/dty_user/openvla_attack/sandbox_home"
    env["TMPDIR"] = "/mnt/sdc/dty_user/openvla_attack/tmp"
    env["OPENVLA_ATTN_IMPLEMENTATION"] = "eager"

    result = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    dt = time.time() - t0

    with open(os.path.join(out_dir, "stdout.log"), "w") as f:
        f.write(result.stdout)
    with open(os.path.join(out_dir, "stderr.log"), "w") as f:
        f.write(result.stderr)

    # Parse result
    success = "SUCCESS" in result.stdout or "success=True" in result.stdout.lower()
    emitted = "EMITTED" in result.stdout or "emit_step" in result.stdout
    print(f"  exit={result.returncode} time={dt:.0f}s success={success} emitted={emitted}")
    if result.stderr:
        err_short = result.stderr[:300].replace('\n', ' | ')
        print(f"  stderr: {err_short}")

    results.append({
        "label": label, "condition": condition, "task_idx": task_idx,
        "state_id": state_id, "exit_code": result.returncode,
        "duration_s": round(dt, 1),
        "success_detected": success, "emitted_detected": emitted,
    })

# Save summary
manifest = {
    "gate": "M0_E2E_OBJECT_POC",
    "checkpoint_sha": hashlib.sha256(open(CKPT, "rb").read()).hexdigest(),
    "model_path": MODEL,
    "gpu": GPU,
    "results": results,
    "n_cells": len(results),
}
json.dump(manifest, open(os.path.join(OUT_BASE, "m0_e2e_manifest.json"), "w"), indent=2)

# Gate check
clean_ok = all(r["success_detected"] for r in results if r["condition"] == "CLEAN")
vis_ok = all(r["emitted_detected"] for r in results if r["condition"] == "TRUE_T10")
rand_ok = all(r["success_detected"] for r in results if r["condition"] == "RAND_T10")
all_ok = clean_ok and vis_ok and rand_ok

print(f"\nCLEAN: {'PASS' if clean_ok else 'FAIL'}")
print(f"TRUE_T10: {'PASS' if vis_ok else 'FAIL'}")
print(f"RAND_T10: {'PASS' if rand_ok else 'FAIL'}")
print(f"M0_E2E: {'PASS' if all_ok else 'FAIL'}")
sys.exit(0 if all_ok else 1)
