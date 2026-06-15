#!/usr/bin/env python3
"""D4.3b: Three-GPU clean-shadow panel orchestrator.

Reads the frozen 30-state panel manifest. Assigns states to 3 GPU pairs
by frozen_order mod 3. Each GPU pair runs reference then shadow sequentially.

GPU mapping:
  slot 0: CUDA_VISIBLE_DEVICES=0,1  render=1
  slot 1: CUDA_VISIBLE_DEVICES=2,6  render=2
  slot 2: CUDA_VISIBLE_DEVICES=4,5  render=4

No attack. No perturbation. Read-only detector.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PIPELINE_ROOT = os.environ.get("L12_PIPELINE_ROOT", "/data/liuyu/l12_e4c2_pipeline")
PYTHON = os.environ.get("L12_PYTHON",
    "/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python")
RUNNER = os.path.join(PIPELINE_ROOT, "scripts", "stageb", "run_d4_clean_shadow.py")
CHECKPOINT = os.path.join(PIPELINE_ROOT, "outputs", "d1b_training", "d1b_detector_best.pt")

GPU_SLOTS = [
    {"slot": 0, "cuda": "1,3", "render": 1},   # GPU 0 excluded (Xid 13/43), GPU 3 healthy
    {"slot": 1, "cuda": "2,6", "render": 2},
    {"slot": 2, "cuda": "4,5", "render": 4},
]

PANEL_N = 30
MAX_RETRIES = 1


def sha256_file(path):
    if not os.path.isfile(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def sha256_hex(s): return hashlib.sha256(s.encode()).hexdigest()


def gpu_compute_processes():
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        if r.returncode != 0: return None
        out = r.stdout.strip()
        if not out: return []
        return [tuple(p.strip() for p in line.split(",")) for line in out.split("\n") if line.strip()]
    except: return None


def episode_failed_before_first_action(episode_dir):
    return not os.path.exists(os.path.join(episode_dir, "FIRST_ACTION_GENERATED.json"))


def run_one_episode(task, state_id, mode, attempt_id, output_dir, launcher_dir, gpu_slot):
    safe_tag = f"{task}_s{state_id}_{mode}_attempt{attempt_id}"
    log_dir = os.path.join(launcher_dir, safe_tag)
    os.makedirs(log_dir, exist_ok=False)
    episode_dir = os.path.join(output_dir, safe_tag)

    cmd = [PYTHON, "-u", RUNNER, "--task", task, "--state-id", str(state_id),
           "--mode", mode, "--attempt-id", str(attempt_id),
           "--episode-dir", episode_dir, "--checkpoint", CHECKPOINT,
           "--render-gpu-device-id", str(gpu_slot["render"]),
           "--model-gpu-device-id", "-1"]

    with open(os.path.join(log_dir, "command.txt"), "w") as f:
        f.write(" ".join(cmd) + f"\nCUDA_VISIBLE_DEVICES={gpu_slot['cuda']}\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_slot["cuda"]
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"

    t0 = time.time()
    with open(os.path.join(log_dir, "stdout.log"), "w") as out_f, \
         open(os.path.join(log_dir, "stderr.log"), "w") as err_f:
        out_f.write(f"=== {safe_tag} ===\n{' '.join(cmd)}\n\n")
        err_f.write(f"=== {safe_tag} ===\n{' '.join(cmd)}\n\n")
        try:
            proc = subprocess.Popen(cmd, env=env, stdout=out_f, stderr=err_f)
            proc.wait(timeout=5400)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(); rc = -1
            err_f.write("\n=== TIMEOUT ===\n")

    dt = time.time() - t0
    with open(os.path.join(log_dir, "returncode.json"), "w") as f:
        json.dump({"returncode": rc, "runtime_sec": round(dt, 1)}, f)
    return rc, episode_dir


def load_manifest(episode_dir):
    path = os.path.join(episode_dir, "episode_manifest.json")
    if not os.path.exists(path): return None
    with open(path) as f: return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-manifest", required=True)
    ap.add_argument("--expected-manifest-sha256", required=True)
    ap.add_argument("--expected-execution-head", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    assert not out.exists() or len(list(out.iterdir())) == 0, f"FATAL: {out} must be empty"
    out.mkdir(parents=True, exist_ok=True)

    # Verify manifest
    msha = sha256_file(args.panel_manifest)
    assert msha == args.expected_manifest_sha256, f"Manifest SHA mismatch: {msha[:16]}..."
    print(f"Manifest SHA: {msha[:16]}... VERIFIED")

    # Verify HEAD
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert head == args.expected_execution_head, f"HEAD mismatch: {head[:16]}..."
    print(f"HEAD: {head[:16]}... VERIFIED")

    # Load panel states
    rows = list(csv.DictReader(open(args.panel_manifest)))
    panel = [r for r in rows if r["subset"] == "panel"]
    assert len(panel) == PANEL_N, f"Expected {PANEL_N} panel states, got {len(panel)}"

    # Assign to GPU slots by frozen_order
    gpu_assignments = {s["slot"]: [] for s in GPU_SLOTS}
    for r in panel:
        order = int(r["frozen_order"])
        slot = (order - 4) % 3  # canary uses orders 0-3
        gpu_assignments[slot].append(r)

    for s in GPU_SLOTS:
        n = len(gpu_assignments[s["slot"]])
        assert n == 10, f"GPU slot {s['slot']}: expected 10 states, got {n}"
        print(f"GPU slot {s['slot']} (CUDA={s['cuda']}): {n} states")

    # Write GPU assignment
    with open(out / "d4_shadow_panel_gpu_assignment.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "task_key", "state_id", "frozen_order", "selection_hash",
            "gpu_slot", "cuda_visible_devices", "render_physical_gpu"])
        w.writeheader()
        for s in GPU_SLOTS:
            for r in gpu_assignments[s["slot"]]:
                w.writerow({"task_key": r["task_key"], "state_id": r["state_id"],
                    "frozen_order": r["frozen_order"], "selection_hash": r["selection_hash"],
                    "gpu_slot": str(s["slot"]),
                    "cuda_visible_devices": s["cuda"],
                    "render_physical_gpu": str(s["render"])})

    # GPU baseline
    gpu_before = gpu_compute_processes()
    assert gpu_before is not None, "GPU query failed"
    assert len(gpu_before) == 0, f"Pre-existing GPU processes: {gpu_before}"
    with open(out / "gpu_processes_before.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["gpu_uuid", "pid", "process_name"]); w.writerows(gpu_before)
    print(f"GPU before: 0 processes")

    launcher_dir = os.path.join(str(out), "launcher_logs")
    os.makedirs(launcher_dir, exist_ok=False)

    # Run panel — sequential per GPU slot for now
    all_results = []
    panel_pass = True
    gate_failures = []

    for s in GPU_SLOTS:
        slot = s["slot"]
        states = gpu_assignments[slot]
        print(f"\n{'='*60}")
        print(f"GPU SLOT {slot}: {len(states)} states — CUDA={s['cuda']}")
        print(f"{'='*60}")

        for r in states:
            task = r["task_key"]
            state_id = int(r["state_id"])
            tag = f"{task}_s{state_id}"

            # Reference
            ref_ok = False; ref_dir = None
            for attempt in range(1, MAX_RETRIES + 2):
                rc, ep_dir = run_one_episode(task, state_id, "reference", attempt, str(out), launcher_dir, s)
                if rc == 0: ref_ok = True; ref_dir = ep_dir; break
                if not episode_failed_before_first_action(ep_dir): break

            # Shadow
            sh_ok = False; sh_dir = None
            for attempt in range(1, MAX_RETRIES + 2):
                rc, ep_dir = run_one_episode(task, state_id, "shadow", attempt, str(out), launcher_dir, s)
                if rc == 0: sh_ok = True; sh_dir = ep_dir; break
                if not episode_failed_before_first_action(ep_dir): break

            ref_m = load_manifest(ref_dir) if ref_dir else None
            sh_m = load_manifest(sh_dir) if sh_dir else None

            if not ref_ok: panel_pass = False; gate_failures.append(f"REF_FAIL:{tag}")
            if not sh_ok: panel_pass = False; gate_failures.append(f"SH_FAIL:{tag}")

            if ref_m and sh_m:
                if ref_m.get("n_steps") != sh_m.get("n_steps"):
                    panel_pass = False; gate_failures.append(f"STEPS:{tag}")
                if ref_m.get("success_primary") != sh_m.get("success_primary"):
                    panel_pass = False; gate_failures.append(f"SUCCESS:{tag}")
                if sh_m.get("action_identity_fail"):
                    panel_pass = False; gate_failures.append(f"IDENTITY:{tag}")
                if sh_m.get("n_invalid_field_steps", 0) > 0:
                    panel_pass = False; gate_failures.append(f"INVALID:{tag}")

            status = "OK" if (ref_ok and sh_ok) else "FAIL"
            ref_steps = ref_m.get("n_steps", -1) if ref_m else -1
            sh_steps = sh_m.get("n_steps", -1) if sh_m else -1
            sh_emit = sh_m.get("detector_emit_step", -1) if sh_m else -1
            print(f"  {tag}: {status} ref={ref_steps} sh={sh_steps} emit={sh_emit}")

            all_results.append({
                "task": task, "state_id": state_id, "gpu_slot": slot,
                "ref_ok": ref_ok, "sh_ok": sh_ok,
                "ref_steps": ref_steps, "sh_steps": sh_steps,
                "ref_success": ref_m.get("success_primary") if ref_m else -1,
                "sh_success": sh_m.get("success_primary") if sh_m else -1,
                "sh_emit": sh_emit,
                "ref_dir": ref_dir, "sh_dir": sh_dir,
            })

        # GPU check between slots (brief)
        time.sleep(2)
        mid = gpu_compute_processes()
        if mid and len(mid) > 0:
            print(f"  WARNING: {len(mid)} GPU processes during panel")

    # GPU cleanup
    time.sleep(3)
    gpu_after = gpu_compute_processes()
    assert gpu_after is not None, "GPU query failed"
    with open(out / "gpu_processes_after.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["gpu_uuid", "pid", "process_name"]); w.writerows(gpu_after)
    if len(gpu_after) > 0:
        panel_pass = False; gate_failures.append(f"GPU_RESIDUAL:{len(gpu_after)}")

    # Summary
    n_ok = sum(1 for r in all_results if r["ref_ok"] and r["sh_ok"])
    result_class = "D4_SHADOW_PANEL_COMPLETE" if panel_pass else "D4_SHADOW_PANEL_FAIL"
    print(f"\n{'='*60}")
    print(f"PANEL: {n_ok}/{PANEL_N} states OK")
    if gate_failures: print(f"FAILURES: {len(gate_failures)}")
    print(f"RESULT: {result_class}")

    report = {"result_class": result_class, "panel_pass": panel_pass,
              "n_states": PANEL_N, "n_ok": n_ok, "gate_failures": gate_failures,
              "timestamp": datetime.now(timezone.utc).isoformat()}
    with open(out / "panel_result.json", "w") as f: json.dump(report, f, indent=2)
    with open(out / "panel_inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader(); w.writerows(all_results)

    print(f"Output: {out}")
    sys.exit(0 if panel_pass else 1)


if __name__ == "__main__":
    main()
