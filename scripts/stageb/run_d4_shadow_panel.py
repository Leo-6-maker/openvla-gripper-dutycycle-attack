#!/usr/bin/env python3
"""D4.3b: Three-GPU parallel clean-shadow panel orchestrator.

Master launches 3 workers via subprocess.Popen. Each worker handles one
GPU pair independently, running reference then shadow for each assigned state.

GPU mapping (GPU 0 excluded — Xid 13/43):
  slot 0: CUDA_VISIBLE_DEVICES=1,3  render=1
  slot 1: CUDA_VISIBLE_DEVICES=2,6  render=2
  slot 2: CUDA_VISIBLE_DEVICES=4,5  render=4
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
    {"slot": 0, "cuda": "1,3", "render": 1},
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


def gpu_compute_processes():
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        if r.returncode != 0: return None
        out = r.stdout.strip()
        return [tuple(p.strip() for p in line.split(",")) for line in out.split("\n") if line.strip()] if out else []
    except: return None


def episode_failed_before_first_action(episode_dir):
    return not os.path.exists(os.path.join(episode_dir, "FIRST_ACTION_GENERATED.json"))


def load_manifest(episode_dir):
    path = os.path.join(episode_dir, "episode_manifest.json")
    if not os.path.exists(path): return None
    with open(path) as f: return json.load(f)


# ── Worker mode ──

def worker_main():
    """Entry point when launched as --worker."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-slot", type=int, default=0)
    ap.add_argument("--worker-cuda", default="1,3")
    ap.add_argument("--worker-render", type=int, default=1)
    ap.add_argument("--worker-states-file", default="")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--launcher-dir", default="")
    args = ap.parse_args()

    if not args.worker:
        return

    gpu_slot = {"slot": args.worker_slot, "cuda": args.worker_cuda, "render": args.worker_render}

    with open(args.worker_states_file) as f:
        states = json.load(f)

    print(f"[Worker {args.worker_slot}] GPU={gpu_slot['cuda']} render={gpu_slot['render']} states={len(states)}", flush=True)

    results = []
    for st in states:
        task = st["task_key"]
        state_id = int(st["state_id"])
        tag = f"{task}_s{state_id}"

        # Reference
        ref_ok = False; ref_dir = None
        for attempt in range(1, MAX_RETRIES + 2):
            rc, ep_dir = _run_one(task, state_id, "reference", attempt, args.output_dir, args.launcher_dir, gpu_slot)
            if rc == 0: ref_ok = True; ref_dir = ep_dir; break
            if not episode_failed_before_first_action(ep_dir): break

        # Shadow
        sh_ok = False; sh_dir = None
        for attempt in range(1, MAX_RETRIES + 2):
            rc, ep_dir = _run_one(task, state_id, "shadow", attempt, args.output_dir, args.launcher_dir, gpu_slot)
            if rc == 0: sh_ok = True; sh_dir = ep_dir; break
            if not episode_failed_before_first_action(ep_dir): break

        ref_m = load_manifest(ref_dir) if ref_dir else None
        sh_m = load_manifest(sh_dir) if sh_dir else None
        status = "OK" if (ref_ok and sh_ok) else "FAIL"
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {tag}: {status}", flush=True)

        results.append({
            "task": task, "state_id": state_id, "gpu_slot": args.worker_slot,
            "ref_ok": ref_ok, "sh_ok": sh_ok,
            "ref_steps": ref_m.get("n_steps", -1) if ref_m else -1,
            "sh_steps": sh_m.get("n_steps", -1) if sh_m else -1,
            "ref_success": ref_m.get("success_primary", -1) if ref_m else -1,
            "sh_success": sh_m.get("success_primary", -1) if sh_m else -1,
            "sh_emit": sh_m.get("detector_emit_step", -1) if sh_m else -1,
            "gate_failures": _check_gates(ref_m, sh_m, tag),
            "ref_dir": ref_dir, "sh_dir": sh_dir,
        })

    # Write slot result
    result_path = os.path.join(args.output_dir, f"slot_{args.worker_slot}_result.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Worker {args.worker_slot}] DONE — {sum(1 for r in results if r['ref_ok'] and r['sh_ok'])}/{len(results)} OK", flush=True)


def _run_one(task, state_id, mode, attempt_id, output_dir, launcher_dir, gpu_slot):
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
        f.write(" ".join(cmd) + f"\nCUDA={gpu_slot['cuda']}\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_slot["cuda"]
    env["MUJOCO_GL"] = "egl"; env["PYOPENGL_PLATFORM"] = "egl"

    t0 = time.time()
    with open(os.path.join(log_dir, "stdout.log"), "w") as of, \
         open(os.path.join(log_dir, "stderr.log"), "w") as ef:
        of.write(f"=== {safe_tag} ===\n"); ef.write(f"=== {safe_tag} ===\n")
        try:
            proc = subprocess.Popen(cmd, env=env, stdout=of, stderr=ef)
            proc.wait(timeout=5400); rc = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(); rc = -1

    with open(os.path.join(log_dir, "returncode.json"), "w") as f:
        json.dump({"returncode": rc, "runtime_sec": round(time.time()-t0, 1)}, f)
    return rc, episode_dir


def _check_gates(ref_m, sh_m, tag):
    failures = []
    if ref_m and sh_m:
        if ref_m.get("n_steps") != sh_m.get("n_steps"): failures.append(f"STEPS:{tag}")
        if ref_m.get("success_primary") != sh_m.get("success_primary"): failures.append(f"SUCCESS:{tag}")
        if sh_m.get("action_identity_fail"): failures.append(f"IDENTITY:{tag}")
        if sh_m.get("n_invalid_field_steps", 0) > 0: failures.append(f"INVALID:{tag}")
    return failures


# ── Master mode ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-slot", type=int, default=0)
    ap.add_argument("--worker-cuda", default="1,3")
    ap.add_argument("--worker-render", type=int, default=1)
    ap.add_argument("--worker-states-file", default="")
    ap.add_argument("--panel-manifest", default="")
    ap.add_argument("--expected-manifest-sha256", default="")
    ap.add_argument("--expected-execution-head", default="")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--launcher-dir", default="")
    args = ap.parse_args()

    # Worker mode
    if args.worker:
        worker_main()
        return

    # Master mode
    out = Path(args.output_dir)
    assert not out.exists() or len(list(out.iterdir())) == 0, f"FATAL: {out} must be empty"
    out.mkdir(parents=True, exist_ok=True)

    msha = sha256_file(args.panel_manifest)
    assert msha == args.expected_manifest_sha256, f"Manifest SHA mismatch"
    print(f"Manifest SHA: {msha[:16]}... VERIFIED")

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert head == args.expected_execution_head, f"HEAD mismatch"
    print(f"HEAD: {head[:16]}... VERIFIED")

    # Load and assign panel states
    rows = list(csv.DictReader(open(args.panel_manifest)))
    panel = [r for r in rows if r["subset"] == "panel"]
    assert len(panel) == PANEL_N

    gpu_assignments = {s["slot"]: [] for s in GPU_SLOTS}
    for r in panel:
        order = int(r["frozen_order"])
        slot = (order - 4) % 3
        gpu_assignments[slot].append(r)

    for s in GPU_SLOTS:
        n = len(gpu_assignments[s["slot"]])
        assert n == 10, f"Slot {s['slot']}: {n} states"
        print(f"GPU slot {s['slot']} (CUDA={s['cuda']}): {n} states")

    # GPU baseline
    gpu_before = gpu_compute_processes()
    assert gpu_before is not None, "GPU query failed"
    assert len(gpu_before) == 0, f"Pre-existing GPU processes: {gpu_before}"
    with open(out / "gpu_processes_before.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["gpu_uuid", "pid", "process_name"]); w.writerows(gpu_before)

    launcher_dir = os.path.join(str(out), "launcher_logs")
    os.makedirs(launcher_dir, exist_ok=False)

    # Launch 3 parallel workers
    workers = []
    for s in GPU_SLOTS:
        states_file = str(out / f"slot_{s['slot']}_states.json")
        with open(states_file, "w") as f:
            json.dump(gpu_assignments[s["slot"]], f)

        cmd = [PYTHON, "-u", __file__,
               "--worker", "--worker-slot", str(s["slot"]),
               "--worker-cuda", s["cuda"], "--worker-render", str(s["render"]),
               "--worker-states-file", states_file,
               "--output-dir", str(out), "--launcher-dir", launcher_dir]
        log_file = str(out / f"worker_{s['slot']}.log")
        print(f"[Master] Launching worker {s['slot']}: GPU={s['cuda']}")
        with open(log_file, "w") as lf:
            lf.write(f"Worker {s['slot']} start {datetime.now().isoformat()}\n")
        proc = subprocess.Popen(cmd, stdout=open(log_file, "a"), stderr=subprocess.STDOUT)
        workers.append((s["slot"], proc, log_file))

    # Wait for all workers
    for slot, proc, log_file in workers:
        proc.wait()
        rc = proc.returncode
        print(f"[Master] Worker {slot}: rc={rc}")

    # Collect results
    all_results = []
    panel_pass = True
    gate_failures = []
    for s in GPU_SLOTS:
        result_path = out / f"slot_{s['slot']}_result.json"
        if result_path.exists():
            slot_results = json.load(open(result_path))
            all_results.extend(slot_results)
            for r in slot_results:
                ok = r["ref_ok"] and r["sh_ok"]
                if not ok: panel_pass = False
                gate_failures.extend(r.get("gate_failures", []))
                if not r["ref_ok"]: gate_failures.append(f"REF_FAIL:{r['task']}_s{r['state_id']}")
                if not r["sh_ok"]: gate_failures.append(f"SH_FAIL:{r['task']}_s{r['state_id']}")
        else:
            panel_pass = False
            gate_failures.append(f"WORKER_MISSING:slot_{s['slot']}")

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
    for r in all_results:
        tag = f"{r['task']}_s{r['state_id']}"
        ok = r["ref_ok"] and r["sh_ok"]
        print(f"  {tag}: {'OK' if ok else 'FAIL'} ref={r['ref_steps']} sh={r['sh_steps']} emit={r['sh_emit']}")
    if gate_failures: print(f"FAILURES ({len(gate_failures)}): {gate_failures[:10]}")
    print(f"RESULT: {result_class}")
    print(f"Output: {out}")

    report = {"result_class": result_class, "panel_pass": panel_pass,
              "n_states": PANEL_N, "n_ok": n_ok, "gate_failures": gate_failures,
              "timestamp": datetime.now(timezone.utc).isoformat()}
    with open(out / "panel_result.json", "w") as f: json.dump(report, f, indent=2)
    if all_results:
        with open(out / "panel_inventory.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[k for k in all_results[0] if k != "gate_failures"])
            w.writeheader()
            for r in all_results:
                w.writerow({k: v for k, v in r.items() if k != "gate_failures"})

    sys.exit(0 if panel_pass else 1)


if __name__ == "__main__":
    main()
