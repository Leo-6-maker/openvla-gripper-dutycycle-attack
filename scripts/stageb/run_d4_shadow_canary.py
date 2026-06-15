#!/usr/bin/env python3
"""D4.3a: Canary orchestrator for paired reference/shadow execution.

Reads the frozen 4-state canary manifest. For each state:
  1. Run CLEAN_REFERENCE (detector disabled).
  2. Run CLEAN_SHADOW (detector enabled, read-only).
  3. Compare sequences.
  4. Check all hard gates.

Retry policy: at most 1 infra retry, only if failure occurs before the
first model action is generated. Never retry for detector output, action
mismatch, task failure, or post-first-action crash.

GPU: physical 2,6 only. No attack. No perturbation.
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
PYTHON = os.environ.get("L12_PYTHON", "/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python")
RUNNER_SCRIPT = os.path.join(PIPELINE_ROOT, "scripts", "stageb", "run_d4_clean_shadow.py")

FROZEN_CHECKPOINT = os.path.join(
    PIPELINE_ROOT, "outputs", "d1b_training", "d1b_detector_best.pt")

GPU_VISIBLE = "2,6"
RENDER_GPU = 2

CANARY_N_STATES = 4
MAX_RETRIES = 1  # only for pre-first-action infra failure


def sha256_file(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_one_episode(task, state_id, mode, attempt_id, output_dir, extra_args=None):
    """Launch one episode via subprocess. Returns (returncode, episode_dir)."""
    safe_tag = f"{task}_s{state_id}_{mode}_attempt{attempt_id}"
    episode_dir = os.path.join(output_dir, safe_tag)

    cmd = [
        PYTHON, "-u", RUNNER_SCRIPT,
        "--task", task,
        "--state-id", str(state_id),
        "--mode", mode,
        "--attempt-id", str(attempt_id),
        "--output-dir", output_dir,
        "--checkpoint", FROZEN_CHECKPOINT,
        "--render-gpu-device-id", str(RENDER_GPU),
        "--model-gpu-device-id", "-1",
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_VISIBLE
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"

    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Launching: {' '.join(cmd)}")
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, env=env, timeout=5400,
            capture_output=True, text=True,
        )
        dt = time.time() - t0
        rc = result.returncode
        stderr_tail = result.stderr[-500:] if result.stderr else ""
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {safe_tag}: rc={rc} dt={dt:.0f}s")
        if rc != 0:
            print(f"  STDERR tail: {stderr_tail[-300:]}")
    except subprocess.TimeoutExpired:
        rc = -1
        dt = time.time() - t0
        stderr_tail = "TIMEOUT (5400s)"
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {safe_tag}: TIMEOUT dt={dt:.0f}s")

    return rc, episode_dir


def episode_failed_before_first_action(episode_dir):
    """Check if episode manifest shows failure before first model action."""
    manifest_path = os.path.join(episode_dir, "episode_manifest.json")
    if not os.path.exists(manifest_path):
        return True  # No manifest = likely before first action
    try:
        with open(manifest_path) as f:
            m = json.load(f)
        # Evidence of reaching first action: n_steps > 0 or sentinel exists
        sentinel = os.path.join(episode_dir, "SENTINEL.txt")
        if os.path.exists(sentinel) and m.get("n_steps", 0) >= 0:
            # Sentinel is written before first model action.
            # If sentinel exists but n_steps == 0 and infra_status contains certain patterns
            infra = m.get("infra_status", "ok")
            if m.get("n_steps", -1) == 0 and infra != "ok" and "action" not in infra.lower():
                return True
        return False
    except Exception:
        return False


def load_episode_manifest(episode_dir):
    path = os.path.join(episode_dir, "episode_manifest.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canary-manifest", required=True,
                    help="tables/d4_shadow/d4_shadow_state_manifest.csv (canary subset)")
    ap.add_argument("--output-dir", required=True,
                    help="Root output directory for canary run")
    args = ap.parse_args()

    out = Path(args.output_dir)
    assert not out.exists() or len(list(out.iterdir())) == 0, (
        f"FATAL: Output directory must be empty: {out}"
    )
    out.mkdir(parents=True, exist_ok=True)

    # ── Load canary states ──
    manifest_rows = list(csv.DictReader(open(args.canary_manifest)))
    canary_states = [r for r in manifest_rows if r["subset"] == "canary"]
    assert len(canary_states) == CANARY_N_STATES, (
        f"FATAL: Expected {CANARY_N_STATES} canary states, got {len(canary_states)}"
    )

    tasks = {r["task_key"] for r in canary_states}
    assert len(tasks) == CANARY_N_STATES, (
        f"FATAL: Canary states must have {CANARY_N_STATES} distinct tasks, got {len(tasks)}"
    )

    print(f"Canary manifest: {CANARY_N_STATES} states, {len(tasks)} distinct tasks")
    for r in canary_states:
        print(f"  {r['task_key']}_s{r['state_id']}  hash={r['selection_hash'][:16]}...")

    # ── Run canary ──
    results = []
    canary_pass = True
    gate_failures = []

    for idx, row in enumerate(canary_states):
        task = row["task_key"]
        state_id = int(row["state_id"])
        print(f"\n{'='*60}")
        print(f"CANARY [{idx+1}/{CANARY_N_STATES}]: {task}_s{state_id}")
        print(f"{'='*60}")

        # ── Reference run ──
        ref_ok = False
        ref_dir = None
        for attempt in range(1, MAX_RETRIES + 2):
            rc, ep_dir = run_one_episode(task, state_id, "reference", attempt, str(out))
            if rc == 0:
                ref_ok = True
                ref_dir = ep_dir
                break
            if not episode_failed_before_first_action(ep_dir):
                # Post-first-action failure — no retry
                break

        if not ref_ok:
            canary_pass = False
            gate_failures.append(f"REFERENCE_FAIL:{task}_s{state_id}")
            print(f"  GATE FAIL: REFERENCE_FAIL for {task}_s{state_id}")
            continue

        ref_manifest = load_episode_manifest(ref_dir)

        # ── Shadow run ──
        sh_ok = False
        sh_dir = None
        for attempt in range(1, MAX_RETRIES + 2):
            rc, ep_dir = run_one_episode(task, state_id, "shadow", attempt, str(out))
            if rc == 0:
                sh_ok = True
                sh_dir = ep_dir
                break
            if not episode_failed_before_first_action(ep_dir):
                break

        if not sh_ok:
            canary_pass = False
            gate_failures.append(f"SHADOW_FAIL:{task}_s{state_id}")
            print(f"  GATE FAIL: SHADOW_FAIL for {task}_s{state_id}")
            continue

        sh_manifest = load_episode_manifest(sh_dir)

        # ── Gate: no detector exception ──
        if sh_manifest.get("detector_exception"):
            canary_pass = False
            gate_failures.append(f"DETECTOR_EXCEPTION:{task}_s{state_id}")

        # ── Gate: no action identity failure ──
        if sh_manifest.get("action_identity_fail"):
            canary_pass = False
            gate_failures.append(f"ACTION_IDENTITY_FAIL:{task}_s{state_id}")

        # ── Gate: reference and shadow identical sequences ──
        for key in ["raw_action_sequence_sha256", "env_action_sequence_sha256",
                     "obs_sequence_sha256"]:
            ref_val = ref_manifest.get(key, "")
            sh_val = sh_manifest.get(key, "")
            if ref_val != sh_val:
                canary_pass = False
                gate_failures.append(f"{key}_MISMATCH:{task}_s{state_id}")

        # ── Gate: episode length match ──
        if ref_manifest.get("n_steps") != sh_manifest.get("n_steps"):
            canary_pass = False
            gate_failures.append(f"EPISODE_LENGTH_MISMATCH:{task}_s{state_id}")

        # ── Gate: success match ──
        if ref_manifest.get("success_primary") != sh_manifest.get("success_primary"):
            canary_pass = False
            gate_failures.append(f"SUCCESS_MISMATCH:{task}_s{state_id}")

        # ── Gate: detector reset before episode ──
        sh_sentinel = os.path.join(sh_dir, "SENTINEL.txt")
        if os.path.exists(sh_sentinel):
            with open(sh_sentinel) as f:
                sentinel_text = f.read()
            if "shadow" not in sentinel_text:
                canary_pass = False
                gate_failures.append(f"SENTINEL_INVALID:{task}_s{state_id}")

        # ── Latency gates ──
        sh_latency_path = os.path.join(sh_dir, "latency.csv")
        if os.path.exists(sh_latency_path):
            latencies = []
            with open(sh_latency_path) as f:
                for row in csv.DictReader(f):
                    du = row.get("detector_update_us", "")
                    if du and du != "DISABLED":
                        latencies.append(int(du))
            if latencies:
                p99 = sorted(latencies)[int(len(latencies) * 0.99)]
                max_lat = max(latencies)
                if p99 > 20000:
                    canary_pass = False
                    gate_failures.append(f"LATENCY_P99:{task}_s{state_id}:{p99}us")
                if max_lat > 50000:
                    canary_pass = False
                    gate_failures.append(f"LATENCY_MAX:{task}_s{state_id}:{max_lat}us")

        results.append({
            "task": task, "state_id": state_id,
            "ref_dir": ref_dir, "sh_dir": sh_dir,
            "ref_n_steps": ref_manifest.get("n_steps", -1),
            "sh_n_steps": sh_manifest.get("n_steps", -1),
            "ref_success": ref_manifest.get("success_primary", -1),
            "sh_success": sh_manifest.get("success_primary", -1),
            "sh_emit_step": sh_manifest.get("detector_emit_step", -1),
            "action_identity_ok": not sh_manifest.get("action_identity_fail", True),
            "ref_ok": ref_ok, "sh_ok": sh_ok,
        })

    # ── Aggregate result ──
    print(f"\n{'='*60}")
    print(f"CANARY RESULTS")
    print(f"{'='*60}")

    for r in results:
        print(f"  {r['task']}_s{r['state_id']}: ref_steps={r['ref_n_steps']} "
              f"sh_steps={r['sh_n_steps']} ref_succ={r['ref_success']} "
              f"sh_succ={r['sh_success']} emit={r['sh_emit_step']} "
              f"identity_ok={r['action_identity_ok']}")

    if gate_failures:
        print(f"\nGATE FAILURES ({len(gate_failures)}):")
        for gf in gate_failures:
            print(f"  {gf}")

    result_class = "D4_SHADOW_CANARY_PASS" if canary_pass else "D4_SHADOW_CANARY_FAIL"
    print(f"\nRESULT: {result_class}")

    # ── Write canary report ──
    report = {
        "result_class": result_class,
        "canary_pass": canary_pass,
        "n_states": CANARY_N_STATES,
        "n_completed": len(results),
        "gate_failures": gate_failures,
        "gpu_visible": GPU_VISIBLE,
        "render_gpu": RENDER_GPU,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    with open(out / "canary_result.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(out / "canary_inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "task", "state_id", "ref_ok", "sh_ok", "ref_n_steps", "sh_n_steps",
            "ref_success", "sh_success", "sh_emit_step", "action_identity_ok",
            "ref_dir", "sh_dir",
        ])
        w.writeheader()
        w.writerows(results)

    # GPU cleanup check
    rc = os.system("nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null")
    print(f"\nGPU status: nvidia-smi rc={rc}")

    print(f"Output: {out}")
    if not canary_pass:
        print("STOP: Canary failed. Do not run panel.")
        sys.exit(1)
    else:
        print("CANARY PASS — panel may proceed after audit.")
        sys.exit(0)


if __name__ == "__main__":
    main()
