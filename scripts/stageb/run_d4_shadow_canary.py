#!/usr/bin/env python3
"""D4.3a: Canary orchestrator for paired reference/shadow execution.

Reads the frozen 4-state canary manifest. For each state:
  1. Run CLEAN_REFERENCE (detector disabled).
  2. Run CLEAN_SHADOW (detector enabled, read-only).
  3. Compare all hard gates.
  4. If all pass → CANARY_PASS; else STOP.

Retry: only if FIRST_ACTION_GENERATED marker does not exist in attempt dir.
Max 1 retry per state per mode.

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
PYTHON = os.environ.get(
    "L12_PYTHON",
    "/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python",
)
RUNNER = os.path.join(PIPELINE_ROOT, "scripts", "stageb", "run_d4_clean_shadow.py")
CHECKPOINT = os.path.join(PIPELINE_ROOT, "outputs", "d1b_training", "d1b_detector_best.pt")

GPU_VISIBLE = "2,6"
RENDER_GPU = 2
CANARY_N = 4
MAX_RETRIES = 1


def sha256_file(path: str) -> str:
    if not os.path.isfile(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def gpu_process_count():
    """Return list of (gpu_uuid, pid, process_name, used_memory) for GPUs 2,6."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception:
        return []
    if not out:
        return []
    procs = []
    for line in out.split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            procs.append(tuple(parts))
    return procs


def gpu_uuids():
    """Return mapping from GPU index to UUID."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception:
        return {}
    mapping = {}
    for line in out.split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            mapping[int(parts[0])] = parts[1]
    return mapping


def episode_failed_before_first_action(episode_dir):
    """Retry allowed only if FIRST_ACTION_GENERATED.json does not exist."""
    marker = os.path.join(episode_dir, "FIRST_ACTION_GENERATED.json")
    return not os.path.exists(marker)


def run_one_episode(task, state_id, mode, attempt_id, output_dir):
    """Launch one episode via subprocess. Saves full stdout/stderr/command."""
    safe_tag = f"{task}_s{state_id}_{mode}_attempt{attempt_id}"
    episode_dir = os.path.join(output_dir, safe_tag)

    # Create attempt directory (orchestrator does this so logs go inside)
    try:
        os.makedirs(episode_dir, exist_ok=False)
    except FileExistsError:
        return -1, episode_dir  # Should not happen if orchestrator manages attempts

    cmd = [
        PYTHON, "-u", RUNNER,
        "--task", task, "--state-id", str(state_id),
        "--mode", mode, "--attempt-id", str(attempt_id),
        "--output-dir", output_dir,
        "--checkpoint", CHECKPOINT,
        "--render-gpu-device-id", str(RENDER_GPU),
        "--model-gpu-device-id", "-1",
    ]

    # Save command
    with open(os.path.join(episode_dir, "command.txt"), "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.write(f"CUDA_VISIBLE_DEVICES={GPU_VISIBLE}\n")

    # Save environment snapshot
    env_snap = {k: str(v) for k, v in sorted(os.environ.items())
                if any(p in k.lower() for p in ["cuda", "gpu", "mujoco", "opengl",
                                                 "openvla", "python", "ld_", "path"])}
    with open(os.path.join(episode_dir, "environment_snapshot.json"), "w") as f:
        json.dump(env_snap, f, indent=2)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_VISIBLE
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"

    t0 = time.time()
    stdout_path = os.path.join(episode_dir, "stdout.log")
    stderr_path = os.path.join(episode_dir, "stderr.log")

    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Launch: {safe_tag}")

    with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
        out_f.write(f"=== {safe_tag} ===\n{' '.join(cmd)}\n\n")
        err_f.write(f"=== {safe_tag} ===\n{' '.join(cmd)}\n\n")
        try:
            proc = subprocess.Popen(
                cmd, env=env, stdout=out_f, stderr=err_f,
            )
            proc.wait(timeout=5400)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
            rc = -1
            err_f.write("\n=== TIMEOUT (5400s) ===\n")

    dt = time.time() - t0

    # Write returncode
    with open(os.path.join(episode_dir, "returncode.json"), "w") as f:
        json.dump({"returncode": rc, "runtime_sec": round(dt, 1),
                    "timeout": rc == -1}, f, indent=2)

    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {safe_tag}: rc={rc} dt={dt:.0f}s")
    return rc, episode_dir


def load_manifest(episode_dir):
    path = os.path.join(episode_dir, "episode_manifest.json")
    if not os.path.exists(path): return None
    with open(path) as f: return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canary-manifest", required=True)
    ap.add_argument("--expected-manifest-sha256", required=True)
    ap.add_argument("--expected-freeze-runner-sha256", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    assert not out.exists() or len(list(out.iterdir())) == 0, (
        f"FATAL: Output directory must be empty: {out}"
    )
    out.mkdir(parents=True, exist_ok=True)

    # ── Verify manifest ──
    manifest_sha = sha256_file(args.canary_manifest)
    assert manifest_sha == args.expected_manifest_sha256, (
        f"FATAL: Manifest SHA mismatch: got {manifest_sha[:16]}..., "
        f"expected {args.expected_manifest_sha256[:16]}..."
    )
    print(f"Manifest SHA: {manifest_sha[:16]}... VERIFIED")

    manifest_rows = list(csv.DictReader(open(args.canary_manifest)))
    canary_states = [r for r in manifest_rows if r["subset"] == "canary"]
    assert len(canary_states) == CANARY_N, (
        f"FATAL: Expected {CANARY_N} canary states, got {len(canary_states)}"
    )
    tasks = {r["task_key"] for r in canary_states}
    assert len(tasks) == CANARY_N, f"FATAL: need 4 distinct tasks, got {len(tasks)}"

    # Verify each canary state's selection hash can be recomputed
    SALT = "D4.3_SHADOW_V1"
    for r in canary_states:
        expected = r["selection_hash"]
        recomputed = sha256_hex(f"{SALT}|{r['task_key']}|{r['state_id']}")
        assert recomputed == expected, (
            f"FATAL: selection hash mismatch for {r['task_key']}_s{r['state_id']}: "
            f"expected {expected[:16]}..., recomputed {recomputed[:16]}..."
        )
    print(f"Canary: {CANARY_N} states, {len(tasks)} tasks — all selection hashes verified")

    for r in canary_states:
        print(f"  {r['task_key']}_s{r['state_id']}  hash={r['selection_hash'][:16]}...")

    # ── GPU baseline ──
    gpu_procs_before = gpu_process_count()
    print(f"GPU processes before canary: {len(gpu_procs_before)}")
    for gp in gpu_procs_before:
        print(f"  {gp}")

    # ── Run canary ──
    results = []
    canary_pass = True
    gate_failures = []

    for idx, row in enumerate(canary_states):
        task = row["task_key"]
        state_id = int(row["state_id"])
        print(f"\n{'='*60}")
        print(f"CANARY [{idx+1}/{CANARY_N}]: {task}_s{state_id}")
        print(f"{'='*60}")

        # ── Reference ──
        ref_ok = False; ref_dir = None
        for attempt in range(1, MAX_RETRIES + 2):
            rc, ep_dir = run_one_episode(task, state_id, "reference", attempt, str(out))
            if rc == 0:
                ref_ok = True; ref_dir = ep_dir; break
            if not episode_failed_before_first_action(ep_dir):
                break  # Post-first-action failure → no retry

        if not ref_ok:
            canary_pass = False
            gate_failures.append(f"REFERENCE_FAIL:{task}_s{state_id}")

        # ── Shadow ──
        sh_ok = False; sh_dir = None
        for attempt in range(1, MAX_RETRIES + 2):
            rc, ep_dir = run_one_episode(task, state_id, "shadow", attempt, str(out))
            if rc == 0:
                sh_ok = True; sh_dir = ep_dir; break
            if not episode_failed_before_first_action(ep_dir):
                break

        if not sh_ok:
            canary_pass = False
            gate_failures.append(f"SHADOW_FAIL:{task}_s{state_id}")

        if not ref_ok or not sh_ok:
            results.append({
                "task": task, "state_id": state_id,
                "ref_ok": ref_ok, "sh_ok": sh_ok,
                "ref_dir": ref_dir, "sh_dir": sh_dir,
            })
            continue

        ref_m = load_manifest(ref_dir)
        sh_m = load_manifest(sh_dir)

        # ── Gate: no detector exception ──
        if sh_m.get("detector_exception"):
            canary_pass = False
            gate_failures.append(f"DETECTOR_EXCEPTION:{task}_s{state_id}")

        # ── Gate: no action identity failure ──
        if sh_m.get("action_identity_fail"):
            canary_pass = False
            gate_failures.append(f"ACTION_IDENTITY_FAIL:{task}_s{state_id}")

        # ── Gate: sequence hashes identical ──
        for key in ["raw_action_sequence_sha256", "env_action_sequence_sha256",
                     "obs_sequence_sha256"]:
            if ref_m.get(key) != sh_m.get(key):
                canary_pass = False
                gate_failures.append(f"{key}_MISMATCH:{task}_s{state_id}")

        # ── Gate: episode length ──
        if ref_m.get("n_steps") != sh_m.get("n_steps"):
            canary_pass = False
            gate_failures.append(f"STEPS_MISMATCH:{task}_s{state_id}")

        # ── Gate: success + done ──
        for sk in ["success_primary", "success_done_any", "success_check_any",
                    "success_step_primary", "done_step"]:
            if ref_m.get(sk) != sh_m.get(sk):
                canary_pass = False
                gate_failures.append(f"{sk}_MISMATCH:{task}_s{state_id}")

        # ── Gate: no invalid critical gripper fields ──
        if sh_m.get("n_invalid_field_steps", 0) > 0:
            canary_pass = False
            gate_failures.append(
                f"INVALID_FIELDS:{task}_s{state_id}:"
                f"{sh_m['n_invalid_field_steps']}_steps"
            )

        # ── Gate: detector reset state before episode ──
        pre = sh_m.get("detector_pre_reset", {})
        if pre:
            if pre.get("next_expected_step") != 0:
                canary_pass = False
                gate_failures.append(f"RESET_STEP:{task}_s{state_id}")
            if pre.get("emit_step") != -1:
                canary_pass = False
                gate_failures.append(f"RESET_EMIT:{task}_s{state_id}")

        # ── Gate: no predictor-abstain emission ──
        sh_cand_path = os.path.join(sh_dir, "detector_candidates.csv")
        if os.path.exists(sh_cand_path) and sh_m.get("detector_emit_step", -1) >= 0:
            emit_step = sh_m["detector_emit_step"]
            with open(sh_cand_path) as f:
                for cand in csv.DictReader(f):
                    if int(cand.get("step", -1)) == emit_step:
                        if cand.get("abstained", "1") == "1" or cand.get("abstain", ""):
                            canary_pass = False
                            gate_failures.append(
                                f"ABSTAIN_EMISSION:{task}_s{state_id}:"
                                f"step={emit_step} reason={cand.get('abstain','')}"
                            )
                        break

        # ── Gate: latency ──
        sh_lat_path = os.path.join(sh_dir, "latency.csv")
        if os.path.exists(sh_lat_path):
            det_lats = []
            model_lats = []
            with open(sh_lat_path) as f:
                for row in csv.DictReader(f):
                    du = row.get("detector_update_us", "")
                    mu = row.get("model_inference_us", "")
                    if du and du != "DISABLED":
                        det_lats.append(int(du))
                    if mu and mu != "DISABLED":
                        model_lats.append(int(mu))
            if det_lats:
                p99 = sorted(det_lats)[int(len(det_lats) * 0.99)]
                mx = max(det_lats)
                median_det = sorted(det_lats)[len(det_lats) // 2]
                median_model = (sorted(model_lats)[len(model_lats) // 2]
                                if model_lats else 1)
                if p99 > 20000:
                    canary_pass = False
                    gate_failures.append(f"LATENCY_P99:{task}_s{state_id}:{p99}us")
                if mx > 50000:
                    canary_pass = False
                    gate_failures.append(f"LATENCY_MAX:{task}_s{state_id}:{mx}us")
                if median_model > 0:
                    overhead_pct = median_det / median_model * 100
                    if overhead_pct > 5.0:
                        canary_pass = False
                        gate_failures.append(
                            f"LATENCY_OVERHEAD:{task}_s{state_id}:{overhead_pct:.1f}%"
                        )

        results.append({
            "task": task, "state_id": state_id,
            "ref_dir": ref_dir, "sh_dir": sh_dir,
            "ref_ok": ref_ok, "sh_ok": sh_ok,
            "ref_n_steps": ref_m.get("n_steps"),
            "sh_n_steps": sh_m.get("n_steps"),
            "ref_success": ref_m.get("success_primary"),
            "sh_success": sh_m.get("success_primary"),
            "sh_emit_step": sh_m.get("detector_emit_step"),
            "action_identity_ok": not sh_m.get("action_identity_fail"),
            "invalid_field_steps": sh_m.get("n_invalid_field_steps", 0),
        })

    # ── GPU cleanup gate ──
    time.sleep(2)  # Let processes finish
    gpu_procs_after = gpu_process_count()
    print(f"\nGPU processes after canary: {len(gpu_procs_after)}")
    for gp in gpu_procs_after:
        print(f"  {gp}")

    # Identify new processes (in 'after' but not in 'before')
    before_set = set(gpu_procs_before)
    new_procs = [p for p in gpu_procs_after if p not in before_set]
    if new_procs:
        canary_pass = False
        gate_failures.append(f"GPU_CLEANUP:{len(new_procs)}_new_processes")
        for np in new_procs:
            print(f"  RESIDUAL: {np}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("CANARY RESULTS")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['task']}_s{r['state_id']}: ref={r['ref_ok']} sh={r['sh_ok']} "
              f"steps={r['ref_n_steps']}/{r['sh_n_steps']} "
              f"succ={r['ref_success']}/{r['sh_success']} "
              f"emit={r['sh_emit_step']} identity={r['action_identity_ok']} "
              f"invalid={r.get('invalid_field_steps',0)}")

    if gate_failures:
        print(f"\nGATE FAILURES ({len(gate_failures)}):")
        for gf in gate_failures:
            print(f"  {gf}")

    result_class = "D4_SHADOW_CANARY_PASS" if canary_pass else "D4_SHADOW_CANARY_FAIL"
    print(f"\nRESULT: {result_class}")

    report = {
        "result_class": result_class, "canary_pass": canary_pass,
        "n_states": CANARY_N, "n_completed": len(results),
        "gate_failures": gate_failures,
        "gpu_visible": GPU_VISIBLE, "render_gpu": RENDER_GPU,
        "gpu_procs_before": len(gpu_procs_before),
        "gpu_procs_after": len(gpu_procs_after),
        "gpu_new_processes": len(new_procs) if canary_pass else -1,
        "manifest_sha": manifest_sha,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    with open(out / "canary_result.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(out / "canary_inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "task", "state_id", "ref_ok", "sh_ok", "ref_n_steps", "sh_n_steps",
            "ref_success", "sh_success", "sh_emit_step", "action_identity_ok",
            "invalid_field_steps", "ref_dir", "sh_dir",
        ])
        w.writeheader(); w.writerows(results)

    print(f"\nOutput: {out}")
    if not canary_pass:
        print("STOP: Canary failed. Do not run panel.")
        sys.exit(1)
    else:
        print("CANARY PASS — panel may proceed after audit.")
        sys.exit(0)


if __name__ == "__main__":
    main()
