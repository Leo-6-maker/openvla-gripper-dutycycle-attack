#!/usr/bin/env python3
"""D4.3a: Canary orchestrator for paired reference/shadow execution.

Orchestrator OWNS launcher logs (command, env, stdout, stderr, returncode).
Runner OWNS episode directory (artifacts, phase markers, provenance).

Orchestrator → launcher_logs/<safe_tag>/
  command.txt  environment_snapshot.json  stdout.log  stderr.log  returncode.json

Runner → <output_dir>/<safe_tag>/
  ATTEMPT_STARTED.json  MODEL_LOADED.json  FIRST_ACTION_GENERATED.json
  step_trace.csv  detector_candidates.csv  ...

Retry: only if FIRST_ACTION_GENERATED.json does not exist in episode dir.
Max 1 retry per state per mode.

GPU: physical 2,6 only.
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
GPU_PHYSICAL_IDS = [2, 6]
RENDER_GPU_PHYSICAL = 2  # first visible physical GPU
CANARY_N = 4
MAX_RETRIES = 1

# Provenance gates — set by main(), read by run_one_episode()
_PROVENANCE_HEAD = ""
_PROVENANCE_BRANCH = ""
_PROVENANCE_CLEAN = False


def sha256_file(path: str) -> str:
    if not os.path.isfile(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ── GPU management ──

def gpu_uuid_map():
    """Return {physical_index: uuid} for all GPUs. Returns None on query failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        out = result.stdout.strip()
    except Exception:
        return None
    m = {}
    for line in out.split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                m[int(parts[0])] = parts[1]
            except ValueError:
                pass
    return m


def gpu_compute_processes(uuids_of_interest):
    """Return [(gpu_uuid, pid, process_name)] or None on query failure."""
    if not uuids_of_interest:
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return None
        out = result.stdout.strip()
    except Exception:
        return None
    if not out:
        return []
    procs = []
    for line in out.split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0] in uuids_of_interest:
            procs.append((parts[0], parts[1], parts[2]))
    return procs


# ── Retry ──

def episode_failed_before_first_action(episode_dir):
    """Only retry if FIRST_ACTION_GENERATED.json does NOT exist."""
    return not os.path.exists(os.path.join(episode_dir, "FIRST_ACTION_GENERATED.json"))


# ── Episode launcher ──

def run_one_episode(task, state_id, mode, attempt_id, output_dir, launcher_dir):
    """Launch episode via subprocess.

    Orchestrator writes launcher logs to launcher_dir/<safe_tag>/.
    Runner creates and owns episode_dir = output_dir/<safe_tag>/.
    """
    safe_tag = f"{task}_s{state_id}_{mode}_attempt{attempt_id}"
    log_dir = os.path.join(launcher_dir, safe_tag)
    os.makedirs(log_dir, exist_ok=False)

    episode_dir = os.path.join(output_dir, safe_tag)

    cmd = [
        PYTHON, "-u", RUNNER,
        "--task", task, "--state-id", str(state_id),
        "--mode", mode, "--attempt-id", str(attempt_id),
        "--episode-dir", episode_dir,
        "--checkpoint", CHECKPOINT,
        "--render-gpu-device-id", str(RENDER_GPU_PHYSICAL),
        "--model-gpu-device-id", "-1",
    ]
    # Pass provenance gates to runner (set by main via closure or globals)
    if _PROVENANCE_HEAD:
        cmd.extend(["--expected-git-head", _PROVENANCE_HEAD])
    if _PROVENANCE_BRANCH:
        cmd.extend(["--expected-branch", _PROVENANCE_BRANCH])
    if _PROVENANCE_CLEAN:
        cmd.append("--require-clean-worktree")

    # Save command + env snapshot
    with open(os.path.join(log_dir, "command.txt"), "w") as f:
        f.write(" ".join(cmd) + "\n")
        f.write(f"CUDA_VISIBLE_DEVICES={GPU_VISIBLE}\n")
        f.write(f"GPU_PHYSICAL={GPU_PHYSICAL_IDS}\n")
        f.write(f"RENDER_GPU_PHYSICAL={RENDER_GPU_PHYSICAL}\n")

    env_snap = {k: str(v) for k, v in sorted(os.environ.items())
                if any(p in k.lower() for p in [
                    "cuda", "gpu", "mujoco", "opengl", "openvla",
                    "python", "ld_", "path", "display",
                ])}
    with open(os.path.join(log_dir, "environment_snapshot.json"), "w") as f:
        json.dump(env_snap, f, indent=2)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_VISIBLE
    env["MUJOCO_GL"] = "egl"
    env["PYOPENGL_PLATFORM"] = "egl"

    t0 = time.time()
    stdout_path = os.path.join(log_dir, "stdout.log")
    stderr_path = os.path.join(log_dir, "stderr.log")

    print(f"  [{datetime.now().strftime('%H:%M:%S')}] Launch: {safe_tag} -> {episode_dir}")

    with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
        out_f.write(f"=== {safe_tag} ===\n{' '.join(cmd)}\n\n")
        err_f.write(f"=== {safe_tag} ===\n{' '.join(cmd)}\n\n")
        try:
            proc = subprocess.Popen(cmd, env=env, stdout=out_f, stderr=err_f)
            proc.wait(timeout=5400)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
            rc = -1
            err_f.write("\n=== TIMEOUT (5400s) ===\n")

    dt = time.time() - t0

    with open(os.path.join(log_dir, "returncode.json"), "w") as f:
        json.dump({"returncode": rc, "runtime_sec": round(dt, 1),
                    "timeout": rc == -1}, f, indent=2)

    # Hash launcher artifacts
    launcher_hashes = {}
    for fn in ["command.txt", "environment_snapshot.json", "stdout.log",
               "stderr.log", "returncode.json"]:
        fp = os.path.join(log_dir, fn)
        if os.path.exists(fp):
            launcher_hashes[fn] = sha256_file(fp)
    with open(os.path.join(log_dir, "launcher_artifact_hashes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["artifact", "sha256"])
        for k, v in launcher_hashes.items():
            w.writerow([k, v])

    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {safe_tag}: rc={rc} dt={dt:.0f}s")
    return rc, episode_dir, log_dir


def load_manifest(episode_dir):
    path = os.path.join(episode_dir, "episode_manifest.json")
    if not os.path.exists(path): return None
    with open(path) as f: return json.load(f)


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canary-manifest", required=True)
    ap.add_argument("--expected-manifest-sha256", required=True)
    ap.add_argument("--expected-freeze-runner-sha256", required=True)
    ap.add_argument("--expected-freeze-commit", required=True,
                    help="Exact 40-char commit that produced the manifest")
    ap.add_argument("--expected-branch", required=True,
                    help="Expected git branch (e.g. exp/l12-production-streaming-adapter-20260615)")
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

    # Verify freeze runner SHA
    freeze_runner_path = os.path.join(
        PIPELINE_ROOT, "scripts", "stageb", "run_d4_freeze_shadow_states.py",
    )
    actual_freeze_sha = sha256_file(freeze_runner_path)
    assert actual_freeze_sha == args.expected_freeze_runner_sha256, (
        f"FATAL: Freeze runner SHA mismatch: got {actual_freeze_sha[:16]}..., "
        f"expected {args.expected_freeze_runner_sha256[:16]}..."
    )
    print(f"Freeze runner SHA: {actual_freeze_sha[:16]}... VERIFIED")

    # ── Provenance verification ──
    current_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
    ).stdout.strip()
    assert current_head, "FATAL: Could not determine git HEAD"

    current_branch = subprocess.run(
        ["git", "branch", "--show-current"], capture_output=True, text=True,
    ).stdout.strip()

    git_status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True,
    ).stdout
    worktree_clean = (git_status.strip() == "")
    assert worktree_clean, (
        f"FATAL: Worktree not clean before canary:\n{git_status[:500]}"
    )

    assert current_branch == args.expected_branch, (
        f"FATAL: Branch mismatch: got {current_branch}, "
        f"expected {args.expected_branch}"
    )

    # Freeze commit verification
    assert current_head == args.expected_freeze_commit, (
        f"FATAL: Execution HEAD {current_head[:16]}... != "
        f"expected freeze commit {args.expected_freeze_commit[:16]}..."
    )
    print(f"Execution HEAD: {current_head[:16]}... VERIFIED (matches expected freeze commit)")
    print(f"Branch: {current_branch}  Worktree: clean")

    # Set provenance globals for runner subprocess
    global _PROVENANCE_HEAD, _PROVENANCE_BRANCH, _PROVENANCE_CLEAN
    _PROVENANCE_HEAD = current_head
    _PROVENANCE_BRANCH = current_branch
    _PROVENANCE_CLEAN = True

    # ── Load and verify canary states ──
    manifest_rows = list(csv.DictReader(open(args.canary_manifest)))
    canary_states = [r for r in manifest_rows if r["subset"] == "canary"]
    assert len(canary_states) == CANARY_N, (
        f"FATAL: Expected {CANARY_N} canary states, got {len(canary_states)}"
    )
    tasks = {r["task_key"] for r in canary_states}
    assert len(tasks) == CANARY_N, f"FATAL: need 4 distinct tasks, got {len(tasks)}"

    SALT = "D4.3_SHADOW_V1"
    for r in canary_states:
        expected = r["selection_hash"]
        recomputed = sha256_hex(f"{SALT}|{r['task_key']}|{r['state_id']}")
        assert recomputed == expected, (
            f"FATAL: selection hash mismatch for {r['task_key']}_s{r['state_id']}"
        )
    print(f"Canary: {CANARY_N} states, {len(tasks)} tasks — all selection hashes verified")

    for r in canary_states:
        print(f"  {r['task_key']}_s{r['state_id']}  hash={r['selection_hash'][:16]}...")

    # ── GPU mapping verification ──
    uuid_map = gpu_uuid_map()
    target_uuids = set()
    for pid in GPU_PHYSICAL_IDS:
        uuid = uuid_map.get(pid)
        assert uuid is not None, f"FATAL: Physical GPU {pid} not found in nvidia-smi"
        target_uuids.add(uuid)
        print(f"  GPU physical {pid} -> UUID {uuid}")

    # GPU baseline
    gpu_before = gpu_compute_processes(target_uuids)
    assert gpu_before is not None, "FATAL: GPU query failed before canary"
    assert len(gpu_before) == 0, (
        f"FATAL: {len(gpu_before)} pre-existing GPU processes on target GPUs: {gpu_before}"
    )
    gpu_before_snapshot = os.path.join(str(out), "gpu_processes_before.csv")
    with open(gpu_before_snapshot, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gpu_uuid", "pid", "process_name"])
        w.writerows(gpu_before)
    print(f"GPU processes before: {len(gpu_before)} (saved to {gpu_before_snapshot})")

    # ── Launcher logs directory ──
    launcher_dir = os.path.join(str(out), "launcher_logs")
    os.makedirs(launcher_dir, exist_ok=False)

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
            rc, ep_dir, log_dir = run_one_episode(
                task, state_id, "reference", attempt, str(out), launcher_dir,
            )
            if rc == 0:
                ref_ok = True; ref_dir = ep_dir; break
            if not episode_failed_before_first_action(ep_dir):
                break

        if not ref_ok:
            canary_pass = False
            gate_failures.append(f"REFERENCE_FAIL:{task}_s{state_id}")

        # ── Shadow ──
        sh_ok = False; sh_dir = None
        for attempt in range(1, MAX_RETRIES + 2):
            rc, ep_dir, log_dir = run_one_episode(
                task, state_id, "shadow", attempt, str(out), launcher_dir,
            )
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

        # Validate manifests exist
        if ref_m is None:
            canary_pass = False
            gate_failures.append(f"REF_MANIFEST_MISSING:{task}_s{state_id}")
            continue
        if sh_m is None:
            canary_pass = False
            gate_failures.append(f"SH_MANIFEST_MISSING:{task}_s{state_id}")
            continue

        tag = f"{task}_s{state_id}"

        # ── Required artifacts must exist ──
        required_artifacts = [
            "ATTEMPT_STARTED.json", "MODEL_LOADED.json",
            "FIRST_ACTION_GENERATED.json", "step_trace.csv",
            "detector_candidates.csv", "detector_emission.json",
            "action_identity.csv", "latency.csv",
            "provenance.csv", "episode_manifest.json",
            "artifact_hashes.csv", "teacher_sidecar.json",
        ]
        for mode_dir, mode_name in [(ref_dir, "ref"), (sh_dir, "sh")]:
            for an in required_artifacts:
                ap = os.path.join(mode_dir, an)
                if not os.path.exists(ap):
                    canary_pass = False
                    gate_failures.append(f"MISSING_ARTIFACT:{tag}_{mode_name}:{an}")

        # ── Row count consistency ──
        n_steps_sh = sh_m.get("n_steps", 0)
        for csv_name in ["step_trace.csv", "action_identity.csv", "latency.csv"]:
            csv_path = os.path.join(sh_dir, csv_name)
            if os.path.exists(csv_path):
                with open(csv_path) as f:
                    n_rows = sum(1 for _ in csv.DictReader(f))
                if n_rows != n_steps_sh:
                    canary_pass = False
                    gate_failures.append(
                        f"ROW_COUNT:{tag}_sh:{csv_name}:{n_rows}!={n_steps_sh}"
                    )

        # Infra status must be "ok"
        if sh_m.get("infra_status") != "ok":
            canary_pass = False
            gate_failures.append(f"INFRA_STATUS:{tag}:{sh_m['infra_status']}")

        # Detector exception
        if sh_m.get("detector_exception"):
            canary_pass = False
            gate_failures.append(f"DETECTOR_EXCEPTION:{tag}")

        # Action identity
        if sh_m.get("action_identity_fail"):
            canary_pass = False
            gate_failures.append(f"ACTION_IDENTITY_FAIL:{tag}")

        # Sequence hashes
        for key in ["raw_action_sequence_sha256", "env_action_sequence_sha256",
                     "obs_sequence_sha256"]:
            if ref_m.get(key) != sh_m.get(key):
                canary_pass = False
                gate_failures.append(f"SEQ_{key}_MISMATCH:{tag}")

        # Steps, success, done
        for sk in ["n_steps", "success_primary", "success_done_any",
                    "success_check_any", "success_step_primary", "done_step"]:
            if ref_m.get(sk) != sh_m.get(sk):
                canary_pass = False
                gate_failures.append(f"{sk}_MISMATCH:{tag}")

        # Invalid fields
        if sh_m.get("n_invalid_field_steps", 0) > 0:
            canary_pass = False
            gate_failures.append(
                f"INVALID_FIELDS:{tag}:{sh_m['n_invalid_field_steps']}_steps"
            )

        # Detector reset
        pre = sh_m.get("detector_pre_reset", {})
        if not pre:
            canary_pass = False
            gate_failures.append(f"RESET_MISSING:{tag}")
        else:
            if pre.get("next_expected_step") != 0:
                canary_pass = False
                gate_failures.append(f"RESET_STEP:{tag}")
            if pre.get("emit_step") != -1:
                canary_pass = False
                gate_failures.append(f"RESET_EMIT:{tag}")
            if pre.get("history_len") != 0:
                canary_pass = False
                gate_failures.append(f"RESET_HISTORY:{tag}")
            if pre.get("candidate_count") != 0:
                canary_pass = False
                gate_failures.append(f"RESET_CANDIDATES:{tag}")

        # Abstain emission
        emit_step = sh_m.get("detector_emit_step", -1)
        if isinstance(emit_step, int) and emit_step >= 0:
            cand_path = os.path.join(sh_dir, "detector_candidates.csv")
            if os.path.exists(cand_path):
                found_emit_cand = False
                with open(cand_path) as f:
                    for cand in csv.DictReader(f):
                        if int(cand.get("step", -1)) == emit_step:
                            found_emit_cand = True
                            if cand.get("abstained") == "1" or cand.get("abstain", ""):
                                canary_pass = False
                                gate_failures.append(
                                    f"ABSTAIN_EMISSION:{tag}:step={emit_step}"
                                )
                if not found_emit_cand:
                    canary_pass = False
                    gate_failures.append(
                        f"EMIT_CANDIDATE_MISSING:{tag}:step={emit_step}"
                    )

        # Latency
        lat_path = os.path.join(sh_dir, "latency.csv")
        if os.path.exists(lat_path):
            det_lats = []; model_lats = []
            with open(lat_path) as f:
                for row in csv.DictReader(f):
                    du = row.get("detector_update_us", "")
                    mu = row.get("model_inference_us", "")
                    if du and du != "DISABLED": det_lats.append(int(du))
                    if mu and mu != "DISABLED": model_lats.append(int(mu))
            if not det_lats:
                canary_pass = False
                gate_failures.append(f"LATENCY_EMPTY:{tag}")
            else:
                p99 = sorted(det_lats)[int(len(det_lats) * 0.99)]
                mx = max(det_lats)
                med_det = sorted(det_lats)[len(det_lats) // 2]
                med_mod = sorted(model_lats)[len(model_lats) // 2] if model_lats else 1
                if p99 > 20000:
                    canary_pass = False
                    gate_failures.append(f"LATENCY_P99:{tag}:{p99}us")
                if mx > 50000:
                    canary_pass = False
                    gate_failures.append(f"LATENCY_MAX:{tag}:{mx}us")
                if med_mod > 0 and med_det / med_mod > 0.05:
                    canary_pass = False
                    gate_failures.append(
                        f"LATENCY_OVERHEAD:{tag}:{med_det/med_mod*100:.1f}%"
                    )
        else:
            canary_pass = False
            gate_failures.append(f"LATENCY_MISSING:{tag}")

        # Artifact hashes
        hash_path = os.path.join(sh_dir, "artifact_hashes.csv")
        if os.path.exists(hash_path):
            with open(hash_path) as f:
                for row in csv.DictReader(f):
                    an = row["artifact"]
                    ap = os.path.join(sh_dir, an)
                    if not os.path.exists(ap):
                        canary_pass = False
                        gate_failures.append(f"HASH_FILE_MISSING:{tag}:{an}")
                    else:
                        actual = sha256_file(ap)
                        if actual != row["sha256"]:
                            canary_pass = False
                            gate_failures.append(f"HASH_MISMATCH:{tag}:{an}")
        else:
            canary_pass = False
            gate_failures.append(f"HASH_MANIFEST_MISSING:{tag}")

        results.append({
            "task": task, "state_id": state_id,
            "ref_ok": ref_ok, "sh_ok": sh_ok,
            "ref_dir": ref_dir, "sh_dir": sh_dir,
            "ref_n_steps": ref_m.get("n_steps"),
            "sh_n_steps": sh_m.get("n_steps"),
            "ref_success": ref_m.get("success_primary"),
            "sh_success": sh_m.get("success_primary"),
            "sh_emit_step": sh_m.get("detector_emit_step"),
            "action_identity_ok": not sh_m.get("action_identity_fail"),
            "invalid_field_steps": sh_m.get("n_invalid_field_steps", 0),
        })

    # ── GPU cleanup gate ──
    time.sleep(3)
    gpu_after = gpu_compute_processes(target_uuids)
    assert gpu_after is not None, "FATAL: GPU query failed after canary"
    gpu_after_snapshot = os.path.join(str(out), "gpu_processes_after.csv")
    with open(gpu_after_snapshot, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gpu_uuid", "pid", "process_name"])
        w.writerows(gpu_after)

    # Compare by (uuid, pid, name) identity
    before_ids = set(gpu_before)
    after_ids = set(gpu_after)
    new_procs = after_ids - before_ids
    if new_procs:
        canary_pass = False
        gate_failures.append(f"GPU_CLEANUP:{len(new_procs)}_new_processes")
        for np_id in sorted(new_procs):
            print(f"  RESIDUAL: {np_id}")

    print(f"GPU processes: before={len(gpu_before)} after={len(gpu_after)} "
          f"new={len(new_procs)}")

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
        "gpu_visible": GPU_VISIBLE,
        "gpu_physical_ids": GPU_PHYSICAL_IDS,
        "render_gpu_physical": RENDER_GPU_PHYSICAL,
        "target_gpu_uuids": sorted(target_uuids),
        "gpu_procs_before": len(gpu_before),
        "gpu_procs_after": len(gpu_after),
        "gpu_new_processes": len(new_procs),
        "manifest_sha": manifest_sha,
        "freeze_runner_sha": actual_freeze_sha,
        "execution_head": current_head,
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
