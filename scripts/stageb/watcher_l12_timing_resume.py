#!/usr/bin/env python3
"""Resume watcher — fixed timeout (900s), lock, retry, unique dirs.

Resumes from partial panel. Does NOT delete existing evidence.
"""
import csv, fcntl, json, hashlib, os, subprocess, sys, time

REPO = os.environ.get("L12_REPO_ROOT", "/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605")
PY = "/data/aviary/envs/openvla_official_libero_20260525/bin/python"
GPU = "2,6"
RENDER = "6"
PANEL_DIR = "/data/liuyu/outputs/l12_timing_panel_v2"
# Write resume to separate dir to avoid overwriting original evidence
OUT_DIR = "/data/liuyu/outputs/l12_timing_panel_v2_resume_r1"
STATE_FILE = os.path.join(OUT_DIR, "resume_state.json")
LOCK_FILE = "/tmp/openvla_l12_gpu26.lock"
TIMEOUT = 900

# Already completed (from audit) — skip these
SKIP_DIRS = {
    "alphabet_soup_s2_reference_attempt1", "alphabet_soup_s2_shadow_attempt1",
    "bbq_sauce_s27_shadow_attempt1", "bbq_sauce_s40_shadow_attempt1",
    "butter_s11_shadow_attempt1", "ketchup_s18_shadow_attempt1",
    "milk_s7_shadow_attempt1", "orange_juice_s29_shadow_attempt1",
    "tomato_sauce_s23_shadow_attempt1",
}

# Remaining: 7 timing + 3 repeats
TIMING = [
    ("salad_dressing", "32", "early"),
    ("cream_cheese", "1", "early"),
    ("cream_cheese", "20", "early"),
    ("salad_dressing", "24", "early"),
    ("salad_dressing", "11", "late"),
    ("ketchup", "34", "miss"),
    ("salad_dressing", "45", "miss"),
]
REPEATS = [
    ("butter", "11"),
    ("tomato_sauce", "23"),
    ("salad_dressing", "11"),
]


def now_utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def write_state(state):
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f: json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def run_cmd(cmd, timeout=TIMEOUT):
    """Run command, return (returncode, stdout, stderr)."""
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def sha256_file(path):
    if not os.path.isfile(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def acquire_lock():
    lf = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lf
    except (IOError, OSError):
        print("FATAL: GPU lock held by another process")
        return None


def gpu_health():
    rc, out, _ = run_cmd(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"], 30)
    if rc != 0: return False
    for line in out.strip().split("\n"):
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2:
            idx = int(parts[0])
            mem = int(parts[1].replace(" MiB", ""))
            if idx in [2, 6] and mem > 200: return False
    rc, out, _ = run_cmd(["dmesg"], 30)
    known = 0
    for l in out.split("\n"):
        if "Xid" in l:
            known += 1
    # Baseline: 5 known Xids from boot (GPU0 x3 + GPU7 x1 + GPU3 x1)
    if known > 5: return False  # new Xid appeared
    return True


def run_episode(task, state_id, mode, tag, attempt_id=1):
    ep_dir = os.path.join(OUT_DIR, tag)
    # NEVER delete existing evidence
    if os.path.exists(ep_dir):
        print(f"WARNING: {ep_dir} exists — using attempt {attempt_id}")
        attempt_id += 1
        tag = f"{task}_s{state_id}_{mode}_attempt{attempt_id}"
        ep_dir = os.path.join(OUT_DIR, tag)

    env_vars = {
        "L12_REPO_ROOT": REPO, "L12_PIPELINE_ROOT": REPO,
        "CUDA_VISIBLE_DEVICES": GPU,
    }
    cmd = [
        PY, f"{REPO}/scripts/stageb/run_d4_clean_shadow.py",
        "--task", task, "--state-id", state_id, "--mode", mode,
        "--attempt-id", str(attempt_id), "--episode-dir", ep_dir,
        "--checkpoint", "outputs/d1b_training/d1b_detector_best.pt",
        "--render-gpu-device-id", RENDER, "--model-gpu-device-id", "-1",
    ]
    if mode == "shadow":
        cmd.extend([
            "--detector-mode", "d5_frozen_online_v1",
            "--d5-checkpoint", "/data/liuyu/outputs/d5_training/d5_candidate_best.pt",
            "--d5-config", "/data/liuyu/outputs/d5_training/d5_frozen_config.json",
            "--enable-privileged-sidecar",
        ])
    try:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                           env={**os.environ, **env_vars}, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT", ep_dir
    last = [l for l in r.stdout.split("\n") if "steps=" in l]
    if last: print(last[-1])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-500:] if r.stderr else "")
    return r.returncode, r.stdout, ep_dir


def verify(ep_dir, expect_success=True):
    st = os.path.join(ep_dir, "step_trace.csv")
    if not os.path.exists(st): return False, "no_trace"
    rows = list(csv.DictReader(open(st)))
    if not rows: return False, "empty"
    succ = rows[-1].get("success_done", "0")
    if expect_success and succ != "1": return False, f"success={succ}"
    return True, f"steps={len(rows)}"


def main():
    lf = acquire_lock()
    if lf is None: return 1
    os.makedirs(OUT_DIR, exist_ok=True)

    state = {"state": "RUNNING", "started": now_utc(), "episodes": {}}
    write_state(state)

    # Verify bundle
    rc, out, _ = run_cmd(
        [PY, f"{REPO}/scripts/stageb/verify_d5_v1_production_bundle.py",
         "--bundle", "configs/d5_v1_production_bundle.json"])
    if rc != 0:
        state["state"] = "FAILED"; state["reason"] = "bundle_verify"; write_state(state); return 1

    if not gpu_health():
        state["state"] = "FAILED"; state["reason"] = "gpu_health"; write_state(state); return 1

    # Run remaining timing parents
    total = len(TIMING) + len(REPEATS)
    idx = 0
    for task, sid, cat in TIMING:
        idx += 1
        tag = f"{task}_s{sid}_shadow_attempt1"
        if tag in SKIP_DIRS:
            print(f"SKIP {tag} (already done)")
            continue
        print(f"[{idx}/{total}] {tag} ({cat})")
        if not gpu_health():
            state["state"] = "FAILED"; state["reason"] = f"gpu_health_{tag}"; write_state(state); return 1
        rc, out, ep_dir = run_episode(task, sid, "shadow", tag)
        ok, msg = verify(ep_dir, cat != "miss")
        state["episodes"][tag] = {"task": task, "sid": sid, "cat": cat, "ok": ok, "msg": msg, "rc": rc}
        write_state(state)
        if rc != 0:
            # One infra retry
            print(f"  INFRA FAIL (rc={rc}), retry once...")
            time.sleep(5)
            if not gpu_health():
                state["state"] = "FAILED"; state["reason"] = f"gpu_health_retry_{tag}"; write_state(state); return 1
            retry_tag = f"{task}_s{sid}_shadow_attempt2"
            rc2, out2, ep_dir2 = run_episode(task, sid, "shadow", retry_tag, attempt_id=2)
            ok2, msg2 = verify(ep_dir2, cat != "miss")
            state["episodes"][retry_tag] = {"task": task, "sid": sid, "cat": cat + "_retry", "ok": ok2, "msg": msg2, "rc": rc2}
            write_state(state)
            if rc2 != 0:
                state["state"] = "FAILED"; state["reason"] = f"double_fail_{tag}"; write_state(state); return 1
        time.sleep(3)

    # Repeatability
    for task, sid in REPEATS:
        idx += 1
        tag = f"{task}_s{sid}_shadow_attempt2"
        print(f"[{idx}/{total}] REPEAT {tag}")
        if not gpu_health():
            state["state"] = "FAILED"; state["reason"] = f"gpu_health_{tag}"; write_state(state); return 1
        rc, out, ep_dir = run_episode(task, sid, "shadow", tag, attempt_id=2)
        ok, msg = verify(ep_dir, True)
        state["episodes"][tag] = {"task": task, "sid": sid, "cat": "repeat", "ok": ok, "msg": msg, "rc": rc}
        write_state(state)
        if rc != 0:
            state["state"] = "FAILED"; state["reason"] = f"repeat_fail_{tag}"; write_state(state); return 1
        time.sleep(3)

    state["state"] = "COMPLETE"
    state["completed"] = now_utc()
    write_state(state)
    lf.close()
    print("RESUME COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
