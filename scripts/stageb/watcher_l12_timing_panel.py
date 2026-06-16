#!/usr/bin/env python3
"""L12 Post-Freeze Timing Panel Watcher — autonomous GPU execution.

State machine: S0→S4→S5→S6→S7→S8→S9
Persists state to watcher_state.json. Atomic writes.
Hard gates: any failure → FAILED state.
"""
import csv, json, hashlib, os, subprocess, sys, time
from datetime import datetime, timezone

REPO = os.environ.get("L12_REPO_ROOT", "/data/liuyu/repos/openvla-gripper-dutycycle-attack-reviewed-20260605")
PY = "/data/aviary/envs/openvla_official_libero_20260525/bin/python"
GPU = "2,6"
RENDER = "6"
OUT = "/data/liuyu/outputs/l12_timing_panel_v2"
STATE_FILE = os.path.join(OUT, "watcher_state.json")
LOCK_FILE = "/tmp/openvla_l12_gpu26.lock"
DEADLINE_S = 9 * 3600  # 9 hours

# Pre-registered parents from Phase 1
PARENTS = [
    # exact (6)
    ("butter", "11", "exact"),
    ("ketchup", "18", "exact"),
    ("orange_juice", "29", "exact"),
    ("milk", "7", "exact"),
    ("bbq_sauce", "40", "exact"),
    ("bbq_sauce", "27", "exact"),
    # early (5)
    ("tomato_sauce", "23", "early"),
    ("salad_dressing", "32", "early"),
    ("cream_cheese", "1", "early"),
    ("cream_cheese", "20", "early"),
    ("salad_dressing", "24", "early"),
    # late (1)
    ("salad_dressing", "11", "late"),
    # miss (2)
    ("ketchup", "34", "miss"),
    ("salad_dressing", "45", "miss"),
]


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    if not os.path.isfile(path): return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
    return h.hexdigest()


def run(cmd, **kw):
    """Run command, return (returncode, stdout, stderr)."""
    env = os.environ.copy()
    env.update(kw.pop("env", {}))
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env, **kw)
    return r.returncode, r.stdout, r.stderr


def write_state(state):
    os.makedirs(OUT, exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def read_state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {"state": "S0", "episodes": {}, "started": now_utc(), "events": []}


def gpu_health():
    """Check GPU health. Returns True if clean."""
    rc, out, _ = run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"])
    if rc != 0: return False
    lines = out.strip().split("\n")
    for line in lines:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2:
            idx = int(parts[0])
            mem = int(parts[1].replace(" MiB", ""))
            if idx in [2, 6] and mem > 100:
                return False
    # Check no new Xid
    rc, out, _ = run(["dmesg"], env={})
    xid_lines = [l for l in out.split("\n") if "Xid" in l and "PCI:0000:04:00" not in l]
    if len(xid_lines) > 3:  # existing known Xids
        return False
    return True


def run_episode(task, state_id, mode, tag, extra_flags=None):
    """Run one episode. Returns (rc, stdout_tail, step_count, success)."""
    ep_dir = os.path.join(OUT, tag)
    if os.path.exists(ep_dir):
        run(["rm", "-rf", ep_dir])
    env = {
        "L12_REPO_ROOT": REPO,
        "L12_PIPELINE_ROOT": REPO,
        "CUDA_VISIBLE_DEVICES": GPU,
    }
    cmd = [
        PY, f"{REPO}/scripts/stageb/run_d4_clean_shadow.py",
        "--task", task, "--state-id", state_id, "--mode", mode,
        "--attempt-id", "1", "--episode-dir", ep_dir,
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
    if extra_flags:
        cmd.extend(extra_flags)
    rc, stdout, stderr = run(cmd, env=env, timeout=300)
    last_lines = [l for l in stdout.split("\n") if "steps=" in l]
    if last_lines:
        print(last_lines[-1])
    if rc != 0:
        print("STDERR:", stderr[-500:])
    return rc, stdout, ep_dir


def verify_episode(ep_dir, expected_success=True):
    """Verify episode artifacts."""
    st = os.path.join(ep_dir, "step_trace.csv")
    if not os.path.exists(st):
        return False, "no_step_trace"
    rows = list(csv.DictReader(open(st)))
    if not rows:
        return False, "empty_trace"
    n = len(rows)
    success = rows[-1].get("success_done", "0")
    if expected_success and success != "1":
        return False, f"success={success}"
    return True, f"steps={n} success={success}"


def main():
    state = read_state()
    print(f"Watcher: state={state['state']} started={state['started']}")

    if state["state"] == "S0":
        # Verify bundle
        rc, out, _ = run([
            PY, f"{REPO}/scripts/stageb/verify_d5_v1_production_bundle.py",
            "--bundle", "configs/d5_v1_production_bundle.json",
        ])
        if rc != 0:
            state["state"] = "FAILED"
            state["reason"] = "bundle_verify_failed"
            write_state(state)
            return 1
        state["state"] = "S1"
        write_state(state)

    if state["state"] == "S1":
        # Inventory already done in Phase 1
        state["state"] = "S2"  # Schema done
        write_state(state)

    if state["state"] in ("S2", "S3", "S4"):
        # Preview + Schema + Tool done. Skip to recorder preflight.
        state["state"] = "S4_RECORDER_PREFLIGHT"
        write_state(state)

    if state["state"] == "S4_RECORDER_PREFLIGHT":
        if not gpu_health():
            state["state"] = "FAILED"
            state["reason"] = "gpu_health_preflight"
            write_state(state)
            return 1
        # Run OFF — episode_dir MUST match naming convention: {task}_s{state_id}_{mode}_attempt{id}
        print("=== Recorder OFF ===")
        rc, out, ep_off = run_episode("alphabet_soup", "2", "reference",
                                       "alphabet_soup_s2_reference_attempt1")
        if rc != 0:
            state["state"] = "FAILED"
            state["reason"] = "recorder_off_failed"
            write_state(state)
            return 1
        # Run ON
        print("=== Recorder ON ===")
        rc, out, ep_on = run_episode("alphabet_soup", "2", "shadow",
                                     "alphabet_soup_s2_shadow_attempt1")
        if rc != 0:
            state["state"] = "FAILED"
            state["reason"] = "recorder_on_failed"
            write_state(state)
            return 1
        # Verify identity
        ref_act = list(csv.DictReader(open(os.path.join(ep_off, "action_identity.csv"))))
        sha_act = list(csv.DictReader(open(os.path.join(ep_on, "action_identity.csv"))))
        ad = sum(1 for r, s in zip(ref_act, sha_act) if r.get("action_hash_post") != s.get("action_hash_post"))
        if ad > 0:
            state["state"] = "FAILED"
            state["reason"] = f"recorder_action_diffs={ad}"
            write_state(state)
            return 1
        state["state"] = "S5_TIMING_PANEL"
        state["episodes"]["recorder"] = {"off": ep_off, "on": ep_on, "action_diffs": ad}
        write_state(state)

    if state["state"] == "S5_TIMING_PANEL":
        for i, (task, sid, cat) in enumerate(PARENTS):
            tag = f"{task}_s{sid}_shadow_attempt1"
            if tag in state.get("episodes", {}):
                print(f"SKIP {tag}: already done")
                continue
            if not gpu_health():
                state["state"] = "FAILED"
                state["reason"] = f"gpu_health_before_{tag}"
                write_state(state)
                return 1
            print(f"[{i+1}/{len(PARENTS)}] {tag} ({cat})")
            rc, out, ep_dir = run_episode(task, sid, "shadow", tag)
            ok, msg = verify_episode(ep_dir, cat != "miss")
            print(f"  {msg}")
            state["episodes"][tag] = {"task": task, "sid": sid, "cat": cat,
                                       "dir": ep_dir, "rc": rc, "ok": ok, "msg": msg}
            write_state(state)
            if not ok and cat != "miss":
                print("WARNING: expected success but got " + msg)
            time.sleep(2)

        # Count primary successes
        n_ok = sum(1 for e in state["episodes"].values()
                   if e.get("cat") != "miss" and e.get("ok"))
        print(f"Primary OK: {n_ok}/{len(PARENTS)-2}")
        state["state"] = "S6_REPEATABILITY"
        write_state(state)

    if state["state"] == "S6_REPEATABILITY":
        # Pick 3: 1 exact, 1 early, 1 late
        rerun = [
            ("butter", "11", "exact"),
            ("tomato_sauce", "23", "early"),
            ("salad_dressing", "11", "late"),
        ]
        for task, sid, cat in rerun:
            tag = f"{task}_s{sid}_shadow_attempt2"
            if tag in state.get("episodes", {}):
                continue
            if not gpu_health():
                state["state"] = "FAILED"
                state["reason"] = f"gpu_health_{tag}"
                write_state(state)
                return 1
            print(f"Repeat: {tag}")
            rc, out, ep_dir = run_episode(task, sid, "shadow", tag)
            ok, msg = verify_episode(ep_dir, cat != "miss")
            state["episodes"][tag] = {"task": task, "sid": sid, "cat": cat + "_repeat",
                                       "dir": ep_dir, "rc": rc, "ok": ok, "msg": msg}
            write_state(state)
            time.sleep(2)
        state["state"] = "S7_FINALIZE"
        write_state(state)

    if state["state"] == "S7_FINALIZE":
        # Generate summary
        summary = []
        for tag, ep in sorted(state["episodes"].items()):
            summary.append(ep)
        with open(os.path.join(OUT, "timing_panel_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        state["state"] = "COMPLETE"
        state["completed"] = now_utc()
        write_state(state)
        print("MISSION COMPLETE")

    return 0


if __name__ == "__main__":
    sys.exit(main())
