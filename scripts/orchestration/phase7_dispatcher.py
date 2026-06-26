#!/usr/bin/env python3
"""Phase 7 GPU Dispatcher — pidfd/polling hybrid, SQLite job queue, deterministic claim.

Monitors GPU worker processes. On exit: audit → update status → claim next eligible job.
No LLM decisions. All scheduling determined by priority + manifest order.
"""
import json, os, signal, sqlite3, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict

REPO = Path("/mnt/sdc/dty_user/openvla_attack")
DB_PATH = REPO / "evidence/orchestration/phase7_jobs.sqlite"
EVENTS_LOG = REPO / "reports/live/PHASE7_DISPATCH_EVENTS.jsonl"
GPU_SLOTS = REPO / "evidence/orchestration/gpu_slots.json"
STATUS_MD = REPO / "reports/live/PHASE7_LIVE_STATUS.md"

PYTHON = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python3"
BRIDGE = str(REPO / "scripts/stageb/run_v2_vis_sc5_mlp_bridge.py")
CHECKPOINT = str(REPO / "outputs/sc5_v2_seed42/sc5_mlp_v2.pt")
CKPT_SHA = "b679e4e072531c70511a336ed68c563cf746938f6864b3cbd14f333e4f0eb09c"
BRIDGE_SHA = "cf125d2393f2ca0a5ec1b62610b22b8d5c17733a647b9824edd2aab19995daa6"
GIT_HEAD = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"]).decode().strip()

STOP_DISPATCH = False

ENV = {
    "MUJOCO_GL": "egl", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "OPENVLA_DTYPE": "bfloat16", "OPENVLA_ATTN_IMPLEMENTATION": "eager",
    "OPENVLA_MODEL_PATH": "/mnt/sdc/dty_user/openvla_attack/models/openvla-7b-finetuned-libero-object",
    "HOME": "/mnt/sdc/dty_user/openvla_attack/sandbox_home",
    "TMPDIR": "/mnt/sdc/dty_user/openvla_attack/tmp",
    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
}

# ── Database ──
def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY, phase TEXT, priority INTEGER DEFAULT 0,
        method TEXT, cell TEXT, state_id INTEGER, perturbation_seed INTEGER,
        task_idx INTEGER, anchor INTEGER, objective TEXT, arm_lock INTEGER DEFAULT 0,
        condition TEXT DEFAULT 'TRUE_T10', trigger_step_override INTEGER,
        keep_running INTEGER DEFAULT 0, status TEXT DEFAULT 'PENDING',
        gpu_id INTEGER, pid INTEGER, attempt INTEGER DEFAULT 0,
        max_attempts INTEGER DEFAULT 2, output_dir TEXT,
        started_at TEXT, finished_at TEXT, exit_code INTEGER,
        audit_status TEXT, failure_class TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    )""")
    conn.commit()
    return conn


def log_event(event, gpu_id, job_id, old_status, new_status, **kwargs):
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(),
             "event": event, "gpu_id": gpu_id, "job_id": job_id,
             "old_status": old_status, "new_status": new_status, **kwargs}
    EVENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_LOG, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def claim_job(conn, gpu_id) -> Optional[Dict]:
    cur = conn.execute("""SELECT job_id, phase, method, cell, state_id, perturbation_seed,
        task_idx, anchor, objective, arm_lock, condition,
        trigger_step_override, keep_running, output_dir
        FROM jobs WHERE status='PENDING' ORDER BY priority ASC, created_at ASC LIMIT 1""")
    row = cur.fetchone()
    if not row: return None
    job_id = row[0]
    cur = conn.execute("UPDATE jobs SET status='CLAIMED', gpu_id=?, started_at=? WHERE job_id=? AND status='PENDING'",
                       (gpu_id, datetime.now(timezone.utc).isoformat(), job_id))
    conn.commit()
    if cur.rowcount == 0: return None
    return dict(zip(["job_id","phase","method","cell","state_id","perturbation_seed",
        "task_idx","anchor","objective","arm_lock","condition",
        "trigger_step_override","keep_running","output_dir"], row))


def launch_job(conn, job, gpu_id):
    outdir = Path(job["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **ENV, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    cmd = [PYTHON, "-u", BRIDGE, "--condition", job["condition"],
           "--state_id", str(job["state_id"]), "--anchor", str(job["anchor"]),
           "--seed_id", str(job["perturbation_seed"]), "--task_idx", str(job["task_idx"]),
           "--attack_objective", job["objective"],
           "--output_dir", str(outdir), "--render_gpu", str(gpu_id),
           "--mlp_path", CHECKPOINT, "--libero_preprocess_backend", "upstream_tf_jpeg"]
    if job["arm_lock"]: cmd.append("--arm_lock")
    if job.get("trigger_step_override") and job["trigger_step_override"] > 0:
        cmd.extend(["--trigger_step_override", str(job["trigger_step_override"])])
    if job.get("keep_running"): cmd.append("--keep_running")
    stdout = open(outdir / "stdout.log", "w")
    stderr = open(outdir / "stderr.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr,
                            preexec_fn=os.setpgrp)
    conn.execute("UPDATE jobs SET status='RUNNING', pid=?, attempt=attempt+1 WHERE job_id=?",
                 (proc.pid, job["job_id"]))
    conn.commit()
    log_event("LAUNCH", gpu_id, job["job_id"], "CLAIMED", "RUNNING", pid=proc.pid,
              message=f"{job['method']} {job['cell']} s{job['perturbation_seed']}")
    return proc


def audit_job(job_id, output_dir):
    outdir = Path(output_dir)
    summary = outdir / "episode_summary.json"
    done = outdir / ".done"
    if not summary.exists(): return "FAILED_RETRYABLE", "missing_summary"
    if not done.exists(): return "FAILED_RETRYABLE", "missing_done"
    try:
        s = json.load(open(summary))
    except Exception:
        return "QUARANTINED", "corrupt_summary"
    if s.get("checkpoint_sha256", "") != CKPT_SHA: return "QUARANTINED", "sha"
    if s.get("preprocess_backend_resolved", "") != "upstream_tf_jpeg": return "QUARANTINED", "backend"
    if s.get("manual_anchor_used", False): return "QUARANTINED", "manual_anchor"
    if s.get("privileged_detector_input_used", False): return "QUARANTINED", "privileged"
    if s.get("invalid_feature_steps", 1) != 0: return "QUARANTINED", "invalid_feat"
    emit = s.get("mlp_emit_step", -1) or -1
    expected_atk = 10 if emit >= 0 else 0
    if s.get("attack_frames", -1) != expected_atk: return "QUARANTINED", "atk_frames"
    return "SUCCESS", None


def update_slot(gpu_id, state, job_id=None, pid=None):
    slots = json.load(open(GPU_SLOTS)) if GPU_SLOTS.exists() else {}
    slots[str(gpu_id)] = {"gpu_id": gpu_id, "state": state, "current_job_id": job_id,
                          "pid": pid, "updated_at": datetime.now(timezone.utc).isoformat()}
    GPU_SLOTS.parent.mkdir(parents=True, exist_ok=True)
    json.dump(slots, open(GPU_SLOTS, "w"), indent=2)


def handle_exit(conn, gpu_id, job_id, output_dir, exit_code):
    global STOP_DISPATCH
    print(f"[{datetime.now():%H:%M:%S}] GPU{gpu_id} exit: {job_id} code={exit_code}")
    if not output_dir:
        # Legacy worker without known output dir — skip audit, just claim next
        next_job = claim_job(conn, gpu_id)
        if next_job:
            proc = launch_job(conn, next_job, gpu_id)
            return proc
        return None

    log_event("EXIT", gpu_id, job_id, "RUNNING", "AUDITING", exit_code=exit_code)
    status, reason = audit_job(job_id, output_dir)
    conn.execute("UPDATE jobs SET status=?, exit_code=?, audit_status=?, failure_class=?, finished_at=? WHERE job_id=?",
                 (status, exit_code, status, reason or "", datetime.now(timezone.utc).isoformat(), job_id))
    conn.commit()
    log_event("AUDIT", gpu_id, job_id, "AUDITING", status, message=str(reason))

    if status == "QUARANTINED":
        STOP_DISPATCH = True
        update_slot(gpu_id, "STOPPED_BY_INCIDENT")
        print(f"STOP-DISPATCH: {job_id} QUARANTINED: {reason}")
        return

    if status == "FAILED_RETRYABLE":
        cur = conn.execute("SELECT attempt FROM jobs WHERE job_id=?", (job_id,))
        row = cur.fetchone()
        if row and row[0] < 2:
            conn.execute("UPDATE jobs SET status='PENDING' WHERE job_id=?", (job_id,))
            conn.commit()
            log_event("RETRY", gpu_id, job_id, "FAILED", "PENDING", message=f"attempt {row[0]}")
        else:
            conn.execute("UPDATE jobs SET status='QUARANTINED' WHERE job_id=?", (job_id,))
            conn.commit()
            STOP_DISPATCH = True
            return

    # Claim next
    next_job = claim_job(conn, gpu_id)
    if next_job:
        proc = launch_job(conn, next_job, gpu_id)
        update_slot(gpu_id, "RUNNING", job_id=next_job["job_id"], pid=proc.pid)
        return proc
    else:
        update_slot(gpu_id, "IDLE_WAITING_FOR_JOB")
        log_event("IDLE", gpu_id, "", "", "IDLE", message="No PENDING jobs")
        return None


def import_manifest(conn, jobs_list):
    """Batch-import jobs, skip existing."""
    for j in jobs_list:
        try:
            conn.execute("""INSERT OR IGNORE INTO jobs (job_id, phase, priority, method, cell,
                state_id, perturbation_seed, task_idx, anchor, objective, arm_lock,
                condition, trigger_step_override, keep_running, output_dir, status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (j["job_id"], j["phase"], j["priority"], j["method"], j["cell"],
                 j["state_id"], j["perturbation_seed"], j["task_idx"], j["anchor"],
                 j["objective"], j["arm_lock"], j.get("condition", "TRUE_T10"),
                 j.get("trigger_step_override"), j.get("keep_running", 0),
                 j["output_dir"], j.get("status", "PENDING")))
        except Exception as e:
            print(f"Import error {j['job_id']}: {e}")
    conn.commit()


def generate_phaseA_jobs():
    """Generate all 87 Phase A jobs."""
    cells = [
        ("salad_dressing_s0", 2, 0, 84), ("bbq_sauce_s0", 3, 0, 128),
        ("ketchup_s0", 4, 0, 95), ("milk_s4", 7, 4, 92),
        ("butter_s2", 6, 2, 100), ("alphabet_soup_s0", 0, 0, 86),
        ("orange_juice_s0", 9, 0, 167), ("butter_s0", 6, 0, 85),
        ("tomato_sauce_s0", 5, 0, 176),
    ]
    noemit_cells = [("cream_cheese_s0", 1, 0, 116), ("chocolate_pudding_s2", 8, 2, 90)]
    seeds_2x2 = [42, 123, 456, 789, 2026]
    seeds_noemit = [42, 123, 456]

    jobs = []
    base = REPO / "evidence/phase7_object/supplement_7h"

    # TMA no-lock: seeds 789, 2026
    for cell, task, state, anchor in cells:
        for seed in [789, 2026]:
            jobs.append({"job_id": f"phaseA_tma_nolock_{cell}_s{seed}", "phase": "A", "priority": 1,
                "method": "TMA_no_lock", "cell": cell, "state_id": state,
                "perturbation_seed": seed, "task_idx": task, "anchor": anchor,
                "objective": "vanilla_tma_gripper_open_ce", "arm_lock": 0,
                "output_dir": str(base / f"phaseA_tma_nolock/{cell}_s{seed}")})

    # TMA ArmLock: seeds 789, 2026
    for cell, task, state, anchor in cells:
        for seed in [789, 2026]:
            jobs.append({"job_id": f"phaseA_tma_armlock_{cell}_s{seed}", "phase": "A", "priority": 1,
                "method": "TMA_ArmLock", "cell": cell, "state_id": state,
                "perturbation_seed": seed, "task_idx": task, "anchor": anchor,
                "objective": "vanilla_tma_gripper_open_ce", "arm_lock": 1,
                "output_dir": str(base / f"phaseA_tma_armlock/{cell}_s{seed}")})

    # Ours ArmLock: seeds 789, 2026
    for cell, task, state, anchor in cells:
        for seed in [789, 2026]:
            jobs.append({"job_id": f"phaseA_ours_armlock_{cell}_s{seed}", "phase": "A", "priority": 1,
                "method": "Ours_ArmLock", "cell": cell, "state_id": state,
                "perturbation_seed": seed, "task_idx": task, "anchor": anchor,
                "objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
                "arm_lock": 1, "output_dir": str(base / f"phaseA_ours_armlock/{cell}_s{seed}")})

    # Ours no-lock: seed 2026 only (789 already done before)
    for cell, task, state, anchor in cells:
        jobs.append({"job_id": f"phaseA_ours_nolock_{cell}_s2026", "phase": "A", "priority": 1,
            "method": "Ours_no_lock", "cell": cell, "state_id": state,
            "perturbation_seed": 2026, "task_idx": task, "anchor": anchor,
            "objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
            "arm_lock": 0, "output_dir": str(base / f"phaseA_ours_nolock/{cell}_s2026")})

    # RAND: seeds 789, 2026
    for cell, task, state, anchor in cells:
        for seed in [789, 2026]:
            jobs.append({"job_id": f"phaseA_rand_{cell}_s{seed}", "phase": "A", "priority": 1,
                "method": "RAND", "cell": cell, "state_id": state,
                "perturbation_seed": seed, "task_idx": task, "anchor": anchor,
                "objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
                "arm_lock": 0, "condition": "RAND_T10",
                "output_dir": str(base / f"phaseA_rand/{cell}_s{seed}")})

    # No-emit formal
    for cell, task, state, anchor in noemit_cells:
        for seed in seeds_noemit:
            jobs.append({"job_id": f"phaseA_noemit_{cell}_s{seed}", "phase": "A", "priority": 1,
                "method": "Ours_ArmLock_noemit", "cell": cell, "state_id": state,
                "perturbation_seed": seed, "task_idx": task, "anchor": anchor,
                "objective": "autoregressive_prefix_gripper_target_token_logratio_arm_v3",
                "arm_lock": 1, "output_dir": str(base / f"phaseA_ours_armlock_noemit/{cell}_s{seed}")})
    return jobs


def generate_phaseB_jobs():
    """Generate all 54 Phase B TMA timing jobs."""
    cells = [
        ("salad_dressing_s0", 2, 0, 84), ("bbq_sauce_s0", 3, 0, 128),
        ("ketchup_s0", 4, 0, 95), ("milk_s4", 7, 4, 92),
        ("butter_s2", 6, 2, 100), ("alphabet_soup_s0", 0, 0, 86),
        ("orange_juice_s0", 9, 0, 167), ("butter_s0", 6, 0, 85),
        ("tomato_sauce_s0", 5, 0, 176),
    ]
    random_windows = {"salad_dressing_s0": 228, "bbq_sauce_s0": 210, "ketchup_s0": 250,
        "milk_s4": 230, "butter_s2": 240, "alphabet_soup_s0": 258,
        "orange_juice_s0": 280, "butter_s0": 220, "tomato_sauce_s0": 201}
    base = REPO / "evidence/phase7_object/supplement_7h"
    jobs = []

    for seed in [42, 123, 456]:
        # TMA Random-Time
        for cell, task, state, anchor in cells:
            trigger = random_windows[cell]
            jobs.append({"job_id": f"phaseB_tma_random_{cell}_s{seed}", "phase": "B", "priority": 5,
                "method": "TMA_RandomTime", "cell": cell, "state_id": state,
                "perturbation_seed": seed, "task_idx": task, "anchor": anchor,
                "objective": "vanilla_tma_gripper_open_ce", "arm_lock": 0,
                "trigger_step_override": trigger, "keep_running": 1,
                "output_dir": str(base / f"phaseB_tma_random/{cell}_tma_random_s{seed}")})

        # TMA Early-Shift
        for cell, task, state, anchor in cells:
            trigger = anchor - 20
            if trigger < 0: continue
            jobs.append({"job_id": f"phaseB_tma_early_{cell}_s{seed}", "phase": "B", "priority": 5,
                "method": "TMA_EarlyShift", "cell": cell, "state_id": state,
                "perturbation_seed": seed, "task_idx": task, "anchor": anchor,
                "objective": "vanilla_tma_gripper_open_ce", "arm_lock": 0,
                "trigger_step_override": trigger,
                "output_dir": str(base / f"phaseB_tma_early/{cell}_tma_early_s{seed}")})
    return jobs


def sync_status(conn):
    """Mark completed jobs as SUCCESS, running jobs as RUNNING."""
    cur = conn.execute("SELECT job_id, output_dir FROM jobs WHERE status IN ('PENDING','CLAIMED','RUNNING')")
    for job_id, outdir in cur.fetchall():
        done = Path(outdir) / ".done"
        if done.exists():
            conn.execute("UPDATE jobs SET status='SUCCESS' WHERE job_id=? AND status!='SUCCESS'", (job_id,))
    conn.commit()

    # Count by status
    for status in ["SUCCESS","RUNNING","PENDING","CLAIMED","QUARANTINED"]:
        count = conn.execute("SELECT COUNT(*) FROM jobs WHERE status=?", (status,)).fetchone()[0]
        if count > 0:
            print(f"  {status}: {count}")


def main():
    global STOP_DISPATCH
    conn = init_db()

    # Import manifests
    all_jobs = generate_phaseA_jobs() + generate_phaseB_jobs()
    import_manifest(conn, all_jobs)
    print(f"Imported {len(all_jobs)} jobs (Phase A: {len(generate_phaseA_jobs())}, Phase B: {len(generate_phaseB_jobs())})")

    # Sync: mark completed from disk
    sync_status(conn)

    # Attach to existing workers
    workers = []
    out = subprocess.check_output(["ps", "-eo", "pid,cmd"]).decode()
    for line in out.split("\n"):
        if "python" not in line or "bridge" not in line: continue
        try:
            pid = int(line.split()[0])
            gpu = "?"
            try:
                with open(f"/proc/{pid}/environ") as f:
                    for e in f.read().split("\0"):
                        if "CUDA_VISIBLE_DEVICES" in e:
                            gpu = e.split("=")[1].strip()
            except: pass
            outdir = ""
            for i, p in enumerate(line.split()):
                if p == "--output_dir": outdir = line.split()[i + 1]
            workers.append({"pid": pid, "gpu": int(gpu) if gpu.isdigit() else -1, "outdir": outdir})
        except: pass

    print(f"\nAttached {len(workers)} existing workers")
    for w in workers:
        update_slot(w["gpu"], "RUNNING", pid=w["pid"])
        print(f"  GPU{w['gpu']} PID={w['pid']}")

    # Main loop: poll every 15s using kill -0 for liveness
    print("\nDispatcher running. Ctrl+C to stop.\n")
    cycle = 0
    while not STOP_DISPATCH:
        time.sleep(15)
        cycle += 1

    for w in workers[:]:
        # Try to find the actual output dir for this worker
        job_outdir = w["outdir"]
        if not job_outdir and w["pid"] > 0:
            try:
                with open(f"/proc/{w['pid']}/cmdline", "rb") as f:
                    cmdline = f.read().decode(errors="replace").replace("\0", " ")
                for i, p in enumerate(cmdline.split()):
                    if p == "--output_dir" and i+1 < len(cmdline.split()):
                        job_outdir = cmdline.split()[i+1]
                        w["outdir"] = job_outdir
            except: pass

        try:
            os.kill(w["pid"], 0)
        except ProcessLookupError:
            try:
                job_id = f"legacy_gpu{w['gpu']}"
                new_proc = handle_exit(conn, w["gpu"], job_id, job_outdir, -1)
                workers.remove(w)
                if new_proc:
                    workers.append({"pid": new_proc.pid, "gpu": w["gpu"], "outdir": ""})
                update_slot(w["gpu"], "RUNNING" if new_proc else "IDLE_WAITING_FOR_JOB",
                            pid=new_proc.pid if new_proc else None)
            except Exception as e:
                print(f"[{datetime.now():%H:%M:%S}] ERROR handling exit GPU{w['gpu']}: {e}")
        except PermissionError:
            pass

        # Scan idle GPUs
        busy_gpus = {w["gpu"] for w in workers if w["gpu"] >= 0}
        for gpu_id in range(7):
            if gpu_id in busy_gpus: continue
            has_process = False
            try:
                gpu_pids = subprocess.check_output(
                    ["nvidia-smi", "-i", str(gpu_id), "--query-compute-apps=pid", "--format=csv,noheader"],
                    timeout=5).decode().strip()
                for p in gpu_pids.split("\n"):
                    if p.strip() and p.strip().isdigit():
                        try: os.kill(int(p.strip()), 0); has_process = True; break
                        except: pass
            except: pass
            if has_process: continue

            try:
                next_job = claim_job(conn, gpu_id)
                if next_job:
                    proc = launch_job(conn, next_job, gpu_id)
                    workers.append({"pid": proc.pid, "gpu": gpu_id, "outdir": ""})
                    update_slot(gpu_id, "RUNNING", job_id=next_job["job_id"], pid=proc.pid)
                    print(f"[{datetime.now():%H:%M:%S}] GPU{gpu_id} idle → launched {next_job['job_id']} PID={proc.pid}")
            except Exception as e:
                print(f"[{datetime.now():%H:%M:%S}] ERROR launching on GPU{gpu_id}: {e}")

        if cycle % 4 == 0:  # every 60s
            pending = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='PENDING'").fetchone()[0]
            running = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='RUNNING'").fetchone()[0]
            success = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='SUCCESS'").fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            print(f"[{datetime.now():%H:%M:%S}] jobs: {success}/{total} done, {running} running, {pending} pending, {len(workers)} workers")

    conn.close()
    print("Dispatcher stopped.")


if __name__ == "__main__":
    main()
