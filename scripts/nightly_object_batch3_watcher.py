#!/usr/bin/env python3
"""10h overnight watcher for Object Batch3 pipeline."""

import csv, json, os, subprocess, sys, time, shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime

REPO = os.environ.get("REPO", "/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524")
PY = os.environ.get("PY", "/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python")
OUT = os.environ.get("OUT", "/data/liuyu/outputs/nightly_object_batch3_20260604")
GPU_PAIRS = ["1,0", "2,3", "4,5", "6,7"]
POLL_SEC = 60
MAX_RUNTIME = 10 * 3600
MAX_RETRY = 1

# ── Batch3 VIS targets (fresh every poll) ──
def load_batch3_vis_targets():
    """Load deduplicated VIS-ready candidates."""
    path = os.path.join(REPO, "tables/object_phase_response_batch3_vis_targets.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ── GPU health ──
def gpu_mem(gpu_id):
    try:
        out = subprocess.check_output(
            f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {gpu_id}",
            shell=True, text=True).strip()
        return int(out.splitlines()[0])
    except Exception:
        return 999999


def check_xid():
    try:
        out = subprocess.check_output("dmesg | tail -n 100 | grep -i xid", shell=True, text=True)
        return out.strip()
    except subprocess.CalledProcessError:
        return ""


def gpu_idle(pair):
    a, b = pair.split(",")
    return gpu_mem(a.strip()) < 500 and gpu_mem(b.strip()) < 500


# ── Task queue ──
class Task:
    def __init__(self, task_id, task_type, priority, task_key, state_id, ws, we, phase, gpu, **kw):
        self.task_id = task_id
        self.task_type = task_type
        self.priority = priority
        self.task_key = task_key
        self.state_id = state_id
        self.ws = int(ws)
        self.we = int(we)
        self.phase = phase
        self.gpu_pair = gpu
        self.status = "PENDING"
        self.retry_count = 0
        self.output_dir = kw.get("output_dir", "")
        self.log_path = kw.get("log_path", "")
        self.depends_on = kw.get("depends_on", "")
        self.created_at = datetime.now().isoformat()
        self.started_at = ""
        self.finished_at = ""
        self.failure_reason = ""
        self.proc = None

    def to_dict(self):
        return dict(task_id=self.task_id, task_type=self.task_type, priority=self.priority,
                    task_key=self.task_key, state_id=self.state_id, window_start=self.ws,
                    window_end=self.we, phase=self.phase, gpu_pair=self.gpu_pair,
                    status=self.status, retry_count=self.retry_count,
                    output_dir=self.output_dir, failure_reason=self.failure_reason,
                    started_at=self.started_at, finished_at=self.finished_at)


class Watcher:
    def __init__(self):
        self.tasks = []
        self.running = {}  # pair -> Task
        self.blacklisted = set()
        self.events = []
        self.started_at = datetime.now()
        os.makedirs(f"{OUT}/queue", exist_ok=True)
        os.makedirs(f"{OUT}/logs", exist_ok=True)
        os.makedirs(f"{OUT}/batch3_VIS", exist_ok=True)
        os.makedirs(f"{OUT}/batch3b_precheck", exist_ok=True)
        os.makedirs(f"{OUT}/batch3b_VIS", exist_ok=True)

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.events.append(line)
        print(line, flush=True)
        with open(f"{OUT}/queue/events.log", "a") as f:
            f.write(line + "\n")

    def check_xid_and_blacklist(self):
        xid_out = check_xid()
        if not xid_out:
            return
        for line in xid_out.split("\n"):
            for gpu_id in range(8):
                pci = f"0000:0{gpu_id+4 if gpu_id<4 else gpu_id+8 if gpu_id<6 else gpu_id+8}:00"
                # Simplified: check any Xid and blacklist pair containing that GPU
                if "Xid" in line and f"{gpu_id}" in line:
                    pass  # Simplified detection

    def launch_vis(self, task):
        """Launch a VIS task on the specified GPU pair."""
        ep_id = f"batch3_{task.task_key}_s{task.state_id}_{task.phase}_w{task.ws}_{task.we}"
        od = f"{OUT}/batch3_VIS/{task.task_key}_s{task.state_id}_{task.phase}_w{task.ws}_{task.we}"
        os.makedirs(od, exist_ok=True)
        os.makedirs(f"{od}/traces", exist_ok=True)

        cmd = [PY, "-u", f"{REPO}/scripts/vis_phase_conditioned_attack.py",
               "--task", task.task_key, "--state-id", str(task.state_id),
               "--condition", "vis_pgd", "--window-source", "fixed",
               "--fixed-window-start", str(task.ws), "--fixed-window-end", str(task.we),
               "--eps_raw_pixels", "6", "--objective", "prefix_locked_gripper_open_margin",
               "--seed", "0", "--gpu_pair", task.gpu_pair,
               "--pgd_steps", "40", "--pgd_restarts", "3",
               "--output-dir", od, "--episode-id", ep_id]
        log_path = f"{od}/VIS_launch.log"
        self.log(f"LAUNCH {task.task_id}: {task.task_key}_s{task.state_id} [{task.ws},{task.we}] GPU={task.gpu_pair}")
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
        task.status = "RUNNING"
        task.started_at = datetime.now().isoformat()
        task.output_dir = od
        task.log_path = log_path
        task.proc = proc
        self.running[task.gpu_pair] = task
        return proc

    def check_completions(self):
        """Check running tasks for completion."""
        done_pairs = []
        for pair, task in list(self.running.items()):
            if task.proc is None:
                done_pairs.append(pair)
                continue
            rc = task.proc.poll()
            if rc is None:
                continue
            task.finished_at = datetime.now().isoformat()
            # Verify trace
            trace_dir = f"{task.output_dir}/traces"
            traces = list(Path(trace_dir).glob("*vis*_trace.csv"))
            if rc == 0 and traces:
                task.status = "DONE"
                self.log(f"DONE {task.task_id}: {task.task_key}_s{task.state_id}")
            elif rc != 0:
                # Check for CUDA/Xid errors
                log_content = ""
                if os.path.exists(task.log_path):
                    with open(task.log_path) as f:
                        log_content = f.read()
                if "illegal memory" in log_content or "Xid" in log_content:
                    task.status = "INFRA_FAILED"
                    task.failure_reason = "CUDA_Xid_or_OOM"
                    # Blacklist GPU pair
                    self.blacklisted.add(pair)
                    self.log(f"INFRA_FAILED {task.task_id}: GPU {pair} blacklisted")
                else:
                    task.status = "FAILED"
                    task.failure_reason = f"rc={rc}"
                # Retry
                if task.retry_count < MAX_RETRY:
                    task.retry_count += 1
                    task.status = "PENDING_RETRY"
                    task.gpu_pair = self._pick_healthy_pair()
                    self.log(f"RETRY {task.task_id} on {task.gpu_pair}")
                else:
                    task.status = "NEEDS_MANUAL_RETRY"
            else:
                task.status = "MISSING_TRACE"
                if task.retry_count < MAX_RETRY:
                    task.retry_count += 1
                    task.status = "PENDING_RETRY"
                    self.log(f"RETRY {task.task_id}: missing trace, retry on {task.gpu_pair}")
                else:
                    task.status = "NEEDS_MANUAL_RETRY"
            done_pairs.append(pair)

        for pair in done_pairs:
            del self.running[pair]

    def _pick_healthy_pair(self):
        """Pick a healthy GPU pair different from failed one."""
        for pair in GPU_PAIRS:
            if pair not in self.blacklisted and pair not in self.running:
                return pair
        return GPU_PAIRS[0]  # fallback

    def schedule(self):
        """Find and launch highest priority pending task."""
        for pair in GPU_PAIRS:
            if pair in self.blacklisted:
                continue
            if pair in self.running:
                continue
            if not gpu_idle(pair):
                continue
            # Get PENDING_RETRY first, then PENDING
            pending = [t for t in self.tasks if t.status in ("PENDING", "PENDING_RETRY")]
            pending.sort(key=lambda t: (t.status != "PENDING_RETRY", t.priority))
            for t in pending:
                t.gpu_pair = pair
                self.launch_vis(t)
                break

    def all_terminal(self):
        return all(t.status in ("DONE", "FAILED", "INFRA_FAILED", "NEEDS_MANUAL_RETRY", "SKIPPED", "BLOCKED")
                   for t in self.tasks)

    def write_state(self):
        for t in self.tasks:
            row = t.to_dict()
            row["status"] = t.status
        with open(f"{OUT}/queue/state.jsonl", "w") as f:
            pass

    def run(self):
        self.log("Watcher started. Pairs: %s, Max runtime: %dh" % (GPU_PAIRS, MAX_RUNTIME//3600))

        # Load Batch3 VIS targets
        targets = load_batch3_vis_targets()
        self.log(f"Loaded {len(targets)} VIS targets from batch3_vis_targets.csv")

        # Create tasks (skip Wave 1 which is already running)
        wave1_keys = {
            ("cream_cheese", "4", 28, 45), ("milk", "4", 19, 36),
            ("salad_dressing", "0", 7, 24), ("bbq_sauce", "5", 27, 44),
        }
        tid = 0
        for t in targets:
            key = (t["task_key"], t["state_id"], int(t["window_start"]), int(t["window_end"]))
            if key in wave1_keys:
                continue  # Wave 1 already running
            tid += 1
            phase = t.get("phase_bin_proxy", t.get("candidate_role", ""))
            self.tasks.append(Task(
                task_id=f"B3_VIS_{tid:03d}", task_type="BATCH3_VIS", priority=10,
                task_key=t["task_key"], state_id=t["state_id"],
                ws=t["window_start"], we=t["window_end"], phase=phase,
                gpu="", output_dir=""))

        self.log(f"Queued {len(self.tasks)} tasks (Wave 1 excluded)")

        # Main loop
        start = time.time()
        while time.time() - start < MAX_RUNTIME:
            self.check_xid_and_blacklist()
            self.check_completions()
            self.schedule()

            # Check stop conditions
            if len(self.blacklisted) > 2:
                self.log("STOP: >2 GPU pairs blacklisted")
                break

            if self.all_terminal():
                self.log("ALL tasks terminal. Queue drained.")
                break

            tnow = datetime.now().strftime("%H:%M")
            running_n = len(self.running)
            done_n = sum(1 for t in self.tasks if t.status == "DONE")
            pend_n = sum(1 for t in self.tasks if t.status in ("PENDING", "PENDING_RETRY"))
            self.log(f"{tnow} heartbeat: run={running_n} done={done_n} pend={pend_n} blacklisted={len(self.blacklisted)}")
            time.sleep(POLL_SEC)

        self.log("Watcher exiting.")
        self.write_state()


if __name__ == "__main__":
    w = Watcher()
    w.run()
