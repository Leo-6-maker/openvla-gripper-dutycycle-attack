#!/usr/bin/env python3
"""10h overnight watcher v2 — hardened Xid, provenance, post-VIS stages."""

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

WAVE1_OUT = "/data/liuyu/outputs/object_phase_response_batch3_VIS_20260604"


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
        out = subprocess.check_output("dmesg | tail -n 200 | grep -i 'xid\\|nvrm'", shell=True, text=True)
        return out.strip().split("\n")
    except subprocess.CalledProcessError:
        return []


def gpu_idle(pair):
    a, b = pair.split(",")
    return gpu_mem(a.strip()) < 500 and gpu_mem(b.strip()) < 500


def verify_trace_metadata(trace_path, task_key, state_id, ws, we):
    if not trace_path or not os.path.exists(trace_path):
        return False, "trace_missing"
    with open(trace_path, newline="") as f:
        r0 = next(csv.DictReader(f), None)
    if r0 is None:
        return False, "trace_empty"
    checks = [
        ("task", str(r0.get("task", "")), str(task_key)),
        ("state_id", str(r0.get("state_id", "")), str(state_id)),
        ("condition", str(r0.get("condition", "")), "vis_pgd"),
        ("window_start", str(r0.get("window_start", "")), str(ws)),
        ("window_end", str(r0.get("window_end", "")), str(we)),
    ]
    failures = [f"{k}={v}!={exp}" for k, v, exp in checks if v != exp]
    if failures:
        return False, "metadata_mismatch: " + ", ".join(failures)
    return True, "ok"


class Watcher:
    def __init__(self):
        self.tasks = []
        self.running = {}
        self.blacklisted = set()
        self.seen_xid_lines = set()
        self.started_at = datetime.now()
        self.stage = "BATCH3_VIS"  # current pipeline stage
        self.audit_done = False
        self.labels_done = False
        os.makedirs(f"{OUT}/queue", exist_ok=True)
        os.makedirs(f"{OUT}/logs", exist_ok=True)
        os.makedirs(f"{OUT}/batch3_VIS", exist_ok=True)

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(f"{OUT}/queue/events.log", "a") as f:
            f.write(line + "\n")

    def check_xid_and_blacklist(self):
        xid_lines = check_xid()
        fresh = [l for l in xid_lines if l not in self.seen_xid_lines]
        if fresh:
            for line in fresh:
                self.seen_xid_lines.add(line)
                self.log(f"XID_DETECTED: {line[:120]}")
            self.log("WARNING: Fresh Xid — pausing scheduling, writing NEEDS_MANUAL_GPU_CHECK")
            self._write_stop_report("fresh_xid_detected", "\n".join(fresh))
            sys.exit(1)  # Conservative: stop on any fresh Xid

    def launch_vis(self, task):
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
        lp = f"{od}/VIS_launch.log"
        self.log(f"LAUNCH {task.task_id}: {task.task_key}_s{task.state_id} [{task.ws},{task.we}] GPU={task.gpu_pair}")
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=open(lp, "w"), stderr=subprocess.STDOUT)
        task.status = "RUNNING"
        task.started_at = datetime.now().isoformat()
        task.output_dir = od
        task.log_path = lp
        task.proc = proc
        self.running[task.gpu_pair] = task

    def check_completions(self):
        done_pairs = []
        for pair, task in list(self.running.items()):
            if task.proc is None:
                done_pairs.append(pair)
                continue
            rc = task.proc.poll()
            if rc is None:
                continue
            task.finished_at = datetime.now().isoformat()
            trace_dir = f"{task.output_dir}/traces"
            traces = list(Path(trace_dir).glob("*vis*trace.csv"))

            if rc == 0 and traces:
                trace_path = str(traces[0])
                ok, meta_reason = verify_trace_metadata(trace_path, task.task_key, task.state_id, task.ws, task.we)
                if ok:
                    task.status = "DONE"
                    self.log(f"DONE {task.task_id}: {task.task_key}_s{task.state_id}")
                else:
                    task.status = "PROVENANCE_FAILED"
                    task.failure_reason = meta_reason
                    self.log(f"PROVENANCE_FAILED {task.task_id}: {meta_reason}")
            elif rc != 0:
                log_content = ""
                if os.path.exists(task.log_path):
                    with open(task.log_path) as f:
                        log_content = f.read()[-2000:]
                if "illegal memory" in log_content or "CUDA error" in log_content:
                    task.status = "INFRA_FAILED"
                    task.failure_reason = "CUDA_illegal_memory"
                    self.blacklisted.add(pair)
                    self.log(f"INFRA_FAILED {task.task_id}: GPU {pair} blacklisted")
                elif "OutOfMemory" in log_content or "OOM" in log_content:
                    task.status = "INFRA_FAILED"
                    task.failure_reason = "CUDA_OOM"
                else:
                    task.status = "FAILED"
                    task.failure_reason = f"rc={rc}"
                if task.retry_count < MAX_RETRY and task.status != "PROVENANCE_FAILED":
                    task.retry_count += 1
                    task.status = "PENDING_RETRY"
                    for p in GPU_PAIRS:
                        if p not in self.blacklisted and p not in self.running:
                            task.gpu_pair = p
                            break
                    self.log(f"RETRY {task.task_id} on {task.gpu_pair}")
                elif task.status != "PROVENANCE_FAILED":
                    task.status = "NEEDS_MANUAL_RETRY"
            else:
                task.status = "MISSING_TRACE"
                if task.retry_count < MAX_RETRY:
                    task.retry_count += 1
                    task.status = "PENDING_RETRY"

            done_pairs.append(pair)
        for pair in done_pairs:
            del self.running[pair]

    def discover_wave1(self):
        """Scan Wave 1 output dir for completed traces."""
        if not os.path.isdir(WAVE1_OUT):
            return
        for root, dirs, files in os.walk(WAVE1_OUT):
            for f in files:
                if "vis_pgd" in f and f.endswith("_trace.csv") and "traces" in root:
                    trace_path = os.path.join(root, f)
                    # Already discovered?
                    known = [t for t in self.tasks if hasattr(t, 'wave1_discovered')]
                    if any(trace_path in getattr(t, 'wave1_path', '') for t in known):
                        continue
                    # Parse task/state from trace
                    try:
                        with open(trace_path) as tf:
                            r0 = next(csv.DictReader(tf), None)
                        if not r0:
                            continue
                        tt = r0.get("task", ""); st = r0.get("state_id", "")
                        ws = r0.get("window_start", ""); we = r0.get("window_end", "")
                        done = r0.get("done", "").lower() == "true"
                        tid = f"W1_{tt}_s{st}_w{ws}_{we}"
                        t = type('obj', (object,), {
                            'task_id': tid, 'task_type': 'BATCH3_VIS', 'priority': 0,
                            'task_key': tt, 'state_id': st, 'ws': int(ws or 0),
                            'we': int(we or 0), 'phase': '', 'gpu_pair': 'wave1',
                            'status': 'EXTERNAL_DONE' if not done else 'EXTERNAL_DONE',
                            'retry_count': 0, 'output_dir': os.path.dirname(os.path.dirname(root)),
                            'log_path': '', 'depends_on': '', 'created_at': '',
                            'started_at': '', 'finished_at': '', 'failure_reason': '',
                            'proc': None, 'wave1_discovered': True, 'wave1_path': trace_path,
                        })()
                        self.tasks.append(t)
                        self.log(f"WAVE1 discovered: {tid} status={t.status}")
                    except Exception as e:
                        self.log(f"WAVE1 parse error: {e}")

    def schedule(self):
        for pair in GPU_PAIRS:
            if pair in self.blacklisted or pair in self.running:
                continue
            if not gpu_idle(pair):
                continue
            pending = [t for t in self.tasks if t.status in ("PENDING", "PENDING_RETRY")]
            pending.sort(key=lambda t: (t.status != "PENDING_RETRY", t.priority))
            if pending:
                pending[0].gpu_pair = pair
                self.launch_vis(pending[0])

    def all_batch3_vis_terminal(self):
        vis_tasks = [t for t in self.tasks if t.task_type == "BATCH3_VIS"]
        if not vis_tasks:
            return False
        terminal_states = {"DONE", "EXTERNAL_DONE", "EXTERNAL_INFRA_FAILED",
                          "INFRA_FAILED", "PROVENANCE_FAILED", "NEEDS_MANUAL_RETRY", "FAILED"}
        return all(t.status in terminal_states for t in vis_tasks)

    def run_audit(self):
        if self.audit_done:
            return
        self.log("Running Batch3 VIS audit...")
        vis_dirs = []
        for t in self.tasks:
            if t.task_type == "BATCH3_VIS" and t.status in ("DONE", "EXTERNAL_DONE"):
                td = os.path.join(t.output_dir, "traces") if not hasattr(t, 'wave1_discovered') else os.path.dirname(t.wave1_path) if hasattr(t, 'wave1_path') else ""
                if td and os.path.isdir(td):
                    vis_dirs.append(td)
        if not vis_dirs:
            # Also scan Wave 1
            if os.path.isdir(WAVE1_OUT):
                for root, dirs, files in os.walk(WAVE1_OUT):
                    if os.path.basename(root) == "traces":
                        vis_dirs.append(root)
        if not vis_dirs:
            self.log("Audit: no VIS trace dirs found yet")
            return
        cmd = [PY, "-u", f"{REPO}/scripts/diagnostics/audit_phase_conditioned_vis.py",
               "--run-dirs"] + vis_dirs + [
               "--output-csv", f"{REPO}/tables/object_phase_response_batch3_vis_provenance.csv",
               "--summary-csv", f"{REPO}/tables/object_phase_response_batch3_vis_summary.csv"]
        self.log(f"Running audit on {len(vis_dirs)} dirs")
        result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            self.audit_done = True
            self.log("Audit PASSED")
        else:
            self.log(f"Audit FAILED: {result.stderr[:300]}")

    def all_terminal(self):
        if not self.tasks:
            return False
        terminal_states = {"DONE", "EXTERNAL_DONE", "FAILED", "INFRA_FAILED",
                          "PROVENANCE_FAILED", "NEEDS_MANUAL_RETRY", "SKIPPED"}
        return all(t.status in terminal_states for t in self.tasks)

    def write_state(self):
        rows = []
        for t in self.tasks:
            rows.append(dict(
                task_id=t.task_id, task_type=t.task_type, task_key=t.task_key,
                state_id=t.state_id, window_start=t.ws, window_end=t.we,
                status=t.status, retry_count=t.retry_count,
                output_dir=t.output_dir, failure_reason=t.failure_reason,
                started_at=t.started_at, finished_at=t.finished_at,
            ))
        with open(f"{OUT}/queue/state.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(f"{OUT}/queue/tasks_snapshot.json", "w") as f:
            json.dump(rows, f, indent=2)

    def _write_stop_report(self, reason, detail=""):
        path = f"{REPO}/reports/NIGHTLY_OBJECT_BATCH3_WATCHER_SUMMARY.md"
        n_done = sum(1 for t in self.tasks if t.status in ("DONE", "EXTERNAL_DONE"))
        n_fail = sum(1 for t in self.tasks if "FAIL" in t.status)
        with open(path, "w") as f:
            f.write(f"# Overnight Batch3 Watcher Summary\n\n")
            f.write(f"**Stop reason**: {reason}\n")
            f.write(f"**Runtime**: {datetime.now() - self.started_at}\n\n")
            f.write(f"| Metric | Value |\n|--------|-------|\n")
            f.write(f"| Total tasks | {len(self.tasks)} |\n")
            f.write(f"| Done | {n_done} |\n")
            f.write(f"| Failed/Infra | {n_fail} |\n")
            f.write(f"| Blacklisted GPUs | {self.blacklisted} |\n")
            if detail:
                f.write(f"\n## Detail\n\n```\n{detail}\n```\n")
        self.log(f"Stop report: {path}")

    def run(self):
        self.log("Watcher v2 started. Pairs: %s, Max: %dh" % (GPU_PAIRS, MAX_RUNTIME // 3600))

        # Load Batch3 VIS targets
        targets_path = f"{REPO}/tables/object_phase_response_batch3_vis_targets.csv"
        if os.path.exists(targets_path):
            with open(targets_path, newline="") as f:
                targets = list(csv.DictReader(f))
            self.log(f"Loaded {len(targets)} VIS targets")

            # Wave 1 keys (already running)
            wave1_keys = {
                ("cream_cheese", "4", 28, 45), ("milk", "4", 19, 36),
                ("salad_dressing", "0", 7, 24), ("bbq_sauce", "5", 27, 44),
            }
            tid = 0
            for t in targets:
                key = (t["task_key"], t["state_id"], int(t["window_start"]), int(t["window_end"]))
                if key in wave1_keys:
                    continue
                tid += 1
                ph = t.get("phase_bin_proxy", t.get("candidate_role", ""))
                tk = type('obj', (object,), {
                    'task_id': f"B3_VIS_{tid:03d}", 'task_type': 'BATCH3_VIS', 'priority': 10,
                    'task_key': t["task_key"], 'state_id': t["state_id"],
                    'ws': int(t["window_start"]), 'we': int(t["window_end"]),
                    'phase': ph, 'gpu_pair': '', 'status': 'PENDING',
                    'retry_count': 0, 'output_dir': '', 'log_path': '',
                    'depends_on': '', 'created_at': datetime.now().isoformat(),
                    'started_at': '', 'finished_at': '', 'failure_reason': '', 'proc': None,
                })()
                self.tasks.append(tk)
            self.log(f"Queued {tid} tasks (Wave 1 excluded)")

        # Main loop
        start = time.time()
        while time.time() - start < MAX_RUNTIME:
            self.check_xid_and_blacklist()
            self.discover_wave1()
            self.check_completions()

            if self.all_batch3_vis_terminal() and not self.audit_done:
                self.run_audit()

            if self.all_terminal():
                self.log("ALL tasks terminal.")
                break

            self.schedule()

            if len(self.blacklisted) > 2:
                self.log("STOP: >2 GPU pairs blacklisted")
                self._write_stop_report("blacklist_count_exceeded")
                break

            tnow = datetime.now().strftime("%H:%M")
            running_n = len(self.running)
            done_n = sum(1 for t in self.tasks if t.status in ("DONE", "EXTERNAL_DONE"))
            pend_n = sum(1 for t in self.tasks if t.status.startswith("PENDING"))
            self.log(f"{tnow} heartbeat: run={running_n} done={done_n} pend={pend_n} bl={len(self.blacklisted)}")
            self.write_state()
            time.sleep(POLL_SEC)

        self.write_state()
        self.log("Watcher exiting.")


if __name__ == "__main__":
    w = Watcher()
    w.run()
