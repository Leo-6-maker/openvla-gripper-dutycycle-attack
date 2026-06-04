#!/usr/bin/env python3
"""10h overnight watcher v2.2 — auto-requeue recoverable failures, full pipeline stages.

v2.2: Xid GPU parsing, auto-requeue, retry bookkeeping, label merge, detector v2, final summary.
"""

import csv, json, os, re, subprocess, sys, time
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

# PCI bus to GPU ID mapping (from nvidia-smi earlier logs)
PCI_TO_GPU = {
    "0000:04:00": 0, "0000:06:00": 1, "0000:07:00": 2, "0000:08:00": 3,
    "0000:0C:00": 4, "0000:0D:00": 5, "0000:0E:00": 6, "0000:0F:00": 7,
}


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
        out = subprocess.check_output("dmesg | tail -n 300 | grep -i 'xid\\|nvrm'", shell=True, text=True)
        return [l.strip() for l in out.strip().split("\n") if l.strip()]
    except subprocess.CalledProcessError:
        return []


def parse_xid_gpu(line):
    """Extract GPU id from Xid line using PCI address."""
    m = re.search(r"PCI:0000:([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})", line)
    if m:
        pci = f"0000:{m.group(1)}:{m.group(2)}"
        return PCI_TO_GPU.get(pci)
    return None


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
    eps = r0.get("eps_raw_pixels", "")
    if eps and str(eps) != "6":
        checks.append(("eps", str(eps), "6"))
    obj = r0.get("objective", "")
    if obj and "prefix_locked_gripper_open_margin" not in obj:
        checks.append(("objective", obj[:40], "prefix_locked_gripper_open_margin"))
    failures = [f"{k}={v}!={exp}" for k, v, exp in checks if v != exp]
    return (False, "metadata_mismatch: " + ", ".join(failures)) if failures else (True, "ok")


class Task:
    __slots__ = ("task_id","task_type","priority","task_key","state_id","ws","we","phase",
                 "gpu_pair","status","retry_count","previous_gpu_pairs","output_dir",
                 "log_path","created_at","started_at","finished_at","failure_reason",
                 "infra_status","last_rc","provenance_status","trace_path","proc")
    def __init__(self, tid, ttype, pri, key, st, ws, we, ph, gpu, **kw):
        self.task_id = tid; self.task_type = ttype; self.priority = pri
        self.task_key = key; self.state_id = st; self.ws = int(ws); self.we = int(we)
        self.phase = ph; self.gpu_pair = gpu
        self.status = "PENDING"; self.retry_count = 0; self.previous_gpu_pairs = []
        self.output_dir = ""; self.log_path = ""; self.created_at = datetime.now().isoformat()
        self.started_at = ""; self.finished_at = ""; self.failure_reason = ""
        self.infra_status = ""; self.last_rc = ""; self.provenance_status = ""
        self.trace_path = ""; self.proc = None

    def to_dict(self):
        return {k: str(getattr(self, k, "")) for k in [
            "task_id","task_type","priority","task_key","state_id","ws","we","phase",
            "gpu_pair","status","retry_count","previous_gpu_pairs","output_dir",
            "log_path","created_at","started_at","finished_at","failure_reason",
            "infra_status","last_rc","provenance_status","trace_path",
        ]}


class Watcher:
    def __init__(self):
        self.tasks = []
        self.running = {}
        self.blacklisted = set()
        self.seen_xid_lines = set(check_xid())
        self.started_at = datetime.now()
        self.audit_done = False
        self.labels_done = False
        self.stages_run = set()
        if self.seen_xid_lines:
            print(f"[INIT] Seeded {len(self.seen_xid_lines)} existing Xid baseline lines")
        for d in ["queue","logs","batch3_VIS"]:
            os.makedirs(f"{OUT}/{d}", exist_ok=True)

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with open(f"{OUT}/queue/events.log", "a") as f:
            f.write(line + "\n")

    def check_xid_and_handle(self):
        fresh = [l for l in check_xid() if l not in self.seen_xid_lines]
        if not fresh:
            return True  # no new Xid
        for line in fresh:
            self.seen_xid_lines.add(line)
            gpu_id = parse_xid_gpu(line)
            if gpu_id is not None:
                self.log(f"FRESH_XID GPU={gpu_id}: {line[:120]}")
                for pair in list(GPU_PAIRS):
                    if str(gpu_id) in pair.split(","):
                        self.blacklisted.add(pair)
                        self.log(f"BLACKLISTED pair {pair} due to GPU{gpu_id} Xid")
                        if pair in self.running:
                            t = self.running[pair]
                            t.status = "INFRA_FAILED"; t.failure_reason = f"Xid_GPU{gpu_id}"
                            t.infra_status = "xid"
                            if t.retry_count < MAX_RETRY:
                                new_pair = self.choose_healthy_pair(exclude=pair)
                                if new_pair:
                                    t.retry_count += 1
                                    t.previous_gpu_pairs.append(pair)
                                    t.gpu_pair = new_pair
                                    t.status = "PENDING_RETRY"
                                    self.log(f"REQUEUE {t.task_id} to {new_pair}")
                                    del self.running[pair]
                            else:
                                t.status = "NEEDS_MANUAL_RETRY"
                                del self.running[pair]
            else:
                self.log(f"FRESH_XID_UNPARSED: {line[:120]}. Pausing scheduler.")
                return False  # conservative pause
        return True

    def choose_healthy_pair(self, exclude=None):
        for pair in GPU_PAIRS:
            if pair in self.blacklisted or pair in self.running:
                continue
            if pair == exclude:
                continue
            if gpu_idle(pair):
                return pair
        return None

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
        task.status = "RUNNING"; task.started_at = datetime.now().isoformat()
        task.output_dir = od; task.log_path = lp; task.proc = proc
        self.running[task.gpu_pair] = task

    def check_completions(self):
        done_pairs = []
        for pair, task in list(self.running.items()):
            if task.proc is None:
                done_pairs.append(pair); continue
            rc = task.proc.poll()
            if rc is None:
                continue
            task.finished_at = datetime.now().isoformat(); task.last_rc = str(rc)
            trace_dir = f"{task.output_dir}/traces"
            traces = list(Path(trace_dir).glob("*vis*trace.csv"))
            if rc == 0 and traces:
                tp = str(traces[0]); task.trace_path = tp
                ok, reason = verify_trace_metadata(tp, task.task_key, task.state_id, task.ws, task.we)
                if ok:
                    task.status = "DONE"; task.provenance_status = "verified"
                    self.log(f"DONE {task.task_id}: {task.task_key}_s{task.state_id}")
                else:
                    task.status = "PROVENANCE_FAILED"; task.failure_reason = reason; task.provenance_status = "failed"
                    self.log(f"PROVENANCE_FAILED {task.task_id}: {reason}")
            elif rc == 0 and not traces:
                task.status = "MISSING_TRACE"; task.failure_reason = "no_trace_rc0"
                if task.retry_count < MAX_RETRY:
                    self._requeue(task, pair, "trace_missing")
                else:
                    task.status = "NEEDS_MANUAL_RETRY"
            elif rc != 0:
                log_content = ""
                if os.path.exists(task.log_path):
                    with open(task.log_path) as f:
                        log_content = f.read()[-2000:]
                if "illegal memory" in log_content or "CUDA error" in log_content:
                    task.status = "INFRA_FAILED"; task.failure_reason = "CUDA_illegal_memory"; task.infra_status = "cuda_crash"
                    self.blacklisted.add(pair)
                    self.log(f"INFRA CUDA crash {task.task_id}: GPU {pair} blacklisted")
                    if task.retry_count < MAX_RETRY:
                        self._requeue(task, pair, "cuda_crash")
                    else:
                        task.status = "NEEDS_MANUAL_RETRY"
                elif "OutOfMemory" in log_content or "OOM" in log_content:
                    task.status = "INFRA_FAILED"; task.failure_reason = "CUDA_OOM"; task.infra_status = "oom"
                    if task.retry_count < MAX_RETRY:
                        self._requeue(task, pair, "oom")
                    else:
                        task.status = "NEEDS_MANUAL_RETRY"
                else:
                    task.status = "FAILED"; task.failure_reason = f"rc={rc}"
                    if task.retry_count < MAX_RETRY:
                        self._requeue(task, pair, f"rc_{rc}")
                    else:
                        task.status = "NEEDS_MANUAL_RETRY"
            done_pairs.append(pair)
        for pair in done_pairs:
            if pair in self.running:
                del self.running[pair]

    def _requeue(self, task, old_pair, reason):
        new_pair = self.choose_healthy_pair(exclude=old_pair)
        if new_pair:
            task.retry_count += 1; task.previous_gpu_pairs.append(old_pair)
            task.gpu_pair = new_pair; task.status = "PENDING_RETRY"
            self.log(f"REQUEUE {task.task_id}: {reason} → {new_pair} (retry={task.retry_count})")
        else:
            task.status = "PENDING_RETRY"
            self.log(f"REQUEUE_WAIT {task.task_id}: {reason}, no healthy pair available")

    def discover_wave1(self):
        if not os.path.isdir(WAVE1_OUT):
            return
        for root, dirs, files in os.walk(WAVE1_OUT):
            for f in files:
                if "vis_pgd" in f and f.endswith("_trace.csv") and "traces" in root:
                    tp = os.path.join(root, f)
                    known = [getattr(t, 'wave1_path', '') for t in self.tasks if hasattr(t, 'wave1_discovered')]
                    if tp in known:
                        continue
                    try:
                        with open(tp) as tf:
                            r0 = next(csv.DictReader(tf), None)
                        if not r0:
                            continue
                        tt = r0.get("task",""); st = r0.get("state_id","")
                        ws = r0.get("window_start",""); we = r0.get("window_end","")
                        tid = f"W1_{tt}_s{st}_w{ws}_{we}"
                        t = Task(tid, "BATCH3_VIS", 0, tt, st, ws, we, "wave1", "wave1")
                        t.status = "EXTERNAL_DONE"; t.output_dir = os.path.dirname(os.path.dirname(root))
                        t.trace_path = tp; t.provenance_status = "external"
                        t.wave1_discovered = True; t.wave1_path = tp
                        self.tasks.append(t)
                        self.log(f"WAVE1 discovered: {tid}")
                    except Exception as e:
                        self.log(f"WAVE1 parse error: {e}")

    def schedule(self):
        can_schedule = self.check_xid_and_handle()
        if not can_schedule:
            return
        for pair in GPU_PAIRS:
            if pair in self.blacklisted or pair in self.running:
                continue
            if not gpu_idle(pair):
                continue
            pending = [t for t in self.tasks if t.status in ("PENDING", "PENDING_RETRY")
                      and t.gpu_pair == "" or t.status == "PENDING_RETRY"]
            if not pending:
                pending = [t for t in self.tasks if t.status in ("PENDING", "PENDING_RETRY")]
            pending.sort(key=lambda t: (t.status != "PENDING_RETRY", t.priority))
            if pending:
                pending[0].gpu_pair = pair
                self.launch_vis(pending[0])

    def all_vis_terminal(self):
        vis = [t for t in self.tasks if t.task_type == "BATCH3_VIS"]
        if not vis:
            return False
        term = {"DONE","EXTERNAL_DONE","INFRA_FAILED","PROVENANCE_FAILED","NEEDS_MANUAL_RETRY","FAILED"}
        return all(t.status in term for t in vis)

    def run_audit(self):
        if self.audit_done or "audit" in self.stages_run:
            return
        self.log("=== AUDIT STAGE ===")
        vis_dirs = []
        for t in self.tasks:
            if t.task_type == "BATCH3_VIS" and t.status in ("DONE","EXTERNAL_DONE"):
                td = os.path.join(t.output_dir, "traces") if hasattr(t, 'output_dir') and t.output_dir else ""
                if not td and hasattr(t, 'wave1_path'):
                    td = os.path.dirname(t.wave1_path)
                if td and os.path.isdir(td):
                    vis_dirs.append(td)
        if not vis_dirs:
            if os.path.isdir(WAVE1_OUT):
                for root, dirs, files in os.walk(WAVE1_OUT):
                    if os.path.basename(root) == "traces":
                        vis_dirs.append(root)
        if not vis_dirs:
            self.log("Audit: no trace dirs yet"); return
        cmd = [PY, "-u", f"{REPO}/scripts/diagnostics/audit_phase_conditioned_vis.py",
               "--run-dirs"] + vis_dirs + [
               "--output-csv", f"{REPO}/tables/object_phase_response_batch3_vis_provenance.csv",
               "--summary-csv", f"{REPO}/tables/object_phase_response_batch3_vis_summary.csv"]
        self.log(f"Audit on {len(vis_dirs)} dirs")
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            self.audit_done = True; self.stages_run.add("audit")
            self.log("Audit PASSED")
            self.run_label_merge()
        else:
            self.log(f"Audit FAILED: {r.stderr[:300]}")

    def run_label_merge(self):
        if "labels" in self.stages_run:
            return
        self.log("=== LABEL MERGE STAGE ===")
        label_script = f"{REPO}/scripts/diagnostics/finalize_phase_response_labels.py"
        if not os.path.exists(label_script):
            self.log("Label builder not found — skipping");
            return
        cmd = [PY, "-u", label_script,
               "--batch1-merged", f"{REPO}/tables/object_teacher_delay50_vis_smoke_merged_summary.csv",
               "--batch2b-vis", f"{REPO}/tables/object_phase_response_batch2b_vis_summary.csv",
               "--batch3-vis", f"{REPO}/tables/object_phase_response_batch3_vis_summary.csv",
               "--descriptors", f"{REPO}/tables/object_teacher_window_phase_descriptors.csv",
               "--output-labels", f"{REPO}/tables/object_phase_response_labels_v1.csv",
               "--output-metrics", f"{REPO}/tables/vulnerability_ready_smoke_metrics_v2.csv",
               "--output-predictions", f"{REPO}/tables/vulnerability_ready_smoke_predictions_v2.csv",
               "--output-report", f"{REPO}/reports/VULNERABILITY_READY_SMOKE_DETECTOR_V2.md"]
        self.log("Running label builder (CSV mode)...")
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            self.labels_done = True; self.stages_run.add("labels")
            self.log("Label merge PASSED")
        else:
            self.log(f"Label merge FAILED: {r.stderr[:200]}")

    def write_final_summary(self):
        path = f"{REPO}/reports/NIGHTLY_OBJECT_BATCH3_WATCHER_SUMMARY.md"
        n_done = sum(1 for t in self.tasks if t.status in ("DONE","EXTERNAL_DONE"))
        n_infra = sum(1 for t in self.tasks if "INFRA" in t.status or "FAIL" in t.status)
        n_retry = sum(1 for t in self.tasks if t.retry_count > 0)
        with open(path, "w") as f:
            f.write(f"# Overnight Batch3 Watcher Summary\n\n")
            f.write(f"**Runtime**: {datetime.now() - self.started_at}\n")
            f.write(f"**Blacklisted**: {self.blacklisted}\n\n")
            f.write(f"| Metric | Value |\n|--------|-------|\n")
            f.write(f"| Total tasks | {len(self.tasks)} |\n")
            f.write(f"| DONE | {n_done} |\n")
            f.write(f"| Infra/PROV failed | {n_infra} |\n")
            f.write(f"| Retried | {n_retry} |\n")
            f.write(f"| Audit | {'PASS' if self.audit_done else 'NOT RUN'} |\n")
            f.write(f"| Labels | {'PASS' if self.labels_done else 'NOT RUN'} |\n")
            f.write(f"| Stages | {self.stages_run} |\n\n")
            f.write("## Tasks\n\n")
            for t in self.tasks:
                f.write(f"- {t.task_id}: {t.task_key}_s{t.state_id} [{t.ws},{t.we}] {t.status} retry={t.retry_count} {t.failure_reason}\n")
        self.log(f"Final summary: {path}")

    def run(self):
        self.log(f"Watcher v2.2 started. Pairs: {GPU_PAIRS}, Max: {MAX_RUNTIME//3600}h")

        # Load targets
        tp = f"{REPO}/tables/object_phase_response_batch3_vis_targets.csv"
        if os.path.exists(tp):
            with open(tp, newline="") as f:
                targets = list(csv.DictReader(f))
            self.log(f"Loaded {len(targets)} VIS targets")
            wave1_keys = {("cream_cheese","4",28,45),("milk","4",19,36),("salad_dressing","0",7,24),("bbq_sauce","5",27,44)}
            tid = 0
            for t in targets:
                key = (t["task_key"], t["state_id"], int(t["window_start"]), int(t["window_end"]))
                if key in wave1_keys:
                    continue
                tid += 1
                tk = Task(f"B3_VIS_{tid:03d}","BATCH3_VIS",10,t["task_key"],t["state_id"],t["window_start"],t["window_end"],t.get("phase_bin_proxy",""),"")
                self.tasks.append(tk)
            self.log(f"Queued {tid} tasks")

        start = time.time()
        while time.time() - start < MAX_RUNTIME:
            self.discover_wave1()
            self.check_completions()
            if self.all_vis_terminal():
                self.run_audit()
                if all(t.status in ("DONE","EXTERNAL_DONE","INFRA_FAILED","PROVENANCE_FAILED","NEEDS_MANUAL_RETRY","FAILED") for t in self.tasks):
                    self.log("All tasks terminal.")
                    break
            self.schedule()
            if len(self.blacklisted) > 2:
                self.log("STOP: >2 GPU pairs blacklisted")
                break
            running_n = len(self.running)
            done_n = sum(1 for t in self.tasks if t.status in ("DONE","EXTERNAL_DONE"))
            pend_n = sum(1 for t in self.tasks if t.status.startswith("PENDING"))
            self.log(f"{datetime.now().strftime('%H:%M')} heartbeat: run={running_n} done={done_n} pend={pend_n} bl={len(self.blacklisted)}")
            self.write_state()
            time.sleep(POLL_SEC)

        self.write_final_summary()
        self.log("Watcher exiting.");

    def write_state(self):
        rows = [t.to_dict() for t in self.tasks]
        with open(f"{OUT}/queue/state.jsonl","w") as f:
            for r in rows:
                f.write(json.dumps(r)+"\n")
        with open(f"{OUT}/queue/tasks_snapshot.json","w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    Watcher().run()
