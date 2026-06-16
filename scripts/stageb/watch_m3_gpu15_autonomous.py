#!/usr/bin/env python3
"""Finite-state watcher for the GPU(1,5) M3 Layer3 autonomous campaign.

This script is intentionally conservative: it starts only preregistered stages,
requires machine-readable gate files between states, and stops on scientific
failures rather than changing attack parameters.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_TOKEN = 31744


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(path))
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def write_json(path: Path, obj: Any) -> None:
    atomic_write(path, json.dumps(obj, indent=2, sort_keys=True))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fieldnames})
    atomic_write(path, buf.getvalue())


class Watcher:
    def __init__(self, cfg_path: Path):
        self.cfg_path = cfg_path
        self.cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        self.output_root = Path(self.cfg["output_root"])
        self.events = self.output_root / "events.jsonl"
        self.state_path = self.output_root / "watcher_state.json"
        self.heartbeat = self.output_root / "heartbeat.json"
        self.current_command = self.output_root / "current_command.txt"
        self.deadline_path = self.output_root / "deadline.txt"
        self.deadline = time.time() + float(self.cfg.get("deadline_hours", 9.0)) * 3600.0
        self.env = os.environ.copy()
        self.env.update(
            {
                "PYTHONHASHSEED": "0",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(self.cfg["authorized_gpu_pair"]),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "TOKENIZERS_PARALLELISM": "false",
                "OPENVLA_ATTN_IMPLEMENTATION": "eager",
            }
        )
        self.py = str(self.cfg["python"])
        self.test_py = str(self.cfg.get("test_python", self.py))

    def event(self, stage: str, status: str, **payload: Any) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        record = {"time": now_iso(), "stage": stage, "status": status, **payload}
        with self.events.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        write_json(self.heartbeat, record)

    def set_state(self, state: str, **payload: Any) -> None:
        write_json(self.state_path, {"time": now_iso(), "state": state, **payload})

    def remaining(self) -> float:
        return self.deadline - time.time()

    def check_deadline(self, min_seconds: float = 900.0) -> None:
        if self.remaining() < min_seconds:
            raise RuntimeError("deadline guard: not enough time to start new GPU job")

    def run_cmd(self, stage: str, cmd: list[str], *, timeout: int, cwd: Path | None = None, extra_env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        self.check_deadline(120.0)
        cwd = cwd or REPO_ROOT
        env = dict(self.env)
        if extra_env:
            env.update(extra_env)
        self.event(stage, "COMMAND_START", command=" ".join(cmd), cwd=str(cwd), timeout=timeout)
        atomic_write(self.current_command, " ".join(cmd) + "\n")
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
        try:
            stdout, stderr = proc.communicate(timeout=min(timeout, max(1, int(self.remaining() - 900))))
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                stdout, stderr = proc.communicate(timeout=int(self.cfg.get("stop_grace_seconds", 120)))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                stdout, stderr = proc.communicate()
            self.event(stage, "TIMEOUT", returncode=proc.returncode, stdout_tail=stdout[-4000:], stderr_tail=stderr[-4000:])
            raise RuntimeError(f"{stage} timed out")
        self.event(stage, "COMMAND_DONE", returncode=proc.returncode, stdout_tail=stdout[-4000:], stderr_tail=stderr[-4000:])
        if proc.returncode != 0:
            raise RuntimeError(f"{stage} command failed rc={proc.returncode}: {' '.join(cmd)}\n{stderr[-4000:]}")
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)

    def git(self, args: list[str]) -> str:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()

    def nvidia_snapshot(self) -> str:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader",
            ],
            text=True,
            env=self.env,
        ).strip()

    def compute_apps(self) -> str:
        return subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            env=self.env,
        ).strip()

    def selected_gpu_processes(self, apps: str) -> list[str]:
        allowed = set(self.cfg["authorized_gpu_uuids"])
        out = []
        for line in apps.splitlines():
            if not line.strip():
                continue
            uuid = line.split(",", 1)[0].strip()
            if uuid in allowed:
                out.append(line)
        return out

    def write_gate(self, stage_dir: Path, status: str, **payload: Any) -> None:
        write_json(stage_dir / "gate_result.json", {"time": now_iso(), "status": status, **payload})

    def s0_init(self) -> None:
        self.set_state("S0_INIT")
        self.output_root.mkdir(parents=True, exist_ok=True)
        atomic_write(self.deadline_path, datetime.fromtimestamp(self.deadline, tz=timezone.utc).isoformat() + "\n")
        head = self.git(["rev-parse", "HEAD"])
        branch = self.git(["rev-parse", "--abbrev-ref", "HEAD"])
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
        if dirty:
            raise RuntimeError("dirty starting worktree")
        if branch != "exp/l3-selective-v3-gpu15-20260617":
            raise RuntimeError(f"wrong branch: {branch}")
        apps = self.compute_apps()
        busy = self.selected_gpu_processes(apps)
        if busy:
            raise RuntimeError("authorized GPU already has compute process: " + "; ".join(busy))
        env_manifest = {
            "head": head,
            "branch": branch,
            "hostname": socket.gethostname(),
            "cfg_path": str(self.cfg_path),
            "cfg_sha256": sha256_file(self.cfg_path),
            "python": self.py,
            "cuda_visible_devices": self.env["CUDA_VISIBLE_DEVICES"],
            "nvidia_smi": self.nvidia_snapshot(),
            "compute_apps": apps,
        }
        write_json(self.output_root / "environment.json", env_manifest)
        self.write_gate(self.output_root / "S0_INIT", "PASS", **env_manifest)

    def s1_gpu_qual(self) -> None:
        self.set_state("S1_GPU_QUAL")
        stage = self.output_root / "S1_GPU_QUAL"
        stage.mkdir(parents=True, exist_ok=True)
        before_xid = subprocess.check_output("dmesg -T 2>/dev/null | egrep -i 'NVRM|Xid' | tail -80 || true", shell=True, text=True)
        atomic_write(stage / "xid_before.txt", before_xid)
        self.run_cmd("S1_GPU_QUAL_NVIDIA", ["nvidia-smi"], timeout=60)
        smoke = textwrap.dedent(
            """
            import json, torch, random, numpy as np
            random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.set_float32_matmul_precision('highest')
            out = {'cuda_count': torch.cuda.device_count(), 'devices': []}
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_device(i)
                a=torch.randn((512,512), device='cuda', dtype=torch.bfloat16, requires_grad=True)
                b=torch.randn((512,512), device='cuda', dtype=torch.bfloat16)
                loss=(a@b).float().square().mean()
                loss.backward()
                out['devices'].append({'logical': i, 'name': torch.cuda.get_device_name(i), 'loss': float(loss.detach().cpu()), 'grad_finite': bool(torch.isfinite(a.grad).all().item())})
            print(json.dumps(out, sort_keys=True))
            """
        )
        proc = self.run_cmd("S1_GPU_QUAL_TORCH", [self.py, "-c", smoke], timeout=120)
        atomic_write(stage / "torch_smoke.json", proc.stdout)
        after_xid = subprocess.check_output("dmesg -T 2>/dev/null | egrep -i 'NVRM|Xid' | tail -80 || true", shell=True, text=True)
        atomic_write(stage / "xid_after.txt", after_xid)
        if after_xid != before_xid:
            raise RuntimeError("new Xid/NVRM lines detected during GPU qual")
        self.write_gate(stage, "PASS", torch_smoke=proc.stdout.strip())

    def s2_cpu_build(self) -> None:
        self.set_state("S2_CPU_BUILD")
        stage = self.output_root / "S2_CPU_BUILD"
        stage.mkdir(parents=True, exist_ok=True)
        cmds = [
            [self.py, "-m", "py_compile", "scripts/stageb/watch_m3_gpu15_autonomous.py", "scripts/stageb/run_m3_step78_true_pgd_fixed_frame.py"],
            [self.test_py, "-m", "pytest", "tests/stageb/test_m3_true_pgd_route_contract.py", "-q"],
        ]
        for idx, cmd in enumerate(cmds):
            proc = self.run_cmd(f"S2_CPU_BUILD_{idx}", cmd, timeout=240)
            atomic_write(stage / f"cmd_{idx}_stdout.log", proc.stdout)
            atomic_write(stage / f"cmd_{idx}_stderr.log", proc.stderr)
        self.write_gate(stage, "PASS")

    def lambda_config(self, lam: float) -> Path:
        cfg = yaml.safe_load(Path(self.cfg["base_config"]).read_text(encoding="utf-8"))
        cfg["stage"] = f"M3_GPU15_SELECTIVE_V3_LAMBDA_{lam}"
        cfg["attack_optimizer"]["arm_preserve_weight"] = float(lam)
        cfg["attack_optimizer"]["objective"] = "autoregressive_prefix_gripper_target_token_logratio_arm_v3"
        cfg["attack_optimizer"]["target_token_id"] = TARGET_TOKEN
        cfg["attack_optimizer"]["epsilon"] = float(self.cfg["tomato"]["epsilon"])
        cfg["gates"]["arm_prefix_min_match_count"] = int(self.cfg["tomato"]["arm_prefix_min_match_count"])
        path = self.output_root / "configs" / f"m3_gpu15_tomato_lambda_{lam}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, yaml.safe_dump(cfg, sort_keys=False))
        return path

    def selected_result(self, out_dir: Path) -> dict[str, dict[str, str]]:
        rows = read_csv(out_dir / "m3_v4_selected_results.csv")
        return {row["condition"]: row for row in rows}

    def tomato_pass(self, out_dir: Path) -> tuple[bool, dict[str, Any]]:
        by_cond = self.selected_result(out_dir)
        true = by_cond.get("TRUE_PGD_TRAJECTORY21_SELECTIVE", {})
        rand = by_cond.get("RAND21_SELECTIVE", {})
        shuffled = by_cond.get("SHUFFLED_GRAD_TRAJECTORY21_SELECTIVE", {})
        result_class = true.get("stage_result", "")
        def f(row: Mapping[str, str], key: str) -> float:
            return float(row[key]) if row.get(key, "") != "" else float("-inf")
        details = {
            "result_class": result_class,
            "true_margin": f(true, "official_target31744_margin"),
            "rand_margin": f(rand, "official_target31744_margin"),
            "shuffled_margin": f(shuffled, "official_target31744_margin"),
            "true_arm": int(true.get("official_arm_match_count") or true.get("arm_match_count") or -1),
            "true_token": int(true.get("official_gripper_token") or -1),
        }
        passed = (
            result_class == "FULL_SELECTIVE_V4_SEED_PASS"
            and details["true_token"] == TARGET_TOKEN
            and details["true_arm"] >= int(self.cfg["tomato"]["arm_prefix_min_match_count"])
            and details["true_margin"] > details["rand_margin"]
            and details["true_margin"] > details["shuffled_margin"]
        )
        return passed, details

    def s3_tomato(self) -> bool:
        self.set_state("S3_TOMATO_SCREEN")
        stage = self.output_root / "S3_TOMATO_SCREEN"
        stage.mkdir(parents=True, exist_ok=True)
        seed = int(self.cfg["tomato"]["attack_seed"])
        lambdas = [float(x) for x in self.cfg["tomato"]["lambda_grid"]]
        render_gpu = str(int(self.cfg.get("render_gpu_device_id", 1)))
        model_gpu = str(int(self.cfg.get("model_gpu_device_id", -1)))
        input_dir = stage / "input_step78"
        first_cfg = self.lambda_config(lambdas[0])
        self.run_cmd(
            "S3_CAPTURE_INPUT",
            [self.py, self.cfg["runner"], "--config", str(first_cfg), "--mode", "capture_input", "--output_dir", str(input_dir), "--attack_seed", str(seed), "--model_gpu_device_id", model_gpu, "--render_gpu_device_id", render_gpu, "--max_steps", "80", "--num_steps_wait", "10"],
            timeout=2400,
        )
        result_rows: list[dict[str, Any]] = []
        pass_rows: list[dict[str, Any]] = []
        for lam in lambdas:
            cfg_path = self.lambda_config(lam)
            lam_root = stage / f"lambda_{lam}"
            preflight_dir = lam_root / "preflight"
            canary_dir = lam_root / "canary"
            self.run_cmd(
                f"S3_PREFLIGHT_LAMBDA_{lam}",
                [self.py, self.cfg["runner"], "--config", str(cfg_path), "--mode", "preflight_zero_step", "--input_dir", str(input_dir), "--output_dir", str(preflight_dir), "--attack_seed", str(seed), "--model_gpu_device_id", model_gpu, "--render_gpu_device_id", render_gpu],
                timeout=1800,
            )
            self.run_cmd(
                f"S3_CANARY_LAMBDA_{lam}",
                [self.py, self.cfg["runner"], "--config", str(cfg_path), "--mode", "canary_v4", "--input_dir", str(input_dir), "--output_dir", str(canary_dir), "--attack_seed", str(seed), "--model_gpu_device_id", model_gpu, "--render_gpu_device_id", render_gpu],
                timeout=3600,
            )
            passed, details = self.tomato_pass(canary_dir)
            row = {"lambda": lam, "passed": passed, "output_dir": str(canary_dir), **details}
            result_rows.append(row)
            if passed:
                pass_rows.append(row)
            write_csv(stage / "m3_v3_tomato_results.csv", result_rows, list(result_rows[0].keys()))
            if len(pass_rows) >= int(self.cfg["tomato"].get("max_control_lambdas", 2)):
                break
        if not pass_rows:
            self.write_gate(stage, "FAIL", failure_class="TOMATO_NO_LAMBDA_PASS", results=result_rows)
            return False
        selected = sorted(pass_rows, key=lambda row: (-int(row["true_arm"]), -(float(row["true_margin"]) - float(row["rand_margin"])), float(row["lambda"])))[0]
        self.write_gate(stage, "PASS", selected=selected, results=result_rows)
        return True

    def s5_multi_parent(self) -> bool:
        self.set_state("S5_MULTI_PARENT")
        stage = self.output_root / "S5_MULTI_PARENT"
        stage.mkdir(parents=True, exist_ok=True)
        tag = self.cfg["multi_parent"]["required_tag"]
        handoff = self.cfg["multi_parent"]["handoff_path"]
        deadline = time.time() + int(self.cfg["multi_parent"].get("wait_for_l12_tag_seconds", 1200))
        found = False
        while time.time() < deadline and self.remaining() > 1200:
            rc = subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).returncode
            if rc == 0:
                try:
                    subprocess.check_call(["git", "show", f"{tag}:{handoff}"], cwd=REPO_ROOT, stdout=(stage / "l12_to_l3_timing_handoff.csv").open("w", encoding="utf-8"))
                    found = True
                    break
                except Exception:
                    pass
            time.sleep(300)
        if not found:
            self.write_gate(stage, "SKIP", reason="L12_G5_TAG_OR_HANDOFF_NOT_AVAILABLE")
            return False
        self.write_gate(stage, "SKIP", reason="MULTI_PARENT_GPU_RUNNER_NOT_WIRED_IN_THIS_CAMPAIGN")
        return False

    def finalize(self, result_class: str) -> None:
        self.set_state("S8_FINALIZE", result_class=result_class)
        rows = []
        tomato_csv = self.output_root / "S3_TOMATO_SCREEN" / "m3_v3_tomato_results.csv"
        if tomato_csv.exists():
            rows = read_csv(tomato_csv)
        write_csv(self.output_root / "tables_m3_v3_tomato_results.csv", rows, list(rows[0].keys()) if rows else ["lambda", "passed"])
        report = [
            "# M3 GPU15 Autonomous Handoff",
            "",
            f"- time: {now_iso()}",
            f"- result_class: {result_class}",
            f"- repo_head: {self.git(['rev-parse', 'HEAD'])}",
            f"- branch: {self.git(['rev-parse', '--abbrev-ref', 'HEAD'])}",
            f"- output_root: {self.output_root}",
            f"- cuda_visible_devices: {self.env['CUDA_VISIBLE_DEVICES']}",
            "",
            "## Claims",
            "",
            "- No broad Layer3 success claim is made by this watcher.",
            "- TRUE_PGD > random may only be claimed for rows whose gate_result.json is PASS and matched controls ran under GPU(1,5).",
            "- Detector-triggered integration was not run unless explicitly recorded in state S7.",
        ]
        atomic_write(self.output_root / "M3_GPU15_AUTONOMOUS_HANDOFF.md", "\n".join(report) + "\n")
        self.write_gate(self.output_root / "S8_FINALIZE", "PASS", result_class=result_class)
        self.event("S8_FINALIZE", "DONE", result_class=result_class)

    def run(self) -> None:
        self.s0_init()
        self.s1_gpu_qual()
        self.s2_cpu_build()
        tomato_ok = self.s3_tomato()
        if not tomato_ok:
            self.finalize("L3-0_TOMATO_FAIL")
            return
        multi_ok = self.s5_multi_parent()
        if not multi_ok:
            self.finalize("L3-1_TOMATO_PASS_ONLY")
            return
        self.finalize("L3-2_OR_HIGHER_NOT_IMPLEMENTED_IN_THIS_WATCHER")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "m3_gpu15_night_plan.yaml"))
    args = ap.parse_args()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    lock_path = Path(cfg["lock_file"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        watcher = Watcher(cfg_path)
        try:
            watcher.run()
        except Exception as exc:
            watcher.event("FAILED", "ERROR", error=repr(exc))
            watcher.set_state("FAILED", error=repr(exc))
            watcher.finalize("FAILED")
            raise


if __name__ == "__main__":
    main()
