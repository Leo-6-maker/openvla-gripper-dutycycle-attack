# Tier1 adaptive window experiment (state4/6/12). VIS attack negative; oracle positive. Establishes window-specificity boundary.
#!/usr/bin/env python3
"""Phase 2 Tier-1 six-condition runner for adaptive black-bowl replication.

Purpose:
    Launch the Tier-1 adaptive-state matrix used to test whether valid clean
    object-z windows outside the strongest state7 case reproduce gripper
    primitive hijack behavior.

Usage:
    Run from the repository root on the remote experiment server. The script
    schedules the clean, oracle, VIS, zero-margin, random-direction, and
    constant-delta controls for Tier-1 states.

Outputs:
    ${OPENVLA_OUTPUT_ROOT}/tier1_negative_20260508/

Paper link:
    Rebuttal/triage evidence. Tier-1 clean and oracle were valid, but VIS was
    negative; this motivates the state7-focused Template-B mechanism claim.
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path

import yaml

PY = os.environ.get("PYTHON", "python")
BASE_MODEL = os.environ.get("OPENVLA_BASE_MODEL_DIR", "${OPENVLA_BASE_MODEL_DIR}")
BASE_ATTACK_CFG = Path("configs/v4_attack.yaml")
TASKS_CONFIG = "configs/v4_tasks_libero.yaml"
DIRECTIONS_CONFIG = "configs/v4_directions.yaml"
THRESH = "outputs/v4/stage2_forceopen_planbv2_20260505/preflight/proxy_local_thresholds_close_transition.json"

ROOT = Path(os.environ.get("OPENVLA_TIER1_OUTPUT_ROOT", "outputs/tier1_negative_20260508"))
PROJECT_LINK = Path("outputs/v4/gripper_prompt_hijack_20260507/phase2_tier1_six_condition_20260508")
RUNS = ROOT / "runs"
LOGS = ROOT / "logs"
TABLES = ROOT / "tables"
VIDEOS = ROOT / "videos"
PREFLIGHT = ROOT / "preflight"
STATE = LOGS / "queue_state.json"

SLOTS = {
    "gpu01": {"cuda": "0,1", "render": "0"},
    "gpu23": {"cuda": "2,3", "render": "2"},
    "gpu45": {"cuda": "4,5", "render": "4"},
    "gpu67": {"cuda": "6,7", "render": "6"},
}
RENDER_TO_SLOT = {v["render"]: k for k, v in SLOTS.items()}

BLACK_BOWL = {
    "task_id": "libero_spatial_black_bowl",
    "model_path": os.environ.get("OPENVLA_SPATIAL_MODEL_PATH", "${OPENVLA_MODEL_ROOT}/openvla-7b-finetuned-libero-spatial"),
    "unnorm_key": "libero_spatial",
    "max_steps": 400,
    "object": "akita_black_bowl_1",
    "receptacle": "plate_1",
}

TIER1 = [
    {"source": "S1", "state": 4, "gate": "z003", "first": 59, "last": 65, "burst": 7},
    {"source": "S1", "state": 6, "gate": "z002", "first": 62, "last": 68, "burst": 7},
    {"source": "S2", "state": 12, "gate": "z002", "first": 72, "last": 76, "burst": 5},
]

CW = {"V4_CW_MARGIN": "5.0"}
COMMON_ENV = {
    "V4_LIFT_CLOSED_GRIPPER_SIGN": "positive",
    "V4_FORCE_OPEN_ENV_SIGN": "negative",
    "V4_TARGET_OBJECT_NAME": BLACK_BOWL["object"],
    "V4_TARGET_RECEPTACLE_NAME": BLACK_BOWL["receptacle"],
}


def state_tag(spec: dict) -> str:
    return f"{spec['source']}_state{spec['state']}_{spec['gate']}"


def job(run_id: str, slot: str, spec: dict, condition: str, trigger: str, objective: str, *, eps=0.10, step=0.020, steps=20, temporal_init="none", env=None, phase="") -> dict:
    return {
        **BLACK_BOWL,
        "run_id": run_id,
        "slot": slot,
        "condition": condition,
        "source": spec["source"],
        "states": str(spec["state"]),
        "state_id": int(spec["state"]),
        "episodes": 1,
        "adaptive_gate": spec["gate"],
        "gate_first_step": int(spec["first"]),
        "gate_last_step": int(spec["last"]),
        "burst": int(spec["burst"]),
        "trigger": trigger,
        "objective": objective,
        "epsilon": float(eps),
        "step_size": float(step),
        "num_steps": int(steps),
        "random_start": False,
        "temporal_init": temporal_init,
        "temporal_smooth_lambda": 0.0,
        "env": dict(env or {}),
        "phase": phase or condition,
    }


def adaptive_env(spec: dict) -> dict:
    return {
        "V4_FIXED_ATTACK_START": str(spec["first"]),
        "V4_FIXED_ATTACK_END": str(spec["last"]),
        "V4_GATE_BURST_STEPS": str(spec["burst"]),
        "V4_ADAPTIVE_GATE_LABEL": str(spec["gate"]),
    }


def jobs_for_spec(spec: dict, slot: str) -> list[dict]:
    tag = state_tag(spec)
    aenv = adaptive_env(spec)
    jobs = [
        job(f"T1_clean_{tag}", slot, spec, "clean", "clean", "", eps=0.03, step=0.006, steps=5, phase="clean"),
        job(
            f"T1_ORACLE_continuous_{tag}",
            slot,
            spec,
            "ORACLE_continuous",
            "fixed_step_window_budgeted",
            "oracle_env_gripper_open",
            eps=0.03,
            step=0.006,
            steps=5,
            env={**aenv, "V4_ORACLE_GRIPPER_PATTERN": "continuous_open", "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE": "-1.0"},
            phase="oracle_continuous_adaptive_window",
        ),
        job(
            f"T1_VIS_margin_prevdelta_{tag}",
            slot,
            spec,
            "VIS_margin_prevdelta",
            "fixed_step_window_budgeted",
            "gripper_logit_margin_cw",
            temporal_init="prev_delta",
            env={**aenv, **CW},
            phase="visual_margin_prevdelta_adaptive_window",
        ),
        job(
            f"T1_CTRL_same_gate_zero_margin_{tag}",
            slot,
            spec,
            "CTRL_same_gate_zero_margin",
            "fixed_step_window_budgeted",
            "gripper_logit_margin_cw",
            env={**aenv, **CW},
            phase="control_same_gate_zero_margin",
        ),
        job(
            f"T1_CTRL_random_direction_{tag}",
            slot,
            spec,
            "CTRL_random_direction",
            "fixed_step_window_budgeted",
            "untargeted_arm_clean_token_ce",
            env=aenv,
            phase="control_random_direction_arm_only",
        ),
        job(
            f"T1_CONSTANT_DELTA_{tag}",
            slot,
            spec,
            "CONSTANT_DELTA",
            "fixed_step_window_budgeted",
            "gripper_logit_margin_cw",
            temporal_init="prev_delta",
            env={**CW, "V4_FIXED_ATTACK_START": "70", "V4_FIXED_ATTACK_END": "80", "V4_GATE_BURST_STEPS": "11", "V4_ADAPTIVE_GATE_LABEL": "constant_70_80"},
            phase="constant_delta_absolute_step_70_80",
        ),
    ]
    jobs[-1]["adaptive_gate"] = "constant_70_80"
    jobs[-1]["gate_first_step"] = 70
    jobs[-1]["gate_last_step"] = 80
    jobs[-1]["burst"] = 11
    return jobs


def initial_jobs() -> dict:
    jobs = {}
    idx = 0
    slots = list(SLOTS)
    all_items = []
    for spec in TIER1:
        all_items.extend(jobs_for_spec(spec, "pending"))
    for item in all_items:
        slot = slots[idx % len(slots)]
        item["slot"] = slot
        item["queue_slot"] = slot
        item["queue_index"] = idx
        jobs[item["run_id"]] = {**item, "launched": False, "analyzed": False, "pid": None}
        idx += 1
    return jobs


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


def as_float(x):
    try:
        return float(x)
    except Exception:
        return None


def run_dir(run_id: str) -> Path:
    return RUNS / run_id / run_id


def progress(run_id: str) -> dict:
    path = run_dir(run_id) / "progress.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"status": "unknown"}


def job_status(run_id: str) -> str:
    return str(progress(run_id).get("status", "missing"))


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"started_at": time.time(), "jobs": initial_jobs()}


def save_state(state: dict) -> None:
    write_json(STATE, state)


def ensure_link() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    PROJECT_LINK.parent.mkdir(parents=True, exist_ok=True)
    if PROJECT_LINK.exists() or PROJECT_LINK.is_symlink():
        return
    PROJECT_LINK.symlink_to(ROOT, target_is_directory=True)


def cfg_path(item: dict) -> Path:
    return PREFLIGHT / f"v4_attack_{item['run_id']}.yaml"


def ensure_cfg(item: dict) -> None:
    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(BASE_ATTACK_CFG.read_text())
    opt = cfg.setdefault("attack_optimizer", {})
    opt.update(
        {
            "method": "token_prefix_pgd",
            "epsilon": item["epsilon"],
            "step_size": item["step_size"],
            "num_steps": item["num_steps"],
            "random_start": item["random_start"],
            "temporal_init": item["temporal_init"],
            "temporal_smooth_lambda": item["temporal_smooth_lambda"],
            "objective": item["objective"] or "targeted_directional_ce",
        }
    )
    if item["objective"] == "gripper_logit_margin_cw":
        opt["cw_margin"] = float(item.get("env", {}).get("V4_CW_MARGIN", 5.0))
    cfg_path(item).write_text(yaml.safe_dump(cfg, sort_keys=False))


def active_slots() -> set[str]:
    out = subprocess.run(
        "pgrep -af 'v4_run_eval_openvla.py' || true",
        shell=True,
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    used = set()
    for line in out.splitlines():
        match = re.search(r"--render_gpu_device_id\s+(\d+)", line)
        if match and match.group(1) in RENDER_TO_SLOT:
            used.add(RENDER_TO_SLOT[match.group(1)])
    return used


def launch(item: dict, slot: str) -> int:
    ensure_cfg(item)
    out = RUNS / item["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            **COMMON_ENV,
            "CUDA_VISIBLE_DEVICES": SLOTS[slot]["cuda"],
            "OPENVLA_CUDA_MAX_MEMORY": "10000MiB",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.update(item.get("env") or {})
    cmd = [
        PY,
        "scripts/v4_run_eval_openvla.py",
        "--tasks_config",
        TASKS_CONFIG,
        "--attack_config",
        str(cfg_path(item)),
        "--directions_config",
        DIRECTIONS_CONFIG,
        "--thresholds",
        THRESH,
        "--task_id",
        item["task_id"],
        "--trigger",
        item["trigger"],
        "--rho",
        "0.03",
        "--seed",
        "0",
        "--episodes",
        str(item["episodes"]),
        "--state_ids",
        item["states"],
        "--max_steps_override",
        str(item["max_steps"]),
        "--model_path",
        item["model_path"],
        "--base_model_code_dir",
        BASE_MODEL,
        "--unnorm_key",
        item["unnorm_key"],
        "--camera_obs_key",
        "agentview_image",
        "--model_gpu_device_id",
        "-1",
        "--render_gpu_device_id",
        SLOTS[slot]["render"],
        "--image_size",
        "256",
        "--openvla_resize_size",
        "224",
        "--success_metric",
        "done",
        "--auto_patch_compat",
        "--libero_official_preprocess",
        "--center_crop",
        "--postprocess_gripper",
        "--deterministic_init_states",
        "--force_open_raw_gripper",
        "1.0",
        "--output_root",
        str(out),
        "--run_id",
        item["run_id"],
    ]
    if item["objective"]:
        cmd += ["--attack_objective", item["objective"]]
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / f"{item['run_id']}.log").open("ab") as fp:
        proc = subprocess.Popen(cmd, cwd=Path.cwd(), env=env, stdout=fp, stderr=subprocess.STDOUT)
    print(f"[launch] {item['run_id']} pid={proc.pid} slot={slot}", flush=True)
    return proc.pid


def max_consecutive_open(attacks: list[dict]) -> int:
    best = cur = 0
    last = None
    for row in sorted(attacks, key=lambda x: (int(x.get("episode_id", -1)), int(x.get("step_idx", -1)))):
        key = (int(row.get("episode_id", -1)), int(row.get("step_idx", -1)))
        open_now = bool((as_float(row.get("executed_gripper_env")) or 0.0) < -0.5)
        if open_now and last and key[0] == last[0] and key[1] == last[1] + 1:
            cur += 1
        elif open_now:
            cur = 1
        else:
            cur = 0
        best = max(best, cur)
        last = key
    return best


def mean(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(item: dict, status: str) -> dict:
    rd = run_dir(item["run_id"])
    steps = read_jsonl(rd / "step_records.jsonl")
    eps = read_jsonl(rd / "episode_records.jsonl")
    attacks = [row for row in steps if row.get("attack_active")]
    phases = {}
    for ep in eps:
        phases[str(ep.get("failure_phase", ""))] = phases.get(str(ep.get("failure_phase", "")), 0) + 1
    qd = [as_float(row.get("physical_gripper_opening_delta")) for row in attacks if as_float(row.get("physical_gripper_opening_delta")) is not None]
    qa = [as_float(row.get("gripper_qpos_abs_sum_after")) for row in attacks if as_float(row.get("gripper_qpos_abs_sum_after")) is not None]
    adv = [as_float(row.get("adv_gripper_open_bin_prob_mass")) for row in attacks if as_float(row.get("adv_gripper_open_bin_prob_mass")) is not None]
    ce = [as_float(row.get("target_ce_final")) for row in attacks if as_float(row.get("target_ce_final")) is not None]
    opened = [1.0 if (as_float(row.get("executed_gripper_env")) is not None and as_float(row.get("executed_gripper_env")) < -0.5) else 0.0 for row in attacks]
    arm_l2 = [as_float(row.get("arm_only_l2")) for row in attacks if as_float(row.get("arm_only_l2")) is not None]
    full_l2 = [as_float(row.get("full_action_l2")) for row in attacks if as_float(row.get("full_action_l2")) is not None]
    grip_delta = [as_float(row.get("gripper_action_delta")) for row in attacks if as_float(row.get("gripper_action_delta")) is not None]
    sr = sum(1 for ep in eps if ep.get("success")) / len(eps) if eps else None
    return {
        "condition": item["condition"],
        "run_id": item["run_id"],
        "slot": item.get("slot", ""),
        "state_id": item["state_id"],
        "source": item["source"],
        "recommended_gate": item["adaptive_gate"],
        "gate_first_step": item["gate_first_step"],
        "gate_last_step": item["gate_last_step"],
        "gate_hit_count": item["burst"],
        "burst": item["burst"],
        "trigger": item["trigger"],
        "objective": item["objective"],
        "attacks": len(attacks),
        "adv_open_mass": mean(adv),
        "target_ce_final": mean(ce),
        "exec_open_rate": (sum(opened) / len(opened)) if opened else None,
        "consecutive_open_streak": max_consecutive_open(attacks),
        "qpos_delta_max": max(qd) if qd else None,
        "qpos_abs_after_max": max(qa) if qa else None,
        "full_action_l2": mean(full_l2),
        "arm_only_l2": mean(arm_l2),
        "gripper_action_delta": mean(grip_delta),
        "SR": sr,
        "failure_phase": json.dumps(phases, sort_keys=True),
        "status": status,
        "phase": item.get("phase", ""),
    }


def write_table(state: dict) -> None:
    rows = []
    for item in state["jobs"].values():
        status = job_status(item["run_id"])
        if item.get("launched") or status != "missing":
            rows.append(summarize(item, status))
    fields = [
        "condition",
        "run_id",
        "slot",
        "state_id",
        "source",
        "recommended_gate",
        "gate_first_step",
        "gate_last_step",
        "gate_hit_count",
        "burst",
        "trigger",
        "objective",
        "attacks",
        "adv_open_mass",
        "target_ce_final",
        "exec_open_rate",
        "consecutive_open_streak",
        "qpos_delta_max",
        "qpos_abs_after_max",
        "full_action_l2",
        "arm_only_l2",
        "gripper_action_delta",
        "SR",
        "failure_phase",
        "status",
        "phase",
    ]
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / "phase2_tier1_mechanism_table.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(TABLES / "phase2_tier1_mechanism_table.json", rows)


def analyze_and_video(item: dict, slot: str) -> None:
    rd = run_dir(item["run_id"])
    if not (rd / "summary.csv").exists():
        return
    out = TABLES / item["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([PY, "scripts/v4_analyze_attack_efficacy.py", "--input_root", str(rd), "--output_dir", str(out)], cwd=Path.cwd(), check=False)
    eps = read_jsonl(rd / "episode_records.jsonl")
    selected = []
    for ep in eps:
        eid = int(ep.get("episode_id", -1))
        if eid >= 0:
            selected.append(eid)
    if selected:
        video_dir = VIDEOS / item["run_id"]
        video_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                PY,
                "scripts/v4_render_episode_video_from_steps.py",
                "--run_dir",
                str(rd),
                "--episode_ids",
                ",".join(map(str, selected[:2])),
                "--output_dir",
                str(video_dir),
                "--tasks_config",
                TASKS_CONFIG,
                "--task_id",
                item["task_id"],
                "--render_gpu_device_id",
                SLOTS[slot]["render"],
                "--image_size",
                "256",
                "--frame_stride",
                "2",
                "--fps",
                "10",
            ],
            cwd=Path.cwd(),
            check=False,
        )


def next_for_slot(state: dict, slot: str) -> dict | None:
    slot_jobs = sorted([item for item in state["jobs"].values() if item["queue_slot"] == slot], key=lambda x: int(x["queue_index"]))
    for item in slot_jobs:
        if not item.get("launched") and job_status(item["run_id"]) == "missing":
            return item
    return None


def all_done(state: dict) -> bool:
    if active_slots():
        return False
    for item in state["jobs"].values():
        status = job_status(item["run_id"])
        if status in {"missing", "starting", "running", "unknown"}:
            return False
        if status == "done" and not item.get("analyzed"):
            return False
    return True


def reap_children() -> None:
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except (ChildProcessError, OSError):
            return
        if pid == 0:
            return


def main() -> None:
    os.chdir(Path(__file__).resolve().parents[1])
    for path in [RUNS, LOGS, TABLES, VIDEOS, PREFLIGHT]:
        path.mkdir(parents=True, exist_ok=True)
    ensure_link()
    while True:
        reap_children()
        state = load_state()
        used = active_slots()
        for run_id, item in list(state["jobs"].items()):
            status = job_status(run_id)
            if item.get("launched") and status == "done" and not item.get("analyzed"):
                analyze_and_video(item, item.get("slot") or item.get("queue_slot") or "gpu01")
                item["analyzed"] = True
                item["completed_at"] = time.time()
            elif item.get("launched") and status in {"error", "failed", "failed_cuda"} and not item.get("analyzed"):
                item["analyzed"] = True
                item["failed_reason"] = progress(run_id).get("error", status)
        used = active_slots()
        for slot in SLOTS:
            if slot in used:
                continue
            item = next_for_slot(state, slot)
            if item is None:
                continue
            pid = launch(item, slot)
            item["launched"] = True
            item["slot"] = slot
            item["pid"] = pid
            item["launched_at"] = time.time()
            used.add(slot)
        write_table(state)
        save_state(state)
        if all_done(state):
            print("[done] all tier1 six-condition jobs completed", flush=True)
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
