# Early overnight gripper primitive queue. Established candidate state5/7/8 phenomena before Template-B freeze; not a public entrypoint.
#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path

import yaml

PY = "python"
BASE_MODEL = "${OPENVLA_BASE_MODEL_DIR}"
BASE_ATTACK_CFG = Path("configs/v4_attack.yaml")
TASKS_CONFIG = "configs/v4_tasks_libero.yaml"
DIRECTIONS_CONFIG = "configs/v4_directions.yaml"
THRESH = "outputs/v4/stage2_forceopen_planbv2_20260505/preflight/proxy_local_thresholds_close_transition.json"

ROOT = Path("${OPENVLA_OUTPUT_ROOT}/openvla_overnight_gripper_primitive_20260508")
PROJECT_LINK = Path("outputs/v4/gripper_prompt_hijack_20260507/overnight_gripper_primitive_10h")
RUNS = ROOT / "runs"
LOGS = ROOT / "logs"
TABLES = ROOT / "tables"
VIDEOS = ROOT / "videos"
PREFLIGHT = ROOT / "preflight"
STATE = LOGS / "queue_state.json"
FINAL_VIDEO_EXPORT_DONE = TABLES / "manual_video_export_done.json"

SLOTS = {
    "gpu01": {"cuda": "0,1", "render": "0"},
    "gpu23": {"cuda": "2,3", "render": "2"},
    "gpu45": {"cuda": "4,5", "render": "4"},
    "gpu67": {"cuda": "6,7", "render": "6"},
}
RENDER_TO_SLOT = {v["render"]: k for k, v in SLOTS.items()}
STABLE_RETRY_SLOTS = ["gpu45", "gpu01"]
DISABLED_PRIMARY_SLOTS = {"gpu23"}
MAX_CUDA_RETRIES = 2

BLACK_BOWL = {
    "task_id": "libero_spatial_black_bowl",
    "model_path": "${OPENVLA_MODEL_ROOT}/openvla-7b-finetuned-libero-spatial",
    "unnorm_key": "libero_spatial",
    "max_steps": 400,
    "object": "akita_black_bowl_1",
    "receptacle": "plate_1",
}


def job(
    run_id: str,
    slot: str,
    seed: int,
    states: str,
    trigger: str,
    objective: str,
    *,
    epsilon: float = 0.10,
    step_size: float = 0.020,
    num_steps: int = 20,
    random_start: bool = False,
    temporal_init: str = "none",
    temporal_smooth_lambda: float = 0.0,
    env: dict | None = None,
    phase: str = "visual",
) -> dict:
    return {
        **BLACK_BOWL,
        "run_id": run_id,
        "slot": slot,
        "seed": int(seed),
        "states": states,
        "episodes": len([x for x in states.split(",") if x.strip()]),
        "trigger": trigger,
        "objective": objective,
        "epsilon": float(epsilon),
        "step_size": float(step_size),
        "num_steps": int(num_steps),
        "random_start": bool(random_start),
        "temporal_init": temporal_init,
        "temporal_smooth_lambda": float(temporal_smooth_lambda),
        "env": dict(env or {}),
        "phase": phase,
    }


Z004 = {"V4_PRIV_LIFT_BOWL_Z_DELTA_MIN": "0.04", "V4_GATE_BURST_STEPS": "10"}
CW = {"V4_CW_MARGIN": "5.0"}

QUEUES = {
    "gpu01": [
        job("VIS_margin_prevdelta_seed1_5_7_8", "gpu01", 1, "5,7,8", "priv_lift_carry_gate_burst_budgeted", "gripper_logit_margin_cw", temporal_init="prev_delta", env={**Z004, **CW}, phase="visual_margin_prevdelta"),
        job("VIS_margin_prevdelta_seed2_5_7_8", "gpu01", 2, "5,7,8", "priv_lift_carry_gate_burst_budgeted", "gripper_logit_margin_cw", temporal_init="prev_delta", env={**Z004, **CW}, phase="visual_margin_prevdelta"),
        job("VIS_margin_prevdelta_repeat_seed0_5_7_8", "gpu01", 0, "5,7,8", "priv_lift_carry_gate_burst_budgeted", "gripper_logit_margin_cw", temporal_init="prev_delta", env={**Z004, **CW}, phase="visual_margin_prevdelta"),
        job("VIS_margin_prevdelta_seed1_state5_only_ep3", "gpu01", 1, "5,5,5", "priv_lift_carry_gate_burst_budgeted", "gripper_logit_margin_cw", temporal_init="prev_delta", env={**Z004, **CW}, phase="visual_margin_prevdelta_state5"),
    ],
    "gpu23": [
        job("CTRL_region_randomstart_seed0_5_7_8", "gpu23", 0, "5,7,8", "priv_lift_carry_gate_burst_budgeted", "gripper_open_region_ce", random_start=True, env=Z004, phase="control_randomstart"),
        job("CTRL_region_zero_seed0_5_7_8", "gpu23", 0, "5,7,8", "priv_lift_carry_gate_burst_budgeted", "gripper_open_region_ce", env=Z004, phase="control_zero"),
        job("CTRL_margin_zero_seed0_5_7_8", "gpu23", 0, "5,7,8", "priv_lift_carry_gate_burst_budgeted", "gripper_logit_margin_cw", env={**Z004, **CW}, phase="control_margin_zero"),
        job("CTRL_region_prevdelta_seed1_5_7_8", "gpu23", 1, "5,7,8", "priv_lift_carry_gate_burst_budgeted", "gripper_open_region_ce", temporal_init="prev_delta", env=Z004, phase="control_region_prevdelta"),
    ],
    "gpu45": [
        job("ORACLE_continuous_seed1_5_7_8", "gpu45", 1, "5,7,8", "priv_lift_carry_gate_burst_budgeted", "oracle_env_gripper_open", epsilon=0.03, step_size=0.006, num_steps=5, env={**Z004, "V4_ORACLE_GRIPPER_PATTERN": "continuous_open", "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE": "-1.0"}, phase="oracle_continuous"),
        job("ORACLE_density60_seed1_5_7_8", "gpu45", 1, "5,7,8", "priv_lift_carry_gate_burst_budgeted", "oracle_env_gripper_open", epsilon=0.03, step_size=0.006, num_steps=5, env={**Z004, "V4_ORACLE_GRIPPER_PATTERN": "density60", "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE": "-1.0"}, phase="oracle_density60"),
        job("ORACLE_alternating_seed1_5_7_8", "gpu45", 1, "5,7,8", "priv_lift_carry_gate_burst_budgeted", "oracle_env_gripper_open", epsilon=0.03, step_size=0.006, num_steps=5, env={**Z004, "V4_ORACLE_GRIPPER_PATTERN": "alternating", "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE": "-1.0"}, phase="oracle_alternating"),
    ],
    "gpu67": [
        job("VIS_region_smooth_seed1_5_7_8", "gpu67", 1, "5,7,8", "priv_lift_carry_gate_burst_budgeted", "gripper_open_region_ce", temporal_init="prev_delta", temporal_smooth_lambda=0.10, env=Z004, phase="visual_region_smooth"),
        job("VIS_region_smooth_seed2_5_7_8", "gpu67", 2, "5,7,8", "priv_lift_carry_gate_burst_budgeted", "gripper_open_region_ce", temporal_init="prev_delta", temporal_smooth_lambda=0.10, env=Z004, phase="visual_region_smooth"),
        job("VIS_margin_prevdelta_S1_0_1_2_3_4_6", "gpu67", 0, "0,1,2,3,4,6", "priv_lift_carry_gate_burst_budgeted", "gripper_logit_margin_cw", temporal_init="prev_delta", env={**Z004, **CW}, phase="cross_state_z004"),
        job("VIS_margin_prevdelta_S2_9_10_11_12_13_14", "gpu67", 0, "9,10,11,12,13,14", "priv_lift_carry_gate_burst_budgeted", "gripper_logit_margin_cw", temporal_init="prev_delta", env={**Z004, **CW}, phase="cross_state_z004"),
    ],
}

BACKUP = [
    job("CLEAN_trace_S1_seed0", "backup", 0, "0,1,2,3,4,6", "clean", "", epsilon=0.03, step_size=0.006, num_steps=5, phase="clean_trace"),
    job("CLEAN_trace_S2_seed0", "backup", 0, "9,10,11,12,13,14", "clean", "", epsilon=0.03, step_size=0.006, num_steps=5, phase="clean_trace"),
    job("ORACLE_continuous_S1_z004", "backup", 0, "0,1,2,3,4,6", "priv_lift_carry_gate_burst_budgeted", "oracle_env_gripper_open", epsilon=0.03, step_size=0.006, num_steps=5, env={**Z004, "V4_ORACLE_GRIPPER_PATTERN": "continuous_open", "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE": "-1.0"}, phase="oracle_cross_state"),
    job("ORACLE_continuous_S2_z004", "backup", 0, "9,10,11,12,13,14", "priv_lift_carry_gate_burst_budgeted", "oracle_env_gripper_open", epsilon=0.03, step_size=0.006, num_steps=5, env={**Z004, "V4_ORACLE_GRIPPER_PATTERN": "continuous_open", "V4_ORACLE_FORCE_GRIPPER_ENV_VALUE": "-1.0"}, phase="oracle_cross_state"),
    job("VIS_margin_prevdelta_seed1_state7_only_ep3", "backup", 1, "7,7,7", "priv_lift_carry_gate_burst_budgeted", "gripper_logit_margin_cw", temporal_init="prev_delta", env={**Z004, **CW}, phase="visual_margin_prevdelta_state7"),
    job("VIS_margin_prevdelta_seed1_state8_only_ep3", "backup", 1, "8,8,8", "priv_lift_carry_gate_burst_budgeted", "gripper_logit_margin_cw", temporal_init="prev_delta", env={**Z004, **CW}, phase="visual_margin_prevdelta_state8"),
]


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
    jobs = {}
    for slot, queue in QUEUES.items():
        for idx, item in enumerate(queue):
            jobs[item["run_id"]] = {**item, "queue_slot": slot, "queue_index": idx, "launched": False, "analyzed": False, "pid": None}
    for idx, item in enumerate(BACKUP):
        jobs[item["run_id"]] = {**item, "queue_slot": "backup", "queue_index": idx, "launched": False, "analyzed": False, "pid": None}
    return {"started_at": time.time(), "jobs": jobs, "slot_history": {}, "backup_cursor": 0}


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
        }
    )
    if item["objective"] == "gripper_logit_margin_cw":
        opt["cw_margin"] = float(item.get("env", {}).get("V4_CW_MARGIN", 5.0))
    cfg_path(item).write_text(yaml.safe_dump(cfg, sort_keys=False))


def active_slots() -> set[str]:
    out = subprocess.run(
        "pgrep -af 'v4_run_eval_openvla.py|v4_gripper_prompt_hijack_phase2b_gcg_opt.py' || true",
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
            "CUDA_VISIBLE_DEVICES": SLOTS[slot]["cuda"],
            "OPENVLA_CUDA_MAX_MEMORY": "10000MiB",
            "PYTHONUNBUFFERED": "1",
            "V4_LIFT_CLOSED_GRIPPER_SIGN": "positive",
            "V4_FORCE_OPEN_ENV_SIGN": "negative",
            "V4_TARGET_OBJECT_NAME": item["object"],
            "V4_TARGET_RECEPTACLE_NAME": item["receptacle"],
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
        str(item["seed"]),
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


def median(vals: list[float]) -> float | None:
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    return vals[len(vals) // 2]


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
    sr = None
    if eps:
        sr = sum(1 for ep in eps if ep.get("success")) / len(eps)
    return {
        "condition": item["run_id"],
        "slot": item.get("slot", ""),
        "seed": item["seed"],
        "state_ids": item["states"],
        "trigger": item["trigger"],
        "objective": item["objective"],
        "attacks": len(attacks),
        "adv_open_mass": median(adv),
        "target_ce_final": median(ce),
        "exec_open_rate": (sum(opened) / len(opened)) if opened else None,
        "consecutive_open_streak": max_consecutive_open(attacks),
        "qpos_delta_max": max(qd) if qd else None,
        "qpos_abs_after_max": max(qa) if qa else None,
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
        "slot",
        "seed",
        "state_ids",
        "trigger",
        "objective",
        "attacks",
        "adv_open_mass",
        "target_ce_final",
        "exec_open_rate",
        "consecutive_open_streak",
        "qpos_delta_max",
        "qpos_abs_after_max",
        "SR",
        "failure_phase",
        "status",
        "phase",
    ]
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / "overnight_mechanism_table.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(TABLES / "overnight_mechanism_table.json", rows)


def analyze_and_video(item: dict, slot: str) -> None:
    rd = run_dir(item["run_id"])
    if not (rd / "summary.csv").exists():
        return
    out = TABLES / item["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([PY, "scripts/v4_analyze_attack_efficacy.py", "--input_root", str(rd), "--output_dir", str(out)], cwd=Path.cwd(), check=False)
    eps = read_jsonl(rd / "episode_records.jsonl")
    selected = []
    success_added = False
    for ep in eps:
        eid = int(ep.get("episode_id", -1))
        if eid < 0:
            continue
        if not ep.get("success"):
            selected.append(eid)
        elif not success_added:
            selected.append(eid)
            success_added = True
    selected = selected[:4]
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
                ",".join(map(str, selected)),
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


def episode_attack_stats(steps: list[dict]) -> dict[int, dict]:
    by_ep: dict[int, dict] = {}
    for row in steps:
        try:
            eid = int(row.get("episode_id", -1))
        except Exception:
            continue
        if eid < 0:
            continue
        rec = by_ep.setdefault(
            eid,
            {
                "attacks": 0,
                "qpos_abs_after_max": None,
                "qpos_delta_max": None,
                "consecutive_open_streak": 0,
            },
        )
        if row.get("attack_active"):
            rec["attacks"] += 1
            qa = as_float(row.get("gripper_qpos_abs_sum_after"))
            qd = as_float(row.get("physical_gripper_opening_delta"))
            if qa is not None:
                rec["qpos_abs_after_max"] = qa if rec["qpos_abs_after_max"] is None else max(rec["qpos_abs_after_max"], qa)
            if qd is not None:
                rec["qpos_delta_max"] = qd if rec["qpos_delta_max"] is None else max(rec["qpos_delta_max"], qd)
    for eid, rec in by_ep.items():
        ep_rows = [row for row in steps if int(row.get("episode_id", -1)) == eid and row.get("attack_active")]
        rec["consecutive_open_streak"] = max_consecutive_open(ep_rows)
    return by_ep


def select_manual_video_episodes(episodes: list[dict], stats: dict[int, dict]) -> list[tuple[int, str]]:
    selected: dict[int, str] = {}
    for ep in episodes:
        eid = int(ep.get("episode_id", -1))
        if eid < 0:
            continue
        rec = stats.get(eid, {})
        qpos = rec.get("qpos_abs_after_max") or 0.0
        streak = rec.get("consecutive_open_streak") or 0
        attacks = rec.get("attacks") or 0
        if not ep.get("success") and attacks > 0:
            selected[eid] = "failure_with_attack"
        elif qpos >= 0.05:
            selected[eid] = "high_qpos"
        elif streak >= 3:
            selected[eid] = "open_streak"
    for ep in episodes:
        eid = int(ep.get("episode_id", -1))
        if eid >= 0 and not ep.get("success") and eid not in selected:
            selected[eid] = "failure_context"
    for ep in episodes:
        eid = int(ep.get("episode_id", -1))
        if eid >= 0 and ep.get("success") and eid not in selected:
            selected[eid] = "success_reference"
            break
    return list(selected.items())[:6]


def render_manual_episode(item: dict, slot: str, episode_id: int) -> Path:
    rd = run_dir(item["run_id"])
    video_dir = VIDEOS / item["run_id"]
    video_dir.mkdir(parents=True, exist_ok=True)
    expected = list(video_dir.glob(f"{item['run_id']}_ep{episode_id:03d}_*.mp4"))
    if expected:
        return expected[0]
    subprocess.run(
        [
            PY,
            "scripts/v4_render_episode_video_from_steps.py",
            "--run_dir",
            str(rd),
            "--episode_ids",
            str(episode_id),
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
    expected = list(video_dir.glob(f"{item['run_id']}_ep{episode_id:03d}_*.mp4"))
    return expected[0] if expected else video_dir / f"{item['run_id']}_ep{episode_id:03d}_missing.mp4"


def final_manual_video_export(state: dict) -> None:
    if FINAL_VIDEO_EXPORT_DONE.exists():
        return
    rows = []
    for item in sorted(state["jobs"].values(), key=lambda x: x["run_id"]):
        if job_status(item["run_id"]) != "done":
            continue
        rd = run_dir(item["run_id"])
        episodes = read_jsonl(rd / "episode_records.jsonl")
        steps = read_jsonl(rd / "step_records.jsonl")
        if not episodes or not steps:
            continue
        stats = episode_attack_stats(steps)
        slot = item.get("slot") if item.get("slot") in SLOTS else "gpu01"
        for eid, reason in select_manual_video_episodes(episodes, stats):
            video = render_manual_episode(item, slot, eid)
            ep = next((row for row in episodes if int(row.get("episode_id", -1)) == eid), {})
            rec = stats.get(eid, {})
            rows.append(
                {
                    "run_id": item["run_id"],
                    "phase": item.get("phase", ""),
                    "slot": slot,
                    "seed": item.get("seed", ""),
                    "state_ids": item.get("states", ""),
                    "episode_id": eid,
                    "state_id": ep.get("state_id", ""),
                    "success": ep.get("success", ""),
                    "failure_phase": ep.get("failure_phase", ""),
                    "selection_reason": reason,
                    "attacks": rec.get("attacks", 0),
                    "consecutive_open_streak": rec.get("consecutive_open_streak", 0),
                    "qpos_delta_max": rec.get("qpos_delta_max"),
                    "qpos_abs_after_max": rec.get("qpos_abs_after_max"),
                    "video_path": str(video),
                }
            )
    fields = [
        "run_id",
        "phase",
        "slot",
        "seed",
        "state_ids",
        "episode_id",
        "state_id",
        "success",
        "failure_phase",
        "selection_reason",
        "attacks",
        "consecutive_open_streak",
        "qpos_delta_max",
        "qpos_abs_after_max",
        "video_path",
    ]
    with (TABLES / "manual_video_review_index.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(TABLES / "manual_video_review_index.json", rows)
    write_json(FINAL_VIDEO_EXPORT_DONE, {"completed_at": time.time(), "videos_indexed": len(rows)})
    print(f"[final-video-export] indexed {len(rows)} videos for manual review", flush=True)


def all_jobs_finished(state: dict) -> bool:
    if active_slots():
        return False
    for item in state["jobs"].values():
        status = job_status(item["run_id"])
        if not item.get("launched") and status == "missing":
            return False
        if status in {"missing", "starting", "running", "unknown"}:
            return False
        if status == "done" and not item.get("analyzed"):
            return False
    return True


def next_for_slot(state: dict, slot: str) -> dict | None:
    retry_jobs = sorted(
        [
            item
            for item in state["jobs"].values()
            if item.get("queue_slot") == slot and item.get("retry_of")
        ],
        key=lambda x: (int(x.get("retry_attempt", 0)), int(x.get("queue_index", 0))),
    )
    for item in retry_jobs:
        if not item.get("launched") and job_status(item["run_id"]) == "missing":
            return item

    if slot in DISABLED_PRIMARY_SLOTS:
        return None

    slot_jobs = sorted(
        [item for item in state["jobs"].values() if item["queue_slot"] == slot],
        key=lambda x: int(x["queue_index"]),
    )
    for item in slot_jobs:
        if item.get("retry_of"):
            continue
        if not item.get("launched") and job_status(item["run_id"]) == "missing":
            return item
    backups = sorted(
        [item for item in state["jobs"].values() if item["queue_slot"] == "backup"],
        key=lambda x: int(x["queue_index"]),
    )
    for item in backups:
        if not item.get("launched") and job_status(item["run_id"]) == "missing":
            return item
    return None


def is_cuda_failure(item: dict, status: str) -> bool:
    text = json.dumps(progress(item["run_id"]), sort_keys=True) + "\n" + str(item.get("failed_reason", ""))
    text = text.lower()
    return status in {"error", "failed", "failed_cuda"} and (
        "cuda" in text or "illegal memory access" in text or "out of memory" in text or "cublas" in text
    )


def retry_attempts_for(state: dict, run_id: str) -> int:
    return sum(1 for item in state["jobs"].values() if item.get("retry_of") == run_id)


def next_retry_slot(state: dict) -> str:
    counts = {slot: 0 for slot in STABLE_RETRY_SLOTS}
    for item in state["jobs"].values():
        if item.get("retry_of") and item.get("queue_slot") in counts and not item.get("analyzed"):
            counts[item["queue_slot"]] += 1
    return min(STABLE_RETRY_SLOTS, key=lambda slot: counts[slot])


def max_queue_index(state: dict, slot: str) -> int:
    vals = [int(item.get("queue_index", 0)) for item in state["jobs"].values() if item.get("queue_slot") == slot]
    return max(vals) if vals else -1


def enqueue_cuda_retry(state: dict, item: dict) -> None:
    if item.get("retry_of"):
        return
    for other in state["jobs"].values():
        if other.get("retry_of") == item["run_id"] and job_status(other["run_id"]) == "done":
            return
        if (
            other.get("retry_of") == item["run_id"]
            and not other.get("analyzed")
            and job_status(other["run_id"]) not in {"error", "failed", "failed_cuda"}
        ):
            return
    attempts = retry_attempts_for(state, item["run_id"])
    if attempts >= MAX_CUDA_RETRIES:
        return
    slot = next_retry_slot(state)
    retry_id = f"{item['run_id']}_retry{attempts + 1}_{slot}"
    if retry_id in state["jobs"]:
        return
    retry = dict(item)
    retry.update(
        {
            "run_id": retry_id,
            "queue_slot": slot,
            "queue_index": max_queue_index(state, slot) + 1,
            "slot": slot,
            "launched": False,
            "analyzed": False,
            "pid": None,
            "retry_of": item["run_id"],
            "retry_attempt": attempts + 1,
            "scheduled_at": time.time(),
        }
    )
    for key in ["launched_at", "completed_at", "failed_reason"]:
        retry.pop(key, None)
    state["jobs"][retry_id] = retry
    print(f"[retry] queued {retry_id} for {item['run_id']} on {slot}", flush=True)


def mark_cuda_failures(state: dict) -> None:
    for item in list(state["jobs"].values()):
        if not item.get("launched"):
            continue
        status = job_status(item["run_id"])
        if status not in {"error", "failed", "failed_cuda"}:
            continue
        if is_cuda_failure(item, status):
            enqueue_cuda_retry(state, item)
        if item.get("analyzed"):
            continue
        item["analyzed"] = True
        item["failed_reason"] = progress(item["run_id"]).get("error", status)


def reap_children() -> None:
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        except OSError:
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
        mark_cuda_failures(state)
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
        if all_jobs_finished(state):
            final_manual_video_export(state)
        save_state(state)
        time.sleep(300)


if __name__ == "__main__":
    main()
