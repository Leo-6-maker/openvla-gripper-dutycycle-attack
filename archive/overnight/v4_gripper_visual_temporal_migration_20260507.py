# Early visual temporal migration experiment. Helped shift the project toward gripper-targeted temporal PGD.
#!/usr/bin/env python3
"""Run visual/temporal PGD migration smokes for gripper primitive hijacking."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

PY = "python"
BASE_MODEL = "${OPENVLA_BASE_MODEL_DIR}"
MODEL = "${OPENVLA_MODEL_ROOT}/openvla-7b-finetuned-libero-spatial"
ROOT = Path("outputs/v4/gripper_prompt_hijack_20260507/visual_temporal_pgd_migration")
RUNS = ROOT / "runs"
TABLES = ROOT / "tables"
LOGS = ROOT / "logs"
VIDEOS = ROOT / "videos"
THRESH = "outputs/v4/stage2_forceopen_planbv2_20260505/preflight/proxy_local_thresholds_close_transition.json"

JOBS = [
    {
        "run_id": "GPH_visual_TC_prevdelta_margin_seed0_gpu67",
        "cfg": "outputs/v4/replication_phase_semantic_physical_gap_20260506/preflight/v4_attack_TC_prevdelta_margin.yaml",
        "objective": "gripper_logit_margin_cw",
    },
    {
        "run_id": "GPH_visual_TC_region_smooth010_seed0_gpu67",
        "cfg": "outputs/v4/replication_phase_semantic_physical_gap_20260506/preflight/v4_attack_TC_prevdelta_region_smooth010.yaml",
        "objective": "gripper_open_region_ce",
    },
]


COMMON = [
    "scripts/v4_run_eval_openvla.py",
    "--tasks_config",
    "configs/v4_tasks_libero.yaml",
    "--directions_config",
    "configs/v4_directions.yaml",
    "--thresholds",
    THRESH,
    "--task_id",
    "libero_spatial_black_bowl",
    "--trigger",
    "priv_lift_carry_gate_burst_budgeted",
    "--rho",
    "0.03",
    "--seed",
    "0",
    "--episodes",
    "3",
    "--state_ids",
    "5,7,8",
    "--max_steps_override",
    "400",
    "--model_path",
    MODEL,
    "--base_model_code_dir",
    BASE_MODEL,
    "--unnorm_key",
    "libero_spatial",
    "--camera_obs_key",
    "agentview_image",
    "--model_gpu_device_id",
    "-1",
    "--render_gpu_device_id",
    "6",
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
]


def run_job(job: dict) -> int:
    out = RUNS / job["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "6,7",
            "OPENVLA_CUDA_MAX_MEMORY": "10000MiB",
            "PYTHONUNBUFFERED": "1",
            "V4_LIFT_CLOSED_GRIPPER_SIGN": "positive",
            "V4_FORCE_OPEN_ENV_SIGN": "negative",
            "V4_GATE_BURST_STEPS": "10",
            "V4_PRIV_LIFT_BOWL_Z_DELTA_MIN": "0.04",
            "V4_TARGET_OBJECT_NAME": "akita_black_bowl_1",
            "V4_TARGET_RECEPTACLE_NAME": "plate_1",
        }
    )
    cmd = [
        PY,
        *COMMON,
        "--attack_config",
        job["cfg"],
        "--attack_objective",
        job["objective"],
        "--output_root",
        str(out),
        "--run_id",
        job["run_id"],
    ]
    log = LOGS / f"{job['run_id']}.log"
    with log.open("w", encoding="utf-8") as fp:
        proc = subprocess.Popen(cmd, cwd=Path.cwd(), env=env, stdout=fp, stderr=subprocess.STDOUT)
        print(f"[launch] {job['run_id']} pid={proc.pid}", flush=True)
        rc = proc.wait()
    print(f"[done] {job['run_id']} rc={rc}", flush=True)
    analyze(job["run_id"])
    return rc


def analyze(run_id: str) -> None:
    run_dir = RUNS / run_id / run_id
    out = TABLES / run_id
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([PY, "scripts/v4_analyze_attack_efficacy.py", "--input_root", str(run_dir), "--output_dir", str(out)], cwd=Path.cwd(), check=False)
    video_dir = VIDEOS / run_id
    video_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            PY,
            "scripts/v4_render_episode_video_from_steps.py",
            "--run_dir",
            str(run_dir),
            "--episode_ids",
            "0,1,2",
            "--output_dir",
            str(video_dir),
            "--tasks_config",
            "configs/v4_tasks_libero.yaml",
            "--task_id",
            "libero_spatial_black_bowl",
            "--render_gpu_device_id",
            "6",
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


def main() -> None:
    os.chdir(Path(__file__).resolve().parents[1])
    for d in [RUNS, TABLES, LOGS, VIDEOS]:
        d.mkdir(parents=True, exist_ok=True)
    for job in JOBS:
        rc = run_job(job)
        if rc != 0:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
