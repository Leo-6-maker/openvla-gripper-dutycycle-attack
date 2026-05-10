# Offline prompt-only audit. Lift/carry gripper-open mass was weak, so this branch is not main evidence.
#!/usr/bin/env python3
from __future__ import annotations
import os, subprocess
from pathlib import Path

ROOT = Path("outputs/v4/gripper_prompt_hijack_20260507")
LOGS = ROOT / "logs"
PY = "python"

CMD = [
    PY,
    "scripts/v4_run_eval_openvla.py",
    "--tasks_config", "configs/v4_tasks_libero.yaml",
    "--directions_config", "configs/v4_directions.yaml",
    "--task_id", "libero_spatial_black_bowl",
    "--trigger", "clean",
    "--rho", "0.0",
    "--seed", "0",
    "--episodes", "3",
    "--state_ids", "5,7,8",
    "--max_steps_override", "400",
    "--model_path", "${OPENVLA_MODEL_ROOT}/openvla-7b-finetuned-libero-spatial",
    "--base_model_code_dir", "${OPENVLA_BASE_MODEL_DIR}",
    "--unnorm_key", "libero_spatial",
    "--camera_obs_key", "agentview_image",
    "--model_gpu_device_id", "-1",
    "--render_gpu_device_id", "0",
    "--image_size", "256",
    "--openvla_resize_size", "224",
    "--auto_patch_compat",
    "--libero_official_preprocess",
    "--center_crop",
    "--postprocess_gripper",
    "--deterministic_init_states",
    "--offline_prompt_audit",
    "--offline_prompt_audit_steps", "12",
    "--prompt_variants_path", "configs/v4_prompt_hijack_phase1_smoke_prompts_20260507.jsonl",
    "--prompt_id", "phase1_smoke_prompt_set",
    "--prompt_type", "offline_prompt_audit_smoke",
    "--target_primitive", "open_gripper",
    "--output_root", str(ROOT / "offline_audit"),
    "--run_id", "P1_offline_prompt_audit_smoke_state5_7_8_seed0",
]


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0,1",
        "OPENVLA_CUDA_MAX_MEMORY": "10000MiB",
        "PYTHONUNBUFFERED": "1",
        "V4_LIFT_CLOSED_GRIPPER_SIGN": "positive",
        "V4_FORCE_OPEN_ENV_SIGN": "negative",
    })
    log = LOGS / "phase1_offline_prompt_audit.log"
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(CMD, cwd=Path.cwd(), env=env, stdout=handle, stderr=subprocess.STDOUT)
    print(f"[launch] P1_offline_prompt_audit_state5_7_8_seed0 pid={proc.pid} log={log}", flush=True)


if __name__ == "__main__":
    main()
