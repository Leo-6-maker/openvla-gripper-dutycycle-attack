#!/usr/bin/env python3
"""Launch runtime profile with environment isolation. MUST be called before any torch/lib import."""
import os, sys, json, argparse, subprocess

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SRC_DIR))

parser = argparse.ArgumentParser()
parser.add_argument("--profile", required=True, choices=["fp32_eager", "bf16_eager", "bf16_flash2"])
parser.add_argument("--cuda_devices", default="6")
parser.add_argument("--model_path", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_steps", type=int, default=220)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--start_ep", type=int, default=0)
parser.add_argument("--end_ep", type=int, default=999)
# Single-episode mode (uses old runner)
parser.add_argument("--task_idx", type=int, default=None)
parser.add_argument("--init_state_idx", type=int, default=None)
# Plan mode (uses new single-process runner)
parser.add_argument("--plan", default=None, help="JSON plan for single-process runner")
args = parser.parse_args()

PROFILES = {
    "fp32_eager": {"dtype": "float32", "attn": "eager", "flash2": False},
    "bf16_eager": {"dtype": "bfloat16", "attn": "eager", "flash2": False},
    "bf16_flash2": {"dtype": "bfloat16", "attn": "flash_attention_2", "flash2": True},
}
profile = PROFILES[args.profile]

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
env["MUJOCO_GL"] = "egl"
env["HF_HUB_OFFLINE"] = "1"
env["TRANSFORMERS_OFFLINE"] = "1"
env["HOME"] = "/mnt/sdc/dty_user/openvla_attack/sandbox_home"
env["TMPDIR"] = "/mnt/sdc/dty_user/openvla_attack/tmp"
env["TOKENIZERS_PARALLELISM"] = "false"

if profile["flash2"]:
    overlay = "/mnt/sdc/dty_user/openvla_attack/envs/flash2_overlay"
    env["PYTHONPATH"] = overlay + ":" + env.get("PYTHONPATH", "")

if args.plan:
    # Single-process runner (load model once, run all plan episodes)
    runner = os.path.join(REPO_ROOT, "scripts/infra/run_single_process_clean.py")
    runner_cmd = [
        sys.executable, runner,
        "--plan", args.plan,
        "--model_path", args.model_path,
        "--output_dir", args.output_dir,
        "--dtype", profile["dtype"],
        "--attn", profile["attn"],
        "--seed", str(args.seed),
        "--max_steps", str(args.max_steps),
        "--start_ep", str(args.start_ep),
        "--end_ep", str(args.end_ep),
    ]
    if args.resume:
        runner_cmd.append("--resume")
else:
    # Old single-episode runner
    runner = os.path.join(REPO_ROOT, "scripts/infra/run_spatial_closed_loop_canary.py")
    runner_cmd = [
        sys.executable, runner,
        "--model_path", args.model_path,
        "--output_dir", args.output_dir,
        "--cuda_visible_devices", args.cuda_devices,
        "--stage", "all" if args.task_idx is not None else "c0",
        "--dtype", profile["dtype"],
        "--attn_implementation", profile["attn"],
        "--seed", str(args.seed),
        "--max_steps", str(args.max_steps),
    ]
    if args.task_idx is not None:
        runner_cmd.extend(["--task_idx", str(args.task_idx)])
    if args.init_state_idx is not None:
        runner_cmd.extend(["--init_state_idx", str(args.init_state_idx)])

print("Profile: %s | dtype=%s attn=%s" % (args.profile, profile["dtype"], profile["attn"]))
print("Cmd: %s" % " ".join(runner_cmd))

result = subprocess.run(runner_cmd, env=env, cwd=REPO_ROOT)
sys.exit(result.returncode)
