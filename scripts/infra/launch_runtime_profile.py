#!/usr/bin/env python3
"""Launch runtime profile with environment isolation. Must be called BEFORE any torch/lib import."""
import os, sys, json, argparse, subprocess

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SRC_DIR))

parser = argparse.ArgumentParser()
parser.add_argument("--profile", required=True, choices=["fp32_eager", "bf16_eager", "bf16_flash2"])
parser.add_argument("--cuda_devices", default="6")
parser.add_argument("--runner", default="scripts/infra/run_spatial_closed_loop_canary.py")
parser.add_argument("--model_path", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--stage", default="all", choices=["c0", "c1", "all"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_steps", type=int, default=220)
parser.add_argument("--task_idx", type=int, default=None)
parser.add_argument("--init_state_idx", type=int, default=None)
args = parser.parse_args()

PROFILES = {
    "fp32_eager": {"dtype": "float32", "attn": "eager", "flash2": False},
    "bf16_eager": {"dtype": "bfloat16", "attn": "eager", "flash2": False},
    "bf16_flash2": {"dtype": "bfloat16", "attn": "flash_attention_2", "flash2": True},
}
profile = PROFILES[args.profile]

# Set environment BEFORE any imports
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

runner_cmd = [
    sys.executable,
    os.path.join(REPO_ROOT, args.runner),
    "--model_path", args.model_path,
    "--output_dir", args.output_dir,
    "--cuda_visible_devices", args.cuda_devices,
    "--stage", args.stage,
    "--dtype", profile["dtype"],
    "--attn_implementation", profile["attn"],
    "--seed", str(args.seed),
    "--max_steps", str(args.max_steps),
]
if args.task_idx is not None:
    runner_cmd.extend(["--task_idx", str(args.task_idx)])
if args.init_state_idx is not None:
    runner_cmd.extend(["--init_state_idx", str(args.init_state_idx)])

print("Profile: %s | dtype=%s attn=%s flash2=%s" % (
    args.profile, profile["dtype"], profile["attn"], profile["flash2"]))
print("Cmd: %s" % " ".join(runner_cmd))

result = subprocess.run(runner_cmd, env=env, cwd=REPO_ROOT)
sys.exit(result.returncode)
