import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_pipeline(*args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_attack_pipeline.py"), *args, "--dry_run"],
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout


def run_pipeline_raw(*args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_attack_pipeline.py"), *args, "--dry_run"],
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_vis_dispatches_paper_pgd_params_and_prev_delta():
    out = run_pipeline("--task", "black_bowl", "--state", "7", "--seed", "1", "--condition", "vis_margin_prevdelta")
    assert "V4_FIXED_ATTACK_START=75" in out
    assert "V4_FIXED_ATTACK_END=84" in out
    assert "--attack_config configs/paper_black_bowl_attack.yaml" in out
    assert "--attack_objective gripper_logit_margin_cw" in out
    assert "--epsilon 0.10" in out
    assert "--step_size 0.020" in out
    assert "--attack_steps 20" in out
    assert "--temporal_init prev_delta" in out
    assert "--cw_margin 5.0" in out
    assert "--state_ids 7" in out


def test_random_dispatches_arm_only_objective():
    out = run_pipeline("--task", "black_bowl", "--state", "7", "--seed", "1", "--condition", "ctrl_random_direction")
    assert "--attack_objective untargeted_arm_clean_token_ce" in out
    assert "--temporal_init none" in out


def test_constant_pregrasp_dispatches_non_pgd_objective():
    out = run_pipeline("--task", "black_bowl", "--state", "7", "--seed", "1", "--condition", "constant_delta_pregrasp")
    assert "V4_FIXED_ATTACK_START=35" in out
    assert "V4_FIXED_ATTACK_END=45" in out
    assert "V4_CONSTANT_DELTA_GRIPPER=-1.0" in out
    assert "--attack_objective constant_delta_pregrasp" in out
    assert "--temporal_init none" in out


def test_constant_pregrasp_rejects_window_override():
    proc = run_pipeline_raw(
        "--task", "black_bowl",
        "--state", "7",
        "--seed", "1",
        "--condition", "constant_delta_pregrasp",
        "--window_start", "75",
    )
    assert proc.returncode != 0
    assert "fixed to the 35-45" in proc.stderr
