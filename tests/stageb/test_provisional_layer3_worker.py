import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.stageb.run_provisional_layer3_two_suite_worker import command_for_job  # noqa: E402


def test_worker_command_uses_suite_model_detector_and_output():
    job = {
        "condition": "VIS",
        "suite": "libero_spatial",
        "model_path": "/models/spatial",
        "unnorm_key": "libero_spatial",
        "task_idx": "3",
        "state_id": "4",
        "teacher_anchor": "50",
        "attack_seed": "81",
        "output_dir": "/tmp/out",
        "render_gpu": "3",
        "detector_path": "/tmp/model.pt",
    }
    cmd = command_for_job(job, python_bin="/py")
    joined = " ".join(cmd)
    assert "--suite libero_spatial" in joined
    assert "--model_path /models/spatial" in joined
    assert "--unnorm_key libero_spatial" in joined
    assert "--mlp_path /tmp/model.pt" in joined
    assert "--write_video" in cmd

