import csv

import numpy as np
import torch

from scripts.stageb.audit_c3r_renderer_determinism import audit_seal
from scripts.stageb.run_c3r_renderer_determinism import action_exactness, array_diff, require_single_visible_gpu
from scripts.stageb.layer3_exact_restore_runner import RealOpenVLAPolicyAdapter


def test_array_diff_reports_small_rgb_drift():
    reference = np.zeros((4, 4, 3), dtype=np.uint8)
    candidate = reference.copy()
    candidate[1, 2, 0] = 1
    candidate[3, 0, 2] = 1

    result = array_diff(reference, candidate, "r1")

    assert result["r1_diff_count"] == 2
    assert result["r1_max_abs"] == 1.0
    assert result["r1_channel_counts"] == '{"0": 1, "1": 0, "2": 1}'
    assert result["r1_bbox"] == "[1, 0, 3, 2]"


def test_action_exactness_uses_raw_and_environment_bytes():
    class Policy:
        def act(self, _obs):
            return [0.0] * 6 + [1.0], [1, 2, 3, 4, 5, 6, 7]

    result = action_exactness(
        policy=Policy(),
        obs={"agentview_image": np.zeros((1, 1, 3), dtype=np.uint8)},
        expected_action=[0.0] * 6 + [1.0],
        expected_tokens=[1, 2, 3, 4, 5, 6, 7],
    )

    assert result["tokens_exact"]
    assert result["raw_action_exact"]
    assert result["env_action_exact"]
    assert result["gripper_semantic_exact"]


def test_audit_seal_detects_post_manifest_mutation(tmp_path):
    payload = tmp_path / "summary.json"
    payload.write_text('{"result":"PASS"}\n', encoding="utf-8")
    from scripts.stageb.audit_c3r_renderer_determinism import sha256_file

    with (tmp_path / "recursive_sha256_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "size", "sha256"])
        writer.writeheader()
        writer.writerow({"path": payload.name, "size": payload.stat().st_size, "sha256": sha256_file(payload)})

    assert audit_seal(tmp_path) == []
    payload.write_text('{"result":"FAIL"}\n', encoding="utf-8")
    assert audit_seal(tmp_path) == ["SHA:summary.json"]


def test_policy_input_stages_are_the_fingerprint_source():
    class Processor:
        def __call__(self, _prompt, image, return_tensors):
            assert return_tensors == "pt"
            array = np.asarray(image)
            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "pixel_values": torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).float(),
            }

    policy = RealOpenVLAPolicyAdapter(
        model=None,
        processor=Processor(),
        device="cpu",
        instruction="test instruction",
        unnorm_key="libero_goal",
        action_dim=7,
    )
    obs = {"agentview_image": np.zeros((8, 8, 3), dtype=np.uint8)}

    raw, prepared, inputs = policy.policy_input_stages(obs)
    fingerprint = policy.policy_input_fingerprint(obs)

    assert raw.shape == (8, 8, 3)
    assert prepared.shape == (224, 224, 3)
    assert inputs["input_ids"].tolist() == [[1, 2, 29871]]
    assert fingerprint["raw_agentview_sha256"]
    assert fingerprint["prepared_image_sha256"]
    assert fingerprint["pixel_values_sha256"]


def test_c3r_requires_one_visible_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert require_single_visible_gpu() == "2"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,4")
    import pytest

    with pytest.raises(Exception, match="exactly one"):
        require_single_visible_gpu()
