import numpy as np
import torch

from gripper_attack.official_detector_features import (
    CANONICAL_25D_FEATURES,
    CLEAN_POLICY_FEATURE_NAMES,
    SC5StreamingFeatureAdapterV2,
    policy_intent_9d,
    top_token_evidence,
)


def test_policy_intent_is_finite_and_frozen_9d():
    logits = torch.zeros(32, dtype=torch.float32)
    values = policy_intent_9d(logits, open_token_ids=(1, 2), close_token_ids=(3, 4))
    assert len(CLEAN_POLICY_FEATURE_NAMES) == 9
    assert len(values) == 9
    assert np.isfinite(np.asarray(values, dtype=np.float32)).all()


def test_streaming_features_keep_canonical_25d_order():
    stream = SC5StreamingFeatureAdapterV2()
    row = stream.update(
        step_id=0,
        raw_gripper=1.0,
        env_gripper=-1.0,
        gripper_qpos=0.0,
        gripper_opening_proxy=0.0,
        eef_x=0.0,
        eef_y=0.0,
        eef_z=0.0,
        eef_vx=0.0,
        eef_vy=0.0,
        eef_vz=0.0,
        action_dx=0.0,
        action_dy=0.0,
        action_dz=0.0,
        action_gripper=1.0,
    )
    assert row["valid"] is True
    values = [row["features"][name] for name in CANONICAL_25D_FEATURES]
    assert len(values) == 25
    assert np.isfinite(np.asarray(values, dtype=np.float32)).all()


def test_top_token_evidence_is_compact_and_ordered():
    ids, values = top_token_evidence(torch.tensor([0.0, 4.0, 2.0, 1.0]), top_k=2)
    assert ids == [1, 2]
    assert values == [4.0, 2.0]
