from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stageb.audit_cross_suite_readiness import REQUIRED_FALSE_FLAGS, load_yaml  # noqa: E402


def test_protocol_disables_target_suite_leakage_channels():
    protocol = load_yaml(REPO_ROOT / "configs" / "sc5_cross_suite_protocol_v1.yaml")
    detector = protocol["detector"]
    for flag in REQUIRED_FALSE_FLAGS:
        assert detector[flag] is False


def test_detector_feature_order_excludes_task_or_suite_identity():
    protocol = load_yaml(REPO_ROOT / "configs" / "sc5_cross_suite_protocol_v1.yaml")
    features = protocol["detector"]["feature_order"]
    joined = " ".join(features).lower()
    forbidden = ["task", "suite", "state_id", "object_name", "target_name", "anchor"]
    assert not any(token in joined for token in forbidden)
