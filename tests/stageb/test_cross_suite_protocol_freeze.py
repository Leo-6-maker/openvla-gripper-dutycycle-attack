from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.stageb.audit_cross_suite_readiness import audit_protocol, load_yaml  # noqa: E402


def test_protocol_freezes_detector_and_runtime_parameters():
    protocol = load_yaml(REPO_ROOT / "configs" / "sc5_cross_suite_protocol_v1.yaml")
    rows = audit_protocol(protocol)
    failures = [row for row in rows if row["status"] != "PASS"]
    assert failures == []
    runtime = protocol["runtime_freeze"]
    assert runtime["k_attack_frames"] == 10
    assert runtime["target_token_id"] == 31744
    assert runtime["pgd_steps"] == 20


def test_protocol_declares_suite_matched_victim_checkpoints():
    protocol = load_yaml(REPO_ROOT / "configs" / "sc5_cross_suite_protocol_v1.yaml")
    policies = protocol["victim_policies"]
    assert policies["libero_spatial"]["unnorm_key"] == "libero_spatial"
    assert policies["libero_goal"]["unnorm_key"] == "libero_goal"
    assert policies["libero_10"]["unnorm_key"] == "libero_10"
