from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_g7_is_read_only_post_freeze_runner() -> None:
    source = (ROOT / "scripts/detector_v5/run_r3_g7_test_evaluation.py").read_text(encoding="utf-8")
    assert "test_read_count\": 1" in source
    assert '"thresholds_frozen_before_test": True' in source
    assert '"model_selection_after_test": False' in source
    assert '"intervention_executed": False' in source
    assert '"v_phys_generated": False' in source
    assert "_train(" not in source
