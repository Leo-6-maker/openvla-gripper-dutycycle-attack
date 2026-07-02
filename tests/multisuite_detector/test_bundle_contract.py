import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.multisuite_detector.export_detector_bundle import load_config
from tools.multisuite_detector.verify_detector_bundle import REQUIRED_FILES


def test_yaml_config_is_parsed(tmp_path):
    config = tmp_path / "detector.yaml"
    config.write_text("condition_id: clean2000\nattack:\n  epsilon: 0\n")

    try:
        import yaml  # noqa: F401
    except ImportError:
        try:
            load_config(str(config))
        except SystemExit as exc:
            assert "PyYAML required" in str(exc)
        else:
            raise AssertionError("YAML config must fail closed without PyYAML")
        return

    parsed = load_config(str(config))
    assert parsed["condition_id"] == "clean2000"
    assert parsed["attack"]["epsilon"] == 0


def test_bundle_verifier_requires_declared_contract_files():
    assert "data_contract.json" in REQUIRED_FILES
    assert "normalization.json" in REQUIRED_FILES


if __name__ == "__main__":
    test_yaml_config_is_parsed(Path(tempfile.mkdtemp()))
    test_bundle_verifier_requires_declared_contract_files()
    print("bundle contract tests passed")
