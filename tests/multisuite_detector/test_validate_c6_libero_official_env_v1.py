from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.multisuite_detector import validate_c6_libero_official_env_v1 as m


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_parse_version_tuple():
    assert m.parse_version_tuple("3.10.12") >= (3, 9)
    assert m.parse_version_tuple("3.8.10") < (3, 9)


def test_missing_python_holds(tmp_path):
    out = tmp_path / "out"
    args = argparse.Namespace(python=str(tmp_path / "missing-python"), repo_root=str(tmp_path), require_cuda=False, output_root=str(out), git_commit="test", tests=[])
    rc = m.run(args)
    assert rc != 0
    assert load(out / "libero_official_env_validation.json")["status"] == "HOLD_ENV_PROBE_FAILED"


def test_current_python_probe_runs(tmp_path):
    out = tmp_path / "out"
    args = argparse.Namespace(python=sys.executable, repo_root=str(Path(__file__).resolve().parents[2]), require_cuda=False, output_root=str(out), git_commit="test", tests=[])
    rc = m.run(args)
    report = load(out / "libero_official_env_validation.json")
    assert "probe" in report
    if sys.version_info >= (3, 9):
        assert rc in (0, 2)
        assert report["status"].startswith("PASS_") or report["status"].startswith("HOLD_")
    else:
        assert rc != 0
