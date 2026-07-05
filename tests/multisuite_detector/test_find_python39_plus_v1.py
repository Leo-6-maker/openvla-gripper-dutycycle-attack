from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.multisuite_detector import find_python39_plus_v1 as m


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_vtuple():
    assert m.vtuple("3.10.12") >= (3, 9)
    assert m.vtuple("3.8.10") < (3, 9)


def test_finder_selects_current_python(tmp_path):
    out = tmp_path / "out"
    args = argparse.Namespace(candidate=[sys.executable], output_root=str(out), git_commit="test", tests=[])
    rc = m.run(args)
    report = load(out / "python39_plus_interpreter_finder.json")
    if sys.version_info >= (3, 9):
        assert rc == 0
        assert report["status"] == m.PASS
        assert report["selected_python"]
    else:
        assert rc != 0


def test_version_for_current_python():
    version, error = m.version_for(sys.executable)
    assert not error
    assert version.startswith(f"{sys.version_info[0]}.{sys.version_info[1]}.")
