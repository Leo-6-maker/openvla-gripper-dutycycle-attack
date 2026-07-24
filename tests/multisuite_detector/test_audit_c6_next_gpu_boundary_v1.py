from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.multisuite_detector import audit_c6_next_gpu_boundary_v1 as m


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_tool(tmp_path: Path, source: Path):
    out = tmp_path / "out"
    args = argparse.Namespace(source_file=str(source), repo_root=str(tmp_path), output_root=str(out), git_commit="test", files_changed=[], tests=[])
    return m.run(args), out


def test_gpu_boundary_passes(tmp_path):
    source = tmp_path / "runner.py"
    source.write_text("def run_real():\n    if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')\n    OffScreenRenderEnv()\n    load_model('x')\n", encoding="utf-8")
    rc, out = run_tool(tmp_path, source)
    assert rc == 0
    assert load(out / "next_gpu_boundary_static_audit.json")["status"] == m.PASS


def test_missing_source_holds(tmp_path):
    rc, out = run_tool(tmp_path, tmp_path / "missing.py")
    assert rc != 0
    assert load(out / "next_gpu_boundary_static_audit.json")["status"] == "HOLD_SOURCE_FILE_NOT_FOUND"


def test_incomplete_evidence_holds(tmp_path):
    source = tmp_path / "runner.py"
    source.write_text("def run_real():\n    load_model('x')\n", encoding="utf-8")
    rc, out = run_tool(tmp_path, source)
    assert rc != 0
    assert load(out / "next_gpu_boundary_static_audit.json")["status"] == "HOLD_GPU_BOUNDARY_EVIDENCE_INCOMPLETE"
