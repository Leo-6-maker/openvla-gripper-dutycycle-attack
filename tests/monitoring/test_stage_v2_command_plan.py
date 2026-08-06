from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.monitoring.register_stage_v2_command_plan import build_plan


def test_command_plan_binds_all_files_and_cpu_only_env(tmp_path: Path) -> None:
    root = tmp_path / "stage-v"
    root.mkdir()
    parent_manifest = tmp_path / "parents.json"
    parent_manifest.write_text("{}\n", encoding="utf-8")
    (root / "SUPERVISOR_START.json").write_text(json.dumps({"parent_manifest": str(parent_manifest)}) + "\n", encoding="utf-8")
    (root / "RUN_MANIFEST.json").write_text("{}\n", encoding="utf-8")
    files = []
    for name in ("runner.py", "auditor.py", "config.json"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        files.append(path)
    args = Namespace(
        goal_root=tmp_path / "goal",
        stage_v_root=root,
        stage_v_source_commit="stage-v-commit",
        stage_v_source_tree="stage-v-tree",
        stage_v2_source_commit="stage-v2-commit",
        stage_v2_source_tree="stage-v2-tree",
        runner=files[0],
        auditor=files[1],
        config=files[2],
        cwd=tmp_path,
        python="python",
    )
    plan = build_plan(args)
    assert plan["schema"] == "STAGE_V2_COMMAND_PLAN_V1"
    assert plan["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert plan["env"]["OMP_NUM_THREADS"] == "1"
    assert plan["expected_run_manifest_sha256"]
    assert "{output_root}" in plan["command_template"]
