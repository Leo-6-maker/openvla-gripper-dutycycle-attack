from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.detector_v5 import run_stage_o_observability as stage_o


def _manifest(path: Path) -> None:
    rows = []
    for suite in stage_o.SUITES:
        for index in range(10):
            rows.append({
                "canonical_parent_key": f"{suite}/task_{index:02d}/state_48",
                "suite": suite, "task_index": index, "state_index": 48,
            })
    path.write_text(json.dumps({"parents": rows}) + "\n", encoding="utf-8")


def test_stage_o_uses_preregistered_seeds_and_dynamic_eight_workers(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    _manifest(manifest)

    def fake_run(argv, **kwargs):
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        job = json.loads((output_dir / "JOB.json").read_text(encoding="utf-8"))
        (output_dir / "RESULT.json").write_text(
            json.dumps({**job, "status": "PASS", "source_commit": "commit", "source_tree": "tree",
                        "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0}) + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(stage_o.subprocess, "run", fake_run)
    args = SimpleNamespace(
        parent_manifest=manifest, output_root=tmp_path / "stage_o",
        runner_command="fake_runner --job-path {job_path} --output-dir {output_dir}",
        source_commit="commit", source_tree="tree", salt="test", gpus=list(range(8)),
    )
    report = stage_o.run(args)
    assert report["status"] == "PASS"
    assert report["jobs"] == 480
    assert report["queue_progress"]["done"] == 480
    manifest_value = json.loads((args.output_root / "STAGE_O_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest_value["seeds"] == [2026080711, 2026080712, 2026080713]
    assert manifest_value["split_counts"]["libero_goal"] == {"train": 6, "validation": 2, "untouched_test": 2}
    audit = json.loads((args.output_root / "STAGE_O_AUDIT.json").read_text(encoding="utf-8"))
    assert audit["verdict"] == "PASS"
    complete = json.loads((args.output_root / "STAGE_O_COMPLETE.json").read_text(encoding="utf-8"))
    assert complete["status"] == "STAGE_O_PASS"
    assert (args.output_root / "SHA256SUMS.sha256").is_file()


def test_stage_o_rejects_non_eight_gpu_formal_pool(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _manifest(manifest)
    args = SimpleNamespace(
        parent_manifest=manifest, output_root=tmp_path / "stage_o",
        runner_command="fake_runner", source_commit="commit", source_tree="tree", salt="test", gpus=[0, 1],
    )
    try:
        stage_o.run(args)
    except RuntimeError as exc:
        assert str(exc) == "STAGE_O_REQUIRES_EIGHT_UNIQUE_GPUS"
    else:
        raise AssertionError("Stage O accepted a non-eight GPU pool")


def test_stage_o_expands_to_nine_sixty_jobs_for_eighty_parent_manifest(tmp_path: Path) -> None:
    rows = []
    for suite in stage_o.SUITES:
        for index in range(20):
            rows.append({
                "canonical_parent_key": f"{suite}/task_{index:02d}/state_48",
                "suite": suite, "task_index": index, "state_index": 48,
            })
    specs = stage_o._job_specs(rows, "test")
    assert len(specs) == 960
    counts = {}
    for job in specs:
        counts[(job["suite"], job["split"])] = counts.get((job["suite"], job["split"]), 0) + 1
    assert counts[("libero_goal", "train")] == 144
    assert counts[("libero_goal", "validation")] == 48
    assert counts[("libero_goal", "untouched_test")] == 48
