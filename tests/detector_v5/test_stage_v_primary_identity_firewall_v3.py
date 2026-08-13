from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.detector_v5 import audit_stage_v_primary_identity_firewall_v3 as firewall


def _identity(index: int) -> str:
    suite = ("libero_10", "libero_goal", "libero_object", "libero_spatial")[index % 4]
    return f"{suite}/task_{index % 10:02d}/state_{index:02d}"


def _write(path: Path, value) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path, overlap: bool = False) -> tuple[Path, Path, Path, Path, Path, list[str]]:
    attempted = [_identity(i) for i in range(55)]
    final = attempted[:40]
    _write(tmp_path / "exact55.json", {"attempted_identities": [{"canonical_parent_key": x} for x in attempted]})
    _write(tmp_path / "final.json", {"parents": [{"canonical_parent_key": x} for x in final]})
    _write(tmp_path / "split.json", {"counts": {"TRAIN": 24, "VAL": 8, "TEST": 8}, "parents": [{"episode_id": x} for x in final]})
    _write(tmp_path / "plan.json", {"status": "PASS", "manifest_status": "PASS_EXACT_40X24_PLAN_ONLY", "parent_count": 40, "probe_count_total": 960, "planned_branch_authority_count": 3840, "outcomes_read": False, "intervention_executed": False, "protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0})
    _write(tmp_path / "plan_manifest.json", {"probes": [{"parent_key": x} for x in final]})
    primary = [attempted[40] if overlap else _identity(100 + i) for i in range(8)]
    specs = []
    for i, item in enumerate(primary):
        path = _write(tmp_path / f"primary_{i}.json", {"identities": [{"episode_id": item}]})
        specs.append(f"P{i}={path}")
    return (tmp_path / "exact55.json", tmp_path / "final.json", tmp_path / "split.json", tmp_path / "plan.json", tmp_path / "plan_manifest.json", specs)


def test_exact55_firewall_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exact55, final, split, plan, plan_manifest, specs = _fixture(tmp_path)
    monkeypatch.setattr(firewall, "rename_noreplace", lambda source, target: source.rename(target))
    result = firewall.audit(exact55_registry=exact55, final_manifest=final, final_split=split, exact_plan_result=plan, exact_plan_manifest=plan_manifest, primary_manifests=specs, historical_quarantine=None, output_root=tmp_path / "out")
    assert result["status"] == "PASS_PRIMARY_DATA_FIREWALL_EXACT55"
    assert result["primary_identity_firewall"]["attempted_overlap_count"] == 0


def test_exact55_firewall_fails_closed_on_overlap(tmp_path: Path) -> None:
    exact55, final, split, plan, plan_manifest, specs = _fixture(tmp_path, overlap=True)
    with pytest.raises(ValueError, match="primary identity overlap"):
        firewall.audit(exact55_registry=exact55, final_manifest=final, final_split=split, exact_plan_result=plan, exact_plan_manifest=plan_manifest, primary_manifests=specs, historical_quarantine=None, output_root=tmp_path / "out")
