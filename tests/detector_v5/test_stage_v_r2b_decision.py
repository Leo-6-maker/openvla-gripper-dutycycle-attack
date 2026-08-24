from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.detector_v5.audit_stage_v_r2b_decision import audit


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            rows.append(f"{sha(path)}  {path.relative_to(root).as_posix()}")
    sums = root / "SHA256SUMS"
    sums.write_text("\n".join(rows) + "\n", encoding="utf-8")
    (root / "SHA256SUMS.sha256").write_text(f"{sha(sums)}  SHA256SUMS\n", encoding="utf-8")


def test_r2b_not_required_decision_audit_is_independent_and_bound(tmp_path: Path) -> None:
    r2a = tmp_path / "r2a"
    write_json(r2a / "STAGE_V_CLOSURE_RECEIPT.json", {
        "status": "STAGE_V_FORMAL_MAP_CLOSED", "accepted_parents": 40, "completed_branches": 2880,
    })
    write_json(r2a / "STAGE_V_COUNTERFACTUAL_AUDIT.json", {"verdict": "PASS"})
    seal(r2a)
    manifest = tmp_path / "r2a_manifest.json"
    candidate = tmp_path / "candidate.json"
    source_manifest = tmp_path / "source_manifest.json"
    write_json(manifest, {"parents": []})
    write_json(candidate, {"parents": []})
    write_json(source_manifest, {"all_candidate_audits": []})
    decision_root = tmp_path / "decision"
    write_json(decision_root / "STAGE_V_R2B_DECISION.json", {
        "schema": "STAGE_V_R2B_PRE_REGISTERED_DECISION_V1",
        "status": "R2B_NOT_REQUIRED", "r2a_root": str(r2a.resolve()),
        "r2a_manifest_sha256": sha(manifest), "candidate_manifest_sha256": sha(candidate),
        "source_clean_parent_manifest": str(source_manifest.resolve()),
        "source_clean_parent_manifest_sha256": sha(source_manifest),
        "source_artifact_binding_mode": "FROZEN_CANDIDATE_METADATA_ONLY",
        "q1_q2_replay_artifacts_reused": False,
        "selected_count": 0, "selected_parents": [], "errors": [],
        "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0,
    })
    result = audit(
        decision_root, r2a_root=r2a, r2a_manifest=manifest, candidate_manifest=candidate,
        source_clean_parent_manifest=source_manifest,
        expected_source_commit="commit", expected_source_tree="tree",
    )
    assert result["verdict"] == "PASS"
    assert json.loads((decision_root / "STAGE_V_R2B_DECISION_AUDIT.json").read_text())["verdict"] == "PASS"
