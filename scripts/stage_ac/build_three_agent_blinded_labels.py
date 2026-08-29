#!/usr/bin/env python3
"""Validate and seal the three-agent AC4 blinded label panel before unblind."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "reports/STAGE_AC_AC4_NEUTRAL_BLIND_MANIFEST_V1.json"
INPUT = ROOT / "reports/STAGE_AC_AC4_THREE_AGENT_BLINDED_AI_ADJUDICATION_INPUT_V1.txt"
LABELS_OUT = ROOT / "reports/STAGE_AC_AC4_THREE_AGENT_BLINDED_AI_ADJUDICATION_LABELS_V1.json"
SEAL_OUT = ROOT / "reports/STAGE_AC_AC4_THREE_AGENT_BLINDED_AI_ADJUDICATION_LABEL_SEAL_V1.json"
EXPECTED_MANIFEST_SHA = "1b1a0aa3d24bf6aa2e21e83eda2a35b3a4751bd27963c1e4ab8ba5fe712e2c8f"
AGENTS = ("A", "B", "C")
LABELS = {
    "STABLE_GRASP",
    "PREMATURE_APERTURE",
    "CONTACT_LOSS",
    "PREMATURE_RELEASE_OR_DROP",
    "OBJECT_DISPLACEMENT",
    "AMBIGUOUS_OR_OCCLUDED",
    "NOT_IDENTIFIABLE",
}
CONFIDENCE = {"LOW", "MEDIUM", "HIGH"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest_sha = sha(MANIFEST)
    if manifest_sha != EXPECTED_MANIFEST_SHA:
        raise SystemExit(f"MANIFEST_SHA_MISMATCH:{manifest_sha}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    present = {
        row["blinded_video_id"]
        for row in manifest["rows"]
        if row.get("availability") == "PRESENT"
    }
    if len(present) != 91:
        raise SystemExit(f"PRESENT_ID_COUNT:{len(present)}")

    by_agent: dict[str, list[dict[str, str]]] = {agent: [] for agent in AGENTS}
    seen: set[tuple[str, str]] = set()
    for line_number, raw in enumerate(INPUT.read_text(encoding="utf-8").splitlines(), 1):
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split("|")
        if len(fields) != 4:
            raise SystemExit(f"INPUT_FIELDS:{line_number}")
        agent, video_id, label, confidence = fields
        if agent not in by_agent or video_id not in present:
            raise SystemExit(f"INPUT_ID_OR_AGENT:{line_number}:{raw}")
        if label not in LABELS or confidence not in CONFIDENCE:
            raise SystemExit(f"INPUT_VOCABULARY:{line_number}:{raw}")
        key = (agent, video_id)
        if key in seen:
            raise SystemExit(f"INPUT_DUPLICATE:{line_number}:{key}")
        seen.add(key)
        by_agent[agent].append({
            "blinded_video_id": video_id,
            "primary_label": label,
            "confidence": confidence,
        })

    for agent in AGENTS:
        rows = by_agent[agent]
        ids = {row["blinded_video_id"] for row in rows}
        if len(rows) != 91 or ids != present:
            raise SystemExit(f"INCOMPLETE_AGENT:{agent}:rows={len(rows)}:ids={len(ids)}")
        rows.sort(key=lambda row: row["blinded_video_id"])

    payload = {
        "schema": "STAGE_AC_AC4_THREE_AGENT_BLINDED_AI_ADJUDICATION_LABELS_V1",
        "status": "STAGE_AC_AC4_THREE_AGENT_LABELS_SEALED_BEFORE_UNBLIND",
        "reviewer_type": "THREE_AGENT_BLINDED_AI_ADJUDICATION",
        "human_review_gate_satisfied": False,
        "manifest_sha256": manifest_sha,
        "agent_count": 3,
        "present_video_count": 91,
        "row_schema": ["blinded_video_id", "primary_label", "confidence"],
        "agents": {
            "AGENT_A": by_agent["A"],
            "AGENT_B": by_agent["B"],
            "AGENT_C": by_agent["C"],
        },
        "reviewer_firewall": {
            "agent_sessions_mapping_exposure": False,
            "agent_sessions_automatic_label_exposure": False,
            "agent_sessions_model_suite_condition_dose_exposure": False,
            "agent_sessions_telemetry_exposure": False,
            "labels_sealed_before_unblind": True,
            "orchestrator_prior_mapping_exposure_before_this_panel": True,
        },
        "scientific_firewall": {
            "new_model_inference": 0,
            "new_env_step": 0,
            "new_open_intervention": 0,
            "new_pgd": 0,
            "new_protected_reads": 0,
            "automatic_labels_rewritten": 0,
        },
    }
    LABELS_OUT.write_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    label_sha = sha(LABELS_OUT)
    seal = {
        "schema": "STAGE_AC_AC4_THREE_AGENT_BLINDED_AI_ADJUDICATION_LABEL_SEAL_V1",
        "status": "STAGE_AC_AC4_THREE_AGENT_LABELS_SEALED",
        "label_sha256": label_sha,
        "label_bytes": LABELS_OUT.stat().st_size,
        "manifest_sha256": manifest_sha,
        "agent_count": 3,
        "row_count_per_agent": 91,
        "labels_sealed_before_unblind": True,
        "hidden_mapping_read_before_label_seal": False,
        "reviewer_sessions_mapping_exposure": False,
        "orchestrator_prior_mapping_exposure_before_this_panel": True,
        "reviewer_type": "THREE_AGENT_BLINDED_AI_ADJUDICATION",
        "human_review_gate_satisfied": False,
        "scientific_firewall": payload["scientific_firewall"],
    }
    SEAL_OUT.write_bytes((json.dumps(seal, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({
        "status": seal["status"],
        "label_bytes": seal["label_bytes"],
        "label_sha256": label_sha,
        "row_count_per_agent": 91,
        "manifest_sha256": manifest_sha,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
