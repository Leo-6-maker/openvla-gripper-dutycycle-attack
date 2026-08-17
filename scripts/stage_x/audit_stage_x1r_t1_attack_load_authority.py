"""Fail-closed audit for the prospective X1R attack-load authority."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value(text: str, key: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(key)}:\s*([^\s#]+)", text, flags=re.MULTILINE)
    return match.group(1) if match else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/STAGE_X_X1R_T1_PROSPECTIVE_DETECTOR_PGD_PROTOCOL_V1.json")
    parser.add_argument("--stage-ix", type=Path, default=ROOT / "configs/STAGE_IX_CANONICAL_PGD_CONTRACT_V1.json")
    parser.add_argument("--fec", type=Path, default=ROOT / "configs/fec_attack_v5_open.yaml")
    parser.add_argument("--fec-region", type=Path, default=ROOT / "configs/fec_attack_v5_open_region.yaml")
    parser.add_argument("--sc5", type=Path, default=ROOT / "configs/sc5_cross_suite_protocol_v1.yaml")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    stage_ix = json.loads(args.stage_ix.read_text(encoding="utf-8"))
    fec_text = args.fec.read_text(encoding="utf-8")
    fec_region_text = args.fec_region.read_text(encoding="utf-8")
    sc5_text = args.sc5.read_text(encoding="utf-8")
    stage_ix_load = stage_ix["optimization"]
    fec_load = {key: value(fec_text, key) for key in ("epsilon", "step_size", "num_steps")}
    fec_region_load = {key: value(fec_region_text, key) for key in ("epsilon", "step_size", "num_steps")}
    sc5_load = {"vis_epsilon": value(sc5_text, "vis_epsilon"), "pgd_steps": value(sc5_text, "pgd_steps")}

    errors = []
    if protocol.get("attack_load_authority", {}).get("status") != "HOLD_ATTACK_LOAD_AUTHORITY":
        errors.append("PROTOCOL_DOES_NOT_DECLARE_ATTACK_LOAD_HOLD")
    if protocol.get("attack_load_authority", {}).get("pgd_authorized") is not False:
        errors.append("PGD_AUTHORIZATION_NOT_FALSE")
    if len({
        (str(stage_ix_load.get("epsilon")), str(stage_ix_load.get("step_size")), str(stage_ix_load.get("num_steps"))),
        tuple(fec_load.values()), tuple(fec_region_load.values()),
        (sc5_load["vis_epsilon"], None, sc5_load["pgd_steps"]),
    }) < 3:
        errors.append("HISTORICAL_LOAD_CONFLICT_NOT_DETECTED")

    files = (args.protocol, args.stage_ix, args.fec, args.fec_region, args.sc5)
    receipt: dict[str, Any] = {
        "schema": "STAGE_X_X1R_T1_ATTACK_LOAD_AUTHORITY_AUDIT_V1",
        "status": "STAGE_X_X1R_T1_HOLD_ATTACK_LOAD_AUTHORITY" if not errors else "HOLD_ATTACK_LOAD_AUDIT_INVALID",
        "pgd_authorized": False,
        "reason": "Do not choose among conflicting historical loads or adapt one after detector evidence; owner must freeze a suite-matched X1R load.",
        "candidates": {
            "stage_ix_canonical": {"path": str(args.stage_ix), "sha256": sha256(args.stage_ix), "status": stage_ix.get("status"), "victim": stage_ix.get("victim", {}).get("model_path"), "optimization": stage_ix_load},
            "fec_open_experimental": {"path": str(args.fec), "sha256": sha256(args.fec), "status": "EXPERIMENTAL_CANARY", "load": fec_load, "target_token_text": "31745 legacy fixed token"},
            "fec_open_region_experimental": {"path": str(args.fec_region), "sha256": sha256(args.fec_region), "status": "EXPERIMENTAL_CANARY", "load": fec_region_load, "objective": "prefix_locked_gripper_open_region_ce"},
            "sc5_historical": {"path": str(args.sc5), "sha256": sha256(args.sc5), "load": sc5_load, "target_token_text": "31744 CLIP_MEDIATED_OPEN historical contract"},
        },
        "incompatibilities": [
            "Stage IX canonical load is epsilon=0.10, step_size=0.020, PGD-20 and is bound to a single L10 F0 contract.",
            "FEC experimental loads are epsilon=0.03, step_size=0.006, num_steps=5 and use legacy 31745 text in one variant.",
            "SC5 historical load uses a different processor epsilon and target semantics.",
            "Current T1 suite-matched victim/native-token contract has no frozen X1R attack load or fresh parent authority.",
        ],
        "protected_counters": {"pgd_calls": 0, "env_step_calls": 0, "attack_outcome_reads": 0, "physical_interventions": 0, "vphys_reads": 0, "eval160_reads": 0, "protected_reads": 0},
        "eval160": "UNREAD", "protected_evaluation": "UNREAD", "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "output": str(args.output), "errors": errors}, sort_keys=True))
    return 0 if receipt["status"].startswith("STAGE_X_X1R_T1_HOLD") else 2


if __name__ == "__main__":
    raise SystemExit(main())
