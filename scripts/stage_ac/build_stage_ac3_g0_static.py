#!/usr/bin/env python3
"""Build the static, treatment-naive AC3 G0 freeze from sealed AC2 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


MODELS = ("M0_OPENVLA", "M1_OPENVLA_OFT", "M2_PI05_LIBERO")
CONDITIONS = (("CLEAN_REFERENCE", 0), ("OPEN_T3", 3), ("OPEN_T5", 5), ("OPEN_T10", 10))
EXPOSURES = ("H0_UNTOUCHED", "HC_CLEAN_ONLY")
BRANCH_SALT = "STAGE_AC_AC3_BRANCH_SEED_V1_20260828"
BLIND_SALT = "STAGE_AC_AC4_BLIND_AUDIT_PARENT_SELECTION_V1_20260828"
BLIND_ID_SALT = "STAGE_AC_AC4_BLIND_VIDEO_ID_V1_20260828"
GATE = "STAGE_AC_AC3_AC4_AC5_TREATMENT_NAIVE_MULTI_MODEL_PHYSICAL_REPLICATION_PROGRAM_V1"
RECEIPT_RE = re.compile(r"^AC2-\d{4}\.json$")

AC2R3_ROOT = "reports/STAGE_AC_AC2R3_ROOT_SEAL_V1.json"
DENOMINATOR = "reports/STAGE_AC_AC2R3_MODEL_SPECIFIC_DENOMINATOR_LEDGER_V1.json"
ELIGIBILITY = "reports/STAGE_AC_AC2R3_ELIGIBILITY_RECOMPUTATION_V1.json"
MANIFEST = "reports/STAGE_AC_AC2_CLEAN_SCREEN_LAUNCH_MANIFEST_V1.json"
AC2_SOURCE = "reports/STAGE_AC_AC2R2_RUNTIME_SOURCE_AUTHORITY_V1.json"
AC2_PROTOCOL = "configs/STAGE_AC_AC2R2_CLEAN_SCREEN_REPAIR_PROTOCOL_V1.json"
SOURCE_FILES = (
    "scripts/stage_ac/run_stage_ac2_clean_screen.py",
    "scripts/stage_ac/run_stage_ac2_family_worker.py",
    "src/stage_ac/eligibility_v2.py",
    "src/stage_z_preparation/z3_contract.py",
    "src/gripper_attack/stage_v_m3_5_physical_taxonomy.py",
    "scripts/stage_aa/run_stage_aa1r1_engineering_branch.py",
    "scripts/stage_aa/run_stage_aa1_engineering_canary.py",
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_binding(path: Path, display: str) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": display, "bytes": len(data), "sha256": sha256(data)}


def rank_digest(salt: str, *parts: str) -> str:
    return sha256((salt + "|" + "|".join(parts)).encode("utf-8"))


def branch_id(condition: str, family: str, parent: str) -> str:
    return "AC3-" + rank_digest(BRANCH_SALT, condition, family, parent)[:20]


def branch_seed(condition: str, family: str, parent: str) -> dict[str, Any]:
    digest = rank_digest(BRANCH_SALT, "SEED", condition, family, parent)
    return {"seed_digest": digest, "seed": int(digest[:15], 16) % (2**31 - 1)}


def blind_id(condition: str, family: str, parent: str) -> str:
    return "W-" + rank_digest(BLIND_ID_SALT, family, parent, condition)[:24]


def source_binding(root: Path, path: str) -> dict[str, Any]:
    candidate = root / path
    require(candidate.is_file(), f"G0_SOURCE_FILE_MISSING:{path}")
    return file_binding(candidate, path)


def receipt_path(raw_root: Path, path_value: str) -> Path:
    name = Path(path_value).name
    require(RECEIPT_RE.fullmatch(name) is not None, f"G0_RECEIPT_NAME_INVALID:{path_value}")
    root = raw_root.resolve()
    candidate = (root / "receipts" / name).resolve()
    require(candidate.parent == (root / "receipts").resolve(), f"G0_RECEIPT_PATH_ESCAPE:{path_value}")
    require(candidate.is_file(), f"G0_RECEIPT_MISSING:{name}")
    return candidate


def compact_action(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": int(row["step"]),
        "boundary": bool(row.get("boundary", row.get("model_boundary", False))),
        "boundary_state_sha256": row.get("boundary_state_sha256"),
        "raw": row.get("raw", row.get("raw_action_7d")),
        "final": row.get("final", row.get("env_action_7d")),
    }


def check_receipt(root: Path, raw_root: Path, item: dict[str, Any], manifest_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cell_id = str(item["cell_id"])
    cell = manifest_by_id.get(cell_id)
    require(cell is not None, f"G0_CELL_NOT_IN_MANIFEST:{cell_id}")
    for key in ("model_family", "suite", "task", "canonical_parent_key", "seed", "state_sha256"):
        require(item.get(key) == cell.get(key), f"G0_CELL_BINDING_MISMATCH:{cell_id}:{key}")
    path = receipt_path(raw_root, str(item["receipt"]["path"]))
    data = path.read_bytes()
    require(len(data) == int(item["receipt"]["bytes"]), f"G0_RECEIPT_BYTES_MISMATCH:{cell_id}")
    require(sha256(data) == item["receipt"]["sha256"], f"G0_RECEIPT_SHA_MISMATCH:{cell_id}")
    receipt = json.loads(data.decode("utf-8"))
    require(receipt.get("status") == "AC2_CLEAN_CELL_COMPLETE", f"G0_RECEIPT_NOT_COMPLETE:{cell_id}")
    for key in ("model_family", "suite", "task", "canonical_parent_key", "seed", "state", "state_id", "state_sha256", "source_task_idx"):
        require(receipt.get(key) == cell.get(key), f"G0_RECEIPT_CELL_BINDING_MISMATCH:{cell_id}:{key}")
    clean = receipt.get("clean") or {}
    require(clean.get("status") and clean.get("rows") and clean.get("actions"), f"G0_CLEAN_EVIDENCE_MISSING:{cell_id}")
    selected = clean.get("selected_critical")
    require(isinstance(selected, dict) and selected.get("eligible") is True, f"G0_CRITICAL_ANCHOR_NOT_ELIGIBLE:{cell_id}")
    step = int(selected["step"])
    require(int(item["selected_critical_step"]) == step, f"G0_SELECTED_STEP_MISMATCH:{cell_id}")
    require(str(item["selected_critical_rank_sha256"]) == str(selected.get("selection_rank_sha256")), f"G0_SELECTED_RANK_MISMATCH:{cell_id}")
    require(bool(selected.get("boundary")), f"G0_SELECTED_POINT_NOT_BOUNDARY:{cell_id}")
    states = clean.get("boundary_states") or {}
    state_entry = states.get(str(step))
    require(isinstance(state_entry, dict) and state_entry.get("state") is not None, f"G0_ANCHOR_STATE_MISSING:{cell_id}")
    anchor_sha = str(state_entry.get("sha256"))
    require(anchor_sha == str(selected.get("boundary_state_sha256")), f"G0_ANCHOR_STATE_SHA_MISMATCH:{cell_id}")
    action_rows = clean["actions"]
    clean_rows = clean["rows"]
    require(step + 20 <= len(action_rows) and step + 20 <= len(clean_rows), f"G0_ANCHOR_CONTINUATION_TOO_SHORT:{cell_id}")
    selected_actions = action_rows[step : step + 20]
    selected_rows = clean_rows[step : step + 20]
    for offset, (action, row) in enumerate(zip(selected_actions, selected_rows)):
        expected_step = step + offset
        require(int(action.get("step", expected_step)) == expected_step, f"G0_ACTION_STEP_MISMATCH:{cell_id}:{expected_step}")
        require(int(row.get("step", expected_step)) == expected_step, f"G0_ROW_STEP_MISMATCH:{cell_id}:{expected_step}")
        require(len(action.get("raw", [])) == 7 and len(action.get("final", [])) == 7, f"G0_ACTION_DIM_INVALID:{cell_id}:{expected_step}")
        require(all(isinstance(v, (int, float)) for v in action["raw"] + action["final"]), f"G0_ACTION_NONNUMERIC:{cell_id}:{expected_step}")
    continuation_digest = sha256(canonical(selected_rows))
    require(str(item["clean_trajectory_digest"]) == str(clean.get("clean_trajectory_digest")), f"G0_TRAJECTORY_DIGEST_MISMATCH:{cell_id}")
    return {
        "cell_id": cell_id,
        "model_family": cell["model_family"],
        "suite": cell["suite"],
        "task": cell["task"],
        "canonical_parent_key": cell["canonical_parent_key"],
        "parent_exposure_class": cell["parent_exposure_class"],
        "seed": int(cell["seed"]),
        "state": cell["state"],
        "state_id": int(cell["state_id"]),
        "state_sha256": cell["state_sha256"],
        "source_task_idx": int(cell["source_task_idx"]),
        "source_receipt": {"path": str(item["receipt"]["path"]), "bytes": len(data), "sha256": sha256(data)},
        "selected_anchor": {
            "step": step,
            "selection_rank_sha256": str(selected["selection_rank_sha256"]),
            "boundary_state_sha256": anchor_sha,
            "boundary_state": state_entry["state"],
            "continuation_steps": [int(row["step"]) for row in selected_rows],
            "continuation_digest": continuation_digest,
            "source_clean_trajectory_digest": str(clean["clean_trajectory_digest"]),
            "actions": [compact_action(action) for action in selected_actions],
        },
    }


def build(root: Path, raw_root: Path, head: str, tree: str) -> dict[str, Any]:
    ac2r3_root = load(root / AC2R3_ROOT)
    denominator = load(root / DENOMINATOR)
    eligibility = load(root / ELIGIBILITY)
    manifest = load(root / MANIFEST)
    ac2_source = load(root / AC2_SOURCE)
    ac2_protocol = load(root / AC2_PROTOCOL)
    require(ac2r3_root.get("status") == "STAGE_AC_AC2R3_THREE_MODEL_DENOMINATORS_FROZEN_READY_STOP_FOR_PI", "G0_AC2R3_ROOT_NOT_ACCEPTED")
    require(len(manifest.get("cells", [])) == 720 and manifest.get("cell_count") == 720, "G0_AC2_MANIFEST_NOT_720")
    require(len(eligibility.get("receipt_index", [])) == 720, "G0_AC2R3_INDEX_NOT_720")
    manifest_by_id = {str(item["cell_id"]): item for item in manifest["cells"]}
    require(len(manifest_by_id) == 720, "G0_MANIFEST_DUPLICATE_CELL_IDS")
    index_by_key = {(str(item["model_family"]), str(item["canonical_parent_key"])): item for item in eligibility["receipt_index"]}
    require(len(index_by_key) == 720, "G0_AC2R3_INDEX_DUPLICATE_MODEL_PARENT")

    selected_units: list[dict[str, Any]] = []
    for family in MODELS:
        frozen = denominator["denominator"][family]["frozen_primary_ranked"]
        require(len(frozen) == 32, f"G0_FROZEN_DENOMINATOR_NOT_32:{family}")
        for rank, ranked in enumerate(frozen, start=1):
            parent = str(ranked["canonical_parent_key"])
            item = index_by_key.get((family, parent))
            require(item is not None, f"G0_SELECTED_PARENT_MISSING:{family}:{parent}")
            compact = check_receipt(root, raw_root, item, manifest_by_id)
            require(compact["parent_exposure_class"] in EXPOSURES, f"G0_EXPOSURE_CLASS_INVALID:{compact['cell_id']}")
            require(str(ranked["rank_sha256"]) == rank_digest(denominator["denominator"][family]["salt"], parent), f"G0_DENOMINATOR_RANK_MISMATCH:{family}:{parent}")
            compact["denominator_rank"] = rank
            compact["denominator_rank_sha256"] = str(ranked["rank_sha256"])
            selected_units.append(compact)
    require(len(selected_units) == 96 and len({(x["model_family"], x["canonical_parent_key"]) for x in selected_units}) == 96, "G0_SELECTED_UNIT_COUNT_INVALID")

    jobs: list[dict[str, Any]] = []
    for unit in selected_units:
        for condition, dose in CONDITIONS:
            seed = branch_seed(condition, unit["model_family"], unit["canonical_parent_key"])
            jobs.append({
                "branch_id": branch_id(condition, unit["model_family"], unit["canonical_parent_key"]),
                "condition": condition,
                "dose": dose,
                "model_family": unit["model_family"],
                "canonical_parent_key": unit["canonical_parent_key"],
                "cell_id": unit["cell_id"],
                "suite": unit["suite"],
                "task": unit["task"],
                "state": unit["state"],
                "state_id": unit["state_id"],
                "state_sha256": unit["state_sha256"],
                "source_task_idx": unit["source_task_idx"],
                "parent_exposure_class": unit["parent_exposure_class"],
                "source_receipt": unit["source_receipt"],
                "selected_anchor": unit["selected_anchor"],
                "branch_seed": seed,
                "scientific_claim": "AC3_TREATMENT_NAIVE_PRIMARY_BRANCH_ONLY",
            })
    require(len(jobs) == 384 and len({x["branch_id"] for x in jobs}) == 384, "G0_BRANCH_MATRIX_INVALID")

    blind_sample: list[dict[str, Any]] = []
    for family in MODELS:
        family_units = [x for x in selected_units if x["model_family"] == family]
        for exposure in EXPOSURES:
            eligible = [x for x in family_units if x["parent_exposure_class"] == exposure]
            require(len(eligible) >= 4, f"G0_AC4_BLIND_STRATUM_TOO_SMALL:{family}:{exposure}")
            chosen = sorted(eligible, key=lambda x: (rank_digest(BLIND_SALT, family, exposure, x["canonical_parent_key"]), x["canonical_parent_key"]))[:4]
            for unit in chosen:
                for condition, dose in CONDITIONS:
                    blind_sample.append({
                        "blinded_video_id": blind_id(condition, family, unit["canonical_parent_key"]),
                        "branch_id": branch_id(condition, family, unit["canonical_parent_key"]),
                        "condition": condition,
                        "dose": dose,
                        "model_family": family,
                        "canonical_parent_key": unit["canonical_parent_key"],
                        "cell_id": unit["cell_id"],
                        "parent_exposure_class": exposure,
                    })
    require(len(blind_sample) == 96 and len({x["blinded_video_id"] for x in blind_sample}) == 96, "G0_AC4_BLIND_SAMPLE_INVALID")

    input_bindings = {}
    for path in (AC2R3_ROOT, DENOMINATOR, ELIGIBILITY, MANIFEST, AC2_SOURCE, AC2_PROTOCOL):
        input_bindings[path] = file_binding(root / path, path)
    source_bindings = {path: source_binding(root, path) for path in SOURCE_FILES}
    protocol = {
        "schema": "STAGE_AC_AC3_AC4_AC5_PROGRAM_PROTOCOL_V1",
        "gate": GATE,
        "status": "STAGE_AC_AC3_G0_STATIC_FREEZE_AUTHORIZED",
        "authorization": {
            "g0_static_freeze": True,
            "g1_consumed_only_engineering": True,
            "g2_primary_branches": True,
            "g3_static_analysis": True,
            "g4_blinded_audit": True,
            "g5_final_synthesis": True,
            "aa_history_immutable": True,
            "no_replacement_or_top_up": True,
        },
        "population": {"model_families": list(MODELS), "model_parent_units": 96, "parents_per_model": 32, "primary_branches": 384, "conditions": [x[0] for x in CONDITIONS]},
        "branch_contract": {
            "action_dim": 7,
            "arm_indices": [0, 1, 2, 3, 4, 5],
            "gripper_index": 6,
            "arm_linf_tolerance": 1e-7,
            "native_open_final": -1.0,
            "native_open_raw": {"M0_OPENVLA": 1.0, "M1_OPENVLA_OFT": 1.0, "M2_PI05_LIBERO": -1.0},
            "doses": [3, 5, 10],
            "physical_horizon": 10,
            "boundaries": {"M0_OPENVLA": "FRESH_PER_STEP", "M1_OPENVLA_OFT": "FRESH_OFT_ACTION_QUEUE", "M2_PI05_LIBERO": "FRESH_PI05_REPLAN"},
        },
        "ac4": {"blind_video_count": 96, "parents_per_model": 8, "exposure_strata": list(EXPOSURES), "selection_salt": BLIND_SALT, "video_id_salt": BLIND_ID_SALT, "selection_is_pre_treatment": True, "ai_only_remains_ai_only": True},
        "seeds": {"branch_seed_salt": BRANCH_SALT, "formula": "uint31(sha256(salt|SEED|condition|model_family|canonical_parent_key)[:15])"},
        "firewall": {"new_inference": 0, "new_env_steps": 0, "open_intervention_steps": 0, "pgd_calls": 0, "attacked_env_steps": 0, "v_phys_reads": 0, "protected_reads": 0, "eval160_reads": 0},
        "source_authority": {"git_commit": head, "git_tree": tree, "input_bindings": input_bindings, "runtime_source_bindings": source_bindings},
        "next_legal_action": "EXECUTE_G1_CONSUMED_ONLY_ENGINEERING_QUALIFICATION",
        "claim_boundary": "G0 static authority freeze only; no treatment or physical susceptibility result",
    }
    source_authority = {
        "schema": "STAGE_AC_AC3_SOURCE_AUTHORITY_V1",
        "gate": GATE,
        "status": "STAGE_AC_AC3_SOURCE_AUTHORITY_FROZEN",
        "git_binding": {"commit": head, "tree": tree, "repository": "Leo-6-maker/openvla-gripper-dutycycle-attack"},
        "input_authorities": input_bindings,
        "runtime_files": source_bindings,
        "historical_authorities_immutable": True,
        "raw_receipt_root": str(raw_root),
        "raw_receipt_consumption": "read-only SHA/byte verification of selected AC2 clean receipts; no mutation",
        "claim_boundary": "AC3 G0/G1/G2 executable source authority; no cross-model result until G5",
    }
    launch = {
        "schema": "STAGE_AC_AC3_G0_LAUNCH_MANIFEST_V1",
        "gate": GATE,
        "status": "STAGE_AC_AC3_PRELAUNCH_AUTHORITY_FROZEN_CONTINUE",
        "git_binding": {"commit": head, "tree": tree},
        "branch_seed_salt": BRANCH_SALT,
        "model_parent_units": selected_units,
        "branch_count": len(jobs),
        "branches": jobs,
        "pre_treatment_counters": {"new_inference": 0, "new_env_steps": 0, "open_intervention_steps": 0, "pgd_calls": 0, "attacked_env_steps": 0, "v_phys_reads": 0, "protected_reads": 0, "eval160_reads": 0},
        "next_legal_action": "EXECUTE_G1_CONSUMED_ONLY_ENGINEERING_QUALIFICATION",
    }
    blind = {
        "schema": "STAGE_AC_AC4_BLIND_AUDIT_SAMPLE_V1",
        "gate": GATE,
        "status": "STAGE_AC_AC4_BLIND_SAMPLE_FROZEN_PRE_TREATMENT",
        "selection_salt": BLIND_SALT,
        "video_id_salt": BLIND_ID_SALT,
        "hide_model_suite_parent_condition": True,
        "sample_count": len(blind_sample),
        "sample": blind_sample,
        "next_legal_action": "MATERIALIZE_BLIND_DERIVATIVES_ONLY_AFTER_G2",
    }
    return {"protocol": protocol, "source_authority": source_authority, "launch": launch, "blind": blind, "input_bindings": input_bindings}


def write_json(path: Path, value: Any) -> dict[str, Any]:
    require(not path.exists(), f"G0_APPEND_ONLY_OUTPUT_EXISTS:{path}")
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return file_binding(path, path.as_posix())


def self_test() -> None:
    assert branch_id("OPEN_T5", "M0_OPENVLA", "x") != branch_id("OPEN_T10", "M0_OPENVLA", "x")
    assert branch_seed("OPEN_T5", "M0_OPENVLA", "x")["seed"] != branch_seed("OPEN_T5", "M1_OPENVLA_OFT", "x")["seed"]
    assert blind_id("OPEN_T5", "M0_OPENVLA", "x").startswith("W-")
    assert len({branch_id(c, m, f"p{n}") for m in MODELS for n in range(32) for c, _ in CONDITIONS}) == 384
    print(json.dumps({"status": "AC3_G0_STATIC_SELF_TEST_PASS", "branches": 384}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--head")
    parser.add_argument("--tree")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.raw_root is None or not args.head or not args.tree:
        parser.error("--raw-root, --head, and --tree are required unless --self-test is used")
    root = args.root.resolve()
    result = build(root, args.raw_root, args.head, args.tree)
    if not args.write:
        print(json.dumps({"status": "AC3_G0_STATIC_DRYRUN_PASS", "model_parent_units": 96, "branches": 384, "blind_sample": 96}, sort_keys=True))
        return 0
    outputs = {}
    outputs["protocol"] = write_json(root / "configs/STAGE_AC_AC3_AC4_AC5_PROGRAM_PROTOCOL_V1.json", result["protocol"])
    outputs["source_authority"] = write_json(root / "reports/STAGE_AC_AC3_SOURCE_AUTHORITY_V1.json", result["source_authority"])
    outputs["launch_manifest"] = write_json(root / "reports/STAGE_AC_AC3_G0_LAUNCH_MANIFEST_V1.json", result["launch"])
    outputs["blind_sample"] = write_json(root / "reports/STAGE_AC_AC4_BLIND_AUDIT_SAMPLE_V1.json", result["blind"])
    payload = {
        "schema": "STAGE_AC_AC3_G0_ROOT_SEAL_V1",
        "gate": GATE,
        "status": "STAGE_AC_AC3_PRELAUNCH_AUTHORITY_FROZEN_CONTINUE",
        "authorization_pi_comment_id": 5434166412,
        "git_binding": {"commit": args.head, "tree": args.tree},
        "artifacts": outputs,
        "inputs": result["input_bindings"],
        "counts": {"selected_model_parent_units": 96, "primary_branches": 384, "ac4_blind_videos": 96},
        "scientific_firewall": {"new_inference": 0, "new_env_steps": 0, "open_intervention_steps": 0, "pgd_calls": 0, "attacked_env_steps": 0, "v_phys_reads": 0, "protected_reads": 0, "eval160_reads": 0, "scientific_parent_exposure": 0},
        "historical_authorities_immutable": True,
        "next_legal_action": "EXECUTE_G1_CONSUMED_ONLY_ENGINEERING_QUALIFICATION",
        "claim_boundary": "G0 static freeze only; no scientific treatment result",
    }
    payload["root_payload_sha256"] = sha256(canonical(payload))
    outputs["root"] = write_json(root / "reports/STAGE_AC_AC3_G0_ROOT_SEAL_V1.json", payload)
    print(json.dumps({"status": payload["status"], "outputs": outputs, "root_payload_sha256": payload["root_payload_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
