#!/usr/bin/env python3
"""Build the fresh B0R1 action-only audit without exposing G2 outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gripper_attack.openvla_libero_exec_spec import raw_gripper_to_env_gripper  # noqa: E402
from stage_aa.action_semantics_v2 import (  # noqa: E402
    MODEL_M0,
    MODEL_M1,
    MODEL_M2,
    validate_action_pair as validate_v2,
)
from stage_z_preparation.action_semantics import validate_action_pair as validate_v1  # noqa: E402


INDEX = ROOT / "reports/STAGE_AC_AC3_G2_BRANCH_RECEIPT_INDEX_V1.json"
V1_SOURCE = ROOT / "src/stage_z_preparation/action_semantics.py"
V2_SOURCE = ROOT / "src/stage_aa/action_semantics_v2.py"
OFFICIAL_SOURCE = ROOT / "src/gripper_attack/openvla_libero_exec_spec.py"
AC3_SOURCE = ROOT / "scripts/stage_ac/run_stage_ac3_g2_model_suite.py"
AA1_SOURCE = ROOT / "scripts/stage_aa/run_stage_aa1_engineering_canary.py"
REMOTE_PYTHON = "/mnt/sdc/dty_user/openvla_attack/envs/openvla-official-a800/bin/python"
REMOTE_INDEX = "/mnt/sdc/dty_user/openvla_attack_worktrees/stage-ac2-clean-screen-c217acfc-lf/reports/STAGE_AC_AC3_G2_BRANCH_RECEIPT_INDEX_V1.json"
REMOTE_ROOT = "/mnt/sdc/dty_user/openvla_attack_outputs/STAGE_AC_AC3_G2_PHYSICAL_V1"
REMOTE_SOURCE = "/mnt/sdc/dty_user/openvla_attack_worktrees/stage-ac2-clean-screen-c217acfc-lf"
TARGET_BRANCH = "AC3-65bcfd948a45dd0be9ac"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def action(gripper: float) -> list[float]:
    return [0.0, 0.1, -0.1, 0.2, -0.2, 0.3, gripper]


def static_authority() -> dict[str, Any]:
    boundary = []
    for raw in (0.4999999, 0.5, 0.5000001):
        expected = float(raw_gripper_to_env_gripper(raw))
        old = validate_v1(MODEL_M1, action(raw), action(expected))
        new = validate_v2(MODEL_M1, action(raw), action(expected))
        boundary.append(
            {
                "raw": raw,
                "official_env": expected,
                "old_v1_accepted": bool(old["accepted"]),
                "v2_accepted": bool(new["accepted"]),
                "v2_semantic_state": new.get("semantic_state"),
            }
        )

    wrong_threshold = []
    for final in (1.0, -1.0):
        result = validate_v2(MODEL_M1, action(0.5), action(final))
        wrong_threshold.append({"final": final, "v2_accepted": bool(result["accepted"]), "reason": result["reason"]})

    compatibility = []
    for family in (MODEL_M0, MODEL_M1):
        for raw in (-1.0, 0.0, 0.499, 0.501, 1.0, 2.0):
            expected = float(raw_gripper_to_env_gripper(raw))
            old = validate_v1(family, action(raw), action(expected))
            new = validate_v2(family, action(raw), action(expected))
            compatibility.append(
                {
                    "family": family,
                    "raw": raw,
                    "old_v1_accepted": bool(old["accepted"]),
                    "v2_accepted": bool(new["accepted"]),
                    "same_open_close_meaning": (not old["accepted"])
                    or new.get("semantic_state") == ("OPEN" if raw > 0.5 else "CLOSE"),
                }
            )
    old_pass = [row for row in compatibility if row["old_v1_accepted"]]
    m2 = validate_v2(MODEL_M2, [2.0, -2.0, 0.2, 0.0, 0.0, 0.0, -0.9986837], [1.0, -1.0, 0.2, 0.0, 0.0, 0.0, -0.9986837])

    return {
        "source_files": {
            "historical_v1": {"path": V1_SOURCE.relative_to(ROOT).as_posix(), "bytes": V1_SOURCE.stat().st_size, "sha256": sha256_file(V1_SOURCE)},
            "aa2r2_v2": {"path": V2_SOURCE.relative_to(ROOT).as_posix(), "bytes": V2_SOURCE.stat().st_size, "sha256": sha256_file(V2_SOURCE)},
            "official_exec_spec": {"path": OFFICIAL_SOURCE.relative_to(ROOT).as_posix(), "bytes": OFFICIAL_SOURCE.stat().st_size, "sha256": sha256_file(OFFICIAL_SOURCE)},
            "ac3_runner": {"path": AC3_SOURCE.relative_to(ROOT).as_posix(), "bytes": AC3_SOURCE.stat().st_size, "sha256": sha256_file(AC3_SOURCE)},
            "legacy_aa1_runner": {"path": AA1_SOURCE.relative_to(ROOT).as_posix(), "bytes": AA1_SOURCE.stat().st_size, "sha256": sha256_file(AA1_SOURCE)},
        },
        "legacy_route_evidence": {
            "ac3_loads_aa1r1_runner": "run_stage_aa1r1_engineering_branch.py" in AC3_SOURCE.read_text(encoding="utf-8"),
            "ac3_routes_through_aa1_semantics": "AA1.model_pairs" in AC3_SOURCE.read_text(encoding="utf-8"),
            "aa1_uses_historical_v1_validator": "src/stage_z_preparation/action_semantics.py" in AA1_SOURCE.read_text(encoding="utf-8"),
        },
        "official_rule": "raw < 0.5 -> env +1; raw == 0.5 -> env 0; raw > 0.5 -> env -1",
        "boundary_probes": boundary,
        "wrong_exact_threshold_final_values": wrong_threshold,
        "m2_clip_probe_accepted": bool(m2["accepted"]),
        "compatibility": {
            "old_pass_count": len(old_pass),
            "rows": compatibility,
            "old_pass_implies_v2_pass": all(row["v2_accepted"] and row["same_open_close_meaning"] for row in old_pass),
        },
    }


REMOTE_CODE = r'''
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, __REMOTE_SOURCE__)
sys.path.insert(0, __REMOTE_SOURCE__ + "/src")

from gripper_attack.openvla_libero_exec_spec import OPENVLA_LIBERO_EXEC_SPEC_VERSION, raw_gripper_to_env_gripper  # noqa: E402
from stage_aa.action_semantics_v2 import validate_action_pair as validate_v2  # noqa: E402
from stage_z_preparation.action_semantics import validate_action_pair as validate_v1  # noqa: E402

REMOTE_ROOT = Path(__REMOTE_ROOT__)
INDEX_PATH = Path(__REMOTE_INDEX__)
TARGET_PATH = REMOTE_ROOT / "receipts" / (__TARGET_BRANCH__ + ".json")
V1_PATH = Path(__REMOTE_SOURCE__) / "src/stage_z_preparation/action_semantics.py"
V2_PATH = Path(__REMOTE_SOURCE__) / "src/stage_aa/action_semantics_v2.py"
OFFICIAL_PATH = Path(__REMOTE_SOURCE__) / "src/gripper_attack/openvla_libero_exec_spec.py"


class View:
    """Whitelist-only view for the action audit projection."""

    def __init__(self, value, allowed):
        if not isinstance(value, dict):
            raise RuntimeError("B0R1_EXPECTED_OBJECT")
        self._value = value
        self._allowed = frozenset(allowed)

    def get(self, key, default=None):
        if key not in self._allowed:
            raise RuntimeError("B0R1_NON_ACTION_FIELD_ACCESS:" + str(key))
        return self._value.get(key, default)

    def __getitem__(self, key):
        if key not in self._allowed:
            raise RuntimeError("B0R1_NON_ACTION_FIELD_ACCESS:" + str(key))
        return self._value[key]


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def audit_pair(family, raw, final):
    if not isinstance(raw, list) or not isinstance(final, list) or len(raw) != 7 or len(final) != 7:
        raise RuntimeError("B0R1_ACTION_PAIR_DIMENSION")
    old = validate_v1(family, raw, final, raw_gripper=float(raw[-1]), final_gripper=float(final[-1]))
    new = validate_v2(family, raw, final, raw_gripper=float(raw[-1]), final_gripper=float(final[-1]))
    return {
        "old_v1_accepted": bool(old.get("accepted")),
        "old_v1_reason": str(old.get("reason")),
        "v2_accepted": bool(new.get("accepted")),
        "v2_reason": str(new.get("reason")),
        "v2_semantic_state": str(new.get("semantic_state")),
    }


def add_pair(summary, result):
    summary["action_pairs"] += 1
    summary["old_v1_accepted"] += int(result["old_v1_accepted"])
    summary["old_v1_rejected"] += int(not result["old_v1_accepted"])
    summary["v2_accepted"] += int(result["v2_accepted"])
    summary["v2_rejected"] += int(not result["v2_accepted"])
    summary["old_pass_v2_fail"] += int(result["old_v1_accepted"] and not result["v2_accepted"])


INDEX_KEYS = {"rows"}
INDEX_ROW_KEYS = {"branch_id", "status", "model_family", "suite", "condition", "receipt", "source_receipt"}
REF_KEYS = {"path", "bytes", "sha256"}
RECEIPT_KEYS = {"schema", "status", "branch_id", "model_family", "suite", "condition", "dose", "action_receipts"}
ACTION_KEYS = {"raw_policy_action", "opened_raw_action", "env_action", "step"}
SOURCE_KEYS = {"schema", "status", "model_family", "action_pair_audit"}
AUDIT_ITEM_KEYS = {"semantics"}
SEMANTICS_KEYS = {"raw_action", "final_action"}

index = View(json.loads(INDEX_PATH.read_text(encoding="utf-8")), INDEX_KEYS)
pass_rows = []
for raw_row in index["rows"]:
    row = View(raw_row, INDEX_ROW_KEYS)
    if row.get("status") == "PASS":
        pass_rows.append(row)

expected_ids = sorted(row["branch_id"] for row in pass_rows)
expected_refs = sorted(
    (
        {
            "branch_id": row["branch_id"],
            "path": View(row["receipt"], REF_KEYS)["path"],
            "bytes": int(View(row["receipt"], REF_KEYS)["bytes"]),
            "sha256": View(row["receipt"], REF_KEYS)["sha256"],
        }
        for row in pass_rows
    ),
    key=lambda item: item["branch_id"],
)
if canonical_hash(expected_ids) != __EXPECTED_ID_DIGEST__ or canonical_hash(expected_refs) != __EXPECTED_REF_DIGEST__:
    raise RuntimeError("B0R1_EXPECTED_INDEX_DIGEST")

all_receipts = []
for directory in (REMOTE_ROOT / "receipts", REMOTE_ROOT / "recovered_receipts_v7"):
    for path in sorted(directory.glob("AC3-*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        receipt = View(data, RECEIPT_KEYS)
        if receipt.get("schema") == "STAGE_AC_AC3_BRANCH_RECEIPT_V1" and receipt.get("status") == "PASS":
            all_receipts.append((path, receipt))

remote_ids = sorted(receipt["branch_id"] for _, receipt in all_receipts)
remote_refs = sorted(
    (
        {"branch_id": receipt["branch_id"], "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path, receipt in all_receipts
    ),
    key=lambda item: item["branch_id"],
)
if len(pass_rows) != __EXPECTED_COUNT__ or len(all_receipts) != __EXPECTED_COUNT__ or remote_ids != expected_ids:
    raise RuntimeError("B0R1_PASS_BRANCH_SET")

summary = {
    "branches": 0,
    "clean_branches": 0,
    "treatment_branches": 0,
    "receipt_sha_matches": 0,
    "source_receipts_verified": 0,
    "action_pairs": 0,
    "old_v1_accepted": 0,
    "old_v1_rejected": 0,
    "v2_accepted": 0,
    "v2_rejected": 0,
    "old_pass_v2_fail": 0,
    "intervention_pairs": 0,
    "clean_source_pairs": 0,
}
for row in pass_rows:
    ref = View(row["receipt"], REF_KEYS)
    path = Path(ref["path"])
    if not path.is_file() or path.stat().st_size != int(ref["bytes"]) or sha256_file(path) != ref["sha256"]:
        raise RuntimeError("B0R1_RECEIPT_REFERENCE_MISMATCH")
    receipt = View(json.loads(path.read_text(encoding="utf-8")), RECEIPT_KEYS)
    if receipt["branch_id"] != row["branch_id"] or receipt["status"] != "PASS":
        raise RuntimeError("B0R1_RECEIPT_BINDING")
    summary["branches"] += 1
    summary["receipt_sha_matches"] += 1
    if row["condition"] == "CLEAN_REFERENCE":
        summary["clean_branches"] += 1
        source_ref = View(row["source_receipt"], REF_KEYS)
        source_path = Path(source_ref["path"])
        if not source_path.is_file() or source_path.stat().st_size != int(source_ref["bytes"]) or sha256_file(source_path) != source_ref["sha256"]:
            raise RuntimeError("B0R1_SOURCE_RECEIPT_REFERENCE_MISMATCH")
        source = View(json.loads(source_path.read_text(encoding="utf-8")), SOURCE_KEYS)
        if source.get("status") != "AC2_CLEAN_CELL_COMPLETE" or source.get("model_family") != row["model_family"]:
            raise RuntimeError("B0R1_SOURCE_RECEIPT_BINDING")
        summary["source_receipts_verified"] += 1
        for raw_item in source.get("action_pair_audit", []):
            item = View(raw_item, AUDIT_ITEM_KEYS)
            semantics = View(item["semantics"], SEMANTICS_KEYS)
            add_pair(summary, audit_pair(row["model_family"], semantics["raw_action"], semantics["final_action"]))
            summary["clean_source_pairs"] += 1
    else:
        summary["treatment_branches"] += 1
        for raw_item in receipt.get("action_receipts", []):
            item = View(raw_item, ACTION_KEYS)
            add_pair(summary, audit_pair(row["model_family"], item["opened_raw_action"], item["env_action"]))
            summary["intervention_pairs"] += 1

target = View(json.loads(TARGET_PATH.read_text(encoding="utf-8")), RECEIPT_KEYS)
target_actions = target.get("action_receipts", [])
target_evidence = {"persisted": bool(target_actions), "action_receipt_count": len(target_actions)}

def source_probe(path):
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}

official_probe = [[float(x), float(raw_gripper_to_env_gripper(x))] for x in (0.4999999, 0.5, 0.5000001)]
print(json.dumps({
    "expected_count": __EXPECTED_COUNT__,
    "index_pass_count": len(pass_rows),
    "remote_pass_count": len(all_receipts),
    "remote_ids_digest": canonical_hash(remote_ids),
    "remote_refs_digest": canonical_hash(remote_refs),
    "receipt_summary": summary,
    "target_action_evidence": target_evidence,
    "remote_source": {
        "historical_v1": source_probe(V1_PATH),
        "aa2r2_v2": source_probe(V2_PATH),
        "official_exec_spec": source_probe(OFFICIAL_PATH),
        "official_version": OPENVLA_LIBERO_EXEC_SPEC_VERSION,
        "official_probe": official_probe,
    },
}, sort_keys=True))
'''


def run_remote(expected_ids: list[str], expected_refs: list[dict[str, Any]]) -> dict[str, Any]:
    replacements = {
        "__REMOTE_SOURCE__": repr(REMOTE_SOURCE),
        "__REMOTE_ROOT__": repr(REMOTE_ROOT),
        "__REMOTE_INDEX__": repr(REMOTE_INDEX),
        "__TARGET_BRANCH__": repr(TARGET_BRANCH),
        "__EXPECTED_COUNT__": str(len(expected_ids)),
        "__EXPECTED_ID_DIGEST__": repr(canonical_hash(expected_ids)),
        "__EXPECTED_REF_DIGEST__": repr(canonical_hash(expected_refs)),
    }
    code = REMOTE_CODE
    for key, value in replacements.items():
        code = code.replace(key, value)
    remote_helper = f"/tmp/codex_b0r1_action_only_{uuid.uuid4().hex}.py"
    try:
        uploaded = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "dty", f"cat > {remote_helper}"],
            input=code.encode("utf-8"),
            capture_output=True,
            timeout=90,
        )
        if uploaded.returncode != 0:
            raise RuntimeError(f"B0R1_REMOTE_HELPER_UPLOAD_FAILED:exit={uploaded.returncode}")
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "dty", f"{REMOTE_PYTHON} {remote_helper}"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            raise RuntimeError(f"B0R1_REMOTE_AUDIT_FAILED:exit={result.returncode}")
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("B0R1_REMOTE_AUDIT_NON_JSON") from exc
    finally:
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "dty", f"rm -f -- {remote_helper}"],
            capture_output=True,
            timeout=30,
        )


def build_report() -> dict[str, Any]:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    pass_rows = [row for row in index["rows"] if row.get("status") == "PASS"]
    expected_ids = sorted(row["branch_id"] for row in pass_rows)
    expected_refs = sorted(
        (
            {
                "branch_id": row["branch_id"],
                "path": row["receipt"]["path"],
                "bytes": int(row["receipt"]["bytes"]),
                "sha256": row["receipt"]["sha256"],
            }
            for row in pass_rows
        ),
        key=lambda item: item["branch_id"],
    )
    static = static_authority()
    remote = run_remote(expected_ids, expected_refs)
    local_official = [[float(x), float(raw_gripper_to_env_gripper(x))] for x in (0.4999999, 0.5, 0.5000001)]
    remote_source = remote["remote_source"]
    source_files = static["source_files"]
    source_reconciliation = {
        "historical_v1_byte_equal": source_files["historical_v1"]["sha256"] == remote_source["historical_v1"]["sha256"],
        "aa2r2_v2_byte_equal": source_files["aa2r2_v2"]["sha256"] == remote_source["aa2r2_v2"]["sha256"],
        "official_spec_byte_equal": source_files["official_exec_spec"]["sha256"] == remote_source["official_exec_spec"]["sha256"],
        "official_spec_semantic_probe_equal": local_official == remote_source["official_probe"],
        "official_version_equal": remote_source["official_version"] == "openvla_libero_exec_spec_v2_official_boundary_20260607",
    }
    receipt = remote["receipt_summary"]
    checks = {
        "static_old_pass_implies_v2_pass": bool(static["compatibility"]["old_pass_implies_v2_pass"]),
        "legacy_route_is_identified": all(static["legacy_route_evidence"].values()),
        "index_and_remote_pass_count_372": len(pass_rows) == 372 and remote["index_pass_count"] == 372 and remote["remote_pass_count"] == 372,
        "remote_receipt_set_exact": remote["remote_ids_digest"] == canonical_hash(expected_ids) and remote["remote_refs_digest"] == canonical_hash(expected_refs),
        "receipt_sha_matches_372": receipt["receipt_sha_matches"] == 372,
        "clean_source_receipts_verified": receipt["source_receipts_verified"] == receipt["clean_branches"],
        "persisted_action_pairs_dimension_checked": receipt["action_pairs"] > 0,
        "persisted_pair_old_pass_implies_v2_pass": receipt["old_pass_v2_fail"] == 0,
        "target_queue_not_persisted": remote["target_action_evidence"]["persisted"] is False,
        "historical_and_v2_source_equal": source_reconciliation["historical_v1_byte_equal"] and source_reconciliation["aa2r2_v2_byte_equal"],
        "official_semantics_reconciled": source_reconciliation["official_spec_semantic_probe_equal"] and source_reconciliation["official_version_equal"],
    }
    status = "STAGE_AC_AC3_G2R1_B0R1_ACTION_ONLY_AUDIT_PASS_CONTINUE" if all(checks.values()) else "STAGE_AC_AC3_G2R1_B0R1_ACTION_ONLY_AUDIT_HOLD_STOP_FOR_PI"
    return {
        "schema": "STAGE_AC_AC3_G2R1_B0R1_ACTION_ONLY_AUDIT_V1",
        "status": status,
        "gate": "STAGE_AC_AC3_G2R1_B0R1_ACTION_ONLY_SEMANTICS_RECONCILIATION_V1",
        "source_commit": "9aae759022b37684a843005417280cd4e80283d7",
        "scope": "action-only projection; no physical_class, V_phys, endpoint, telemetry, or treatment-outcome field is used by this audit",
        "static_authority": static,
        "source_reconciliation": source_reconciliation,
        "pass_branch_set": {"count": len(pass_rows), "ids_sha256": canonical_hash(expected_ids), "refs_sha256": canonical_hash(expected_refs)},
        "remote_action_only_audit": remote,
        "checks": checks,
        "target_branch": {"branch_id": TARGET_BRANCH, "action_evidence": remote["target_action_evidence"]},
        "scientific_firewall": {
            "new_model_inference_calls": 0,
            "new_env_step_calls": 0,
            "new_open_intervention_steps": 0,
            "new_physical_telemetry_reads": 0,
            "new_physical_endpoint_reads": 0,
            "new_v_phys_reads": 0,
            "new_attack_outcome_reads": 0,
            "new_protected_reads": 0,
        },
        "next_legal_action": "B1_TARGET_INFERENCE_ONLY_REPLAY" if status.endswith("PASS_CONTINUE") else "STOP_FOR_PI",
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        static = static_authority()
        assert static["compatibility"]["old_pass_implies_v2_pass"]
        assert all(row["v2_accepted"] for row in static["boundary_probes"])
        assert all(not row["v2_accepted"] for row in static["wrong_exact_threshold_final_values"])
        print(json.dumps({"status": "B0R1_ACTION_ONLY_STATIC_SELF_TEST_PASS"}, sort_keys=True))
        return 0

    report_path = ROOT / "reports/STAGE_AC_AC3_G2R1_B0R1_ACTION_ONLY_AUDIT_V1.json"
    root_path = ROOT / "reports/STAGE_AC_AC3_G2R1_B0R1_ROOT_SEAL_V1.json"
    report = build_report()
    write_json(report_path, report)
    report_ref = {"path": report_path.relative_to(ROOT).as_posix(), "bytes": report_path.stat().st_size, "sha256": sha256_file(report_path)}
    root_payload = {
        "gate": report["gate"],
        "status": report["status"],
        "report": report_ref,
        "source_commit": report["source_commit"],
        "scientific_firewall": report["scientific_firewall"],
    }
    root = {
        "schema": "STAGE_AC_AC3_G2R1_B0R1_ROOT_SEAL_V1",
        "status": report["status"],
        "gate": report["gate"],
        "root_payload": root_payload,
        "root_payload_sha256": canonical_hash(root_payload),
        "next_legal_action": report["next_legal_action"],
    }
    write_json(root_path, root)
    print(json.dumps({"status": report["status"], "report": str(report_path), "root": str(root_path), "checks": report["checks"]}, sort_keys=True))
    return 0 if report["status"].endswith("PASS_CONTINUE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
