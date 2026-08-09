"""CPU-only T0-D preflight for the FIT670 Teacher permission chain.

No episode payload is opened here.  The command validates only sealed metadata,
source files and permission manifests; it allocates no GPU and emits no labels.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audit_r3_contact_input import sha256_file, verify_seal
from gripper_attack.seal_utils import rename_noreplace


FORBIDDEN_PARTS = {"cal", "check", "g10", "t2r-d", "protected", "attack"}
PERMISSIONS = {
    "fit_episode_read": True,
    "teacher_label_generation": True,
    "student_dataset_generation": False,
    "student_training": False,
    "detector_load": False,
    "rollout": False,
    "shadow": False,
    "attack": False,
    "protected_payload_read": False,
    "CAL_READ": False,
    "CHECK_READ": False,
    "G10_READ": False,
    "T2R_D_READ": False,
}


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_seal(root: Path) -> str:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    if not files:
        raise ValueError("cannot seal empty preflight")
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}\n" for p in files), encoding="utf-8")
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def _safe(path: Path) -> Path:
    resolved = path.resolve()
    if any(part.lower() in FORBIDDEN_PARTS for part in resolved.parts) or resolved.is_symlink():
        raise ValueError(f"forbidden or symlinked path: {resolved}")
    return resolved


def _runtime_census() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    nvidia = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    ps = subprocess.run(["ps", "-eo", "pid,comm", "--no-headers"], capture_output=True, text=True, check=False)
    return {
        "pid": os.getpid(),
        "python": sys.executable,
        "platform": platform.platform(),
        "cuda_visible_devices": visible,
        "nvidia_smi_returncode": nvidia.returncode,
        "nvidia_smi_rows": [line.strip() for line in nvidia.stdout.splitlines() if line.strip()],
        "process_count": len([line for line in ps.stdout.splitlines() if line.strip()]),
        "process_census_returncode": ps.returncode,
    }


def preflight(input_audit_root: Path, fit_to_teacher_root: Path, teacher_contract: Path, teacher_runner: Path, protocol: Path, output_root: Path, *, expected_audit_seal: str, expected_transition_seal: str) -> dict[str, Any]:
    input_audit_root = _safe(input_audit_root)
    fit_to_teacher_root = _safe(fit_to_teacher_root)
    teacher_contract = _safe(teacher_contract)
    teacher_runner = _safe(teacher_runner)
    protocol = _safe(protocol)
    output_root = output_root.resolve()
    if output_root.is_symlink() or any(part.lower() in FORBIDDEN_PARTS for part in output_root.parts) or output_root.exists() or output_root.parent != input_audit_root.parent:
        raise FileExistsError("preflight output must be a new sibling of the T0-A audit root")
    audit_seal = verify_seal(input_audit_root)
    if audit_seal["sha256sums_sha256"] != expected_audit_seal:
        raise ValueError("T0-A seal mismatch")
    audit = _read_json(input_audit_root / "FORMAL_INPUT_MANIFEST.json")
    if audit.get("status") != "PASS_FORMAL_INPUT_CONSUMABLE" or audit.get("episode_count") != 670 or audit.get("protected_reads") != 0:
        raise ValueError("T0-A metadata is not consumable")
    if audit.get("teacher_labels_generated") is not False or audit.get("labels_generated") is not False:
        raise ValueError("T0-A already contains labels")

    transition_seal = verify_seal(fit_to_teacher_root)
    if transition_seal["sha256sums_sha256"] != expected_transition_seal:
        raise ValueError("FIT_TO_TEACHER seal mismatch")
    transition = _read_json(fit_to_teacher_root / "FIT_TO_TEACHER_TRANSITION.json")
    if transition.get("schema") != "FIT_TO_TEACHER_TRANSITION_V1" or transition.get("status") != "PASS_FIT_TO_TEACHER_AUTHORIZATION":
        raise ValueError("FIT_TO_TEACHER transition is not authorized")
    if transition.get("permissions") != PERMISSIONS or transition.get("protected_reads") != 0:
        raise ValueError("FIT_TO_TEACHER permissions are not exact")
    if transition.get("input_audit_manifest_sha256") != sha256_file(input_audit_root / "FORMAL_INPUT_MANIFEST.json"):
        raise ValueError("FIT_TO_TEACHER input audit binding mismatch")
    if transition.get("input_audit_seal_sha256sums_sha256") != audit_seal["sha256sums_sha256"]:
        raise ValueError("FIT_TO_TEACHER input audit seal binding mismatch")
    if transition.get("teacher_contract_sha256") != sha256_file(teacher_contract) or transition.get("teacher_runner_sha256") != sha256_file(teacher_runner):
        raise ValueError("Teacher source binding mismatch")
    if transition.get("protocol_sha256") != sha256_file(protocol):
        raise ValueError("protocol binding mismatch")
    for field in ("formal_root", "output_root", "identity_set_digest", "episode_seal_digest", "runner_source_commit", "runner_source_tree"):
        if not transition.get(field):
            raise ValueError(f"FIT_TO_TEACHER binding missing: {field}")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible not in {"", "-1"}:
        raise ValueError("T0-D requires CUDA_VISIBLE_DEVICES empty/-1")
    payload_read_paths: list[str] = []
    runtime_census = _runtime_census()
    if runtime_census["nvidia_smi_returncode"] != 0 or runtime_census["process_census_returncode"] != 0:
        raise RuntimeError("T0-D runtime census command failed")
    created_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema": "V5_R3_TEACHER_EXECUTION_PREFLIGHT_V1",
        "status": "PASS_TEACHER_EXECUTION_READY",
        "created_at": created_at,
        "input_audit_root": str(input_audit_root),
        "input_audit_seal_sha256sums_sha256": audit_seal["sha256sums_sha256"],
        "fit_to_teacher_root": str(fit_to_teacher_root),
        "fit_to_teacher_seal_sha256sums_sha256": transition_seal["sha256sums_sha256"],
        "teacher_contract_sha256": sha256_file(teacher_contract),
        "teacher_runner_sha256": sha256_file(teacher_runner),
        "protocol_sha256": sha256_file(protocol),
        "identity_count": 670,
        "episode_payload_read": len(payload_read_paths),
        "episode_payload_paths": payload_read_paths,
        "labels_generated": 0,
        "student_started": 0,
        "gpu_allocation": 0 if cuda_visible in {"", "-1"} else 1,
        "runtime_evidence": {**runtime_census, "payload_read_paths": payload_read_paths},
        "protected_reads": 0,
        "training_authorized": False,
        "rollout_authorized": False,
        "shadow_authorized": False,
        "attack_authorized": False,
        "cal_check_g10_t2rd_reads": 0,
        "execution_command_class": "METADATA_ONLY_CPU_PREFLIGHT",
        "output_root": str(output_root),
    }
    staging = output_root.with_name(f".{output_root.name}.staging")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    (staging / "TEACHER_EXECUTION_PREFLIGHT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / "PERMISSION_MATRIX.json").write_text(json.dumps(PERMISSIONS, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = _write_seal(staging)
    rename_noreplace(staging, output_root)
    report["sha256sums_sha256"] = digest
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-audit-root", type=Path, required=True)
    parser.add_argument("--fit-to-teacher-root", type=Path, required=True)
    parser.add_argument("--teacher-contract", type=Path, required=True)
    parser.add_argument("--teacher-runner", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-audit-seal", required=True)
    parser.add_argument("--expected-transition-seal", required=True)
    args = parser.parse_args()
    print(json.dumps(preflight(args.input_audit_root, args.fit_to_teacher_root, args.teacher_contract, args.teacher_runner, args.protocol, args.output_root, expected_audit_seal=args.expected_audit_seal, expected_transition_seal=args.expected_transition_seal), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
