"""Seal the eligible-head Teacher -> Student development transition.

This is a metadata-only gate.  It binds the already sealed T0/T2/T3 roots and
authorizes development work only for heads that pass the frozen coverage gate.
It never copies episode labels and never grants formal, rollout, shadow, or
attack authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts" / "detector_v5") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "detector_v5"))

from audit_r3_contact_input import sha256_file, verify_seal
from gripper_attack.seal_utils import rename_noreplace


HEADS = (
    "physical_criticality",
    "k10_feasibility",
    "safe_release",
    "instability",
    "gripper_closing_state",
)
ELIGIBLE_HEADS = tuple(head for head in HEADS if head != "safe_release")
FORBIDDEN_PARTS = {"cal", "check", "g10", "t2r-d", "protected", "attack"}
SOURCE_FILES = (
    "configs/R3_DEV_PROTOCOL.json",
    "configs/R3_SC5_FEATURE_BINDING_V1.json",
    "scripts/detector_v5/run_r3_v23_formal_teacher.py",
    "src/gripper_attack/sc5_streaming_features_v2.py",
    "src/gripper_attack/v5_r3_features.py",
    "src/gripper_attack/v5_r3_student.py",
    "src/gripper_attack/v5_r3_teacher.py",
    "scripts/detector_v5/run_r3_full670_student_development.py",
)


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or symlinked JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha(value: Any, field: str, length: int = 64) -> str:
    result = str(value)
    if not re.fullmatch(rf"[0-9a-f]{{{length}}}", result):
        raise ValueError(f"{field} is not a lowercase SHA{length}")
    return result


def _safe_root(path: Path) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"root path must be absolute and lexical-safe: {path}")
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"symlink in root path: {current}")
        current = current.parent
    resolved = path.resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"missing or symlinked root: {resolved}")
    if any(part.lower() in FORBIDDEN_PARTS for part in resolved.parts):
        raise ValueError(f"forbidden-looking root: {resolved}")
    return resolved


def _safe_file(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be absolute and lexical-safe: {path}")
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink: {current}")
        current = current.parent
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{label} is missing or not a regular file: {resolved}")
    if any(part.lower() in FORBIDDEN_PARTS for part in resolved.parts):
        raise ValueError(f"{label} is under a forbidden-looking root: {resolved}")
    return resolved


def _safe_output(path: Path, *, parent: Path) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"output root must be absolute and lexical-safe: {path}")
    if path.exists():
        raise FileExistsError(path)
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            raise ValueError(f"output parent contains a symlink: {current}")
        current = current.parent
    resolved = path.resolve()
    if resolved.parent != parent.resolve():
        raise ValueError("output root must be a new sibling of the teacher root")
    if any(part.lower() in FORBIDDEN_PARTS for part in resolved.parts):
        raise ValueError(f"output root is under a forbidden-looking root: {resolved}")
    return resolved


def _nested_or_same(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _assert_role_roots_disjoint(roles: Mapping[str, Path]) -> None:
    items = list(roles.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1:]:
            if _nested_or_same(left, right):
                raise ValueError(f"role roots overlap or nest: {left_name} vs {right_name}")


def _sealed_json(root: Path, filename: str) -> tuple[dict[str, Any], dict[str, Any]]:
    seal = verify_seal(root)
    path = root / filename
    return _json(path), seal


def _source_bindings(repo_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = repo_root / relative
        if path.is_symlink() or not path.is_file() or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"source binding is missing or unsafe: {relative}")
        result[relative] = sha256_file(path)
    return result


def _git_snapshot(repo_root: Path) -> tuple[str, str]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True, stderr=subprocess.STDOUT).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(f"cannot verify git snapshot for {repo_root}") from exc

    commit = run("rev-parse", "HEAD")
    tree = run("rev-parse", "HEAD^{tree}")
    _sha(commit, "git HEAD", 40)
    _sha(tree, "git HEAD tree", 40)
    return commit, tree


def _require_clean_git(repo_root: Path) -> None:
    try:
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot verify clean git worktree for {repo_root}") from exc
    if status.strip():
        raise ValueError("code snapshot worktree is dirty")


def _feature_binding(repo_root: Path) -> dict[str, Any]:
    relative = Path("configs/R3_SC5_FEATURE_BINDING_V1.json")
    data = _json(repo_root / relative)
    order = data.get("feature_order")
    if not isinstance(order, list) or len(order) != 25 or not all(isinstance(item, str) for item in order):
        raise ValueError("R3 feature order is not the frozen 25D list")
    order_sha = hashlib.sha256(json.dumps(order, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    if data.get("feature_order_sha256") != order_sha:
        raise ValueError("R3 feature-order SHA mismatch")
    adapter_relative = data.get("adapter_source")
    if not isinstance(adapter_relative, str) or Path(adapter_relative).is_absolute() or ".." in Path(adapter_relative).parts:
        raise ValueError("unsafe R3 adapter source path")
    adapter_path = repo_root / adapter_relative
    if adapter_path.is_symlink() or not adapter_path.is_file():
        raise ValueError("R3 adapter source is missing")
    adapter_sha = hashlib.sha256(adapter_path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")).hexdigest()
    if data.get("adapter_source_sha256") != adapter_sha:
        raise ValueError("R3 adapter source SHA mismatch")
    if data.get("future_fields_used") is not False or data.get("teacher_fields_used") is not False or data.get("outcome_fields_used") is not False or data.get("attack_enabled") is not False:
        raise ValueError("R3 feature binding is not causal/fail-closed")
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(repo_root / relative),
        "feature_order": order,
        "feature_order_sha256": order_sha,
        "adapter_source": adapter_relative,
        "adapter_source_sha256": adapter_sha,
    }


def _validate_teacher_manifest(manifest: Mapping[str, Any], *, root_seal: str) -> None:
    expected = {
        "schema": "V5_R3_V23_TEACHER_FORMAL_V1",
        "status": "DEVELOPMENT_NONCONSUMABLE",
        "selection_mode": "FULL_FORMAL_T2",
        "input_status": "PASS_CONSUMABLE_FINAL",
        "identity_count": 670,
        "step_count": 196483,
        "protected_reads": 0,
        "teacher_labels_generated": True,
        "unknown_to_negative": False,
        "formal_inference_authorized": False,
        "formal_training_authorized": False,
        "attack_authorized": False,
        "future_fields_used": False,
        "outcome_fields_used": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"teacher manifest mismatch: {key}")
    if set(manifest.get("heads", [])) != set(HEADS):
        raise ValueError("teacher manifest head set mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", root_seal):
        raise ValueError("teacher root seal is invalid")


def _validate_t0b_permissions(permissions: Any) -> None:
    expected = {
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
    if permissions != expected:
        raise ValueError("T0-B nested permission matrix is not exactly fail-closed")


def _validate_nested_input_binding(
    teacher_manifest: Mapping[str, Any],
    *,
    t0a_manifest: Mapping[str, Any],
    t0a_manifest_sha: str,
    t0a_seal: Mapping[str, Any],
    t0a_root: Path,
    t0b_manifest: Mapping[str, Any],
    t0b_manifest_sha: str,
    t0b_seal: Mapping[str, Any],
    t0b_path: Path,
) -> None:
    binding = teacher_manifest.get("input_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("T2 input_binding is missing")
    if binding.get("schema") != "FIT670_V2_FORMAL_CONSUMABLE_INPUT_V1" or binding.get("status") != "PASS_CONSUMABLE_FINAL":
        raise ValueError("T2 input_binding schema/status mismatch")
    if binding.get("identity_count") != 670 or binding.get("step_count") != 196483 or binding.get("protected_reads") != 0:
        raise ValueError("T2 input_binding cardinality/boundary mismatch")
    if binding.get("formal_root") != t0a_manifest.get("formal_root"):
        raise ValueError("T2 input_binding formal root mismatch")
    if binding.get("formal_inference_authorized") is not False or binding.get("formal_training_authorized") is not False or binding.get("attack_authorized") is not False:
        raise ValueError("T2 input_binding grants a forbidden permission")

    audit = binding.get("input_audit")
    if not isinstance(audit, Mapping) or audit.get("manifest") != dict(t0a_manifest):
        raise ValueError("T2 input_audit nested manifest is not an exact T0-A copy")
    if audit.get("manifest_sha256") != t0a_manifest_sha or audit.get("root") != str(t0a_root):
        raise ValueError("T2 input_audit manifest/path binding mismatch")
    if audit.get("seal_sha256sums_sha256") != t0a_seal.get("sha256sums_sha256"):
        raise ValueError("T2 input_audit seal binding mismatch")
    if audit.get("seal", {}).get("sha256sums_sha256") != t0a_seal.get("sha256sums_sha256"):
        raise ValueError("T2 input_audit nested seal binding mismatch")

    transition = binding.get("fit_to_teacher_transition")
    if not isinstance(transition, Mapping) or transition.get("manifest") != dict(t0b_manifest):
        raise ValueError("T2 T0-B nested manifest is not an exact copy")
    if transition.get("manifest_sha256") != t0b_manifest_sha or transition.get("manifest_path") != str(t0b_path):
        raise ValueError("T2 T0-B manifest/path binding mismatch")
    if transition.get("seal_sha256sums_sha256") != t0b_seal.get("sha256sums_sha256"):
        raise ValueError("T2 T0-B seal binding mismatch")
    if transition.get("seal", {}).get("sha256sums_sha256") != t0b_seal.get("sha256sums_sha256"):
        raise ValueError("T2 T0-B nested seal binding mismatch")

    selection = binding.get("selection")
    if not isinstance(selection, Mapping) or selection.get("schema") != "V5_R3_FULL_FORMAL_SELECTION_FROM_T0_A_V1" or selection.get("status") != "PASS_FULL_FORMAL_T2_SELECTION" or selection.get("identity_count") != 670:
        raise ValueError("T2 full-formal selection binding is incomplete")
    if selection.get("manifest_sha256") != t0a_manifest_sha or selection.get("seal_sha256sums_sha256") != t0a_seal.get("sha256sums_sha256"):
        raise ValueError("T2 full-formal selection does not bind T0-A")

    finalization = binding.get("finalization")
    if not isinstance(finalization, Mapping) or finalization.get("identity_set_digest") != t0a_manifest.get("identity_set_digest") or finalization.get("episode_seal_digest") != t0b_manifest.get("episode_seal_digest"):
        raise ValueError("T2 finalization binding mismatch")

    transition_copy = binding.get("transition")
    if not isinstance(transition_copy, Mapping) or transition_copy.get("manifest_sha256") != t0b_manifest.get("parent_transition_manifest_sha256") or transition_copy.get("seal_sha256sums_sha256") != t0b_manifest.get("parent_transition_sha256sums_sha256"):
        raise ValueError("T2 parent transition binding mismatch")


def _validate_t3(coverage: Mapping[str, Any], *, teacher_root: Path, teacher_seal: str) -> tuple[list[str], dict[str, Any]]:
    if coverage.get("schema") != "V5_R3_TEACHER_COVERAGE_AUDIT_V1" or coverage.get("status") != "HOLD_COVERAGE":
        raise ValueError("T3 coverage report is not the expected development HOLD")
    if coverage.get("identity_count") != 670 or coverage.get("step_count") != 196483:
        raise ValueError("T3 coverage cardinality mismatch")
    if coverage.get("protected_reads") != 0 or coverage.get("unknown_as_negative") is not False or coverage.get("right_censored_as_negative") is not False:
        raise ValueError("T3 coverage boundary is not closed")
    protected_audit = coverage.get("protected_read_audit")
    if not isinstance(protected_audit, Mapping) or protected_audit.get("status") != "PASS" or protected_audit.get("forbidden_root_parts") != []:
        raise ValueError("T3 protected-read audit is not PASS")
    if Path(str(coverage.get("input_root", ""))).resolve() != teacher_root.resolve():
        raise ValueError("T3 teacher root path mismatch")
    if coverage.get("input_sha256sums_sha256") != teacher_seal:
        raise ValueError("T3 teacher root seal mismatch")
    raw = coverage.get("coverage")
    if not isinstance(raw, Mapping) or set(raw) != set(HEADS):
        raise ValueError("T3 coverage head set mismatch")
    if any(not isinstance(raw[head], Mapping) or type(raw[head].get("pass")) is not bool for head in HEADS):
        raise ValueError("T3 coverage pass fields must be explicit booleans")
    if raw["safe_release"].get("pass") is not False:
        raise ValueError("safe_release must be explicitly HOLD_COVERAGE")
    eligible = [head for head in HEADS if raw[head].get("pass") is True]
    held = {head: raw[head] for head in HEADS if raw[head].get("pass") is not True}
    if eligible != list(ELIGIBLE_HEADS):
        raise ValueError(f"unexpected eligible head set: {eligible}")
    return eligible, held


def build(
    *,
    teacher_root: Path,
    coverage_root: Path,
    input_audit_root: Path,
    fit_transition: Path,
    protocol: Path,
    feature_binding: Path,
    output_root: Path,
    repo_root: Path,
    code_commit: str,
    code_tree: str,
    environment: str,
) -> dict[str, Any]:
    teacher_root = _safe_root(teacher_root)
    coverage_root = _safe_root(coverage_root)
    input_audit_root = _safe_root(input_audit_root)
    fit_transition = _safe_file(fit_transition, label="T0-B transition")
    protocol = _safe_file(protocol, label="R3 protocol")
    feature_binding = _safe_file(feature_binding, label="R3 feature binding")
    repo_root = _safe_root(repo_root)
    output_root = _safe_output(output_root, parent=teacher_root.parent)
    fit_transition_root = _safe_root(fit_transition.parent)
    _assert_role_roots_disjoint({
        "teacher": teacher_root,
        "coverage": coverage_root,
        "input_audit": input_audit_root,
        "fit_transition": fit_transition_root,
        "output": output_root,
    })
    code_commit = _sha(code_commit, "code_commit", 40)
    code_tree = _sha(code_tree, "code_tree", 40)
    _require_clean_git(repo_root)
    actual_commit, actual_tree = _git_snapshot(repo_root)
    if (code_commit, code_tree) != (actual_commit, actual_tree):
        raise ValueError("code snapshot does not match the repository worktree")
    protocol_data = _json(protocol)
    if protocol_data.get("schema") != "V5_TEACHER_STUDENT_R3_DEV_PROTOCOL_V1_AMENDED_FAST_CLOSURE":
        raise ValueError("R3 protocol schema mismatch")
    protocol_sha = sha256_file(protocol)
    feature = _feature_binding(repo_root)
    if feature_binding != (repo_root / feature["path"]).resolve():
        raise ValueError("feature binding path must be the repository frozen binding")
    if protocol != (repo_root / "configs/R3_DEV_PROTOCOL.json").resolve():
        raise ValueError("protocol must be the repository frozen R3 protocol")
    source_bindings = _source_bindings(repo_root)

    teacher_manifest, teacher_seal = _sealed_json(teacher_root, "teacher_manifest.json")
    _validate_teacher_manifest(teacher_manifest, root_seal=teacher_seal["sha256sums_sha256"])
    coverage, coverage_seal = _sealed_json(coverage_root, "coverage_report.json")
    eligible, held = _validate_t3(coverage, teacher_root=teacher_root, teacher_seal=teacher_seal["sha256sums_sha256"])

    t0a_manifest, t0a_seal = _sealed_json(input_audit_root, "FORMAL_INPUT_MANIFEST.json")
    if t0a_manifest.get("schema") != "V5_R3_FORMAL_INPUT_AUDIT_V1" or t0a_manifest.get("status") != "PASS_FORMAL_INPUT_CONSUMABLE":
        raise ValueError("T0-A is not consumable")
    if t0a_manifest.get("episode_count") != 670 or t0a_manifest.get("protected_reads") != 0 or t0a_manifest.get("teacher_labels_generated") is not False:
        raise ValueError("T0-A boundary/cardinality mismatch")
    fit_transition_root = _safe_root(fit_transition.parent)
    fit_manifest, fit_seal = _sealed_json(fit_transition_root, fit_transition.name)
    if fit_manifest.get("schema") != "FIT_TO_TEACHER_TRANSITION_V1" or fit_manifest.get("status") != "PASS_FIT_TO_TEACHER_AUTHORIZATION":
        raise ValueError("T0-B transition is not consumable")
    if fit_manifest.get("input_audit_manifest_sha256") != sha256_file(input_audit_root / "FORMAL_INPUT_MANIFEST.json"):
        raise ValueError("T0-B does not bind T0-A manifest")
    if fit_manifest.get("input_audit_seal_sha256sums_sha256") != t0a_seal["sha256sums_sha256"]:
        raise ValueError("T0-B does not bind T0-A seal")
    if fit_manifest.get("protected_reads") != 0 or fit_manifest.get("student_training_authorized") is not False or fit_manifest.get("attack_authorized") is not False:
        raise ValueError("T0-B permission boundary mismatch")
    for label, bound in (
        ("T2 teacher", teacher_manifest.get("protocol_sha256")),
        ("T3 coverage", coverage.get("protocol_sha256")),
        ("T0-B protocol", fit_manifest.get("protocol_sha256")),
    ):
        if bound != protocol_sha:
            raise ValueError(f"{label} protocol SHA mismatch")
    if fit_manifest.get("teacher_contract_sha256") != source_bindings["src/gripper_attack/v5_r3_teacher.py"]:
        raise ValueError("T0-B teacher contract SHA mismatch")
    if fit_manifest.get("teacher_runner_sha256") != source_bindings["scripts/detector_v5/run_r3_v23_formal_teacher.py"]:
        raise ValueError("T0-B teacher runner SHA mismatch")
    _validate_t0b_permissions(fit_manifest.get("permissions"))
    _validate_nested_input_binding(
        teacher_manifest,
        t0a_manifest=t0a_manifest,
        t0a_manifest_sha=sha256_file(input_audit_root / "FORMAL_INPUT_MANIFEST.json"),
        t0a_seal=t0a_seal,
        t0a_root=input_audit_root,
        t0b_manifest=fit_manifest,
        t0b_manifest_sha=sha256_file(fit_transition),
        t0b_seal=fit_seal,
        t0b_path=fit_transition,
    )

    teacher_records = teacher_root / "teacher_records.jsonl"
    if not teacher_records.is_file() or teacher_records.is_symlink():
        raise ValueError("teacher record stream missing")
    created_at = datetime.now(timezone.utc).isoformat()
    held_heads = {
        head: {"status": "HOLD_COVERAGE", "coverage": held[head]}
        for head in HEADS if head not in eligible
    }
    report = {
        "schema": "V5_R3_TEACHER_STUDENT_TRANSITION_V1",
        "status": "PASS_DEVELOPMENT_ELIGIBLE_HEADS",
        "created_at": created_at,
        "code_snapshot": {"commit": code_commit, "tree": code_tree},
        "teacher_root": str(teacher_root),
        "teacher_root_sha256sums_sha256": teacher_seal["sha256sums_sha256"],
        "teacher_manifest_sha256": sha256_file(teacher_root / "teacher_manifest.json"),
        "teacher_records_sha256": sha256_file(teacher_records),
        "teacher_identity_count": 670,
        "teacher_step_count": 196483,
        "coverage_root": str(coverage_root),
        "coverage_root_sha256sums_sha256": coverage_seal["sha256sums_sha256"],
        "coverage_report_sha256": sha256_file(coverage_root / "coverage_report.json"),
        "coverage_status": coverage["status"],
        "eligible_heads": eligible,
        "held_heads": held_heads,
        "full_five_status": "HOLD_COVERAGE",
        "t0_a": {"root": str(input_audit_root), "manifest_sha256": sha256_file(input_audit_root / "FORMAL_INPUT_MANIFEST.json"), "seal_sha256sums_sha256": t0a_seal["sha256sums_sha256"]},
        "t0_b": {"manifest": str(fit_transition), "manifest_sha256": sha256_file(fit_transition), "seal_sha256sums_sha256": fit_seal["sha256sums_sha256"]},
        "protocol": {"path": str(protocol), "sha256": protocol_sha},
        "feature_binding": feature,
        "source_bindings": source_bindings,
        "environment": environment,
        "permissions": {
            "teacher_label_read": True,
            "student_dataset_generation": True,
            "student_training": True,
            "student_training_scope": "DEVELOPMENT_ONLY",
            "development_student_training_authorized": True,
            "development_inference": True,
            "development_inference_authorized": True,
            "formal_training_authorized": False,
            "formal_inference_authorized": False,
            "shadow_authorized": False,
            "rollout_authorized": False,
            "protected_reads": 0,
            "CAL_READ": False,
            "CHECK_READ": False,
            "G10_READ": False,
            "T2R_D_READ": False,
            "attack_authorized": False,
        },
        "safe_release_training_authorized": False,
        "labels_copied": False,
        "protected_reads": 0,
        "formal_training_authorized": False,
        "attack_authorized": False,
    }
    staging = output_root.with_name(f".{output_root.name}.staging")
    if staging.exists() or output_root.exists():
        raise FileExistsError(f"T4 output or staging already exists: {output_root}")
    staging.mkdir(parents=True)
    (staging / "TEACHER_STUDENT_TRANSITION.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / "HEAD_ELIGIBILITY.json").write_text(json.dumps({"eligible_heads": eligible, "held_heads": held_heads, "safe_release_training_authorized": False}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (staging / "PERMISSION_MATRIX.json").write_text(json.dumps(report["permissions"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = _write_seal(staging)
    rename_noreplace(staging, output_root)
    report["sha256sums_sha256"] = digest
    return report


def _write_seal(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.sha256"})
    if not files:
        raise ValueError("cannot seal empty T4 transition")
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files), encoding="utf-8")
    digest = sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(f"{digest}  SHA256SUMS\n", encoding="utf-8")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--coverage-root", type=Path, required=True)
    parser.add_argument("--input-audit-root", type=Path, required=True)
    parser.add_argument("--fit-transition", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--feature-binding", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--code-tree", required=True)
    parser.add_argument("--environment", required=True)
    args = parser.parse_args()
    print(json.dumps(build(teacher_root=args.teacher_root, coverage_root=args.coverage_root, input_audit_root=args.input_audit_root, fit_transition=args.fit_transition, protocol=args.protocol, feature_binding=args.feature_binding, output_root=args.output_root, repo_root=args.repo_root, code_commit=args.code_commit, code_tree=args.code_tree, environment=args.environment), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
