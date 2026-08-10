#!/usr/bin/env python3
"""Independent, stdlib-first audit of the prospective M3.5 V1.3 freeze."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping


SCHEMA = "STAGE_V_M3_5_STATIC_INDEPENDENT_AUDIT_V1_3"
COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
DOSES = ("T3", "T5", "T10")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return dict(value)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT).strip()


def _tree_binding(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.resolve().rglob("*"), key=lambda item: item.relative_to(root.resolve()).as_posix()):
        if path.is_dir():
            if path.is_symlink():
                raise ValueError(f"DIRECTORY_SYMLINK_UNSUPPORTED:{path}")
            continue
        if not path.is_file():
            raise ValueError(f"DIRECTORY_ENTRY_UNSUPPORTED:{path}")
        rows.append({
            "path": path.relative_to(root.resolve()).as_posix(), "size": path.stat().st_size,
            "sha256": _sha256(path), "symlink_target": os.readlink(path) if path.is_symlink() else None,
        })
    return {
        "algorithm": "sha256(canonical_json(sorted(relative_path,size,file_sha256,symlink_target)))",
        "tree_sha256": _sha256_json(rows), "file_count": len(rows),
        "total_bytes": sum(int(row["size"]) for row in rows),
    }


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": str(detail)})


def _manual_pair(selection_sha256: str, parent_key: str) -> dict[str, Any]:
    candidates = [
        {
            "canonical_parent_key": parent_key,
            "probe_id": f"Q{probe:02d}",
            "repetition": repetition,
            "dose": dose,
        }
        for probe in range(24)
        for repetition in range(3)
        for dose in DOSES
    ]
    return min(
        candidates,
        key=lambda row: hashlib.sha256(
            f"M35_V1_3_MANUAL_PAIR::{selection_sha256}::{parent_key}::{row['probe_id']}::R{row['repetition']}::{row['dose']}".encode("utf-8")
        ).hexdigest(),
    )


def _repo_binding(checks: list[dict[str, Any]], repo: Path, name: str, binding: Any) -> Path | None:
    if not isinstance(binding, Mapping):
        _check(checks, f"binding:{name}", False, "mapping required")
        return None
    path = repo / str(binding.get("path", binding.get("module_path", "")))
    _check(checks, f"binding:{name}:exists", path.is_file(), path)
    _check(checks, f"binding:{name}:sha256", path.is_file() and _sha256(path) == binding.get("sha256"), binding.get("sha256"))
    return path if path.is_file() else None


def _git_input(checks: list[dict[str, Any]], name: str, binding: Any) -> None:
    if not isinstance(binding, Mapping):
        _check(checks, f"runtime_input:{name}", False, "mapping required")
        return
    root = Path(str(binding.get("path", ""))).resolve()
    try:
        actual = (_git(root, "rev-parse", "HEAD"), _git(root, "rev-parse", "HEAD^{tree}"), _git(root, "status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError) as exc:
        _check(checks, f"runtime_input:{name}", False, type(exc).__name__)
        return
    expected = (str(binding.get("git_commit", "")), str(binding.get("git_tree", "")), "")
    _check(checks, f"runtime_input:{name}", actual == expected, actual)
    if binding.get("adapter_path"):
        adapter = Path(str(binding["adapter_path"]))
        _check(checks, f"runtime_input:{name}:adapter", adapter.is_file() and _sha256(adapter) == binding.get("adapter_sha256"), adapter)


def _exact_regression_valid(
    repo: Path,
    binding: Any,
    receipt: Any,
    runtime_python: Any,
) -> tuple[bool, dict[str, Any]]:
    if not isinstance(binding, Mapping) or not isinstance(receipt, Mapping):
        return False, {"error": "binding and receipt mappings required"}
    test_files = binding.get("test_files")
    tested_bindings = binding.get("tested_bindings")
    counts = {name: receipt.get(name) for name in ("collected", "passed", "skipped", "failed", "errors", "deselected")}
    counts_valid = all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in counts.values())
    paths_valid = isinstance(test_files, list) and bool(test_files) and all(
        isinstance(path, str) and path.startswith("tests/detector_v5/test_stage_v") and path.endswith(".py")
        for path in test_files
    )
    paths_valid = paths_valid and len(test_files) == len(set(test_files))
    bindings_valid = isinstance(tested_bindings, Mapping) and bool(tested_bindings)
    if bindings_valid:
        for relative, expected_sha in tested_bindings.items():
            path = Path(str(relative))
            resolved = (repo / path).resolve()
            if (
                not isinstance(relative, str)
                or not isinstance(expected_sha, str)
                or len(expected_sha) != 64
                or any(character not in "0123456789abcdef" for character in expected_sha)
                or path.is_absolute()
                or ".." in path.parts
                or not resolved.is_relative_to(repo)
                or not resolved.is_file()
                or _sha256(resolved) != expected_sha
            ):
                bindings_valid = False
                break
    valid = (
        receipt.get("schema") == "STAGE_V_M3_5_EXACT_A800_REGRESSION_RECEIPT_V1"
        and receipt.get("status") == "PASS"
        and receipt.get("runtime_python") == binding.get("runtime_python") == runtime_python
        and all(
            isinstance(binding.get(name), str)
            and len(binding[name]) == 40
            and all(character in "0123456789abcdef" for character in binding[name])
            for name in ("source_commit", "source_tree")
        )
        and receipt.get("source_commit") == binding.get("source_commit")
        and receipt.get("source_tree") == binding.get("source_tree")
        and receipt.get("source_status_porcelain") == ""
        and receipt.get("cuda_visible_devices") == ""
        and receipt.get("test_files") == test_files
        and receipt.get("tested_bindings") == tested_bindings
        and paths_valid
        and bindings_valid
        and all(path in tested_bindings for path in test_files)
        and counts_valid
        and counts["collected"] == binding.get("expected_collected")
        and counts["collected"] == counts["passed"] + counts["skipped"]
        and counts["failed"] == counts["errors"] == counts["deselected"] == 0
        and receipt.get("py_compile_status") == "PASS"
        and receipt.get("protected_counters") == COUNTERS
    )
    return valid, {
        "source_commit": receipt.get("source_commit"),
        "source_tree": receipt.get("source_tree"),
        "test_file_count": len(test_files) if isinstance(test_files, list) else None,
        "tested_binding_count": len(tested_bindings) if isinstance(tested_bindings, Mapping) else None,
        "counts": counts,
    }


def audit(repo_root: Path, protocol_path: Path) -> dict[str, Any]:
    repo = repo_root.resolve()
    protocol_path = protocol_path.resolve()
    protocol = _load(protocol_path)
    checks: list[dict[str, Any]] = []
    actual_commit = _git(repo, "rev-parse", "HEAD")
    actual_tree = _git(repo, "rev-parse", "HEAD^{tree}")
    actual_status = _git(repo, "status", "--porcelain")
    _check(checks, "source_worktree_clean", not actual_status, actual_status or "clean")
    _check(checks, "protocol_schema", protocol.get("schema") == "STAGE_V_M3_5_DIAGNOSTIC_PROTOCOL_V1_3" and protocol.get("version") == "V1.3.1", {"schema": protocol.get("schema"), "version": protocol.get("version")})
    _check(checks, "protocol_revision", protocol.get("supersedes") == "configs/STAGE_V_M3_5_DIAGNOSTIC_PROTOCOL_V1_3.json" and protocol.get("revision_reason") == "correct physical EGL binding before any simulator step or rollout", {"supersedes": protocol.get("supersedes"), "revision_reason": protocol.get("revision_reason")})
    _check(checks, "protocol_status", protocol.get("status") == "FROZEN_PROSPECTIVE_RUNTIME_READY_PENDING_INDEPENDENT_AUDIT", protocol.get("status"))
    _check(checks, "runtime_authorized", protocol.get("runtime_authorized") is True and protocol.get("requires_explicit_owner_authorization") is True and protocol.get("launch_policy", {}).get("runtime_authorized") is True, "explicit owner-authorized diagnostic only")
    _check(checks, "protected_eval160", protocol.get("protected_eval160") == {"reads_allowed": False, "rollouts_allowed": False, "hard_stop": True}, protocol.get("protected_eval160"))
    _check(checks, "protected_counters", protocol.get("protected_counters") == COUNTERS, protocol.get("protected_counters"))
    _check(checks, "exact_python", Path(sys.executable).as_posix() == str(protocol.get("source_binding", {}).get("runtime_python")), sys.executable)

    bindings = protocol.get("contract_bindings") if isinstance(protocol.get("contract_bindings"), Mapping) else {}
    bound_paths = {name: _repo_binding(checks, repo, name, binding) for name, binding in bindings.items()}
    label_path = bound_paths.get("label_contract")
    label = _load(label_path) if label_path else {}
    _check(checks, "label_schema", label.get("schema") == "STAGE_V_M3_5_LABEL_AND_QUALIFICATION_CONTRACT_V1_3" and label.get("version") == "V1.3", label.get("schema"))
    probe = label.get("probe_contract", {})
    _check(checks, "corridor_quantiles", probe.get("probe_count_per_parent") == 24 and probe.get("selection") == "24 deterministic round-half-up quantiles over timestep-sorted eligible corridor" and probe.get("phase_quota") is None and probe.get("minimum_observed_remaining_steps") == 20, probe)
    accounting = label.get("execution_accounting", {})
    _check(checks, "diagnostic_accounting", accounting.get("physical_diagnostic_executions_per_parent") == 288 and accounting.get("treatment_repetition_observations_per_parent") == 216 and accounting.get("repeatability_collapsed_probe_dose_labels_per_parent") == 72, accounting)
    _check(checks, "formal_accounting_separate", accounting.get("formal_m4_physical_executions_per_parent") == 96 and accounting.get("formal_m4_matched_labels_per_parent") == 72, accounting)
    horizon = label.get("horizon_contract", {})
    _check(checks, "dose_and_horizons", horizon.get("dose_steps") == {"T3": 3, "T5": 5, "T10": 10} and horizon.get("H_phys") == 10 and horizon.get("primary_required_steps") == {"T3": 13, "T5": 15, "T10": 20} and horizon.get("H_task") == "not consumed by M3.5; task success inside the frozen physical window is descriptive only", horizon)
    treatment = label.get("treatment_contract", {})
    _check(checks, "surgical_treatment", treatment.get("forced_raw_gripper") == 1.0 and treatment.get("forced_env_gripper") == -1.0 and treatment.get("arm_delta_linf_tolerance") == 1e-7 and treatment.get("post_treatment_mode") == "own frozen-policy closed-loop from intervention-altered state", treatment)
    taxonomy = label.get("physical_taxonomy", {})
    _check(checks, "fixture_only_exclusion", taxonomy.get("fixture_only_goal_policy") == "INELIGIBLE; never infer a movable object from a fixture" and taxonomy.get("target_tracking") == "one prospectively bound goal-source object per probe and branch", taxonomy)
    _check(
        checks,
        "physical_thresholds",
        taxonomy.get("thresholds") == {
            "contact_absence_consecutive_frames": 2,
            "object_release_l2_threshold": 0.01,
            "object_drop_height_threshold": 0.02,
            "already_open_aperture_min": 0.03,
            "aperture_response_delta_min": 0.005,
        }
        and taxonomy.get("physical_failure_classes") == ["GRIPPER_CONTACT_LOSS", "PREMATURE_OBJECT_RELEASE", "OBJECT_DROP"]
        and taxonomy.get("unknown_is_not_negative") is True,
        taxonomy,
    )
    repeatability = label.get("repeatability_gate", {})
    _check(checks, "repeatability", repeatability.get("repetitions") == 3 and repeatability.get("pass_rule") == "3/3 identical registered class and 3/3 compliant; no majority vote", repeatability)
    lineage = label.get("lineage_contract", {})
    _check(checks, "matched_control_lineage", lineage.get("required_fields") == ["shared_control_branch_id", "shared_control_result_sha256"] and lineage.get("orphan_control_allowed") is False, lineage)
    causal = label.get("causal_state_contract", {})
    required_causal = {"full_simulator_state_sha256", "policy_input_sha256", "prompt_sha256", "input_ids_sha256", "pixel_values_sha256", "decode_config_sha256", "model_tree_sha256", "source_commit", "source_tree", "gpu_uuid"}
    _check(checks, "causal_bindings", set(causal.get("required_bindings", [])) == required_causal and causal.get("sampling") is False and causal.get("generation_passes_per_step") == 1, causal)
    manual = protocol.get("blinded_manual_taxonomy_audit", {})
    selected_pairs = manual.get("selected_pairs") if isinstance(manual, Mapping) else None
    selection_parents = protocol.get("diagnostic_parent_selection", {})
    _check(checks, "manual_audit_preregistered", isinstance(selected_pairs, list) and len(selected_pairs) == 8 and manual.get("condition_identity_visible_to_reviewer") is False and manual.get("major_disagreement_rule") == "manual FAILURE vs automatic NO_FAILURE, or reverse" and manual.get("pair_selection_algorithm") == "minimum sha256 over all 24x3x3 pairs using M35_V1_3_MANUAL_PAIR, selection sha256, and parent key", manual)
    if isinstance(selected_pairs, list):
        pair_keys = [row.get("canonical_parent_key") for row in selected_pairs if isinstance(row, Mapping)]
        pair_valid = len(pair_keys) == len(selected_pairs) == len(set(pair_keys)) == 8 and all(row.get("probe_id") in {f"Q{i:02d}" for i in range(24)} and isinstance(row.get("repetition"), int) and not isinstance(row.get("repetition"), bool) and row.get("repetition") in range(3) and row.get("dose") in set(DOSES) for row in selected_pairs if isinstance(row, Mapping))
        _check(checks, "manual_audit_pair_identity", pair_valid, pair_keys)

    selection_path = Path(str(selection_parents.get("path", ""))).resolve()
    selection = _load(selection_path) if selection_path.is_file() else {}
    _check(checks, "selection_sha", selection_path.is_file() and _sha256(selection_path) == selection_parents.get("sha256"), selection_path)
    selection_keys = [row.get("canonical_parent_key") for row in selection.get("selected_parents", []) if isinstance(row, Mapping)]
    _check(checks, "selection_contract", selection.get("schema") == "STAGE_V_M3_5_DIAGNOSTIC_PARENT_SELECTION_V2" and selection.get("status") == "FROZEN_FOR_VALIDATION" and selection.get("selected_count") == len(selection_keys) == len(set(selection_keys)) == 8 and selection.get("selected_counts_by_suite") == {suite: 2 for suite in SUITES}, selection.get("selected_counts_by_suite"))
    _check(checks, "selection_outcome_blind", selection.get("selection_reads", {}).get("branch_results_read") is False and selection.get("selection_reads", {}).get("counterfactual_outcomes_read") is False and selection.get("protected_counters") == COUNTERS, selection.get("selection_reads"))
    _check(checks, "selection_manual_pair_match", isinstance(selected_pairs, list) and set(selection_keys) == {row.get("canonical_parent_key") for row in selected_pairs if isinstance(row, Mapping)}, selection_keys)
    expected_pairs = [_manual_pair(str(selection_parents.get("sha256", "")), str(key)) for key in selection_keys]
    actual_pairs = [dict(row) for row in selected_pairs if isinstance(row, Mapping)] if isinstance(selected_pairs, list) else []
    _check(checks, "selection_manual_pair_deterministic", len(actual_pairs) == 8 and sorted(actual_pairs, key=lambda row: str(row.get("canonical_parent_key"))) == sorted(expected_pairs, key=lambda row: str(row.get("canonical_parent_key"))), expected_pairs)
    selected_evidence_ok = all(
        row.get("clean_success") is True and row.get("prospective_probe_plan_status") == "PASS"
        and len(row.get("prospective_probe_steps", [])) == 24 and row.get("protected_counters") == COUNTERS
        and Path(str(row.get("clean_result_path", ""))).is_file() and _sha256(Path(str(row["clean_result_path"]))) == row.get("clean_result_sha256")
        and Path(str(row.get("clean_trajectory_path", ""))).is_file() and _sha256(Path(str(row["clean_trajectory_path"]))) == row.get("clean_trajectory_file_sha256")
        for row in selection.get("selected_parents", []) if isinstance(row, Mapping)
    )
    _check(checks, "selected_clean_evidence", len(selection_keys) == 8 and selected_evidence_ok, "sealed file hashes and 24 probes")

    dose_sanity = protocol.get("dose_sanity_contract", {})
    _check(
        checks,
        "dose_sanity_contract",
        dose_sanity.get("delivered_steps") == {"T3": 3, "T5": 5, "T10": 10}
        and dose_sanity.get("physical_failure_monotonicity_required") is False
        and dose_sanity.get("nonmonotonic_triplet_rate_hold_threshold") == 0.25,
        dose_sanity,
    )

    runtime_inputs = protocol.get("runtime_inputs") if isinstance(protocol.get("runtime_inputs"), Mapping) else {}
    _git_input(checks, "official_snapshot", runtime_inputs.get("official_snapshot"))
    _git_input(checks, "upstream", runtime_inputs.get("upstream"))
    environment = runtime_inputs.get("runtime_environment", {})
    actual_packages = {}
    package_ok = isinstance(environment.get("packages"), Mapping) and environment.get("python_version") == platform.python_version()
    for package, expected in environment.get("packages", {}).items() if isinstance(environment.get("packages"), Mapping) else ():
        try:
            actual_packages[str(package)] = metadata.version(str(package))
            package_ok = package_ok and actual_packages[str(package)] == str(expected)
        except metadata.PackageNotFoundError:
            package_ok = False
    _check(checks, "runtime_environment", package_ok, {"python_version": platform.python_version(), "packages": actual_packages})
    models = runtime_inputs.get("models") if isinstance(runtime_inputs.get("models"), Mapping) else {}
    for suite in SUITES:
        binding = models.get(suite) if isinstance(models, Mapping) else None
        if not isinstance(binding, Mapping) or not binding.get("path"):
            _check(checks, f"model:{suite}", False, "binding path required")
            continue
        root = Path(str(binding["path"])).resolve()
        try:
            actual_model = _tree_binding(root) if root.is_dir() else {}
        except (OSError, ValueError) as exc:
            actual_model = {"error": f"{type(exc).__name__}:{exc}"}
        expected_model = {key: binding.get(key) for key in ("algorithm", "tree_sha256", "file_count", "total_bytes")} if isinstance(binding, Mapping) else {}
        _check(checks, f"model:{suite}", actual_model == expected_model, actual_model)

    gpu_map = protocol.get("resource_contract", {}).get("gpu_uuid_by_index", {})
    admitted_gpus = protocol.get("resource_contract", {}).get("admitted_gpu_indices")
    _check(
        checks,
        "gpu_admission_contract",
        admitted_gpus == [0, 1, 2, 4, 5, 6, 7]
        and protocol.get("resource_contract", {}).get("excluded_gpu_indices") == [3]
        and protocol.get("resource_contract", {}).get("minimum_free_memory_mib") == 20480
        and protocol.get("resource_contract", {}).get("maximum_project_workers_per_gpu") == 1,
        protocol.get("resource_contract", {}),
    )
    _check(
        checks,
        "logical_cuda_egl_binding",
        protocol.get("resource_contract", {}).get("cuda_visible_devices") == "one admitted physical GPU per worker"
        and protocol.get("resource_contract", {}).get("torch_logical_device") == 0
        and protocol.get("resource_contract", {}).get("mujoco_egl_device_id") == "admitted physical GPU index"
        and protocol.get("resource_contract", {}).get("env_render_gpu_device_id") == "admitted physical GPU index",
        protocol.get("resource_contract", {}),
    )
    gpu_result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    actual_gpu_map = {}
    if gpu_result.returncode == 0:
        for line in gpu_result.stdout.splitlines():
            index, uuid = (part.strip() for part in line.split(",", 1))
            actual_gpu_map[index] = uuid.lower().removeprefix("gpu-")
    expected_gpu_map = {str(key): str(value).lower().removeprefix("gpu-") for key, value in gpu_map.items()} if isinstance(gpu_map, Mapping) else {}
    _check(checks, "gpu_uuid_map", actual_gpu_map == expected_gpu_map and len(actual_gpu_map) == 8, actual_gpu_map)

    test_binding = protocol.get("exact_a800_regression", {})
    test_path = Path(str(test_binding.get("path", ""))).resolve()
    test_receipt = _load(test_path) if test_path.is_file() else {}
    regression_valid, regression_detail = _exact_regression_valid(
        repo, test_binding, test_receipt, protocol.get("source_binding", {}).get("runtime_python")
    )
    _check(
        checks,
        "exact_a800_regression",
        test_path.is_file() and _sha256(test_path) == test_binding.get("sha256") and regression_valid,
        regression_detail,
    )
    for name, path in bound_paths.items():
        if path and path.suffix == ".py":
            result = subprocess.run([sys.executable, "-m", "py_compile", str(path)], cwd=repo, capture_output=True, text=True)
            _check(checks, f"py_compile:{name}", result.returncode == 0, result.stderr.strip() or "compiled")
    runner_text = bound_paths.get("runner").read_text(encoding="utf-8") if bound_paths.get("runner") else ""
    forbidden = ("open_eval160", "read_eval160", "protected_eval160_root", "eval160_path")
    _check(checks, "runner_no_eval160_reader", not any(token in runner_text.lower() for token in forbidden), "no protected reader token")
    diff = subprocess.run(["git", "diff", "--check"], cwd=repo, capture_output=True, text=True)
    _check(checks, "git_diff_check", diff.returncode == 0, diff.stdout.strip() or diff.stderr.strip() or "clean")

    failures = [row for row in checks if row["status"] != "PASS"]
    return {
        "schema": SCHEMA, "status": "PASS" if not failures else "FAIL",
        "protocol": str(protocol_path), "protocol_sha256": _sha256(protocol_path),
        "actual_source_commit": actual_commit, "actual_source_tree": actual_tree,
        "actual_source_status": actual_status, "checks": checks,
        "check_count": len(checks), "failure_count": len(failures),
        "branch_results_read": False, "protected_counters": dict(COUNTERS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"REFUSE_OVERWRITE:{args.output}")
    report = audit(args.repo_root, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output.with_name(args.output.name + ".sha256").write_text(f"{_sha256(args.output)}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "check_count": report["check_count"], "failure_count": report["failure_count"]}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
