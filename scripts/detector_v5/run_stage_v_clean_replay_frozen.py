"""Run one clean replay through the frozen Official V3 worker snapshot."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
from typing import Any, Mapping


OFFICIAL_WORKER_SHA256 = "a8e230f1ef10f51ee61c847c49969b444ab57697ac7312100b06e64d03491311"
OFFICIAL_WORKER_COMMIT = "943b02749dce4414ec6791b15ceec87dbd3be1ba"
OFFICIAL_WORKER_TREE = "07a4efa9663edf415c957374e86d12db30854fbe"
DEPENDENCY_SHA256 = {
    "src/gripper_attack/official_openvla_adapter.py": "0e8c1c2476b50609627cee2000467966fb1cbf68b01050147471867f13710aae",
    "src/gripper_attack/official_libero_protocol.py": "72f58fd8e5b06ce2c0d4c0bf93fd33ea30bc52b6384cc1b721f608d573dff5fa",
    "src/gripper_attack/official_generation_contract.py": "84acf46889033bfb907c340012417929e20e2c051ebbe9f7eee1d7c9b1586950",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(all_finite(item) for item in value)
    return True


def git_value(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def verify_frozen_worker(worker_script: Path) -> dict[str, Any]:
    worker_script = worker_script.resolve()
    repo_root = worker_script.parents[1]
    checks: dict[str, Any] = {
        "worker_script": str(worker_script),
        "worker_script_sha256": sha256_file(worker_script),
        "worker_git_commit": git_value(repo_root, "rev-parse", "HEAD"),
        "worker_git_tree": git_value(repo_root, "rev-parse", "HEAD^{tree}"),
        "worker_git_status": git_value(repo_root, "status", "--porcelain"),
        "dependencies": {},
    }
    for relative, expected in DEPENDENCY_SHA256.items():
        path = repo_root / relative
        checks["dependencies"][relative] = {"path": str(path), "sha256": sha256_file(path) if path.is_file() else None, "expected_sha256": expected}
    if checks["worker_script_sha256"] != OFFICIAL_WORKER_SHA256:
        raise RuntimeError("FROZEN_OFFICIAL_WORKER_SHA256_MISMATCH")
    if checks["worker_git_commit"] != OFFICIAL_WORKER_COMMIT or checks["worker_git_tree"] != OFFICIAL_WORKER_TREE or checks["worker_git_status"]:
        raise RuntimeError("FROZEN_OFFICIAL_WORKER_GIT_MISMATCH")
    if any(item["sha256"] != item["expected_sha256"] for item in checks["dependencies"].values()):
        raise RuntimeError("FROZEN_OFFICIAL_WORKER_DEPENDENCY_MISMATCH")
    checks["verified"] = True
    return checks


def load_candidate(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("candidate must be a JSON object")
    required = {"canonical_parent_key", "suite", "task_index", "state_index"}
    if not required.issubset(value):
        raise ValueError(f"candidate missing fields: {sorted(required - set(value))}")
    return dict(value)


def copy_provenance(source: Path, output_root: Path) -> tuple[dict[str, Any], str, str]:
    source = source.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    destination = output_root / "provenance" / "UPSTREAM_PROVENANCE.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    return payload, sha256_file(source), sha256_file(destination)


def write_worker_manifest(path: Path, row: Mapping[str, Any], initial_state_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["canonical_parent_key", "suite", "task_idx", "state_id", "split", "initial_state_sha256"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "canonical_parent_key": str(row["canonical_parent_key"]),
            "suite": str(row["suite"]),
            "task_idx": int(row["task_index"]),
            "state_id": int(row["state_index"]),
            "split": str(row.get("split") or row.get("source_split") or "STAGE_V_R2_QUALIFICATION"),
            "initial_state_sha256": initial_state_sha256,
        })


def last_jsonl(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    last: Mapping[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, Mapping):
                last = value
    return last


def _bind_runtime_attention() -> str:
    if os.environ.get("OPENVLA_ATTN_IMPLEMENTATION") != "eager":
        return "UNMODIFIED"
    from transformers import AutoModelForVision2Seq

    original = AutoModelForVision2Seq.from_pretrained

    def eager_from_pretrained(*args: Any, **kwargs: Any) -> Any:
        kwargs["attn_implementation"] = "eager"
        return original(*args, **kwargs)

    AutoModelForVision2Seq.from_pretrained = staticmethod(eager_from_pretrained)
    return "OPENVLA_UPSTREAM_ATTENTION_OVERRIDE_V1"


def build_start_provenance(module: Mapping[str, Any], worker_script: Path, artifact_provenance: Mapping[str, Any]) -> dict[str, Any]:
    repo_root = worker_script.resolve().parents[1]
    return {
        "worker_start_epoch": __import__("time").time(),
        "worker_start_git_head": git_value(repo_root, "rev-parse", "HEAD"),
        "worker_start_worktree_clean": not bool(git_value(repo_root, "status", "--porcelain")),
        "worker_start_script_tracked_at_head": bool(git_value(repo_root, "ls-files", "--error-unmatch", "scripts/official_clean_worker.py")),
        "worker_start_script_sha256": sha256_file(worker_script),
        "worker_start_adapter_sha256": sha256_file(repo_root / "src/gripper_attack/official_openvla_adapter.py"),
        "worker_start_protocol_sha256": sha256_file(repo_root / "src/gripper_attack/official_libero_protocol.py"),
        "worker_start_generation_contract_sha256": sha256_file(repo_root / "src/gripper_attack/official_generation_contract.py"),
        "worker_start_model_tree_sha256": artifact_provenance.get("checkpoint_tree_sha256_actual"),
        "worker_start_processor_tokenizer_sha256": artifact_provenance.get("processor_tokenizer_sha256_actual"),
    }


def worker_identity_row(row: Mapping[str, Any], initial_state_sha256: str) -> dict[str, Any]:
    """Bridge the qualification identity names to the frozen worker contract."""
    return {
        **row,
        "task_idx": int(row["task_index"]),
        "state_id": int(row["state_index"]),
        "initial_state_sha256": initial_state_sha256,
        "split": str(row.get("split") or row.get("source_split") or "STAGE_V_R2_QUALIFICATION"),
    }


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_pythonpath_prefixes = []
    for prefix in reversed(args.pythonpath_prefix):
        resolved_prefix = prefix.resolve()
        if not resolved_prefix.is_dir() or resolved_prefix.is_symlink():
            raise ValueError(f"runtime pythonpath prefix missing or symlinked: {resolved_prefix}")
        runtime_pythonpath_prefixes.append(str(resolved_prefix))
        sys.path.insert(0, str(resolved_prefix))
    row = load_candidate(args.candidate_path)
    frozen = verify_frozen_worker(args.worker_script)
    upstream_provenance, provenance_source_sha256, provenance_copy_sha256 = copy_provenance(args.provenance_source, output_dir)
    suite = str(row["suite"])
    model_path = Path(args.model_path) if args.model_path else Path(upstream_provenance["checkpoints"][suite]["path"])
    manifest_path = output_dir / "OFFICIAL_CLEAN_REPLAY_MANIFEST.csv"
    control: dict[str, Any]
    runtime_adapter = "NOT_APPLIED"
    try:
        runtime_adapter = _bind_runtime_attention()
        old_argv = sys.argv[:]
        old_pycache = os.environ.get("PYTHONDONTWRITEBYTECODE")
        try:
            os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
            sys.argv = [str(args.worker_script), "--suite", suite, "--gpu", str(args.gpu), "--worker-id", args.worker_id,
                        "--model-path", str(model_path), "--manifest", str(manifest_path), "--output-root", str(output_dir),
                        "--upstream-root", str(args.upstream_root), "--seed", str(args.seed)]
            module = runpy.run_path(str(args.worker_script), run_name="stage_v_frozen_clean_worker")
        finally:
            sys.argv = old_argv
            if old_pycache is None:
                os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
            else:
                os.environ["PYTHONDONTWRITEBYTECODE"] = old_pycache

        from libero.libero import benchmark
        suite_instance = benchmark.get_benchmark_dict()[suite]()
        task_idx = int(row["task_index"])
        state_idx = int(row["state_index"])
        task = suite_instance.get_task(task_idx)
        initial_state = copy.deepcopy(suite_instance.get_task_init_states(task_idx)[state_idx])
        initial_state_sha256 = str(module["state_sha"](initial_state))
        supplied_state_sha256 = row.get("initial_state_sha256")
        if supplied_state_sha256 and str(supplied_state_sha256) != initial_state_sha256:
            raise RuntimeError("CANDIDATE_INITIAL_STATE_SHA256_MISMATCH")
        write_worker_manifest(manifest_path, row, initial_state_sha256)
        row = worker_identity_row(row, initial_state_sha256)
        artifact_provenance = module["load_artifact_provenance"]()
        model, processor, device, unnorm_key = module["load_policy"]()
        adapter = module["OfficialOpenVLAActionAdapter"](
            model, processor, device, unnorm_key, center_crop=True, base_vla_name=str(model_path),
        )
        module["WORKER_START_PROVENANCE"] = build_start_provenance(module, args.worker_script, artifact_provenance)
        episode = dict(module["run_episode"](adapter, task, initial_state, row, output_dir, model_path, artifact_provenance))
        metadata = json.loads((output_dir / "episode_metadata.json").read_text(encoding="utf-8"))
        artifact_manifest = json.loads((output_dir / "artifact_sha256.json").read_text(encoding="utf-8")) if (output_dir / "artifact_sha256.json").is_file() else {}
        sidecar_last = last_jsonl(output_dir / "privileged_teacher_sidecar.jsonl")
        finite = all_finite(metadata) and all_finite(sidecar_last or {})
        for file_name in ("step_records.jsonl", "policy_intent_records.jsonl", "privileged_teacher_sidecar.jsonl"):
            for line in (output_dir / file_name).read_text(encoding="utf-8").splitlines():
                if line.strip() and not all_finite(json.loads(line)):
                    finite = False
        clean_success = bool(metadata.get("success") is True and episode.get("status") == "PASS")
        expected_identity = {"canonical_parent_key": row["canonical_parent_key"], "suite": suite, "task_index": task_idx, "state_index": state_idx, "initial_state_sha256": initial_state_sha256}
        terminal_state = sidecar_last or {"canonical_parent_key": row["canonical_parent_key"], "state": "MISSING"}
        remaining = max(int(metadata.get("official_horizon", 0)) - int(episode.get("steps", metadata.get("steps", 0))), 0)
        artifact_valid = bool(module["artifact_checksum_valid"](output_dir))
        control = {
            "schema": "STAGE_V_R2_CLEAN_CONTROL_RESULT_V1",
            "status": "PASS" if clean_success else "TASK_FAILURE",
            "exit_code": 0,
            "replicate": args.replicate,
            "canonical_parent_key": row["canonical_parent_key"],
            "suite": suite,
            "task_index": task_idx,
            "state_index": state_idx,
            "clean_success": clean_success,
            "snapshot_restore_valid": bool(metadata.get("env_reset_called") is True and initial_state_sha256 == str(metadata.get("initial_state_sha256"))),
            "runtime_valid": metadata.get("runtime_valid") is True,
            "task_identity_valid": all(metadata.get(field) == value for field, value in (("canonical_parent_key", row["canonical_parent_key"]), ("suite", suite), ("task_idx", task_idx), ("state_id", state_idx))),
            "metrics_finite": finite,
            "remaining_policy_steps": remaining,
            "remaining_horizon_complete": bool(clean_success and remaining >= args.min_remaining_steps),
            "terminal_outcome": "SUCCESS" if clean_success else "TASK_FAILURE",
            "terminal_state_sha256": sha256_json(terminal_state),
            "key_state_identity_sha256": sha256_json(expected_identity),
            "source_commit": args.source_commit,
            "source_tree": args.source_tree,
            "clean_runner_snapshot": frozen,
            "clean_runner_commit": frozen["worker_git_commit"],
            "clean_runner_tree": frozen["worker_git_tree"],
            "clean_runner_sha256": frozen["worker_script_sha256"],
            "provenance_source_sha256": provenance_source_sha256,
            "provenance_copy_sha256": provenance_copy_sha256,
            "artifact_recursive_sha256": artifact_manifest.get("recursive_sha256"),
            "artifact_validation_pass": artifact_valid,
            "old_artifacts_reused": False,
            "eval160_reads": 0,
            "protected_eval_reads": 0,
            "vis_pgd_attack_rollouts": 0,
            "attack_rollouts": 0,
            "worker_gpu": int(args.gpu),
            "runtime_environment": {"OPENVLA_ATTN_IMPLEMENTATION": os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "")},
            "runtime_adapter": runtime_adapter,
            "runtime_pythonpath_prefixes": runtime_pythonpath_prefixes,
            "generated_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        }
    except Exception as exc:
        control = {
            "schema": "STAGE_V_R2_CLEAN_CONTROL_RESULT_V1", "status": "FAIL", "exit_code": 1,
            "replicate": args.replicate, "canonical_parent_key": row.get("canonical_parent_key"), "suite": suite,
            "clean_success": False, "snapshot_restore_valid": False, "runtime_valid": False,
            "task_identity_valid": False, "metrics_finite": False, "remaining_horizon_complete": False,
            "terminal_outcome": "ERROR", "terminal_state_sha256": None, "key_state_identity_sha256": None,
            "source_commit": args.source_commit, "source_tree": args.source_tree,
            "error": f"{type(exc).__name__}:{str(exc)[:1000]}", "old_artifacts_reused": False,
            "eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0, "attack_rollouts": 0,
            "runtime_environment": {"OPENVLA_ATTN_IMPLEMENTATION": os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "")},
            "runtime_adapter": runtime_adapter,
            "runtime_pythonpath_prefixes": runtime_pythonpath_prefixes,
        }
    atomic_write_json(output_dir / "CONTROL_RESULT.json", control)
    return int(control["exit_code"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicate", required=True, choices=["A", "B"])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", default="stage-v-r2-clean")
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--provenance-source", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-remaining-steps", type=int, default=10)
    parser.add_argument("--pythonpath-prefix", type=Path, action="append", default=[])
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
