"""Freeze the Q2 clean-control protocol before any fresh replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

try:
    from .stage_v_dynamic_common import atomic_write_json, sha256_file, utc_now
except ImportError:  # direct server execution
    from stage_v_dynamic_common import atomic_write_json, sha256_file, utc_now


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _tool_binding(path: Path) -> dict[str, Any]:
    binding = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    repo = next((parent for parent in (path.resolve(), *path.resolve().parents) if (parent / ".git").exists()), path.resolve().parents[-1])
    binding.update({
        "git_commit": _git(repo, "rev-parse", "HEAD"),
        "git_tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "git_blob_sha256": binding["sha256"],
    })
    return binding


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output.iterdir() if path.name not in {"STAGE_Q2_PROTOCOL.json", "STAGE_Q2_PROTOCOL.md", "STAGE_Q2_PROTOCOL.sha256"}}
    if existing:
        raise ValueError(f"protocol directory is not clean: {sorted(existing)}")
    candidate = _load(args.candidate_universe)
    if candidate.get("schema") != "D8_STAGE_V_CLEAN_PROBE_CANDIDATE_POOL_V1":
        raise ValueError("candidate universe is not the frozen clean-only pool")
    candidate_sha = sha256_file(args.candidate_universe)
    if args.expected_candidate_sha256 and candidate_sha != args.expected_candidate_sha256:
        raise ValueError("candidate universe SHA256 mismatch")
    if candidate.get("selection_frozen_before_new_rollouts") is not True:
        raise ValueError("candidate selection was not frozen before new rollouts")
    q1_hashes = {}
    for label, path in (("q1_matrix", args.q1_matrix), ("q1_semantic_audit", args.q1_semantic_audit)):
        q1_hashes[label] = {"path": str(path), "sha256": sha256_file(path)}
    tools: dict[str, Any] = {}
    for label, path in (
        ("q2_producer", getattr(args, "q2_producer", None)), ("q2_independent_auditor", getattr(args, "q2_auditor", None)),
        ("q2_supervisor", getattr(args, "q2_supervisor", None)),
        ("frozen_clean_wrapper", getattr(args, "frozen_clean_wrapper", None)), ("official_clean_worker", getattr(args, "official_clean_worker", None)),
        ("upstream_provenance", getattr(args, "upstream_provenance", None)),
    ):
        if path:
            if not path.is_file():
                raise ValueError(f"tool binding missing: {path}")
            tools[label] = _tool_binding(path) if path.suffix == ".py" else {"path": str(path.resolve()), "sha256": sha256_file(path)}
    protocol = {
        "schema": "STAGE_Q2_PROTOCOL_V1", "status": "FROZEN", "protocol_id": "STAGE_V_R2_Q2_20260807",
        "source_commit": args.source_commit, "source_tree": args.source_tree,
        "candidate_universe_path": str(args.candidate_universe), "candidate_universe_sha256": candidate_sha,
        "candidate_universe_schema": candidate.get("schema"), "candidate_universe_count": candidate.get("candidate_count"),
        "candidate_universe_counts_by_suite": {suite: sum(item.get("suite") == suite for item in candidate.get("candidates", [])) for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial")},
        "salt": "STAGE_V_R2_Q2_CONTROL_QUALIFICATION_20260807",
        "initial_per_suite": 20, "expansion_batch_size": 10, "target_per_suite": 10,
        "max_infrastructure_retries": 1, "retry_policy": "one_retry_only_when_no_valid_scientific_result",
        "qualification_semantics": {
            "engineering_required": ["exit_code_zero", "clean_result_schema", "snapshot_restore_valid", "task_identity_valid", "runtime_valid", "metrics_finite", "artifact_validation_pass", "source_commit_tree_exact", "canonical_parent_exact", "initial_state_identity_present", "boundary_zero"],
            "parent_qualified": "A_and_B_engineering_valid_and_clean_success_and_canonical_parent_exact_and_A_B_initial_state_identity_exact",
            "terminal_state_sha256": "descriptive_only",
            "remaining_horizon_complete": "descriptive_only",
            "clean_success_false_with_valid_artifact": "CLEAN_REPEATABILITY_FAIL_no_retry",
            "engineering_invalid_after_retry": "hard_stop",
        },
        "fresh_output_required": True, "q1_artifacts_reused": False,
        "q1_forensic_bindings": q1_hashes,
        "tool_bindings": tools,
        "boundaries": {"eval160_reads": 0, "protected_eval_reads": 0, "vis_pgd_attack_rollouts": 0, "attack_rollouts": 0},
        "scientific_scope": "clean_control_only; no OPEN/VIS/PGD/attack/vulnerability labels",
        "generated_utc": utc_now(),
    }
    atomic_write_json(output / "STAGE_Q2_PROTOCOL.json", protocol)
    markdown = "\n".join([
        "# Stage V R2 Q2 protocol",
        "",
        "Status: `FROZEN`",
        "",
        f"- Source commit: `{args.source_commit}`",
        f"- Source tree: `{args.source_tree}`",
        f"- Candidate universe SHA256: `{candidate_sha}`",
        "- Salt: `STAGE_V_R2_Q2_CONTROL_QUALIFICATION_20260807`",
        "- Initial sample: 20 per suite",
        "- Expansion: +10 only for underfilled suites, deterministic rank order",
        "- Quota: 10 qualified parents per suite",
        "- Qualification: both fresh A/B clean-success results with exact parent and initial-state identity",
        "- Terminal-state hash equality: descriptive only",
        "- Remaining-horizon predicate: descriptive only",
        "- Engineering invalid: one infrastructure retry, then hard stop",
        "- Prohibited reads/rollouts: Eval160, protected eval, OPEN, VIS, PGD, attack, vulnerability labels",
        "",
    ])
    (output / "STAGE_Q2_PROTOCOL.md").write_text(markdown, encoding="utf-8")
    protocol_sha = sha256_file(output / "STAGE_Q2_PROTOCOL.json")
    (output / "STAGE_Q2_PROTOCOL.sha256").write_text(f"{protocol_sha}  STAGE_Q2_PROTOCOL.json\n", encoding="utf-8")
    return protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-universe", type=Path, required=True)
    parser.add_argument("--q1-matrix", type=Path, required=True)
    parser.add_argument("--q1-semantic-audit", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--expected-candidate-sha256", default="")
    parser.add_argument("--q2-producer", type=Path)
    parser.add_argument("--q2-auditor", type=Path)
    parser.add_argument("--q2-supervisor", type=Path)
    parser.add_argument("--frozen-clean-wrapper", type=Path)
    parser.add_argument("--official-clean-worker", type=Path)
    parser.add_argument("--upstream-provenance", type=Path)
    args = parser.parse_args(argv)
    freeze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
