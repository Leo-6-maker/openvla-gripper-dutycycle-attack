"""Strict generation-count and token/score alignment checks for CLEAN artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def validate_generation_contract(path: Path) -> None:
    """Fail closed unless every valid step records one real generation pass."""
    step_path = path / "step_records.jsonl"
    policy_path = path / "policy_intent_records.jsonl"
    summary = json.loads((path / "episode_summary.json").read_text(encoding="utf-8"))
    runtime = json.loads((path / "runtime_audit.json").read_text(encoding="utf-8"))
    metadata = json.loads((path / "episode_metadata.json").read_text(encoding="utf-8"))

    step_rows = [json.loads(line) for line in step_path.read_text(encoding="utf-8").splitlines() if line]
    policy_rows = [json.loads(line) for line in policy_path.read_text(encoding="utf-8").splitlines() if line]
    if not step_rows or len(step_rows) != len(policy_rows) or summary.get("steps") != len(step_rows):
        raise ValueError("GENERATION_RECORD_LENGTH_MISMATCH")
    if metadata.get("generation_passes_per_step") != 1 or runtime.get("generation_passes_per_step") != 1:
        raise ValueError("GENERATION_PASS_METADATA_INVALID")
    for index, (step, policy) in enumerate(zip(step_rows, policy_rows)):
        if step.get("generation_passes_per_step") != 1 or policy.get("generation_passes_per_step") != 1:
            raise ValueError(f"GENERATION_PASS_FIELD_INVALID:{index}")
        if step.get("single_generation_parity_pass") is not True or policy.get("single_generation_parity_pass") is not True:
            raise ValueError(f"SINGLE_GENERATION_PARITY_INVALID:{index}")
        if step.get("score_adapter_parity_pass") is not True or policy.get("score_adapter_parity_pass") is not True:
            raise ValueError(f"SCORE_ACTION_PARITY_INVALID:{index}")
        if not isinstance(step.get("action_token_ids"), list) or len(step["action_token_ids"]) != 7:
            raise ValueError(f"ACTION_TOKEN_COUNT_INVALID:{index}")
        if not isinstance(step.get("score_head_summary"), list) or len(step["score_head_summary"]) != 7:
            raise ValueError(f"SCORE_COUNT_INVALID:{index}")
        if policy.get("action_token_ids") != step.get("action_token_ids"):
            raise ValueError(f"TOKEN_RECORD_MISMATCH:{index}")


__all__ = ["validate_generation_contract"]
