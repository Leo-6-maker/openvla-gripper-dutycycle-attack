#!/usr/bin/env python3
"""Forward-smoke compatibility-only Official student inputs; never reads Teacher labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from gripper_attack.b3_stateful import B3_HEADS, B3_25D, B3_25D9D
from scripts.detector.run_b3_stateful_gpu_smoke import (
    script_in_head_tree,
    write_evidence_seals,
)


MODEL_CLASSES = {"B3_25D": B3_25D, "B3_25D9D": B3_25D9D}
STUDENT_FIELDS = {
    "canonical_parent_key", "clean_policy_intent_9d", "features_25d",
    "state_id", "step", "suite", "task_idx",
}
FORBIDDEN_FILES = {"teacher_retention_records.jsonl", "retention_events.json"}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def source_fingerprint(source_root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for name in ("materialization_manifest.json", "student_input_records.jsonl")
        for path in source_root.rglob(name)
    )
    for path in files:
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_episodes(source_root: Path, expected_episodes: int) -> list[dict[str, Any]]:
    if not source_root.is_dir():
        raise ValueError(f"source root does not exist: {source_root}")
    forbidden = sorted(
        path.relative_to(source_root).as_posix()
        for name in FORBIDDEN_FILES
        for path in source_root.rglob(name)
    )
    if forbidden:
        raise ValueError(f"Teacher files present; refusing to open: {forbidden}")
    student_files = sorted(source_root.rglob("student_input_records.jsonl"))
    if len(student_files) != expected_episodes:
        raise ValueError(f"expected {expected_episodes} student-only episodes, found {len(student_files)}")

    episodes = []
    for student_path in student_files:
        manifest_path = student_path.with_name("materialization_manifest.json")
        if not manifest_path.is_file():
            raise ValueError(f"missing compatibility manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("mode") != "compatibility-only":
            raise ValueError(f"unexpected materialization mode: {manifest_path}")
        if manifest.get("teacher_materialization") != "NOT_RUN":
            raise ValueError(f"Teacher materialization is not held: {manifest_path}")
        if manifest.get("student_forbidden_fields_absent") is not True:
            raise ValueError(f"student-only field contract is not closed: {manifest_path}")
        if manifest.get("source_schema") != "OFFICIAL_25D_V1":
            raise ValueError(f"unexpected source feature schema: {manifest_path}")
        rows = []
        with student_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if set(row) != STUDENT_FIELDS:
                    raise ValueError(f"student schema mismatch at {student_path}:{line_number}")
                if len(row["features_25d"]) != 25 or len(row["clean_policy_intent_9d"]) != 9:
                    raise ValueError(f"feature width mismatch at {student_path}:{line_number}")
                if not all(math.isfinite(float(value)) for value in row["features_25d"] + row["clean_policy_intent_9d"]):
                    raise ValueError(f"non-finite student input at {student_path}:{line_number}")
                rows.append(row)
        if not rows or [row["step"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"student steps are not contiguous: {student_path}")
        for key in ("canonical_parent_key", "suite", "task_idx", "state_id"):
            if len({row[key] for row in rows}) != 1:
                raise ValueError(f"student identity changes within episode: {student_path}")
        source_identity = manifest.get("source_identity", {})
        for key in ("canonical_parent_key", "suite", "task_idx", "state_id"):
            if source_identity.get(key) != rows[0][key]:
                raise ValueError(f"manifest/student identity mismatch: {student_path}")
        if manifest.get("step_count") != len(rows):
            raise ValueError(f"manifest/student step count mismatch: {student_path}")
        episodes.append({"path": student_path.relative_to(source_root).as_posix(), "rows": rows, "manifest": manifest})
    return episodes


def padded_inputs(episodes: list[dict[str, Any]], device: torch.device, length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(episodes)
    x25 = torch.zeros(batch, length, 25, device=device)
    x9 = torch.zeros(batch, length, 9, device=device)
    mask = torch.zeros(batch, length, dtype=torch.bool, device=device)
    for index, episode in enumerate(episodes):
        rows = episode["rows"]
        if len(rows) > length:
            raise ValueError("episode exceeds configured smoke length")
        x25[index, :len(rows)] = torch.tensor([row["features_25d"] for row in rows], device=device)
        x9[index, :len(rows)] = torch.tensor([row["clean_policy_intent_9d"] for row in rows], device=device)
        mask[index, :len(rows)] = True
    return x25, x9, mask


def hidden_norm(hidden: Any) -> float:
    if isinstance(hidden, tuple):
        return max(hidden_norm(value) for value in hidden)
    return float(hidden.detach().norm(dim=-1).max().item())


def forward_smoke(episodes: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    max_input_length = max(len(episode["rows"]) for episode in episodes)
    tested_length = max(520, max_input_length)
    x25, x9, mask = padded_inputs(episodes, device, tested_length)
    input_values = [value for episode in episodes for row in episode["rows"] for value in row["features_25d"]]
    intent_values = [value for episode in episodes for row in episode["rows"] for value in row["clean_policy_intent_9d"]]
    stats = {}
    for name, values in (("features_25d", input_values), ("clean_policy_intent_9d", intent_values)):
        tensor = torch.tensor(values, dtype=torch.float64)
        stats[name] = {
            "min": float(tensor.min()), "max": float(tensor.max()),
            "mean": float(tensor.mean()), "std": float(tensor.std(unbiased=False)),
            "finite": bool(torch.isfinite(tensor).all()),
        }
    results = {}
    for name, model_cls in MODEL_CLASSES.items():
        torch.manual_seed(20260716)
        model = model_cls(hidden_dim=128).to(device).eval()
        model_x9 = x9 if name == "B3_25D9D" else None
        with torch.no_grad():
            sequence, sequence_hidden = model.forward_sequence(x25, model_x9, mask=mask)
        step_outputs = {f"{head}_logit": [] for head in B3_HEADS}
        hidden = None
        hidden_norms = []
        with torch.no_grad():
            for step in range(tested_length):
                output, hidden = model.step(x25[:, step], None if model_x9 is None else x9[:, step], hidden, mask[:, step])
                for head, value in output.items():
                    step_outputs[head].append(value)
                hidden_norms.append(hidden_norm(hidden))
        stepped = {name: torch.stack(values, dim=1) for name, values in step_outputs.items()}
        step_error = max(
            float((sequence[name][mask] - stepped[name][mask]).abs().max().item()) for name in sequence
        )
        reset_error = 0.0
        for index, episode in enumerate(episodes):
            length = len(episode["rows"])
            single_model = model_cls(hidden_dim=128).to(device).eval()
            single_model.load_state_dict(model.state_dict())
            with torch.no_grad():
                single, _ = single_model.forward_sequence(
                    x25[index:index + 1, :length],
                    None if model_x9 is None else x9[index:index + 1, :length],
                )
            reset_error = max(reset_error, max(
                float((single[head] - sequence[head][index:index + 1, :length]).abs().max().item())
                for head in sequence
            ))
        finite = all(bool(torch.isfinite(value).all()) for value in sequence.values())
        results[name] = {
            "pass": finite and step_error <= 1e-6 and reset_error <= 1e-6,
            "sequence_step_max_abs": step_error,
            "episode_reset_max_abs": reset_error,
            "logits_finite": finite,
            "logit_abs_max": max(float(value.abs().max().item()) for value in sequence.values()),
            "hidden_norm_min": min(hidden_norms),
            "hidden_norm_max": max(hidden_norms),
            "tested_length": tested_length,
        }
    return {"pass": all(result["pass"] for result in results.values()), "input_stats": stats, "variants": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-episodes", type=int, default=31)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit("output root already exists; use a new full-commit root")
    output_root.mkdir(parents=True, exist_ok=False)
    source_root = args.source_root.resolve()
    before = source_fingerprint(source_root)
    episodes = read_episodes(source_root, args.expected_episodes)
    device = torch.device(args.device)
    forward = forward_smoke(episodes, device)
    after = source_fingerprint(source_root)
    git_head = git("rev-parse", "HEAD")
    git_status = git("status", "--porcelain")
    head_match = git_head == args.expected_head
    worktree_clean = not bool(git_status)
    script_tracked = script_in_head_tree(Path(__file__).resolve())
    status = "PASS" if forward["pass"] and before == after and head_match and worktree_clean and script_tracked else "FAIL"
    manifest = {
        "schema": "B3_REAL_STUDENT_FORWARD_SMOKE_V1",
        "status": status,
        "smoke_kind": "REAL_STUDENT_FORWARD_SMOKE",
        "not_model_selection": True,
        "no_teacher_labels": True,
        "teacher_files_opened": False,
        "metrics_produced": False,
        "source_root": str(source_root),
        "source_read_scope": ["materialization_manifest.json", "student_input_records.jsonl"],
        "source_sha256_before": before,
        "source_sha256_after": after,
        "source_unchanged": before == after,
        "episode_count": len(episodes),
        "git_head": git_head,
        "expected_head": args.expected_head,
        "head_match": head_match,
        "worktree_clean": worktree_clean,
        "git_status": git_status,
        "script_in_head_tree": script_tracked,
        "device": str(device),
        "torch_version": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "forward": forward,
        "evidence_sealed": True,
    }
    status_path = output_root / "REAL_STUDENT_FORWARD_SMOKE_STATUS.json"
    status_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = [
        "# B3 Real Student Forward Smoke",
        "",
        f"Status: `{status}`",
        "",
        "Compatibility-only student inputs were read. Teacher labels and effect metrics were not read or produced.",
        "",
        f"Episodes: `{len(episodes)}`",
        f"Source unchanged: `{before == after}`",
        f"HEAD match: `{head_match}`",
        f"Worktree clean: `{worktree_clean}`",
        f"Script in HEAD tree: `{script_tracked}`",
        f"Max forward length tested: `{max(result['tested_length'] for result in forward['variants'].values())}`",
    ]
    (output_root / "REAL_STUDENT_FORWARD_SMOKE_SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    write_evidence_seals(output_root, status_path)
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
