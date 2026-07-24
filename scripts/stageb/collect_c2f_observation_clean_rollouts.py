#!/usr/bin/env python3
"""C2f observation-rich CLEAN rollout collector scaffold.

Purpose
-------
Collect clean rollouts with RGB frames + task language + 25D temporal features
+ teacher event labels for the post-D7 C2f detector route.

This script intentionally does NOT run attacks and must not read D7B2 outcomes.
It is a strict artifact writer + adapter boundary. DeepSeek should implement the
LIBERO/OpenVLA-specific adapter in `make_runtime_adapter()` without changing the
artifact schema.

CPU/GPU boundary
----------------
- May use an idle GPU for OpenVLA inference if the adapter needs it.
- Does not modify D7B2 worker/detector/thresholds.
- Writes to a separate C2f output root only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

SCHEMA = "C2F_OBS_LANG_CLEAN_EPISODE_V1"
EVENT_ROLES = {
    "primary_attackable",
    "auxiliary_manipulation",
    "distractor_or_setup",
    "unsupported_or_abstain",
}

CANONICAL_25D_FEATURES = [
    "gripper_command", "gripper_qpos", "gripper_opening_proxy",
    "eef_x", "eef_y", "eef_z", "eef_vx", "eef_vy", "eef_vz",
    "action_dx", "action_dy", "action_dz", "action_gripper",
    "recent_close_streak", "recent_open_streak", "recent_gripper_flip_count",
    "close_onset", "time_since_close", "eef_speed", "eef_z_delta_since_close",
    "qpos_delta_1", "qpos_delta_3", "opening_proxy_delta_3",
    "opening_proxy_variance_5", "eef_speed_variance_5",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text())
        if isinstance(obj, list):
            return obj
        if "episodes" in obj:
            return list(obj["episodes"])
        raise ValueError("JSON manifest must be a list or contain an 'episodes' list")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported manifest suffix: {path.suffix}")


@dataclass
class StepRecord:
    step: int
    rgb_array: Any  # np.ndarray-like HxWx3 uint8, or None if adapter writes rgb_path itself
    rgb_path: Optional[str]
    features_25d: List[float]
    task_language: str
    teacher_hazard: int
    teacher_primary_attackable: int
    teacher_release_safe: int
    teacher_event_role: str
    teacher_phase: str = ""


class RuntimeAdapter:
    """LIBERO/OpenVLA adapter interface.

    DeepSeek should implement this interface using the repo's existing OpenVLA
    bridge/worker utilities. The artifact writer below is intentionally stable.
    """

    def run_clean_episode(self, episode_cfg: Dict[str, Any]) -> Iterable[StepRecord]:
        raise NotImplementedError

    def close(self) -> None:
        pass


def make_runtime_adapter(args: argparse.Namespace) -> RuntimeAdapter:
    """Create the actual LIBERO/OpenVLA adapter.

    TODO(DeepSeek): bind to existing suite model paths and clean policy runner.
    Required output per step:
      - RGB frame as np.uint8 HxWx3 or adapter-provided rgb_path
      - canonical 25D feature vector
      - task language
      - clean-only teacher labels from privileged state, labels only
    """
    if args.adapter_module:
        import importlib
        mod_name, _, fn_name = args.adapter_module.partition(":")
        if not mod_name or not fn_name:
            raise ValueError("--adapter-module must be 'module.path:function_name'")
        mod = importlib.import_module(mod_name)
        return getattr(mod, fn_name)(args)
    raise RuntimeError(
        "No runtime adapter configured. Pass --adapter-module module:function. "
        "This scaffold is intentionally adapter-free to preserve the C2f schema."
    )


def save_rgb_png(path: Path, rgb_array: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError("Pillow is required to save RGB frames") from e
    Image.fromarray(rgb_array).save(path)


def validate_step(row: Dict[str, Any]) -> None:
    if len(row.get("features_25d", [])) != 25:
        raise ValueError(f"features_25d must have length 25, got {len(row.get('features_25d', []))}")
    if not str(row.get("task_language", "")).strip():
        raise ValueError("task_language must be non-empty for C2f visual-language grounding")
    role = row.get("teacher_event_role", "")
    if role not in EVENT_ROLES:
        raise ValueError(f"Invalid teacher_event_role={role!r}; expected one of {sorted(EVENT_ROLES)}")
    for k in ["teacher_hazard", "teacher_primary_attackable", "teacher_release_safe"]:
        if int(row.get(k, -1)) not in (0, 1):
            raise ValueError(f"{k} must be 0/1")


def collect_one_episode(adapter: RuntimeAdapter, out_root: Path, episode_cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    suite = str(episode_cfg["suite"])
    parent_key = str(episode_cfg.get("parent_key") or episode_cfg.get("episode_id") or f"{suite}_{int(time.time()*1000)}")
    ep_dir = out_root / "episodes" / suite / parent_key
    rgb_dir = ep_dir / "rgb"
    step_jsonl = ep_dir / "step_records.jsonl"
    if step_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f"Existing episode artifacts: {ep_dir}; pass --overwrite to replace")
    if args.overwrite and ep_dir.exists():
        import shutil
        shutil.rmtree(ep_dir)
    ep_dir.mkdir(parents=True, exist_ok=True)

    n_steps = 0
    clean_success_manifest = bool(episode_cfg.get("clean_success", False))
    clean_success = clean_success_manifest
    first_task_language = ""
    t0 = time.time()

    for rec in adapter.run_clean_episode(episode_cfg):
        rel_rgb = rec.rgb_path
        if rel_rgb is None:
            rel_rgb = f"rgb/frame_{rec.step:06d}.png"
            save_rgb_png(ep_dir / rel_rgb, rec.rgb_array)
        row = {
            "step": int(rec.step),
            "rgb_path": rel_rgb,
            "features_25d": [float(x) for x in rec.features_25d],
            "task_language": str(rec.task_language),
            "teacher_hazard": int(rec.teacher_hazard),
            "teacher_primary_attackable": int(rec.teacher_primary_attackable),
            "teacher_release_safe": int(rec.teacher_release_safe),
            "teacher_event_role": str(rec.teacher_event_role),
            "teacher_phase": str(rec.teacher_phase),
        }
        validate_step(row)
        if not first_task_language and row["task_language"].strip():
            first_task_language = row["task_language"].strip()
        append_jsonl(step_jsonl, row)
        n_steps += 1

    adapter_info = getattr(adapter, "_last_episode_info", {}) or {}
    if "clean_success_observed" in adapter_info:
        clean_success = bool(adapter_info.get("clean_success_observed", False))
    resolved_task_language = str(adapter_info.get("task_language") or first_task_language or episode_cfg.get("task_language", ""))

    meta = {
        "schema": SCHEMA,
        "suite": suite,
        "task_index": int(episode_cfg.get("task_index", -1)),
        "task_name": str(episode_cfg.get("task_name") or adapter_info.get("task_name_resolved", "")),
        "task_language": resolved_task_language,
        "task_language_source": str(adapter_info.get("task_language_source", "")),
        "parent_key": parent_key,
        "condition": "CLEAN",
        "n_steps": n_steps,
        "clean_success": clean_success,
        "clean_success_manifest": clean_success_manifest,
        "clean_success_observed": bool(adapter_info.get("clean_success_observed", False)),
        "adapter_episode_info": adapter_info,
        "student_allowed_modalities": ["rgb", "task_language", "features_25d", "context_108d"],
        "student_forbidden_modalities": ["object_pose", "target_pose", "attack_outcome", "manual_failure_label"],
        "source_commit": args.source_commit,
        "collector_commit": args.git_commit,
        "runtime_seconds": time.time() - t0,
    }
    write_json(ep_dir / "episode_metadata.json", meta)
    return {"suite": suite, "parent_key": parent_key, "n_steps": n_steps, "episode_dir": str(ep_dir), "clean_success": clean_success}


def write_sha256s(root: Path) -> None:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.sha256"}:
            rows.append((sha256_file(p), p.relative_to(root).as_posix()))
    sums = root / "SHA256SUMS"
    sums.write_text("".join(f"{h}  {rel}\n" for h, rel in rows))
    (root / "SHA256SUMS.sha256").write_text(f"{sha256_file(sums)}  SHA256SUMS\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect C2f observation-rich CLEAN rollouts")
    ap.add_argument("--manifest", required=True, help="JSON/JSONL/CSV episode manifest")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--adapter-module", default="", help="module:function returning RuntimeAdapter")
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--suite", default="", help="optional suite filter")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--git-commit", required=True)
    ap.add_argument("--source-commit", default="")
    args = ap.parse_args()

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    episodes = read_manifest(Path(args.manifest))
    if args.suite:
        episodes = [e for e in episodes if str(e.get("suite")) == args.suite]
    if args.max_episodes > 0:
        episodes = episodes[:args.max_episodes]

    adapter = make_runtime_adapter(args)
    results = []
    try:
        for i, ep in enumerate(episodes):
            print(f"[C2f collect] {i+1}/{len(episodes)} suite={ep.get('suite')} parent={ep.get('parent_key','')}", flush=True)
            results.append(collect_one_episode(adapter, out, ep, args))
    finally:
        adapter.close()

    manifest = {
        "schema": "C2F_OBS_LANG_CLEAN_COLLECTION_V1",
        "created_at_unix": time.time(),
        "git_commit": args.git_commit,
        "source_commit": args.source_commit,
        "n_episodes": len(results),
        "episodes": results,
        "boundaries": {
            "condition": "CLEAN_ONLY",
            "attack": "NOT_PERFORMED",
            "d7b2_outcome_read": False,
        },
    }
    write_json(out / "manifest.json", manifest)
    write_sha256s(out)
    print(json.dumps({"status": "C2F_COLLECTION_WRITTEN", "n_episodes": len(results), "output_root": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
