#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml
from PIL import Image, ImageDraw


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def norm_name(x) -> str:
    return "".join(ch.lower() for ch in str(x) if ch.isalnum() or ch == "_")


def resolve_task_index(bench, task_name: str) -> int:
    names = list(bench.get_task_names())
    if task_name in names:
        return names.index(task_name)
    wanted = norm_name(task_name)
    for idx, name in enumerate(names):
        if norm_name(name) == wanted:
            return idx
    raise ValueError(f"task_name not found: {task_name}; available={names}")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_episode_ids(text: str, available: list[int], *, max_auto: int = 2) -> list[int]:
    if text.lower() == "all":
        return available
    if text.lower() == "auto":
        return available[:max_auto]
    ids: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        ids.append(int(part))
    return ids


def parse_state_ids_from_manifest(run_dir: Path) -> list[int]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        argv = shlex.split(str(manifest.get("command", "")))
        if "--state_ids" in argv:
            raw = argv[argv.index("--state_ids") + 1]
            return [int(x.strip()) for x in raw.split(",") if x.strip()]
    except Exception:
        return []
    return []


def overlay_frame(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    # Draw two rectangles for readable text without depending on custom fonts.
    pad = 4
    text_bbox = draw.textbbox((0, 0), label)
    w = text_bbox[2] - text_bbox[0] + 2 * pad
    h = text_bbox[3] - text_bbox[1] + 2 * pad
    draw.rectangle((0, 0, min(image.width, w), h), fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255))
    return np.asarray(image)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks_config", default="configs/v4_tasks_libero.yaml")
    ap.add_argument("--task_id", required=True)
    ap.add_argument("--run_dir", required=True, help="Directory containing step_records.jsonl and episode_records.jsonl")
    ap.add_argument("--episode_ids", default="auto", help="Comma-separated ids, 'all', or 'auto'")
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--camera_obs_key", default="agentview_image")
    ap.add_argument("--render_gpu_device_id", type=int, default=0)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--frame_stride", type=int, default=2)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--action_field", default="env_action", choices=["env_action", "executed_action", "clean_action"])
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--horizon_override", type=int, default=0)
    ap.add_argument("--no_overlay", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    step_path = run_dir / "step_records.jsonl"
    episode_path = run_dir / "episode_records.jsonl"
    if not step_path.exists():
        raise SystemExit(f"missing {step_path}")
    if not episode_path.exists():
        raise SystemExit(f"missing {episode_path}")

    tasks = load_yaml(args.tasks_config)["tasks"]
    task = next(t for t in tasks if t["task_id"] == args.task_id)

    steps = read_jsonl(step_path)
    episodes = read_jsonl(episode_path)
    steps_by_ep: dict[int, list[dict]] = {}
    for row in steps:
        steps_by_ep.setdefault(int(row["episode_id"]), []).append(row)
    for rows in steps_by_ep.values():
        rows.sort(key=lambda r: int(r["step_idx"]))

    episode_meta = {int(e["episode_id"]): e for e in episodes}
    available = sorted(steps_by_ep)
    episode_ids = parse_episode_ids(args.episode_ids, available)
    state_ids = parse_state_ids_from_manifest(run_dir)
    if not episode_ids:
        raise SystemExit("no episode ids selected")

    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    bench = get_benchmark(task["suite"])()
    task_idx = resolve_task_index(bench, task["task_name"])
    init_states = bench.get_task_init_states(task_idx)
    max_replay_step = max((int(row.get("step_idx", 0)) for row in steps), default=0) + 2
    max_episode_steps = max((int(e.get("num_steps", 0)) for e in episodes), default=0) + 2
    horizon = int(args.horizon_override or max(int(task.get("max_steps", 400)), max_replay_step, max_episode_steps))
    env = OffScreenRenderEnv(
        bddl_file_name=bench.get_task_bddl_file_path(task_idx),
        camera_heights=int(args.image_size),
        camera_widths=int(args.image_size),
        render_gpu_device_id=int(args.render_gpu_device_id),
        horizon=horizon,
    )
    try:
        env.seed(0)
    except Exception:
        pass

    for ep in episode_ids:
        rows = steps_by_ep.get(ep, [])
        if not rows:
            print(f"[skip] episode {ep}: no steps", flush=True)
            continue
        meta = episode_meta.get(ep, {})
        state_id = state_ids[ep] if ep < len(state_ids) else ep % len(init_states)
        obs = env.reset()
        obs = env.set_init_state(init_states[int(state_id)])
        success = bool(meta.get("success", False))
        timeout = bool(meta.get("timeout", False))
        trigger_name = str(meta.get("trigger_name", rows[0].get("trigger_name", "")))
        run_id = str(rows[0].get("run_id", run_dir.name))
        status = "success" if success else ("timeout" if timeout else "failure")
        out_path = out_dir / f"{run_id}_ep{ep:03d}_{status}.mp4"

        with imageio.get_writer(out_path, fps=int(args.fps), macro_block_size=1) as writer:
            frames_written = 0
            first = np.asarray(obs[args.camera_obs_key]).astype(np.uint8)
            label = f"ep={ep} init={state_id} t=init {trigger_name} {status}"
            writer.append_data(first if args.no_overlay else overlay_frame(first, label))
            frames_written += 1
            for row in rows:
                action = np.asarray(row[args.action_field], dtype=np.float32)
                try:
                    obs, _, done, _ = env.step(action)
                except ValueError as exc:
                    if "terminated episode" in str(exc):
                        break
                    raise
                step_idx = int(row.get("step_idx", 0))
                if step_idx % max(1, int(args.frame_stride)) == 0 or done:
                    frame = np.asarray(obs[args.camera_obs_key]).astype(np.uint8)
                    if not args.no_overlay:
                        attack = "A" if row.get("attack_active") else "-"
                        label = f"ep={ep} t={step_idx} {trigger_name} {status} attack={attack}"
                        frame = overlay_frame(frame, label)
                    writer.append_data(frame)
                    frames_written += 1
                    if args.max_frames and frames_written >= int(args.max_frames):
                        break
                if done:
                    break
        print(f"[ok] wrote {out_path} frames={frames_written}", flush=True)


if __name__ == "__main__":
    main()
