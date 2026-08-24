#!/usr/bin/env python3
"""Render fixed, anonymous D1M0 review copies from sealed clean MP4s."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
MAPPING = REPO / "reports/STAGE_X_X1R_T1D1M0_MANUAL_REVIEW_MAPPING_V1.json"
FREEZE = REPO / "reports/STAGE_X_X1R_T1D1M0_PREVIDEO_FREEZE_V1.json"
OUT_MANIFEST = REPO / "reports/STAGE_X_X1R_T1D1M0_REVIEW_RENDER_MANIFEST_V1.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def frame_count(path: Path) -> int:
    value = run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path),
    ])
    for line in value.splitlines():
        if line.strip().isdigit():
            return int(line.strip())
    raise RuntimeError(f"FRAME_COUNT_UNAVAILABLE:{path}")


def filter_graph(start: int, end: int, emit: int, tile: bool = False) -> str:
    marker = (
        f"drawbox=x=0:y=0:w=iw-1:h=ih-1:color=red@0.95:t=8:enable='eq(n,{emit})',"
        f"drawtext=text='T_EMIT':x=12:y=12:fontcolor=white:fontsize=24:box=1:boxcolor=red@0.9:enable='eq(n,{emit})'"
    )
    selection = f"select='between(n,{start},{end})',setpts=N/FRAME_RATE/TB"
    if tile:
        return f"{marker},{selection},scale=320:-2,tile=5x5:padding=4:margin=4:color=white"
    return f"{marker},{selection}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("STAGE_X_X1R_T1D1M0_HOLD_FFMPEG_UNAVAILABLE")
    mapping = load(MAPPING)
    freeze = load(FREEZE)
    if mapping.get("status") != "FROZEN_BLINDED_ORDER_BEFORE_VIDEO_PIXEL_ACCESS":
        raise SystemExit("D1M0_MAPPING_NOT_FROZEN")
    if freeze.get("status") != "PASS_PREVIDEO_FREEZE" or freeze.get("video_pixels_opened") is not False:
        raise SystemExit("D1M0_PREVIDEO_FREEZE_INVALID")
    rows = mapping.get("rows", [])
    if len(rows) != 14:
        raise SystemExit(f"D1M0_MAPPING_COUNT_INVALID:{len(rows)}")
    root = args.root.resolve()
    out_root = (args.output_root or (root / "T1D1M0_REVIEW_PACKET")).resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"D1M0_REVIEW_OUTPUT_EXISTS:{out_root}")
    out_root.mkdir(parents=True, exist_ok=False)
    rendered: list[dict[str, Any]] = []
    for row in rows:
        review_id = str(row["review_id"])
        raw = Path(str(row["raw_clean_video_path"]))
        if not raw.is_file() or sha(raw) != row["raw_clean_video_sha256"]:
            raise SystemExit(f"D1M0_RAW_VIDEO_BINDING_MISMATCH:{review_id}")
        source_frames = frame_count(raw)
        if source_frames != int(row["policy_steps_executed"]):
            raise SystemExit(f"STAGE_X_X1R_T1D1M0_HOLD_VIDEO_FRAME_MAPPING:{review_id}:{source_frames}:{row['policy_steps_executed']}")
        start, end, emit = int(row["context_start"]), int(row["context_end"]), int(row["first_emit_step"])
        expected_clip_frames = end - start + 1
        candidate_dir = out_root / review_id
        candidate_dir.mkdir()
        clip = candidate_dir / "review_clip.mp4"
        strip = candidate_dir / "review_frame_strip.png"
        graph = filter_graph(start, end, emit, tile=False)
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw), "-vf", graph, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "20", str(clip)])
        clip_frames = frame_count(clip)
        if clip_frames != expected_clip_frames:
            raise SystemExit(f"STAGE_X_X1R_T1D1M0_HOLD_REVIEW_CLIP_FRAME_COUNT:{review_id}:{clip_frames}:{expected_clip_frames}")
        strip_graph = filter_graph(start, end, emit, tile=True)
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw), "-vf", strip_graph, "-frames:v", "1", "-an", str(strip)])
        rendered.append({
            "review_id": review_id,
            "review_clip_path": str(clip),
            "review_clip_sha256": sha(clip),
            "review_clip_bytes": clip.stat().st_size,
            "review_clip_frame_count": clip_frames,
            "review_frame_strip_path": str(strip),
            "review_frame_strip_sha256": sha(strip),
            "review_frame_strip_bytes": strip.stat().st_size,
            "raw_clean_video_sha256": row["raw_clean_video_sha256"],
            "source_frame_count": source_frames,
            "context_start": start,
            "context_end": end,
            "emit_step": emit,
            "marker": "T_EMIT",
        })
    manifest = {
        "schema": "STAGE_X_X1R_T1D1M0_REVIEW_RENDER_MANIFEST_V1",
        "status": "PASS_FIXED_REVIEW_COPIES_RENDERED",
        "source": "sealed D1R raw clean videos only",
        "video_pixels_opened": True,
        "review_root": str(out_root),
        "candidate_count": len(rendered),
        "rows": rendered,
        "raw_videos_unchanged": True,
        "prohibited_execution": {"model_inference": 0, "student_inference": 0, "env_steps": 0, "pgd_calls": 0, "physical_interventions": 0, "vphys_reads": 0, "eval160_reads": 0, "protected_reads": 0},
        "next_gate": "OWNER_MANUAL_CONTACT_LABELS_REQUIRED",
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "candidate_count": len(rendered), "review_root": str(out_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
