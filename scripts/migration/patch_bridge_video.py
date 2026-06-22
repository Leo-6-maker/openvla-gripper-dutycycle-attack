#!/usr/bin/env python3
"""Patch A800 bridge to add --save_video support. Makes .bak backup."""
import sys, shutil

BRIDGE_PATH = "/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_v2_vis_sc5_mlp_bridge.py"

with open(BRIDGE_PATH) as f:
    code = f.read()

shutil.copy(BRIDGE_PATH, BRIDGE_PATH + ".bak")
print("Backup: %s.bak" % BRIDGE_PATH)

patches_ok = 0

# P1: Add argparse video args
old = 'ap.add_argument("--task_idx", type=int, default=6, help="LIBERO task index (default 6=butter)")\nargs = ap.parse_args()'
new = ('ap.add_argument("--task_idx", type=int, default=6, help="LIBERO task index (default 6=butter)")\n'
       'ap.add_argument("--save_video", action="store_true", default=False)\n'
       'ap.add_argument("--source_commit", default="", help="Git commit SHA (required when --save_video)")\n'
       'ap.add_argument("--video_fps", type=int, default=20)\n'
       'ap.add_argument("--frame_stride", type=int, default=1)\n'
       'ap.add_argument("--save_raw_frames", action="store_true", default=False)\n'
       'args = ap.parse_args()\n'
       '\n'
       'if args.save_video and not args.source_commit:\n'
       '    raise ValueError("--source_commit is required when --save_video is enabled")')
assert old in code, "P1: old block not found"
code = code.replace(old, new)
patches_ok += 1
print("P1 OK")

# P2: Add copy, hashlib to imports
old = 'import argparse, csv, json, os, sys, time, numpy as np, torch'
new = 'import argparse, copy, csv, hashlib, json, os, sys, time, numpy as np, torch'
assert old in code, "P2: old imports not found"
code = code.replace(old, new)
patches_ok += 1
print("P2 OK")

# P3: Video init after detector load
old = 'print("MLP detector loaded, dataset_sha256=%s" % detector.dataset_sha256[:16])'
new = ('print("MLP detector loaded, dataset_sha256=%s" % detector.dataset_sha256[:16])\n'
       '\n'
       '# Video recording init\n'
       '_video_raw_frames = []\n'
       'if args.save_video:\n'
       '    print("Video recording ENABLED (fps=%d, stride=%d)" % (args.video_fps, args.frame_stride))')
assert old in code, "P3: old detector print not found"
code = code.replace(old, new)
patches_ok += 1
print("P3 OK")

# P4: Frame capture after env.step
old = 'obs, _, done, _ = env.step(env_action_final)'
new = ('obs, _, done, _ = env.step(env_action_final)\n'
       '            if args.save_video and step % args.frame_stride == 0:\n'
       '                try:\n'
       '                    _raw = obs.get("agentview_image", None)\n'
       '                    if _raw is not None:\n'
       '                        _raw_copy = copy.deepcopy(_raw)\n'
       '                        _video_raw_frames.append(np.asarray(_raw_copy))\n'
       '                except Exception:\n'
       '                    pass')
assert old in code, "P4: old env.step line not found"
code = code.replace(old, new)
patches_ok += 1
print("P4 OK")

# P5: Video encoding before output directory creation
old = 'out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)'
new = ('# Video encoding\n'
       '_video_manifest = {}\n'
       'if args.save_video and _video_raw_frames:\n'
       '    try:\n'
       '        from imageio.v2 import mimwrite as _mimwrite\n'
       '        out_vdir = Path(args.output_dir)\n'
       '        out_vdir.mkdir(parents=True, exist_ok=True)\n'
       '        _raw_path = out_vdir / "rollout_raw.mp4"\n'
       '        _mimwrite(str(_raw_path), [np.asarray(f) for f in _video_raw_frames],\n'
       '                  fps=args.video_fps, codec="libx264", quality=8,\n'
       '                  output_params=["-preset", "fast"])\n'
       '        print("Video saved: %s (%d frames)" % (_raw_path, len(_video_raw_frames)))\n'
       '        _video_manifest = {\n'
       '            "raw_video_path": str(_raw_path),\n'
       '            "frame_count": len(_video_raw_frames),\n'
       '            "fps": args.video_fps,\n'
       '            "stride": args.frame_stride,\n'
       '            "source_commit": args.source_commit,\n'
       '        }\n'
       '    except Exception as _ve:\n'
       '        print("Video encoding failed: %s" % _ve)\n'
       '\n'
       'out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)')
assert old in code, "P5: old output dir line not found"
code = code.replace(old, new)
patches_ok += 1
print("P5 OK")

# P6: Add video manifest before summary write
old = 'with open(out / "episode_summary.json", "w") as f:\n    json.dump(summary, f, indent=2, default=str)'
new = ('if _video_manifest:\n    summary["video"] = _video_manifest\n'
       'with open(out / "episode_summary.json", "w") as f:\n    json.dump(summary, f, indent=2, default=str)')
assert old in code, "P6: old summary write not found"
code = code.replace(old, new)
patches_ok += 1
print("P6 OK")

# P7: Tag print line with [VIDEO]
old = '    args.condition, STATE_ID, ANCHOR, _mlp_emit,'
new = '    args.condition, STATE_ID, ANCHOR, _mlp_emit, " [VIDEO]" if args.save_video else "",'
assert old in code, "P7: old print line not found"
code = code.replace(old, new)
patches_ok += 1
print("P7 OK")

with open(BRIDGE_PATH, "w") as f:
    f.write(code)

print("\nAll %d patches applied. Bridge updated." % patches_ok)
