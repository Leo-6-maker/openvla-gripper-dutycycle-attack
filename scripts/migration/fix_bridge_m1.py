#!/usr/bin/env python3
"""Fix A800 bridge for M1: dtype/attn env-vars, --save_video, runtime attestation."""
import sys, shutil

BRIDGE = "/mnt/sdc/dty_user/openvla_attack/scripts/stageb/run_v2_vis_sc5_mlp_bridge.py"

with open(BRIDGE) as f:
    code = f.read()

shutil.copy(BRIDGE, BRIDGE + ".m1bak")
print("Backup: %s.m1bak" % BRIDGE)
ok = 0

# === FIX 1: Replace model loading with env-var based dtype/attn ===
old_model_load = '''model = AutoModelCls.from_pretrained(
    MODEL_PATH, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, device_map="cuda:0", attn_implementation="eager")
model_dtype = next(model.parameters()).dtype
device = "cuda:0"'''

new_model_load = '''# M1: dtype and attention from env vars with fail-closed validation
_dtype_name = os.environ.get("OPENVLA_DTYPE", "bfloat16")
_attn_name = os.environ.get("OPENVLA_ATTN_IMPLEMENTATION", "eager")
_dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
if _dtype_name not in _dtype_map:
    raise RuntimeError("OPENVLA_DTYPE must be bfloat16 or float32, got: %s" % _dtype_name)
_torch_dtype = _dtype_map[_dtype_name]
model = AutoModelCls.from_pretrained(
    MODEL_PATH, trust_remote_code=True, local_files_only=True, torch_dtype=_torch_dtype,
    low_cpu_mem_usage=True, device_map="cuda:0", attn_implementation=_attn_name)
model_dtype = next(model.parameters()).dtype
device = "cuda:0"
# Runtime self-attestation
_actual_dtype_str = str(model_dtype).replace("torch.", "")
_actual_attn = getattr(model.config, "_attn_implementation", "unknown")
print("Model on %s dtype=%s attn=%s (requested: dtype=%s attn=%s)" % (
    device, _actual_dtype_str, _actual_attn, _dtype_name, _attn_name))'''

assert old_model_load in code, "F1: old model load block not found"
code = code.replace(old_model_load, new_model_load)
ok += 1
print("F1 (dtype/attn env-var): OK")

# === FIX 2: Add --save_video args and validation ===
old_args = 'ap.add_argument("--task_idx", type=int, default=6, help="LIBERO task index (default 6=butter)")\nargs = ap.parse_args()'
new_args = ('ap.add_argument("--task_idx", type=int, default=6, help="LIBERO task index (default 6=butter)")\n'
    'ap.add_argument("--save_video", action="store_true", default=False)\n'
    'ap.add_argument("--source_commit", default="", help="Git commit SHA (required when --save_video)")\n'
    'ap.add_argument("--video_fps", type=int, default=20)\n'
    'ap.add_argument("--frame_stride", type=int, default=1)\n'
    'args = ap.parse_args()\n'
    '\n'
    'if args.save_video and not args.source_commit:\n'
    '    raise ValueError("--source_commit is required when --save_video is enabled")')
assert old_args in code, "F2: old args block not found"
code = code.replace(old_args, new_args)
ok += 1
print("F2 (save_video args): OK")

# === FIX 3: Add copy import ===
old_imports = 'import argparse, csv, json, os, sys, time, numpy as np, torch'
new_imports = 'import argparse, copy, csv, hashlib, json, os, sys, time, numpy as np, torch'
assert old_imports in code, "F3: old imports not found"
code = code.replace(old_imports, new_imports)
ok += 1
print("F3 (imports): OK")

# === FIX 4: Video init after detector ===
old_det = 'print("MLP detector loaded, dataset_sha256=%s" % detector.dataset_sha256[:16])'
new_det = ('print("MLP detector loaded, dataset_sha256=%s" % detector.dataset_sha256[:16])\n'
    '\n'
    '_video_raw_frames = []\n'
    'if args.save_video:\n'
    '    print("Video recording ENABLED (fps=%d, stride=%d)" % (args.video_fps, args.frame_stride))')
assert old_det in code, "F4: old detector print not found"
code = code.replace(old_det, new_det)
ok += 1
print("F4 (video init): OK")

# === FIX 5: Frame capture after env.step (correct indentation) ===
old_step = 'obs, _, done, _ = env.step(env_action_final)'
new_step = ('obs, _, done, _ = env.step(env_action_final)\n'
    '    if args.save_video and step % args.frame_stride == 0:\n'
    '        try:\n'
    '            _raw = obs.get("agentview_image", None)\n'
    '            if _raw is not None:\n'
    '                _raw_copy = copy.deepcopy(_raw)\n'
    '                _video_raw_frames.append(np.asarray(_raw_copy))\n'
    '        except Exception:\n'
    '            pass')
assert old_step in code, "F5: old env.step not found"
code = code.replace(old_step, new_step)
ok += 1
print("F5 (frame capture): OK")

# === FIX 6: Video encoding before output dir creation ===
old_out = 'out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)'
new_out = ('_video_manifest = {}\n'
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
assert old_out in code, "F6: old output dir not found"
code = code.replace(old_out, new_out)
ok += 1
print("F6 (video encoding): OK")

# === FIX 7: Runtime attestation + video in episode_summary ===
old_summary = 'with open(out / "episode_summary.json", "w") as f:\n    json.dump(summary, f, indent=2, default=str)'
new_summary = ('summary["requested_dtype"] = _dtype_name\n'
    'summary["actual_dtype"] = _actual_dtype_str\n'
    'summary["requested_attn"] = _attn_name\n'
    'summary["actual_attn"] = _actual_attn\n'
    'if _video_manifest:\n'
    '    summary["video"] = _video_manifest\n'
    'with open(out / "episode_summary.json", "w") as f:\n    json.dump(summary, f, indent=2, default=str)')
assert old_summary in code, "F7: old summary write not found"
code = code.replace(old_summary, new_summary)
ok += 1
print("F7 (runtime attestation): OK")

# === FIX 8: Tag print with [VIDEO] ===
old_print = '    args.condition, STATE_ID, ANCHOR, _mlp_emit,'
new_print = '    args.condition, STATE_ID, ANCHOR, _mlp_emit, " [VIDEO]" if args.save_video else "",'
assert old_print in code, "F8: old print line not found"
code = code.replace(old_print, new_print)
ok += 1
print("F8 (print tag): OK")

# === FIX 9: Add full action recording to telemetry ===
# The telemetry already has raw_gripper and env_gripper but may lack full 7D action
# Add raw_action_7d and env_action_7d if not present
old_tel_action = "_tel[\"raw_gripper\"] = raw_grip"
new_tel_action = ('_tel["raw_gripper"] = raw_grip\n'
    '        _tel["env_gripper"] = env_grip\n'
    '        _tel["raw_action_7d\"] = json.dumps([float(x) for x in action]) if \"action\" in dir() else \"[]"\n'
    '        _tel["env_action_7d\"] = json.dumps([float(x) for x in env_action_final])')
if old_tel_action in code:
    code = code.replace(old_tel_action, new_tel_action)
    ok += 1
    print("F9 (full action telemetry): OK")
else:
    print("F9 (full action telemetry): SKIPPED — raw_grip line not found, may already have full actions")

# Write
with open(BRIDGE, "w") as f:
    f.write(code)

print("\n%d fixes applied." % ok)
print("Bridge: %s" % BRIDGE)
