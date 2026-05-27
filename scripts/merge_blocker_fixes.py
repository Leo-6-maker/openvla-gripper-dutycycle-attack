#!/usr/bin/env python3
"""Apply merge-blocker fixes to v4 runner and attack adapter."""
from pathlib import Path

REPO = Path("/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524")
RUNNER = REPO / "scripts/v4_run_eval_openvla.py"
ADAPTER = REPO / "src/gripper_attack/attack_adapter.py"

# Backup
for f in [RUNNER, ADAPTER]:
    bak = Path(str(f) + ".merge_fix_backup")
    if not bak.exists():
        bak.write_bytes(f.read_bytes())

text = RUNNER.read_text()

# Fix 1: Remove the tf_jpeg_legacy override in prepare_openvla_image
old = '''    backend = str(libero_preprocess_backend or "official_pil_lanczos")
    # Legacy boolean flag overrides to tf_jpeg_legacy for backward compat
    if libero_official_preprocess and backend == "official_pil_lanczos":
        backend = "tf_jpeg_legacy"'''
new = '''    backend = str(libero_preprocess_backend or "official_pil_lanczos")
    # --libero_official_preprocess is a deprecated compatibility alias.
    # It does NOT switch to tf_jpeg_legacy. Use --libero_preprocess_backend explicitly.'''
if old in text:
    text = text.replace(old, new)
    print("Fix 1: Removed tf_jpeg_legacy override")
else:
    print("Fix 1: SKIP (not found)")

# Fix 2: Update docstring
old2 = '''    When ``libero_official_preprocess`` is True (legacy flag), uses tf_jpeg_legacy.
    The ``libero_preprocess_backend`` flag takes precedence over the legacy boolean.'''
new2 = '''    The ``--libero_official_preprocess`` flag is a deprecated compatibility alias;
    it no longer switches the backend. Use --libero_preprocess_backend to select.'''
if old2 in text:
    text = text.replace(old2, new2)
    print("Fix 2: Updated docstring")
else:
    print("Fix 2: SKIP (not found)")

# Fix 3: Update help text for --libero_official_preprocess
old3 = '''ap.add_argument("--libero_official_preprocess",action="store_true",help="(Legacy) Use TF JPEG preprocessing. Prefer --libero_preprocess_backend official_pil_lanczos for official-aligned runs.")'''
new3 = '''ap.add_argument("--libero_official_preprocess",action="store_true",help="(Deprecated) Compatibility alias. Does NOT switch preprocessing backend. Use --libero_preprocess_backend to select. Default is official_pil_lanczos.")'''
if old3 in text:
    text = text.replace(old3, new3)
    print("Fix 3: Updated --libero_official_preprocess help")
else:
    print("Fix 3: SKIP (not found)")

# Fix 4: Pass libero_preprocess_backend to attack adapter
old4 = '''preprocess_kwargs={"libero_official_preprocess": args.libero_official_preprocess, "center_crop": args.center_crop, "resize_size": args.openvla_resize_size, "postprocess_gripper": args.postprocess_gripper}'''
new4 = '''preprocess_kwargs={"libero_official_preprocess": args.libero_official_preprocess, "libero_preprocess_backend": args.libero_preprocess_backend, "center_crop": args.center_crop, "resize_size": args.openvla_resize_size, "postprocess_gripper": args.postprocess_gripper}'''
if old4 in text:
    text = text.replace(old4, new4)
    print("Fix 4: Pass libero_preprocess_backend to attack adapter")
else:
    print("Fix 4: SKIP (not found)")

# Fix 5: Pass libero_preprocess_backend to decode_with_scores calls in main loop
# Line 1095: clean decode
old5a = '''clean,prefix_logits,Tclean,out_clean=decode_with_scores(model,processor,device,obs[args.camera_obs_key],instruction,unnorm,int(cfg["uncertainty"]["K_trigger"]),libero_official_preprocess=args.libero_official_preprocess,center_crop=args.center_crop,resize_size=args.openvla_resize_size,drop_attention_mask=(not args.keep_attention_mask))'''
new5a = '''clean,prefix_logits,Tclean,out_clean=decode_with_scores(model,processor,device,obs[args.camera_obs_key],instruction,unnorm,int(cfg["uncertainty"]["K_trigger"]),libero_official_preprocess=args.libero_official_preprocess,libero_preprocess_backend=args.libero_preprocess_backend,center_crop=args.center_crop,resize_size=args.openvla_resize_size,drop_attention_mask=(not args.keep_attention_mask))'''
if old5a in text:
    text = text.replace(old5a, new5a)
    print("Fix 5a: Backend passthrough in clean decode (main loop)")
else:
    print("Fix 5a: SKIP (not found)")

# Fix 5b: executed decode (line ~1156)
old5b = '''executed,_,adv_decode_sec,adv_gen=decode_with_scores(model,processor,device,noisy_img_uint8,instruction,unnorm,int(cfg["uncertainty"]["K_trigger"]),libero_official_preprocess=args.libero_official_preprocess,center_crop=args.center_crop,resize_size=args.openvla_resize_size,drop_attention_mask=(not args.keep_attention_mask))'''
new5b = '''executed,_,adv_decode_sec,adv_gen=decode_with_scores(model,processor,device,noisy_img_uint8,instruction,unnorm,int(cfg["uncertainty"]["K_trigger"]),libero_official_preprocess=args.libero_official_preprocess,libero_preprocess_backend=args.libero_preprocess_backend,center_crop=args.center_crop,resize_size=args.openvla_resize_size,drop_attention_mask=(not args.keep_attention_mask))'''
if old5b in text:
    text = text.replace(old5b, new5b)
    print("Fix 5b: Backend passthrough in attack decode (noisy)")
else:
    print("Fix 5b: SKIP (not found)")

# Fix 5c: second executed decode (line ~1164)
old5c = '''executed,_,adv_decode_sec,adv_gen=decode_with_scores(model,processor,device,adv_img,instruction,unnorm,int(cfg["uncertainty"]["K_trigger"]),libero_official_preprocess=args.libero_official_preprocess,center_crop=args.center_crop,resize_size=args.openvla_resize_size,drop_attention_mask=(not args.keep_attention_mask))'''
new5c = '''executed,_,adv_decode_sec,adv_gen=decode_with_scores(model,processor,device,adv_img,instruction,unnorm,int(cfg["uncertainty"]["K_trigger"]),libero_official_preprocess=args.libero_official_preprocess,libero_preprocess_backend=args.libero_preprocess_backend,center_crop=args.center_crop,resize_size=args.openvla_resize_size,drop_attention_mask=(not args.keep_attention_mask))'''
if old5c in text:
    text = text.replace(old5c, new5c)
    print("Fix 5c: Backend passthrough in attack decode (adv)")
else:
    print("Fix 5c: SKIP (not found)")

# Fix 5d: offline audit clean decode (line ~976)
old5d = '''base_action,_,base_dt,base_gen=decode_with_scores(model,processor,device,obs[args.camera_obs_key],base_instruction,unnorm,int(cfg["uncertainty"]["K_trigger"]),libero_official_preprocess=args.libero_official_preprocess,center_crop=args.center_crop,resize_size=args.openvla_resize_size,drop_attention_mask=(not args.keep_attention_mask))'''
new5d = '''base_action,_,base_dt,base_gen=decode_with_scores(model,processor,device,obs[args.camera_obs_key],base_instruction,unnorm,int(cfg["uncertainty"]["K_trigger"]),libero_official_preprocess=args.libero_official_preprocess,libero_preprocess_backend=args.libero_preprocess_backend,center_crop=args.center_crop,resize_size=args.openvla_resize_size,drop_attention_mask=(not args.keep_attention_mask))'''
if old5d in text:
    text = text.replace(old5d, new5d)
    print("Fix 5d: Backend passthrough in offline audit clean decode")
else:
    print("Fix 5d: SKIP (not found)")

# Fix 5e: offline audit variant decode (line ~983)
old5e = '''action,_,dt,gen=decode_with_scores(model,processor,device,obs[args.camera_obs_key],variant["instruction"],unnorm,int(cfg["uncertainty"]["K_trigger"]),libero_official_preprocess=args.libero_official_preprocess,center_crop=args.center_crop,resize_size=args.openvla_resize_size,drop_attention_mask=(not args.keep_attention_mask))'''
new5e = '''action,_,dt,gen=decode_with_scores(model,processor,device,obs[args.camera_obs_key],variant["instruction"],unnorm,int(cfg["uncertainty"]["K_trigger"]),libero_official_preprocess=args.libero_official_preprocess,libero_preprocess_backend=args.libero_preprocess_backend,center_crop=args.center_crop,resize_size=args.openvla_resize_size,drop_attention_mask=(not args.keep_attention_mask))'''
if old5e in text:
    text = text.replace(old5e, new5e)
    print("Fix 5e: Backend passthrough in offline audit variant decode")
else:
    print("Fix 5e: SKIP (not found)")

RUNNER.write_text(text)
print(f"\nRunner patched: {RUNNER}")
print("Running syntax check...")
import subprocess, sys
result = subprocess.run(["/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python", "-m", "py_compile", str(RUNNER)], capture_output=True, text=True)
if result.returncode == 0:
    print("Runner syntax: OK")
else:
    print(f"Runner syntax: FAILED\n{result.stderr}")

# Now fix attack adapter
adapter_text = ADAPTER.read_text()

# Fix: update prepare_openvla_image_for_attack to accept and use backend
old_adapt = '''def prepare_openvla_image_for_attack(image_np, *, libero_official_preprocess: bool = False, center_crop: bool = False, resize_size: int = 224) -> Image.Image:'''
new_adapt = '''def prepare_openvla_image_for_attack(image_np, *, libero_official_preprocess: bool = False, center_crop: bool = False, resize_size: int = 224, libero_preprocess_backend: str = "official_pil_lanczos", **kwargs) -> Image.Image:'''

if old_adapt in adapter_text:
    adapter_text = adapter_text.replace(old_adapt, new_adapt)
    print("\nAdapter Fix 1: Added backend parameter to prepare_openvla_image_for_attack")

    # Now replace the body to use the v4 runner's prepare_openvla_image
    # Find and update the function body
    old_body = '''    if libero_official_preprocess:
        return _official_tf_libero_image(image_np, center_crop=center_crop, resize_size=resize_size)'''
    new_body = '''    from scripts.v4_run_eval_openvla import prepare_openvla_image as _v4_prepare
    return _v4_prepare(image_np, libero_official_preprocess=libero_official_preprocess,
                       center_crop=center_crop, resize_size=resize_size,
                       libero_preprocess_backend=libero_preprocess_backend)'''
    if old_body in adapter_text:
        adapter_text = adapter_text.replace(old_body, new_body)
        print("Adapter Fix 2: Replaced TF-only body with v4 prepare_openvla_image delegation")
    else:
        print("Adapter Fix 2: SKIP (body not found)")
        # Check if there's a different pattern
        import re
        matches = list(re.finditer(r'libero_official_preprocess', adapter_text))
        print(f"  Found {len(matches)} libero_official_preprocess references in adapter")
else:
    print("\nAdapter Fix 1: SKIP (function signature not found)")

ADAPTER.write_text(adapter_text)
result = subprocess.run(["/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python", "-m", "py_compile", str(ADAPTER)], capture_output=True, text=True)
if result.returncode == 0:
    print("Adapter syntax: OK")
else:
    print(f"Adapter syntax: FAILED\n{result.stderr}")

print("\nAll fixes applied.")
