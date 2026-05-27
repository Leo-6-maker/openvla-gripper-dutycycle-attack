#!/usr/bin/env python3
"""Extract shared preprocessing functions to src/gripper_attack/openvla_preprocess.py"""
from pathlib import Path

REPO = Path("/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524")
RUNNER = REPO / "scripts/v4_run_eval_openvla.py"
ADAPTER = REPO / "src/gripper_attack/attack_adapter.py"
MODULE = REPO / "src/gripper_attack/openvla_preprocess.py"

text = RUNNER.read_text()
lines = text.split("\n")

# Find function boundaries
funcs = {}
for i, line in enumerate(lines):
    for fname in ["_pil_center_crop_resize", "_official_pil_libero_image",
                   "_official_tf_libero_image", "prepare_openvla_image"]:
        if f"def {fname}" in line:
            funcs[fname] = i

print(f"Functions found at: {funcs}")

# Find where prepare_openvla_image ends: first blank line after its closing brace
# The function ends with "return image" then a blank line
start = funcs["prepare_openvla_image"]
end_line = None
for i in range(start + 1, len(lines)):
    if lines[i].strip() == "" and i > start + 15:
        # Check if next non-blank line starts a new top-level def
        j = i + 1
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        if j < len(lines) and lines[j].strip().startswith("def ") and not lines[j].strip().startswith("def _official"):
            end_line = i
            break

if end_line is None:
    end_line = start + 35  # fallback: approximate 35 lines for prepare_openvla_image

print(f"prepare_openvla_image ends at line {end_line}")

# Build module
module = [
    '#!/usr/bin/env python3',
    '"""Shared OpenVLA LIBERO image preprocessing.',
    '',
    'Imported by v4_run_eval_openvla.py and attack_adapter.py.',
    'Three backends: official_pil_lanczos (default), tf_jpeg_legacy, none/pil_fallback.',
    '"""',
    '',
    'from __future__ import annotations',
    '',
    'import numpy as np',
    'from PIL import Image',
    '',
]
# Include all four preprocess functions
for i in range(funcs["_pil_center_crop_resize"], end_line):
    module.append(lines[i])

MODULE.write_text("\n".join(module) + "\n")
print(f"Written {len(module)} lines to {MODULE}")

# Verify syntax
import subprocess
r = subprocess.run(
    ["/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python", "-m", "py_compile", str(MODULE)],
    capture_output=True, text=True,
)
print("Module syntax:", "OK" if r.returncode == 0 else f"FAIL: {r.stderr}")

# Update attack_adapter.py to import from openvla_preprocess
adapter_text = ADAPTER.read_text()
old_import = "from scripts.v4_run_eval_openvla import prepare_openvla_image"
new_import = "from gripper_attack.openvla_preprocess import prepare_openvla_image"
if old_import in adapter_text:
    adapter_text = adapter_text.replace(old_import, new_import)
    ADAPTER.write_text(adapter_text)
    print("Adapter import updated: scripts -> gripper_attack")
elif new_import in adapter_text:
    print("Adapter already imports from gripper_attack.openvla_preprocess (OK)")
else:
    print("Adapter import pattern not found")
    for i, line in enumerate(adapter_text.split("\n")):
        if "prepare_openvla_image" in line:
            print(f"  Line {i}: {line}")

r = subprocess.run(
    ["/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python", "-m", "py_compile", str(ADAPTER)],
    capture_output=True, text=True,
)
print("Adapter syntax:", "OK" if r.returncode == 0 else f"FAIL: {r.stderr}")

# Update v4 runner to import from openvla_preprocess
# The runner defines the functions inline, so we need to replace with import
# Actually, the functions are still needed in the runner for direct use.
# The adapter import is the only one that changes.

print("\nDone. Module created and adapter updated.")
