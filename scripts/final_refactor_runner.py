#!/usr/bin/env python3
"""Final refactor: v4 runner imports preprocess from shared module instead of duplicating."""
from pathlib import Path
import subprocess

REPO = Path("/data/liuyu/repos/openvla-gripper-dutycycle-attack-clean-main-20260524")
RUNNER = REPO / "scripts/v4_run_eval_openvla.py"

text = RUNNER.read_text()
lines = text.split("\n")

# Find the import block (around line 7-10) and add the import
# Look for: from PIL import Image
import_section_end = None
for i, line in enumerate(lines):
    if "from PIL import Image" in line:
        import_section_end = i
        break

# Find the preprocess function block to remove (lines 172-276, 0-indexed 171-275)
# We know:
# _pil_center_crop_resize starts at line 172 (0-indexed: 171)
# prepare_openvla_image ends at line 275 (0-indexed: 274)

# Verify the start and end
assert "_pil_center_crop_resize" in lines[171], f"Expected _pil_center_crop_resize at line 171, got: {lines[171][:60]}"
assert "return image" in lines[274], f"Expected return image at line 274, got: {lines[274][:60]}"
# Line 275 should be blank
assert lines[275].strip() == "", f"Expected blank line at 275, got: {lines[275][:60]}"

# Remove lines 171-275 (inclusive)
new_lines = lines[:171] + lines[276:]

# Add import after PIL import
new_lines.insert(import_section_end + 1, "from gripper_attack.openvla_preprocess import prepare_openvla_image  # shared preprocess module")

new_text = "\n".join(new_lines)
RUNNER.write_text(new_text)

# Verify
result = subprocess.run(
    ["/home/liuyu/.conda/envs/openvla_official_libero_20260525/bin/python", "-m", "py_compile", str(RUNNER)],
    capture_output=True, text=True,
)
if result.returncode == 0:
    print("Runner syntax: OK")
else:
    print(f"Runner syntax: FAIL\n{result.stderr}")

# Verify no duplicate function remains
if "_pil_center_crop_resize" in new_text:
    print("WARNING: _pil_center_crop_resize still present!")
if "_official_pil_libero_image" in new_text:
    print("WARNING: _official_pil_libero_image still present!")
if "_official_tf_libero_image" in new_text:
    print("WARNING: _official_tf_libero_image still present!")

# Verify the import exists
if "from gripper_attack.openvla_preprocess import prepare_openvla_image" in new_text:
    print("Import: OK")
else:
    print("WARNING: Import not found!")

# Verify prepare_openvla_image is still referenced correctly
count = new_text.count("prepare_openvla_image")
print(f"prepare_openvla_image references: {count}")
print(f"Lines after refactor: {len(new_text.split(chr(10)))}")
print(f"Lines removed: {len(lines) - len(new_text.split(chr(10)))}")
print("Done.")
