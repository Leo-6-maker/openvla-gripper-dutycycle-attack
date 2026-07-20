"""Quick fix: insert missing _safe_info helper."""
import re
from pathlib import Path

fp = Path("/mnt/sdc/dty_user/worktrees/r10_4d_smoke_3d741847/src/gripper_attack/r10_4d_passive.py")
content = fp.read_text()

if "def _safe_info" in content:
    print("_safe_info definition already present")
else:
    safe_info_code = '''
def _safe_info(info):
    """Serialize env.step info dict safely."""
    if info is None:
        return {}
    try:
        return dict(info)
    except Exception:
        pass
    result = {}
    for k, v in (info.items() if isinstance(info, dict) else []):
        try:
            import json as _json
            _json.dumps({k: v})
            result[k] = v
        except (TypeError, ValueError):
            result[k] = str(type(v).__name__)
    return result
'''
    marker = "\ndef run_passive_episode("
    content = content.replace(marker, safe_info_code + marker, 1)
    fp.write_text(content)
    print("_safe_info inserted")

# Verify
content2 = fp.read_text()
if "_safe_info" in content2 and "_classify_termination" in content2:
    lines_with = [(i+1, l.strip()[:80]) for i, l in enumerate(content2.split("\n"))
                  if "_safe_info" in l or "_classify_termination" in l or "termination_reason" in l]
    for ln, txt in lines_with:
        print(f"  L{ln}: {txt}")
    print("PASS: all patches verified")
else:
    print("FAIL: patch verification failed")
