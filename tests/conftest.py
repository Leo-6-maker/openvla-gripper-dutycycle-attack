from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for rel in ("src", "scripts"):
    p = str(ROOT / rel)
    if p not in sys.path:
        sys.path.insert(0, p)
