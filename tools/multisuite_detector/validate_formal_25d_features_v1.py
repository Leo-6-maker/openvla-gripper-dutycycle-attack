#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.multisuite_detector.extract_formal_25d_features_v1 import main

if __name__ == "__main__":
    raise SystemExit(main(["validate", *sys.argv[1:]]))
