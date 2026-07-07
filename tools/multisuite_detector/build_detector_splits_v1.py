#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.multisuite_detector.detector_dataset_closure_v1 import main

if __name__ == "__main__":
    raise SystemExit(main(["split", *sys.argv[1:]]))
