#!/usr/bin/env python3
"""Write detector dataset metadata only from a feature binding manifest."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.multisuite_detector.feature_binding_manifest_v1 import FeatureBindingError, write_metadata_outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    try:
        report = write_metadata_outputs(args.binding_manifest, args.output_root)
    except (OSError, json.JSONDecodeError, FeatureBindingError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
