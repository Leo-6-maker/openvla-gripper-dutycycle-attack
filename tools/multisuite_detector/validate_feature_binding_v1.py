#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.multisuite_detector.feature_binding_manifest_v1 import FeatureBindingError, validate_binding_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding-manifest", required=True)
    parser.add_argument("--expected-label-mode", default="formal-ledger-build", choices=["synthetic-dry-run", "formal-ledger-build"])
    args = parser.parse_args(argv)
    try:
        report = validate_binding_manifest(args.binding_manifest, expected_label_mode=args.expected_label_mode)
    except (OSError, json.JSONDecodeError, FeatureBindingError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
