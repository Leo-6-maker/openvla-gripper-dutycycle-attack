#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.multisuite_detector.feature_binding_manifest_v1 import FeatureBindingError, summary_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding-manifest", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        report = summary_report(args.binding_manifest)
    except (OSError, json.JSONDecodeError, FeatureBindingError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
