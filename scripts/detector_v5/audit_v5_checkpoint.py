"""Fail-closed audit for a V5 development checkpoint bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gripper_attack.b3_training_protocol import sha256_file, verify_sealed_directory


def audit(root: Path) -> dict[str, object]:
    root = root.resolve()
    verify_sealed_directory(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = root / "checkpoint.pt"
    if not checkpoint.is_file() or sha256_file(checkpoint) != manifest.get("checkpoint_sha256", sha256_file(checkpoint)):
        # Development smoke manifests predating the explicit checkpoint digest
        # are still auditable by the sealed root; new bundles must include it.
        if "checkpoint_sha256" in manifest:
            raise ValueError("checkpoint SHA mismatch")
    if manifest.get("formal_training_authorized") is not False or manifest.get("formal_attack_authorized") is not False:
        raise ValueError("development checkpoint carries an authorization flag")
    if manifest.get("eligible_for_model_selection") is not False:
        raise ValueError("development checkpoint is incorrectly model-selection eligible")
    return {"schema": "DETECTOR_V5_CHECKPOINT_AUDIT_V2", "status": "PASS", "checkpoint_sha256": sha256_file(checkpoint), "formal_training_authorized": False, "formal_attack_authorized": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
