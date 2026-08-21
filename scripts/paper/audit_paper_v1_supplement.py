#!/usr/bin/env python3
"""CPU-only completeness check for the Paper V1 supplement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUPPLEMENT = ROOT / "paper/PAPER_V1_SUPPLEMENT_REPRODUCIBILITY.md"
BINDINGS = ROOT / "paper/PAPER_V1_SUPPLEMENT_BINDINGS_V1.json"
AUTHORITY = ROOT / "paper/PAPER_V1_EVIDENCE_AUTHORITY_MAP_V1.json"
E4_ROOT = ROOT / "reports/STAGE_X1R2_E4_FACTORIZATION_FAILURE_DECOMPOSITION_20260821/E4_ROOT_SEAL_V1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    text = SUPPLEMENT.read_text(encoding="utf-8")
    bindings = json.loads(BINDINGS.read_text(encoding="utf-8"))
    authority = json.loads(AUTHORITY.read_text(encoding="utf-8"))
    assert bindings["status"] == "PAPER_V1_SUPPLEMENT_REPRODUCIBILITY_PASS"
    assert bindings["tokenizer_authority"]["non_bijective_boundary"] == "31744<->31745"
    assert bindings["protected_boundary"]["eval160"] == "UNREAD"
    assert bindings["protected_boundary"]["protected_evaluation"] == "UNREAD"
    for required in ("S1. Evidence hierarchy", "S2. Causal architecture", "S3. Tokenizer", "S5. Timing-selector", "S8. Victim provenance", "S9. No-rerun", "S11. Final boundary"):
        assert required in text, required
    assert "72 E3 candidate slots are a within-parent ordered audit" in text
    assert "E3/E4 are not upgraded to physical efficacy or impossibility" in text
    e4 = next(source for source in authority["sources"] if source["id"] == "E3_E4")
    assert e4["sealed_artifact_sha256"][-1] == sha256(E4_ROOT)
    print("PAPER_V1_SUPPLEMENT_REPRODUCIBILITY_PASS sections=7 tokenizer=bound protected=UNREAD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
