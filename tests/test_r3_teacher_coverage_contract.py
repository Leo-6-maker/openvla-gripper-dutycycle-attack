import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "detector_v5"))

from audit_r3_teacher_coverage import _require_transition_audit_binding  # noqa: E402


def test_fit_to_teacher_binds_exact_t0_a_manifest_and_seal():
    audit = {"manifest_sha256": "a" * 64, "seal_sha256sums_sha256": "b" * 64}
    transition = {
        "input_audit_manifest_sha256": audit["manifest_sha256"],
        "input_audit_seal_sha256sums_sha256": audit["seal_sha256sums_sha256"],
    }
    _require_transition_audit_binding(transition, audit)


@pytest.mark.parametrize("field", ["input_audit_manifest_sha256", "input_audit_seal_sha256sums_sha256"])
def test_fit_to_teacher_rejects_mismatched_t0_a_binding(field):
    audit = {"manifest_sha256": "a" * 64, "seal_sha256sums_sha256": "b" * 64}
    transition = {
        "input_audit_manifest_sha256": audit["manifest_sha256"],
        "input_audit_seal_sha256sums_sha256": audit["seal_sha256sums_sha256"],
    }
    transition[field] = "c" * 64
    with pytest.raises(ValueError, match="audit .* binding mismatch"):
        _require_transition_audit_binding(transition, audit)
