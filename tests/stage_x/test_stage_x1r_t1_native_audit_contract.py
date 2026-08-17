from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def test_native_audit_is_not_an_attack_or_environment_runner():
    text = (REPO / "scripts" / "stage_x" / "audit_stage_x1r_t1_native_token_authority.py").read_text(encoding="utf-8")
    assert ".step(" not in text
    assert ".attack(" not in text
    assert "pgd_calls" in text
    assert "vphys_reads" in text

