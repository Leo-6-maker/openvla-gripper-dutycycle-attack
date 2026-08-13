import sys
from pathlib import Path

from scripts.detector_v5.capture_stage_v_runtime_provenance import build


def test_runtime_provenance_receipt_binds_python_source_modules_and_artifact(tmp_path: Path):
    artifact = tmp_path / "checkpoint.bin"
    artifact.write_bytes(b"checkpoint")
    receipt = build(
        python_path=Path(sys.executable),
        source_worktree=Path(__file__).resolve().parents[2],
        modules=("json",),
        artifacts=(("checkpoint", artifact),),
        files=(artifact,),
    )
    assert receipt["status"] == "PASS_RUNTIME_PROVENANCE_CAPTURED"
    assert receipt["official_python"]["version"]
    assert receipt["source_worktree"]["commit"]
    assert receipt["imported_modules"][0]["origin_sha256"]
    assert receipt["artifacts"][0]["sha256"]
    assert receipt["protected_counters"]["protected_reads"] == 0
