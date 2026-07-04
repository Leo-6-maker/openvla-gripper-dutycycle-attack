import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.multisuite_detector.test_detector_dataset_closure_v1 import make_label_artifact
from tests.multisuite_detector.test_extract_formal_25d_features_v1 import make_sources, rewrite
from tools.multisuite_detector.extract_formal_25d_features_v1 import FormalFeatureError, build_feature_artifact


@pytest.mark.parametrize("field", ["episode_key", "parent_key", "source_record_path"])
def test_initial_state_hash_must_not_be_identifier_derived(tmp_path, field):
    label_root, label_rows = make_label_artifact(tmp_path)
    approved_root, source_csv, rows = make_sources(tmp_path, label_rows)
    rows = [
        dict(row, initial_state_hash=hashlib.sha256(row[field].encode()).hexdigest())
        if row["episode_key"] == "ep_obj_a"
        else row
        for row in rows
    ]
    rewrite(source_csv, rows)

    with pytest.raises(FormalFeatureError, match=f"forbidden {field}"):
        build_feature_artifact(source_csv, label_root, tmp_path / "out", approved_root)
