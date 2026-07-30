import hashlib

import pytest

from scripts.detector_v5.build_r3_g1_binding_receipt import SPLIT_FILES, _regular_repo_file


def test_g1_binding_receipt_has_exact_six_split_keys():
    assert tuple(SPLIT_FILES) == (
        "episode_train", "episode_validation", "episode_test",
        "task_train", "task_validation", "task_test",
    )


def test_g1_binding_receipt_rejects_absolute_or_parent_repo_paths():
    with pytest.raises(ValueError):
        _regular_repo_file("/etc/passwd")
    with pytest.raises(ValueError):
        _regular_repo_file("../outside")


def test_identity_digest_is_deterministic():
    identities = ["libero_10/task_00/state_00", "libero_10/task_00/state_01"]
    digest = hashlib.sha256(("\n".join(identities) + "\n").encode("utf-8")).hexdigest()
    assert len(digest) == 64
