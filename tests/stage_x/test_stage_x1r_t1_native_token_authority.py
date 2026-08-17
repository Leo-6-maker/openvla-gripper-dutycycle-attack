from __future__ import annotations

import numpy as np

from gripper_attack.stage_x_t1_native_token_authority import (
    LEGACY_HELPER_STATUS,
    NativeActionTokenAuthorityV2,
    SuiteActionTokenBinding,
)


def _authority(suite: str, *, gripper_low: float = 0.0, gripper_high: float = 1.0) -> NativeActionTokenAuthorityV2:
    bins = np.linspace(-1.0, 1.0, 256, dtype=np.float64)
    centers = (bins[:-1] + bins[1:]) / 2.0
    binding = SuiteActionTokenBinding(
        suite=suite,
        checkpoint_path=f"/models/{suite}",
        checkpoint_config_sha256=f"config-{suite}",
        tokenizer_source="official/action_tokenizer.py",
        tokenizer_source_sha256="action-tokenizer-sha",
        model_decoder_source_sha256="modeling-sha",
        tokenizer_files=(("tokenizer.json", f"tokenizer-{suite}"),),
        tokenizer_vocab_size=32000,
        n_action_bins=256,
        bins=tuple(bins.tolist()),
        bin_centers=tuple(centers.tolist()),
        q01=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_low),
        q99=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, gripper_high),
        mask=(False, False, False, False, False, False, True),
    )
    return NativeActionTokenAuthorityV2(binding)


def test_native_quantizer_uses_digitize_and_preserves_endpoints():
    authority = _authority("libero_goal")
    bins = np.asarray(authority.binding.bins)
    centers = np.asarray(authority.binding.bin_centers)
    for value in centers:
        got = int(authority.encode_normalized(np.full(7, value))[-1])
        expected = 32000 - int(np.digitize(value, bins))
        assert got == expected
    for edge in bins:
        for value in (np.nextafter(edge, -np.inf), edge, np.nextafter(edge, np.inf)):
            got = int(authority.encode_normalized(np.full(7, value))[-1])
            expected = 32000 - int(np.digitize(np.clip(value, -1.0, 1.0), bins))
            assert got == expected
    receipt = authority.endpoint_receipt()
    assert receipt["native_endpoint_non_bijective"] is True
    assert receipt["roundtrip_endpoint_equality_required"] is False


def test_suite_binding_is_not_global_target_token_authority():
    authorities = [_authority(suite) for suite in ("libero_10", "libero_goal", "libero_object", "libero_spatial")]
    assert all(a.open_token_id() == 31744 for a in authorities)
    assert len({a.binding.suite for a in authorities}) == 4
    assert all(a.receipt()["legacy_helper_status"] == LEGACY_HELPER_STATUS for a in authorities)


def test_gripper_semantics_cover_open_boundary_close():
    authority = _authority("libero_object")
    close = authority.encode_raw(np.array([0.0] * 7))[-1]
    boundary = authority.encode_raw(np.array([0.5] * 7))[-1]
    open_token = authority.open_token_id()
    assert authority.decode_gripper(int(close))["raw_class"] == "close"
    assert authority.decode_gripper(int(boundary))["raw_class"] == "boundary_or_neutral"
    assert authority.decode_gripper(open_token)["env_class"] == "open"


def test_checkpoint_binding_rejects_missing_identity():
    binding = _authority("libero_spatial").binding
    broken = binding.__class__(**{**binding.__dict__, "checkpoint_config_sha256": ""})
    try:
        broken.validate()
    except ValueError as exc:
        assert str(exc) == "TOKEN_AUTHORITY_SOURCE_BINDING_REQUIRED"
    else:
        raise AssertionError("missing checkpoint identity was accepted")
