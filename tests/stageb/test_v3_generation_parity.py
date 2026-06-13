from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gripper_attack.v3_generation_parity import (
    ACTION_DIM,
    REPLAY_SCHEMA_VERSION,
    arm_match_rate,
    classify_disc_and_raw,
    classify_path_diagnosis,
    classify_token_simple,
    extract_exact_new_tokens,
    extract_new_tokens_from_generation,
    generation_score_audit_from_row,
    make_safe_replay_stem,
    path_result_schema,
    require_token_list,
    sanitize_component,
    sha256_file,
    summarize_score_row,
    validate_generation_score_invariant,
    validate_prefix_invariant,
    validate_replay_bundle,
)
from stageb.diagnose_v3_generation_parity import (
    run_forward_path,
    run_generate_path,
)

VOCAB = 20
NBINS = 3
CLOSE_TOKEN = 19
BOUNDARY_TOKEN = 18
OPEN_TOKEN = 17
ARM_TOKEN = 14


def stats(mask_last=True):
    return {
        "q01": np.zeros(7, dtype=np.float32),
        "q99": np.ones(7, dtype=np.float32),
        "mask": np.array([True] * 6 + [mask_last], dtype=bool),
    }


def bin_centers():
    return np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)


def good_score_audit(emitted=BOUNDARY_TOKEN):
    row = torch.full((VOCAB,), -10.0)
    row[emitted] = 5.0
    row[OPEN_TOKEN] = 4.0
    row[CLOSE_TOKEN] = 3.0
    return generation_score_audit_from_row(
        row,
        emitted_token=emitted,
        vocab_eff=VOCAB,
        n_bins=NBINS,
        bin_centers=bin_centers(),
        action_stats=stats(),
        surrogate_top_token=OPEN_TOKEN,
    )


def good_bundle(tmp_path, *, tensor=True):
    tensor_name = "adv.pt"
    tensor_path = tmp_path / tensor_name
    value = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    if tensor:
        torch.save(value, tensor_path)
        tensor_sha = sha256_file(tensor_path)
    else:
        tensor_sha = "a" * 64
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "task": "butter",
        "state_id": 2,
        "attack_seed": 811,
        "seed": 811,
        "job_id": "job/with spaces",
        "step": 12,
        "condition": "online_vis_pgd",
        "objective": "autoregressive_prefix_gripper_open_execspec_v3",
        "objective_tag": "v3_ar_prefix",
        "model_path": "/model",
        "model_dtype": "torch.bfloat16",
        "runner_sha256": "a" * 64,
        "adapter_sha256": "b" * 64,
        "semantics_sha256": "c" * 64,
        "exec_spec_sha256": "d" * 64,
        "prompt_input_ids": [[3, 29871]],
        "prompt_input_ids_shape": [1, 2],
        "prompt_input_ids_dtype": "torch.int64",
        "adv_tensor_filename": tensor_name,
        "adv_tensor_sha256": tensor_sha,
        "adv_pixel_values_shape": [1, 1, 1, 1],
        "adv_tensor_dtype": "torch.float32",
        "surrogate_generated_arm_prefix_token_ids": [ARM_TOKEN] * 6,
        "generated_arm_prefix": [ARM_TOKEN] * 6,
        "official_generated_action_token_ids": [ARM_TOKEN] * 6 + [BOUNDARY_TOKEN],
        "full_ar_tokens": [ARM_TOKEN] * 6 + [BOUNDARY_TOKEN],
        "surrogate_global_top_token": OPEN_TOKEN,
        "official_generation_score_audit": good_score_audit(BOUNDARY_TOKEN),
        "surrogate_token_execution": classify_disc_and_raw(
            OPEN_TOKEN, VOCAB, NBINS, bin_centers(), stats()),
        "official_token_execution": classify_disc_and_raw(
            BOUNDARY_TOKEN, VOCAB, NBINS, bin_centers(), stats()),
        "ar_token_execution": classify_disc_and_raw(
            BOUNDARY_TOKEN, VOCAB, NBINS, bin_centers(), stats()),
        "generation_score_argmax": BOUNDARY_TOKEN,
        "generation_config": {
            "do_sample": False,
            "max_new_tokens": 7,
            "default_use_cache": True,
            "effective_use_cache": True,
            "eos_token_id": None,
            "pad_token_id": None,
        },
        "clean_generated_action_token_ids": [ARM_TOKEN] * 6 + [CLOSE_TOKEN],
        "clean_generated_arm_prefix_token_ids": [ARM_TOKEN] * 6,
        "retokenized_clean_action_arm_prefix": [CLOSE_TOKEN] * 6,
        "adv_generated_arm_prefix": [ARM_TOKEN] * 6,
        "adv_vs_clean_generated_arm_match_rate": 1.0,
        "surrogate_top_matches_generation": False,
        "transfer_classification": "SURROGATE_TO_GENERATION_TOP1_MISMATCH",
        "v3_transfer_class": "SURROGATE_TO_GENERATION_TOP1_MISMATCH",
    }


def test_exact_new_token_extraction_pass_fail():
    seq = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    assert extract_exact_new_tokens(seq, prompt_len=2, expected_new_tokens=7) == [3, 4, 5, 6, 7, 8, 9]
    with pytest.raises(ValueError, match="expected 7 new tokens"):
        extract_exact_new_tokens(seq, prompt_len=3, expected_new_tokens=7)
    gen = SimpleNamespace(sequences=seq)
    assert extract_new_tokens_from_generation(gen, prompt_len=2, expected_new_tokens=7)[-1] == 9


def test_prefix_invariant_and_match_rate():
    assert validate_prefix_invariant([1] * 6, [1] * 6 + [2])["ok"] is True
    mismatch = validate_prefix_invariant([1] * 6, [1, 1, 1, 1, 1, 9, 2])
    assert mismatch["ok"] is False
    assert mismatch["failure_type"] == "SURROGATE_PREFIX_FULL_AR_PREFIX_MISMATCH"
    assert arm_match_rate([1, 2, 3, 4, 5, 6], [1, 2, 0, 4, 0, 6]) == pytest.approx(4 / 6)


def test_score_invariant_pass_missing_mismatch():
    ok, failure = validate_generation_score_invariant(good_score_audit(), BOUNDARY_TOKEN)
    assert ok and failure == ""
    ok, failure = validate_generation_score_invariant({}, BOUNDARY_TOKEN)
    assert not ok and failure == "GENERATE_SCORE_AUDIT_MISSING"
    audit = good_score_audit()
    audit["processed_score_argmax_token"] = OPEN_TOKEN
    ok, failure = validate_generation_score_invariant(audit, BOUNDARY_TOKEN)
    assert not ok and failure == "GENERATE_SCORE_ARGMAX_MISMATCH"


def test_official_decode_native_boundary_clip_edges_and_mask_false():
    assert classify_disc_and_raw(OPEN_TOKEN, VOCAB, NBINS, bin_centers(), stats())["execution_class"] == "NATIVE_OPEN"
    assert classify_disc_and_raw(CLOSE_TOKEN, VOCAB, NBINS, bin_centers(), stats())["execution_class"] == "NATIVE_CLOSE"
    assert classify_disc_and_raw(BOUNDARY_TOKEN, VOCAB, NBINS, bin_centers(), stats())["execution_class"] == "NATIVE_BOUNDARY"
    assert classify_disc_and_raw(25, VOCAB, NBINS, bin_centers(), stats())["execution_class"] == "CLIP_MEDIATED_CLOSE"
    assert classify_disc_and_raw(-1, VOCAB, NBINS, bin_centers(), stats())["execution_class"] == "CLIP_MEDIATED_OPEN"
    assert classify_disc_and_raw(OPEN_TOKEN, VOCAB, NBINS, bin_centers(), stats(mask_last=False))["decoded_raw_gripper"] == 1.0
    assert classify_token_simple(None, VOCAB, NBINS, 0.0, False) == "UNKNOWN"


def test_score_row_summary_retains_required_competition_fields():
    row = torch.zeros((VOCAB,), dtype=torch.float32)
    row[OPEN_TOKEN] = 3.0
    row[BOUNDARY_TOKEN] = 5.0
    row[CLOSE_TOKEN] = 4.0
    summary = summarize_score_row(
        row,
        vocab_eff=VOCAB,
        n_bins=NBINS,
        bin_centers=bin_centers(),
        action_stats=stats(),
        surrogate_top_token=OPEN_TOKEN,
    )
    assert summary["top1_token"] == BOUNDARY_TOKEN
    assert summary["top2_token"] == CLOSE_TOKEN
    assert summary["top1_minus_top2_gap"] == pytest.approx(1.0)
    assert summary["score_token_31744"] is None
    assert summary["score_token_31872"] is None
    assert summary["best_native_open_token"] == OPEN_TOKEN
    assert summary["best_native_close_token"] == CLOSE_TOKEN
    assert summary["best_native_boundary_token"] == BOUNDARY_TOKEN
    assert summary["surrogate_top_score"] == pytest.approx(3.0)


def test_safe_replay_stem_sanitizes_job_id_and_adds_digest():
    stem = make_safe_replay_stem(
        task="butter", state_id=2, objective_tag="v3", condition="online",
        job_id="bad/id with spaces", seed=811, step=4)
    assert "/" not in stem and " " not in stem
    assert "bad_id_with_spaces" in stem
    assert len(stem.rsplit("_", 1)[-1]) == 12
    assert sanitize_component("///") == "none"


def test_replay_validation_rejects_bad_hash_token_lengths_missing_tensor_and_sha(tmp_path):
    bundle = good_bundle(tmp_path)
    assert validate_replay_bundle(bundle, bundle_dir=tmp_path, verify_tensor=True) == []

    bad_hash = dict(bundle)
    bad_hash["runner_sha256"] = "zz"
    assert any("runner_sha256" in x for x in validate_replay_bundle(bad_hash, bundle_dir=tmp_path, verify_tensor=True))

    bad_tokens = dict(bundle)
    bad_tokens["official_generated_action_token_ids"] = [1, 2]
    assert any("official_generated_action_token_ids" in x for x in validate_replay_bundle(bad_tokens, bundle_dir=tmp_path, verify_tensor=True))

    missing = good_bundle(tmp_path / "missing", tensor=False)
    assert any("FILE_MISSING" in x for x in validate_replay_bundle(missing, bundle_dir=tmp_path / "missing", verify_tensor=True))

    bad_sha = dict(bundle)
    bad_sha["adv_tensor_sha256"] = "e" * 64
    assert any("FILE_HASH_MISMATCH" in x for x in validate_replay_bundle(bad_sha, bundle_dir=tmp_path, verify_tensor=True))


def test_replay_validation_rejects_shape_dtype_prompt_execution_and_schema(tmp_path):
    bundle = good_bundle(tmp_path)
    bad_shape = dict(bundle)
    bad_shape["adv_pixel_values_shape"] = [9]
    assert any("TENSOR_SHAPE_MISMATCH" in x for x in validate_replay_bundle(bad_shape, bundle_dir=tmp_path, verify_tensor=True))

    bad_dtype = dict(bundle)
    bad_dtype["adv_tensor_dtype"] = "torch.bfloat16"
    assert any("TENSOR_DTYPE_MISMATCH" in x for x in validate_replay_bundle(bad_dtype, bundle_dir=tmp_path, verify_tensor=True))

    bad_prompt = dict(bundle)
    bad_prompt["prompt_input_ids_shape"] = [1, 99]
    assert "prompt_input_ids_shape:mismatch" in validate_replay_bundle(bad_prompt, bundle_dir=tmp_path, verify_tensor=True)

    bad_exec = dict(bundle)
    bad_exec["official_token_execution"] = {"token_id": 1}
    assert any("official_token_execution.execution_class" in x for x in validate_replay_bundle(bad_exec, bundle_dir=tmp_path, verify_tensor=True))

    bad_schema = dict(bundle)
    bad_schema["schema_version"] = "old"
    assert any("schema_version:INVALID" in x for x in validate_replay_bundle(bad_schema, bundle_dir=tmp_path, verify_tensor=True))


class TinyDiagnosticModel(torch.nn.Module):
    def __init__(self, emitted=BOUNDARY_TOKEN, forward_top=OPEN_TOKEN, d_tokens=None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(vocab_size=VOCAB),
            pad_to_multiple_of=0,
        )
        self.bin_centers = bin_centers()
        self.emitted = emitted
        self.forward_top = forward_top
        self.d_tokens = d_tokens

    def get_action_stats(self, key):
        return stats()

    def generate(self, input_ids, pixel_values, max_new_tokens, do_sample=False,
                 return_dict_in_generate=True, output_scores=True, use_cache=None):
        toks = self.d_tokens if use_cache is False and self.d_tokens is not None else [ARM_TOKEN] * 6 + [self.emitted]
        toks = torch.tensor([toks[:max_new_tokens]], dtype=torch.long, device=input_ids.device)
        scores = []
        for idx in range(max_new_tokens):
            row = torch.full((1, VOCAB), -10.0, device=input_ids.device)
            emitted = int(toks[0, idx])
            row[0, emitted] = 5.0
            row[0, OPEN_TOKEN] = max(float(row[0, OPEN_TOKEN]), 4.0)
            row[0, CLOSE_TOKEN] = max(float(row[0, CLOSE_TOKEN]), 3.0)
            scores.append(row)
        return SimpleNamespace(sequences=torch.cat([input_ids, toks], dim=1), scores=tuple(scores))

    def forward(self, input_ids, pixel_values, use_cache=False, return_dict=True):
        logits = torch.zeros((1, input_ids.shape[1], VOCAB), dtype=torch.float32, device=input_ids.device)
        top = BOUNDARY_TOKEN if use_cache else self.forward_top
        logits[0, -1, top] = 5.0
        logits[0, -1, OPEN_TOKEN] = max(float(logits[0, -1, OPEN_TOKEN]), 4.0)
        logits[0, -1, CLOSE_TOKEN] = max(float(logits[0, -1, CLOSE_TOKEN]), 3.0)
        return SimpleNamespace(logits=logits)


class BadScoreDiagnosticModel(TinyDiagnosticModel):
    def generate(self, input_ids, pixel_values, max_new_tokens, do_sample=False,
                 return_dict_in_generate=True, output_scores=True, use_cache=None):
        out = super().generate(
            input_ids, pixel_values, max_new_tokens,
            do_sample=do_sample,
            return_dict_in_generate=return_dict_in_generate,
            output_scores=output_scores,
            use_cache=use_cache,
        )
        rows = []
        for idx, row in enumerate(out.scores):
            row = row.clone()
            if idx == max_new_tokens - 1:
                row[0, OPEN_TOKEN] = 9.0
                row[0, BOUNDARY_TOKEN] = 5.0
            rows.append(row)
        out.scores = tuple(rows)
        return out


def test_path_a_d_exact_token_score_checks_and_four_path_schema(tmp_path):
    model = TinyDiagnosticModel()
    bundle = good_bundle(tmp_path)
    input_ids = torch.tensor(bundle["prompt_input_ids"], dtype=torch.long)
    pixel_values = torch.zeros((1, 1, 1, 1))
    prefix = torch.tensor([ARM_TOKEN] * 6, dtype=torch.long)

    a = run_generate_path(model, input_ids, pixel_values, bundle, path_name="A", use_cache_arg=None)
    d = run_generate_path(model, input_ids, pixel_values, bundle, path_name="D", use_cache_arg=False)
    b = run_forward_path(model, input_ids, pixel_values, prefix, bundle, path_name="B", use_cache=False)
    c = run_forward_path(model, input_ids, pixel_values, prefix, bundle, path_name="C", use_cache=True)
    assert a["generated_tokens"] == [ARM_TOKEN] * 6 + [BOUNDARY_TOKEN]
    assert d["processed_score_top_token"] == BOUNDARY_TOKEN
    assert b["raw_logit_top_token"] == OPEN_TOKEN
    assert c["raw_logit_top_token"] == BOUNDARY_TOKEN
    for item in (a, b, c, d):
        assert "path" in item and "cache_behavior" in item
        assert "processed_score_top2_token" in item
        assert "raw_logit_top2_token" in item


def test_path_a_reproduction_mismatch_raises(tmp_path):
    model = TinyDiagnosticModel(emitted=OPEN_TOKEN)
    bundle = good_bundle(tmp_path)
    input_ids = torch.tensor(bundle["prompt_input_ids"], dtype=torch.long)
    with pytest.raises(RuntimeError, match="Path A official reproduction mismatch"):
        run_generate_path(model, input_ids, torch.zeros((1, 1, 1, 1)), bundle, path_name="A", use_cache_arg=None)


def test_path_generation_score_invariant_mismatch_raises(tmp_path):
    model = BadScoreDiagnosticModel()
    bundle = good_bundle(tmp_path)
    input_ids = torch.tensor(bundle["prompt_input_ids"], dtype=torch.long)
    with pytest.raises(RuntimeError, match="GENERATE_SCORE_ARGMAX_MISMATCH"):
        run_generate_path(model, input_ids, torch.zeros((1, 1, 1, 1)), bundle, path_name="A", use_cache_arg=None)


def test_diagnosis_categories_report_supporting_differences(tmp_path):
    bundle = good_bundle(tmp_path)
    a = path_result_schema(
        path="A", cache_behavior="default", prefix_tokens=[ARM_TOKEN] * 6,
        generated_tokens=[ARM_TOKEN] * 6 + [BOUNDARY_TOKEN],
        emitted_gripper_token=BOUNDARY_TOKEN,
        processed_score_summary={"top1_token": BOUNDARY_TOKEN, "top1_score": 5.0, "top2_token": OPEN_TOKEN, "top2_score": 4.0, "top1_minus_top2_gap": 1.0},
    )
    b = path_result_schema(path="B", cache_behavior="use_cache=False", raw_logit_summary={"top1_token": OPEN_TOKEN, "top1_score": 5.0})
    c = path_result_schema(path="C", cache_behavior="use_cache=True", raw_logit_summary={"top1_token": BOUNDARY_TOKEN, "top1_score": 5.0})
    d = path_result_schema(path="D", cache_behavior="use_cache=False", generated_tokens=[ARM_TOKEN] * 6 + [BOUNDARY_TOKEN], emitted_gripper_token=BOUNDARY_TOKEN)
    assert classify_path_diagnosis({"A": a, "B": b, "C": c, "D": d}, bundle) == "CACHE_PATH_MISMATCH_CANDIDATE"

    c2 = path_result_schema(path="C", cache_behavior="use_cache=True", raw_logit_summary={"top1_token": OPEN_TOKEN, "top1_score": 5.0})
    d2 = path_result_schema(path="D", cache_behavior="use_cache=False", generated_tokens=[ARM_TOKEN] * 6 + [OPEN_TOKEN], emitted_gripper_token=OPEN_TOKEN)
    assert classify_path_diagnosis({"A": a, "B": b, "C": c2, "D": d2}, bundle) == "GENERATION_SCORE_PROCESSING_MISMATCH_CANDIDATE"

    a_tie = dict(a)
    a_tie["processed_score_top1_minus_top2_gap"] = 1e-5
    assert classify_path_diagnosis({"A": a_tie, "B": b, "C": c2, "D": d}, bundle) == "NEAR_TIE_NUMERICAL_SENSITIVITY_CANDIDATE"


def test_require_token_list_real_exception():
    with pytest.raises(ValueError):
        require_token_list([1, 2], expected_len=7, label="tokens")
