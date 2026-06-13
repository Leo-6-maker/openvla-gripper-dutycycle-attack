"""Shared V3 generation-parity helpers.

These helpers are intentionally pure and importable by the runner, standalone
diagnostic, and CPU tests. They are the single source of truth for replay
schema validation, official token execution classification, score-row audit,
and four-path diagnostic record shape.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .openvla_libero_exec_spec import (
    raw_gripper_is_close,
    raw_gripper_is_open,
    raw_gripper_to_env_gripper,
)

REPLAY_SCHEMA_VERSION = "v3_generation_parity_replay_v2"
VALID_REPLAY_SCHEMA_VERSIONS = {REPLAY_SCHEMA_VERSION}
DIAGNOSTIC_TOKENS = (31744, 31872)
ACTION_DIM = 7
ARM_PREFIX_LEN = 6


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def sanitize_component(value: Any) -> str:
    text = str(value if value is not None else "none").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "none"


def make_safe_replay_stem(
    *,
    task: Any,
    state_id: Any,
    objective_tag: Any,
    condition: Any,
    job_id: Any,
    seed: Any,
    step: Any,
) -> str:
    parts = [
        "v3_parity",
        sanitize_component(task),
        f"s{sanitize_component(state_id)}",
        sanitize_component(objective_tag),
        sanitize_component(condition),
        f"job{sanitize_component(job_id)}",
        f"seed{sanitize_component(seed)}",
        f"step{sanitize_component(step)}",
    ]
    raw = "|".join(str(x) for x in (task, state_id, objective_tag, condition, job_id, seed, step))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return "_".join(parts + [digest])


def _tolist_int(values: Any) -> list[int]:
    if values is None:
        raise ValueError("missing token sequence")
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    if isinstance(values, tuple):
        values = list(values)
    if not isinstance(values, list):
        raise ValueError(f"token sequence must be list-like, got {type(values).__name__}")
    if len(values) == 1 and isinstance(values[0], list):
        values = values[0]
    return [int(v) for v in values]


def require_token_list(values: Any, *, expected_len: int, label: str) -> list[int]:
    out = _tolist_int(values)
    if len(out) != int(expected_len):
        raise ValueError(f"{label}: expected {expected_len} tokens, got {len(out)}")
    return out


def extract_exact_new_tokens(sequences: Any, *, prompt_len: int, expected_new_tokens: int = ACTION_DIM) -> list[int]:
    if hasattr(sequences, "detach"):
        seq = sequences.detach().cpu()
        if seq.ndim != 2 or int(seq.shape[0]) != 1:
            raise ValueError(f"sequences must have shape [1, T], got {tuple(seq.shape)}")
        values = [int(x) for x in seq[0].tolist()]
    else:
        values = require_token_list(sequences, expected_len=int(prompt_len) + int(expected_new_tokens), label="sequences")
    prompt_len = int(prompt_len)
    if prompt_len < 0:
        raise ValueError("prompt_len must be non-negative")
    new = values[prompt_len:]
    if len(new) != int(expected_new_tokens):
        raise ValueError(f"expected {expected_new_tokens} new tokens, got {len(new)}")
    return new


def extract_new_tokens_from_generation(gen: Any, *, prompt_len: int, expected_new_tokens: int = ACTION_DIM) -> list[int]:
    if gen is None or not hasattr(gen, "sequences"):
        raise ValueError("generation output is missing sequences")
    return extract_exact_new_tokens(gen.sequences, prompt_len=prompt_len, expected_new_tokens=expected_new_tokens)


def infer_prompt_len_from_generation(gen: Any, *, expected_new_tokens: int = ACTION_DIM) -> int:
    if gen is None or not hasattr(gen, "sequences"):
        raise ValueError("generation output is missing sequences")
    seq = gen.sequences
    if hasattr(seq, "shape"):
        total = int(seq.shape[1])
    else:
        total = len(require_token_list(seq, expected_len=0, label="sequences"))  # unreachable for normal callers
    prompt_len = total - int(expected_new_tokens)
    if prompt_len < 0:
        raise ValueError(f"generation total length {total} < expected new tokens {expected_new_tokens}")
    return prompt_len


def validate_prefix_invariant(generated_prefix: Any, full_ar_tokens: Any) -> dict[str, Any]:
    prefix = require_token_list(generated_prefix, expected_len=ARM_PREFIX_LEN, label="generated_prefix")
    ar = require_token_list(full_ar_tokens, expected_len=ACTION_DIM, label="full_ar_tokens")
    ar_prefix = ar[:ARM_PREFIX_LEN]
    ok = prefix == ar_prefix
    return {
        "ok": ok,
        "generated_prefix": prefix,
        "full_ar_arm_prefix": ar_prefix,
        "failure_type": "" if ok else "SURROGATE_PREFIX_FULL_AR_PREFIX_MISMATCH",
    }


def arm_match_rate(left: Any, right: Any) -> float:
    a = require_token_list(left, expected_len=ARM_PREFIX_LEN, label="left_arm_prefix")
    b = require_token_list(right, expected_len=ARM_PREFIX_LEN, label="right_arm_prefix")
    return float(sum(int(x == y) for x, y in zip(a, b)) / ARM_PREFIX_LEN)


def classify_disc_and_raw(token_id: int, vocab_eff: int, n_bins: int, bin_centers, unnorm_key_stats):
    """Decode a token through the frozen official token->action pipeline."""
    import numpy as np

    token_id = int(token_id)
    disc_before = int(vocab_eff - token_id - 1)
    disc_after = max(0, min(int(n_bins) - 1, disc_before))
    clipped = disc_before != disc_after
    center = float(bin_centers[disc_after])
    stats = unnorm_key_stats
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi = np.asarray(stats["q99"], dtype=np.float32)
    lo = np.asarray(stats["q01"], dtype=np.float32)
    gripper_dim = len(hi) - 1
    if bool(mask[gripper_dim]):
        raw = float(0.5 * (center + 1.0) * (hi[gripper_dim] - lo[gripper_dim]) + lo[gripper_dim])
    else:
        raw = float(center)
    env = raw_gripper_to_env_gripper(raw)
    if clipped:
        if env < -0.5:
            exec_class = "CLIP_MEDIATED_OPEN"
        elif env > 0.5:
            exec_class = "CLIP_MEDIATED_CLOSE"
        else:
            exec_class = "CLIP_MEDIATED_NEUTRAL"
    elif abs(raw - 0.5) <= 1e-9:
        exec_class = "NATIVE_BOUNDARY"
    elif raw_gripper_is_open(raw):
        exec_class = "NATIVE_OPEN"
    elif raw_gripper_is_close(raw):
        exec_class = "NATIVE_CLOSE"
    else:
        exec_class = "NATIVE_UNCLASSIFIED"
    return {
        "token_id": token_id,
        "disc_before": disc_before,
        "disc_after": disc_after,
        "clipped": bool(clipped),
        "decoded_raw_gripper": round(raw, 8),
        "executed_env_gripper": round(env, 6),
        "execution_class": exec_class,
    }


def classify_token_simple(token_id, vocab_eff, n_bins, executed_raw, gripper_clipped):
    tid = int(token_id) if token_id not in (None, "") else None
    if tid is None:
        return "UNKNOWN"
    disc = int(vocab_eff - tid - 1)
    clipped = bool(gripper_clipped) or disc < 0 or disc >= int(n_bins)
    if clipped:
        if isinstance(executed_raw, (int, float)) and executed_raw > 0.5:
            return "CLIP_MEDIATED_OPEN"
        return "CLIP_MEDIATED_CLOSE"
    if isinstance(executed_raw, (int, float)):
        if executed_raw > 0.5:
            return "NATIVE_OPEN"
        if executed_raw < 0.5:
            return "NATIVE_CLOSE"
        if abs(executed_raw - 0.5) <= 1e-9:
            return "NATIVE_BOUNDARY"
    return "NATIVE_UNCLASSIFIED"


def _score_at(row: torch.Tensor, token_id: int) -> float | None:
    token_id = int(token_id)
    if token_id < 0 or token_id >= int(row.numel()):
        return None
    return float(row[token_id].detach().cpu())


def summarize_score_row(
    row: Any,
    *,
    vocab_eff: int,
    n_bins: int,
    bin_centers,
    action_stats,
    surrogate_top_token: int | None = None,
    diagnostic_tokens: Sequence[int] = DIAGNOSTIC_TOKENS,
) -> dict[str, Any]:
    if not hasattr(row, "detach"):
        row = torch.as_tensor(row)
    row = row.detach().float().cpu()
    if row.ndim != 1:
        raise ValueError(f"score row must be rank-1, got shape {tuple(row.shape)}")
    k = min(2, int(row.numel()))
    top = torch.topk(row, k)
    top1_token = int(top.indices[0])
    top1_score = float(top.values[0])
    top2_token = int(top.indices[1]) if k > 1 else None
    top2_score = float(top.values[1]) if k > 1 else None
    native_start = max(0, int(vocab_eff) - int(n_bins))
    native_end = min(int(vocab_eff), int(row.numel()))
    best = {
        "open": {"token": None, "score": None},
        "close": {"token": None, "score": None},
        "boundary": {"token": None, "score": None},
    }
    for tid in range(native_start, native_end):
        exec_info = classify_disc_and_raw(tid, vocab_eff, n_bins, bin_centers, action_stats)
        cls = exec_info["execution_class"]
        if cls == "NATIVE_OPEN":
            key = "open"
        elif cls == "NATIVE_CLOSE":
            key = "close"
        elif cls == "NATIVE_BOUNDARY":
            key = "boundary"
        else:
            continue
        score = float(row[tid])
        if best[key]["score"] is None or score > float(best[key]["score"]):
            best[key] = {"token": int(tid), "score": score}
    summary = {
        "top1_token": top1_token,
        "top1_score": top1_score,
        "top2_token": top2_token,
        "top2_score": top2_score,
        "top1_minus_top2_gap": None if top2_score is None else float(top1_score - top2_score),
        "best_native_open_token": best["open"]["token"],
        "best_native_open_score": best["open"]["score"],
        "best_native_close_token": best["close"]["token"],
        "best_native_close_score": best["close"]["score"],
        "best_native_boundary_token": best["boundary"]["token"],
        "best_native_boundary_score": best["boundary"]["score"],
        "surrogate_top_token": None if surrogate_top_token is None else int(surrogate_top_token),
        "surrogate_top_score": None if surrogate_top_token is None else _score_at(row, int(surrogate_top_token)),
    }
    for tid in diagnostic_tokens:
        summary[f"score_token_{int(tid)}"] = _score_at(row, int(tid))
    return summary


def generation_score_audit_from_row(row: Any, *, emitted_token: int, **kwargs) -> dict[str, Any]:
    summary = summarize_score_row(row, **kwargs)
    summary.update({
        "emitted_token": int(emitted_token),
        "processed_score_argmax_token": int(summary["top1_token"]),
        "generation_score_argmax": int(summary["top1_token"]),  # backward-compatible alias
        "argmax_matches_emitted": int(summary["top1_token"]) == int(emitted_token),
    })
    return summary


def validate_generation_score_invariant(score_audit: Mapping[str, Any] | None, emitted_gripper_token: int):
    if not score_audit:
        return False, "GENERATE_SCORE_AUDIT_MISSING"
    argmax = score_audit.get("processed_score_argmax_token", score_audit.get("generation_score_argmax"))
    if argmax is None:
        return False, "GENERATE_SCORE_AUDIT_MISSING"
    if int(argmax) != int(emitted_gripper_token):
        return False, "GENERATE_SCORE_ARGMAX_MISMATCH"
    return True, ""


def determine_v3_transfer_class(surrogate_top_token, ar_gripper_token, surrogate_exec_class, ar_exec_class):
    if int(surrogate_top_token) != int(ar_gripper_token):
        return "SURROGATE_TO_GENERATION_TOP1_MISMATCH"
    if ar_exec_class == "NATIVE_OPEN":
        return "SURROGATE_TOP_MATCH_NATIVE_OPEN"
    if ar_exec_class == "CLIP_MEDIATED_OPEN":
        return "SURROGATE_TOP_MATCH_CLIP_MEDIATED_OPEN"
    return "SURROGATE_TOP_MATCH_NONOPEN"


def classify_path_diagnosis(paths: Mapping[str, Mapping[str, Any]], bundle: Mapping[str, Any]) -> str:
    a = paths.get("A") or {}
    b = paths.get("B") or {}
    c = paths.get("C") or {}
    d = paths.get("D") or {}
    a_tok = a.get("emitted_gripper_token")
    b_tok = b.get("raw_logit_top_token")
    c_tok = c.get("raw_logit_top_token")
    d_tok = d.get("emitted_gripper_token")
    a_gap = a.get("processed_score_top1_minus_top2_gap")
    if a_tok is not None and b_tok is not None and int(a_tok) != int(b_tok):
        if c_tok is not None and int(c_tok) == int(a_tok):
            return "CACHE_PATH_MISMATCH_CANDIDATE"
        if d_tok is not None and int(d_tok) != int(a_tok):
            return "GENERATION_SCORE_PROCESSING_MISMATCH_CANDIDATE"
        if a_gap is not None and abs(float(a_gap)) < 1e-3:
            return "NEAR_TIE_NUMERICAL_SENSITIVITY_CANDIDATE"
        surr = bundle.get("surrogate_token_execution") or {}
        if surr.get("execution_class") in {"NATIVE_BOUNDARY", "NATIVE_CLOSE"}:
            return "COMPETITION_SET_INCOMPLETENESS_CONFIRMED"
        return "LARGE_UNEXPLAINED_PATH_DIFFERENCE"
    return "PATHS_AGREE_OR_INSUFFICIENT_DIFFERENCE"


def path_result_schema(
    *,
    path: str,
    cache_behavior: str,
    prefix_tokens: Sequence[int] | None = None,
    generated_tokens: Sequence[int] | None = None,
    emitted_gripper_token: int | None = None,
    raw_logit_summary: Mapping[str, Any] | None = None,
    processed_score_summary: Mapping[str, Any] | None = None,
    token_execution: Mapping[str, Any] | None = None,
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    def prefixed(prefix: str, source: Mapping[str, Any] | None) -> dict[str, Any]:
        source = source or {}
        return {
            f"{prefix}_top_token": source.get("top1_token"),
            f"{prefix}_top_score": source.get("top1_score"),
            f"{prefix}_top2_token": source.get("top2_token"),
            f"{prefix}_top2_score": source.get("top2_score"),
            f"{prefix}_top1_minus_top2_gap": source.get("top1_minus_top2_gap"),
            f"{prefix}_score_token_31744": source.get("score_token_31744"),
            f"{prefix}_score_token_31872": source.get("score_token_31872"),
            f"{prefix}_best_native_open_token": source.get("best_native_open_token"),
            f"{prefix}_best_native_open_score": source.get("best_native_open_score"),
            f"{prefix}_best_native_close_token": source.get("best_native_close_token"),
            f"{prefix}_best_native_close_score": source.get("best_native_close_score"),
            f"{prefix}_best_native_boundary_token": source.get("best_native_boundary_token"),
            f"{prefix}_best_native_boundary_score": source.get("best_native_boundary_score"),
            f"{prefix}_surrogate_top_token": source.get("surrogate_top_token"),
            f"{prefix}_surrogate_top_score": source.get("surrogate_top_score"),
        }

    out = {
        "path": path,
        "cache_behavior": cache_behavior,
        "prefix_tokens": None if prefix_tokens is None else [int(x) for x in prefix_tokens],
        "generated_tokens": None if generated_tokens is None else [int(x) for x in generated_tokens],
        "emitted_gripper_token": None if emitted_gripper_token is None else int(emitted_gripper_token),
        "unavailable_reason": unavailable_reason,
        "token_execution": dict(token_execution or {}) if token_execution is not None else None,
    }
    out.update(prefixed("raw_logit", raw_logit_summary))
    out.update(prefixed("processed_score", processed_score_summary))
    return out


REPLAY_BUNDLE_REQUIRED_FIELDS = (
    "schema_version", "task", "state_id", "attack_seed", "job_id", "step", "condition",
    "objective", "objective_tag", "model_path", "model_dtype",
    "runner_sha256", "adapter_sha256", "semantics_sha256", "exec_spec_sha256",
    "prompt_input_ids", "prompt_input_ids_shape", "prompt_input_ids_dtype",
    "adv_tensor_filename", "adv_tensor_sha256", "adv_pixel_values_shape", "adv_tensor_dtype",
    "surrogate_generated_arm_prefix_token_ids", "official_generated_action_token_ids",
    "official_generation_score_audit", "surrogate_token_execution", "official_token_execution",
    "generation_config", "clean_generated_action_token_ids", "clean_generated_arm_prefix_token_ids",
    "retokenized_clean_action_arm_prefix", "adv_generated_arm_prefix",
    "adv_vs_clean_generated_arm_match_rate", "transfer_classification",
)


def _empty(value: Any) -> bool:
    return value is None or value == "" or value == []


def validate_execution_dict(value: Any, label: str) -> list[str]:
    issues: list[str] = []
    if not isinstance(value, dict):
        return [f"{label}:expected dict"]
    for key in ("token_id", "disc_before", "disc_after", "clipped", "decoded_raw_gripper", "executed_env_gripper", "execution_class"):
        if key not in value:
            issues.append(f"{label}.{key}:MISSING")
        elif value[key] is None or value[key] == "":
            issues.append(f"{label}.{key}:EMPTY")
    return issues


def validate_replay_bundle(
    bundle: Mapping[str, Any],
    *,
    bundle_dir: str | Path | None = None,
    verify_tensor: bool = False,
    require_official_tokens: Sequence[int] | None = None,
) -> list[str]:
    issues: list[str] = []
    for key in REPLAY_BUNDLE_REQUIRED_FIELDS:
        if key not in bundle:
            issues.append(f"{key}:MISSING")
        elif _empty(bundle[key]):
            issues.append(f"{key}:EMPTY")
    if issues:
        return issues
    if bundle.get("schema_version") not in VALID_REPLAY_SCHEMA_VERSIONS:
        issues.append(f"schema_version:INVALID:{bundle.get('schema_version')}")
    for key in ("runner_sha256", "adapter_sha256", "semantics_sha256", "exec_spec_sha256", "adv_tensor_sha256"):
        if not is_sha256_hex(bundle.get(key)):
            issues.append(f"{key}:expected 64-char hex")
    for key, n in (
        ("surrogate_generated_arm_prefix_token_ids", ARM_PREFIX_LEN),
        ("clean_generated_arm_prefix_token_ids", ARM_PREFIX_LEN),
        ("retokenized_clean_action_arm_prefix", ARM_PREFIX_LEN),
        ("adv_generated_arm_prefix", ARM_PREFIX_LEN),
        ("official_generated_action_token_ids", ACTION_DIM),
        ("clean_generated_action_token_ids", ACTION_DIM),
    ):
        try:
            require_token_list(bundle.get(key), expected_len=n, label=key)
        except Exception as exc:
            issues.append(f"{key}:{exc}")
    if list(bundle.get("prompt_input_ids_shape")) != [len(bundle.get("prompt_input_ids")), len(bundle.get("prompt_input_ids")[0])]:
        issues.append("prompt_input_ids_shape:mismatch")
    score = bundle.get("official_generation_score_audit")
    if not isinstance(score, dict):
        issues.append("official_generation_score_audit:expected dict")
    else:
        for key in ("emitted_token", "processed_score_argmax_token", "top1_token", "top1_score", "top2_token", "top2_score", "top1_minus_top2_gap"):
            if score.get(key) is None:
                issues.append(f"official_generation_score_audit.{key}:MISSING")
        ok, failure = validate_generation_score_invariant(score, bundle["official_generated_action_token_ids"][-1])
        if not ok:
            issues.append(f"official_generation_score_invariant:{failure}")
    issues.extend(validate_execution_dict(bundle.get("surrogate_token_execution"), "surrogate_token_execution"))
    issues.extend(validate_execution_dict(bundle.get("official_token_execution"), "official_token_execution"))
    cfg = bundle.get("generation_config")
    if not isinstance(cfg, dict):
        issues.append("generation_config:expected dict")
    else:
        for key in ("do_sample", "max_new_tokens", "default_use_cache", "effective_use_cache", "eos_token_id", "pad_token_id"):
            if key not in cfg:
                issues.append(f"generation_config.{key}:MISSING")
    try:
        inv = validate_prefix_invariant(bundle["surrogate_generated_arm_prefix_token_ids"], bundle["official_generated_action_token_ids"])
        if not inv["ok"]:
            issues.append(f"prefix_invariant:{inv['failure_type']}")
    except Exception as exc:
        issues.append(f"prefix_invariant:{exc}")
    if require_official_tokens is not None:
        expected = [int(x) for x in require_official_tokens]
        actual = require_token_list(bundle.get("official_generated_action_token_ids"), expected_len=ACTION_DIM, label="official_generated_action_token_ids")
        if actual != expected:
            issues.append("official_generated_action_token_ids:REPRODUCTION_MISMATCH")
    if verify_tensor:
        if bundle_dir is None:
            issues.append("tensor_verification:bundle_dir_missing")
        else:
            tensor_path = Path(bundle_dir) / str(bundle.get("adv_tensor_filename"))
            if not tensor_path.exists():
                issues.append("adv_tensor_filename:FILE_MISSING")
            else:
                actual_sha = sha256_file(tensor_path)
                if actual_sha != bundle.get("adv_tensor_sha256"):
                    issues.append("adv_tensor_sha256:FILE_HASH_MISMATCH")
                try:
                    tensor = torch.load(tensor_path, map_location="cpu")
                    if list(tensor.shape) != list(bundle.get("adv_pixel_values_shape")):
                        issues.append("adv_pixel_values_shape:TENSOR_SHAPE_MISMATCH")
                    if str(tensor.dtype) != str(bundle.get("adv_tensor_dtype")):
                        issues.append("adv_tensor_dtype:TENSOR_DTYPE_MISMATCH")
                except Exception as exc:
                    issues.append(f"adv_tensor:LOAD_FAILED:{exc}")
    return issues


def check_finite_or_fail(value, label, record_idx):
    if value is None or value == "":
        raise RuntimeError(f"V6 HARD FAIL: v3 record[{record_idx}].{label} is missing/empty")
    v = float(value)
    if not math.isfinite(v):
        raise RuntimeError(f"V6 HARD FAIL: v3 record[{record_idx}].{label}={v} not finite")
    return v
