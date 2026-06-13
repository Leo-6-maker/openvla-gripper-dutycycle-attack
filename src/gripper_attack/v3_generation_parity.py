"""Shared V3 generation parity helpers.

Used by runner, diagnostic script, and tests. Single source of truth for:
- Token execution classification (official decode pipeline)
- Generation score invariant
- V3 transfer class
- Replay bundle schema
"""

from __future__ import annotations

import math

from .openvla_libero_exec_spec import (
    raw_gripper_is_open,
    raw_gripper_is_close,
    raw_gripper_to_env_gripper,
)

# ── Token execution classification ──


def classify_disc_and_raw(token_id: int, vocab_eff: int, n_bins: int,
                          bin_centers, unnorm_key_stats):
    """Decode a token through the FULL official pipeline.

    Returns dict with: disc_before, disc_after, clipped, decoded_raw,
    executed_env, execution_class.

    Steps: token→disc→clip→bin_center→q01/q99 unnormalize→raw→env.
    """
    import numpy as np
    disc_before = int(vocab_eff - token_id - 1)
    disc_after = max(0, min(int(n_bins) - 1, disc_before))
    clipped = disc_before != disc_after

    # Official unnormalize: bin_center → raw action
    center = float(bin_centers[disc_after])
    stats = unnorm_key_stats
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    hi = np.asarray(stats["q99"], dtype=np.float32)
    lo = np.asarray(stats["q01"], dtype=np.float32)
    gripper_dim = len(hi) - 1
    if bool(mask[gripper_dim]):
        raw = float(0.5 * (center + 1.0) * (hi[gripper_dim] - lo[gripper_dim]) + lo[gripper_dim])
    else:
        raw = float(center)  # official: np.where(mask, unnormalize, center)

    env = raw_gripper_to_env_gripper(raw)

    # Classify by official execution
    if clipped:
        if env < -0.5:
            exec_class = 'CLIP_MEDIATED_OPEN'
        elif env > 0.5:
            exec_class = 'CLIP_MEDIATED_CLOSE'
        else:
            exec_class = 'CLIP_MEDIATED_NEUTRAL'
    elif abs(raw - 0.5) <= 1e-9:
        exec_class = 'NATIVE_BOUNDARY'
    elif raw_gripper_is_open(raw):
        exec_class = 'NATIVE_OPEN'
    elif raw_gripper_is_close(raw):
        exec_class = 'NATIVE_CLOSE'
    else:
        exec_class = 'NATIVE_UNCLASSIFIED'

    return {
        'token_id': int(token_id),
        'disc_before': disc_before,
        'disc_after': disc_after,
        'clipped': clipped,
        'decoded_raw_gripper': round(raw, 8),
        'executed_env_gripper': round(env, 6),
        'execution_class': exec_class,
    }


def classify_token_simple(token_id, vocab_eff, n_bins, executed_raw,
                          gripper_clipped):
    """Lightweight classification using pre-computed raw + clip status.

    For use when full unnormalize pipeline isn't available (e.g. in-runner
    where raw/env are already computed).
    """
    tid = int(token_id) if token_id not in (None, '') else None
    if tid is None:
        return 'UNKNOWN'
    disc = int(vocab_eff - tid - 1)
    if disc < 0 or disc >= n_bins:
        clipped = True
    else:
        clipped = bool(gripper_clipped)
    if clipped:
        if isinstance(executed_raw, (int, float)) and executed_raw > 0.5:
            return 'CLIP_MEDIATED_OPEN'
        return 'CLIP_MEDIATED_CLOSE'
    if isinstance(executed_raw, (int, float)):
        if executed_raw > 0.5:
            return 'NATIVE_OPEN'
        if executed_raw < 0.5:
            return 'NATIVE_CLOSE'
        if abs(executed_raw - 0.5) <= 1e-9:
            return 'NATIVE_BOUNDARY'
    return 'NATIVE_UNCLASSIFIED'


# ── Generation score invariant ──


def validate_generation_score_invariant(score_audit, emitted_gripper_token):
    """Validate official generation score argmax == emitted token.

    Returns (ok, failure_type):
      - (True, '') if invariant holds
      - (False, 'GENERATE_SCORE_ARGMAX_MISMATCH') if violated
      - (False, 'GENERATE_SCORE_AUDIT_MISSING') if score_audit absent
    """
    if not score_audit:
        return False, 'GENERATE_SCORE_AUDIT_MISSING'
    argmax = score_audit.get('generation_score_argmax')
    if argmax is None:
        return False, 'GENERATE_SCORE_AUDIT_MISSING'
    if int(argmax) != int(emitted_gripper_token):
        return False, 'GENERATE_SCORE_ARGMAX_MISMATCH'
    return True, ''


# ── V3 transfer class ──


def determine_v3_transfer_class(surrogate_top_token, ar_gripper_token,
                                surrogate_exec_class, ar_exec_class):
    """Determine the v3_transfer_class for one attack opportunity.

    Returns one of:
      SURROGATE_TO_GENERATION_TOP1_MISMATCH
      SURROGATE_TOP_MATCH_NATIVE_OPEN
      SURROGATE_TOP_MATCH_NONOPEN
      SURROGATE_TOP_MATCH_CLIP_MEDIATED_OPEN
    """
    top_match = (int(surrogate_top_token) == int(ar_gripper_token))
    if not top_match:
        return 'SURROGATE_TO_GENERATION_TOP1_MISMATCH'
    if ar_exec_class == 'NATIVE_OPEN':
        return 'SURROGATE_TOP_MATCH_NATIVE_OPEN'
    if ar_exec_class == 'CLIP_MEDIATED_OPEN':
        return 'SURROGATE_TOP_MATCH_CLIP_MEDIATED_OPEN'
    return 'SURROGATE_TOP_MATCH_NONOPEN'


# ── Replay bundle schema ──


REPLAY_BUNDLE_REQUIRED_FIELDS = (
    'schema_version', 'step', 'task', 'state_id', 'job_id', 'condition',
    'objective', 'objective_tag', 'seed',
    'runner_sha256', 'adapter_sha256', 'semantics_sha256',
    'exec_spec_sha256', 'model_path', 'model_dtype',
    'prompt_input_ids', 'prompt_input_ids_shape',
    'adv_pixel_values_shape', 'adv_tensor_dtype',
    'adv_tensor_filename', 'adv_tensor_sha256',
    'generated_arm_prefix', 'full_ar_tokens',
    'surrogate_global_top_token', 'surrogate_token_execution',
    'ar_token_execution',
    'generation_score_argmax',
    'surrogate_top_matches_generation', 'v3_transfer_class',
)


def validate_replay_bundle(bundle):
    """Check replay bundle has all required fields with non-empty values.

    Returns list of issues (missing keys or empty values).
    """
    issues = []
    for k in REPLAY_BUNDLE_REQUIRED_FIELDS:
        if k not in bundle:
            issues.append(f'{k}:MISSING')
            continue
        v = bundle[k]
        if v is None or v == '':
            issues.append(f'{k}:EMPTY')
            continue
        # Type-specific checks
        if k == 'generated_arm_prefix':
            if not isinstance(v, list) or len(v) != 6:
                issues.append(f'{k}:expected list[6], got {type(v).__name__}')
        if k == 'full_ar_tokens':
            if not isinstance(v, list) or len(v) != 7:
                issues.append(f'{k}:expected list[7], got {type(v).__name__}')
        if k in ('adv_tensor_sha256', 'runner_sha256', 'adapter_sha256',
                 'semantics_sha256', 'exec_spec_sha256'):
            if isinstance(v, str) and len(v) != 64:
                issues.append(f'{k}:expected 64-char hex, got len={len(v)}')
        if k == 'surrogate_token_execution':
            if not isinstance(v, dict) or 'execution_class' not in v:
                issues.append(f'{k}:expected dict with execution_class')
        if k == 'ar_token_execution':
            if not isinstance(v, dict) or 'execution_class' not in v:
                issues.append(f'{k}:expected dict with execution_class')
    return issues


# ── Finite + non-empty check ──


def check_finite_or_fail(value, label, record_idx):
    """Validate a numeric value is finite. Raises RuntimeError if not."""
    if value is None or value == '':
        raise RuntimeError(
            f"V6 HARD FAIL: v3 record[{record_idx}].{label} is missing/empty")
    v = float(value)
    if not math.isfinite(v):
        raise RuntimeError(
            f"V6 HARD FAIL: v3 record[{record_idx}].{label}={v} not finite")
    return v
