from __future__ import annotations
import numpy as np


def _pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=np.float32), q)) if xs else 0.0


def normalized_action_discrepancy_cleanref(
    clean_action,
    executed_action,
    action_low=None,
    action_high=None,
    dims=None,
    eps: float = 1e-8,
) -> float:
    """Baseline-inspired clean-reference Normalized Action Discrepancy.

    Baseline NAD normalizes |prediction - ground_truth| by the maximum possible
    discrepancy to the action bounds. Online LIBERO rollouts after divergence do
    not expose synchronized ground-truth actions, so V4 reports a clean-reference
    variant: |executed - clean_policy| normalized by the max distance from the
    clean policy action to [low, high] on the evaluated DoFs.
    """
    clean = np.asarray(clean_action, dtype=np.float32)
    executed = np.asarray(executed_action, dtype=np.float32)
    if clean.shape != executed.shape:
        raise ValueError(f"clean/executed action shape mismatch: {clean.shape} vs {executed.shape}")
    if dims is None:
        idx = np.arange(clean.size)
    else:
        idx = np.asarray(list(dims), dtype=np.int64)
    if idx.size == 0:
        return 0.0
    c = clean[idx]
    e = executed[idx]
    if action_low is None or action_high is None:
        # Fallback for dry tests only: unnormalized denominator. Real V4 runs
        # must log action bounds from OpenVLA q01/q99 stats.
        denom = np.ones_like(c, dtype=np.float32)
    else:
        low = np.asarray(action_low, dtype=np.float32)[idx]
        high = np.asarray(action_high, dtype=np.float32)[idx]
        denom = np.maximum(np.abs(c - low), np.abs(c - high))
    return float(np.mean(np.abs(e - c) / (denom + float(eps))))


def aggregate_episode_from_steps(step_records: list[dict], success: bool, timeout: bool, invalid: bool) -> dict:
    n = max(1, len(step_records))
    atk = [r for r in step_records if r.get("attack_active")]
    raw = [r for r in step_records if r.get("trigger_active_raw")]
    req = [r for r in step_records if r.get("trigger_request_active", r.get("trigger_active_raw"))]
    blocked = [r for r in step_records if r.get("budget_blocked")]
    grasp_gate = [r for r in step_records if r.get("grasp_gate_active")]
    proxy_gate = [r for r in step_records if r.get("proxy_grasp_gate_active")]
    attack_in_grasp = [r for r in atk if r.get("grasp_gate_active")]
    attack_in_proxy = [r for r in atk if r.get("proxy_grasp_gate_active")]
    align = [float(r.get("directional_alignment", 0.0)) for r in step_records]
    align_atk = [float(r.get("directional_alignment", 0.0)) for r in atk]
    delta_l2 = [float(r.get("delta_l2", 0.0)) for r in step_records]
    delta_l2_atk = [float(r.get("delta_l2", 0.0)) for r in atk]
    delta_linf = [float(r.get("delta_linf", 0.0)) for r in step_records]
    delta_linf_atk = [float(r.get("delta_linf", 0.0)) for r in atk]
    nad_clean = [float(r.get("nad_cleanref_step", 0.0)) for r in step_records]
    nad_clean_atk = [float(r.get("nad_cleanref_step", 0.0)) for r in atk]
    return {
        "success": bool(success), "failure": not bool(success), "timeout": bool(timeout), "invalid": bool(invalid),
        "num_steps": len(step_records), "num_attack_active_steps": len(atk),
        "attacked_step_ratio": len(atk) / float(n), "raw_trigger_rate": len(raw) / float(n),
        "trigger_request_rate": len(req) / float(n),
        "budget_blocked_rate": len(blocked) / float(n),
        "grasp_gate_rate": len(grasp_gate) / float(n),
        "proxy_grasp_gate_rate": len(proxy_gate) / float(n),
        "attack_in_gate_rate": len(attack_in_grasp) / float(max(len(atk), 1)),
        "attack_in_proxy_gate_rate": len(attack_in_proxy) / float(max(len(atk), 1)),
        "first_attack_step_relative_to_grasp": next((r.get("first_attack_step_relative_to_grasp") for r in atk if r.get("first_attack_step_relative_to_grasp") is not None), None),
        "action_delta_l2_all": float(np.mean(delta_l2)) if delta_l2 else 0.0,
        "action_delta_l2_attacked": float(np.mean(delta_l2_atk)) if delta_l2_atk else 0.0,
        "action_delta_linf_all": float(np.mean(delta_linf)) if delta_linf else 0.0,
        "action_delta_linf_attacked": float(np.mean(delta_linf_atk)) if delta_linf_atk else 0.0,
        "nad_cleanref_all": float(np.mean(nad_clean)) if nad_clean else 0.0,
        "nad_cleanref_attacked": float(np.mean(nad_clean_atk)) if nad_clean_atk else 0.0,
        "mean_alignment_all": float(np.mean(align)) if align else 0.0,
        "mean_alignment_attacked": float(np.mean(align_atk)) if align_atk else 0.0,
        "targeted_alignment_rate": float(np.mean([x > 0.0 for x in align_atk])) if align_atk else 0.0,
        "latency_total_p50": _pct([r.get("Ttotal", 0.0) for r in step_records], 50),
        "latency_total_p95": _pct([r.get("Ttotal", 0.0) for r in step_records], 95),
        "latency_trigger_p50": _pct([r.get("Ttrig", 0.0) for r in step_records], 50),
        "latency_trigger_p95": _pct([r.get("Ttrig", 0.0) for r in step_records], 95),
        "latency_attack_p50": _pct([r.get("Tattack", 0.0) for r in step_records], 50),
        "latency_attack_p95": _pct([r.get("Tattack", 0.0) for r in step_records], 95),
        "signal_availability_rate": float(np.mean([bool(r.get("signal_available")) for r in step_records])) if step_records else 0.0,
        "fallback_rate": float(np.mean([bool(r.get("fallback")) for r in step_records])) if step_records else 0.0,
    }


def aggregate_run(episode_records: list[dict], clean_reference_records: list[dict] | None = None) -> dict:
    sr = float(np.mean([bool(e.get("success")) for e in episode_records])) if episode_records else 0.0
    out = {
        "episodes": len(episode_records),
        "SR_attack": sr,
        "FR_attack": 1.0 - sr,
        "attacked_step_ratio_mean": float(np.mean([float(e.get("attacked_step_ratio", 0.0)) for e in episode_records])) if episode_records else 0.0,
        "NAD_cleanref_mean": float(np.mean([float(e.get("nad_cleanref_all", 0.0)) for e in episode_records])) if episode_records else 0.0,
        "NAD_cleanref_attacked_mean": float(np.mean([float(e.get("nad_cleanref_attacked", 0.0)) for e in episode_records])) if episode_records else 0.0,
        "action_delta_l2_mean": float(np.mean([float(e.get("action_delta_l2_all", e.get("nad_all", 0.0))) for e in episode_records])) if episode_records else 0.0,
        "action_delta_l2_attacked_mean": float(np.mean([float(e.get("action_delta_l2_attacked", e.get("nad_attacked", 0.0))) for e in episode_records])) if episode_records else 0.0,
        "timeout_rate": float(np.mean([bool(e.get("timeout")) for e in episode_records])) if episode_records else 0.0,
        "invalid_rate": float(np.mean([bool(e.get("invalid")) for e in episode_records])) if episode_records else 0.0,
        "stable_success_50_mean": float(np.mean([float(e.get("stable_success_50", 0.0)) for e in episode_records])) if episode_records and any("stable_success_50" in e for e in episode_records) else "",
        "stable_success_100_mean": float(np.mean([float(e.get("stable_success_100", 0.0)) for e in episode_records])) if episode_records and any("stable_success_100" in e for e in episode_records) else "",
        "grasp_gate_rate_mean": float(np.mean([float(e.get("grasp_gate_rate", 0.0)) for e in episode_records])) if episode_records else 0.0,
        "proxy_grasp_gate_rate_mean": float(np.mean([float(e.get("proxy_grasp_gate_rate", 0.0)) for e in episode_records])) if episode_records else 0.0,
        "attack_in_gate_rate_mean": float(np.mean([float(e.get("attack_in_gate_rate", 0.0)) for e in episode_records])) if episode_records else 0.0,
    }
    if clean_reference_records:
        sr_clean = float(np.mean([bool(e.get("success")) for e in clean_reference_records]))
        out["SR_clean_matched"] = sr_clean
        out["FR_drop"] = sr_clean - sr
    else:
        out["SR_clean_matched"] = ""
        out["FR_drop"] = ""
    return out


def bootstrap_ci(values: list[float], n: int = 10000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float32)
    if x.size == 0:
        return (0.0, 0.0)
    rng = np.random.RandomState(seed)
    means = [float(np.mean(rng.choice(x, size=x.size, replace=True))) for _ in range(int(n))]
    return (float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2))))


def paired_delta_by_task_seed(rows_a, rows_b, metric: str) -> dict:
    idx = {(r.get("task_id"), r.get("seed")): float(r.get(metric, 0.0)) for r in rows_b}
    vals = []
    for r in rows_a:
        k = (r.get("task_id"), r.get("seed"))
        if k in idx:
            vals.append(float(r.get(metric, 0.0)) - idx[k])
    lo, hi = bootstrap_ci(vals) if vals else (0.0, 0.0)
    return {"n": len(vals), "mean": float(np.mean(vals)) if vals else 0.0, "ci_low": lo, "ci_high": hi, "metric": metric}
