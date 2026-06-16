"""Dynamic LIBERO suite/task registry — not hardcoded to any single suite.

Provides runtime discovery of available benchmark suites, tasks, init states,
and verification of model compatibility.
"""
from __future__ import annotations

import os
from typing import Optional


# ── Known suite identifiers ──
SUITE_ALIASES = {
    "object": "libero_object",
    "spatial": "libero_spatial",
    "goal": "libero_goal",
    "10": "libero_10",
    "90": "libero_90",
}


def resolve_suite(suite_name: str) -> str:
    """Resolve suite name to canonical benchmark key."""
    return SUITE_ALIASES.get(suite_name.lower(), suite_name)


def get_benchmark_dict():
    """Lazy-import benchmark dict. Returns dict of suite_name → task_suite_class."""
    from libero.libero import benchmark
    return benchmark.get_benchmark_dict()


def list_available_suites() -> list[str]:
    """Return list of canonical suite names available in this LIBERO install."""
    return sorted(get_benchmark_dict().keys())


def is_suite_available(suite_name: str) -> bool:
    return resolve_suite(suite_name) in get_benchmark_dict()


def get_suite(suite_name: str):
    """Get task suite instance. Raises KeyError if not found."""
    canonical = resolve_suite(suite_name)
    bm = get_benchmark_dict()
    if canonical not in bm:
        raise KeyError(f"Suite '{canonical}' not found. Available: {sorted(bm.keys())}")
    return bm[canonical]()


def get_task_names(suite_name: str) -> list[str]:
    """Return sorted list of task names in a suite."""
    suite = get_suite(suite_name)
    return sorted(suite.tasks.keys())


def get_task_index(suite_name: str, task_name: str) -> int:
    """Get task index from task name. Raises KeyError if not found."""
    suite = get_suite(suite_name)
    for idx, (name, _) in enumerate(suite.tasks.items()):
        if name == task_name:
            return idx
    raise KeyError(f"Task '{task_name}' not found in suite '{suite_name}'")


def get_init_state_count(suite_name: str, task_name: str) -> int:
    """Get number of init states for a task."""
    suite = get_suite(suite_name)
    init_states = suite.get_task_init_states(get_task_index(suite_name, task_name))
    return len(init_states)


def get_task_language(suite_name: str, task_name: str) -> str:
    """Get language instruction for a task."""
    suite = get_suite(suite_name)
    idx = get_task_index(suite_name, task_name)
    task_obj = suite.get_task(idx)
    return task_obj.language


def get_bddl_path(suite_name: str, task_name: str) -> str:
    """Get BDDL file path for a task."""
    from libero.libero import get_libero_path
    suite = get_suite(suite_name)
    idx = get_task_index(suite_name, task_name)
    task_obj = suite.get_task(idx)
    return os.path.join(get_libero_path("bddl_files"), task_obj.problem_folder, task_obj.bddl_file)


def probe_suite_contract(suite_name: str) -> dict:
    """Runtime probe of suite contract. Returns detailed dict."""
    canonical = resolve_suite(suite_name)
    result = {
        "suite_name": suite_name,
        "canonical": canonical,
        "available": False,
        "n_tasks": 0,
        "task_names": [],
        "total_init_states": 0,
        "errors": [],
    }

    try:
        suite = get_suite(suite_name)
        result["available"] = True
    except (KeyError, Exception) as e:
        result["errors"].append(f"suite_load_failed: {e}")
        return result

    try:
        task_names = get_task_names(suite_name)
        result["task_names"] = task_names
        result["n_tasks"] = len(task_names)
    except Exception as e:
        result["errors"].append(f"task_enum_failed: {e}")
        return result

    total_states = 0
    task_details = []
    for tn in task_names:
        try:
            lang = get_task_language(suite_name, tn)
            n_states = get_init_state_count(suite_name, tn)
            bddl = get_bddl_path(suite_name, tn)
            bddl_ok = os.path.exists(bddl)
            total_states += n_states
            task_details.append({
                "name": tn, "language": lang, "n_states": n_states,
                "bddl_exists": bddl_ok, "bddl_path": bddl,
            })
        except Exception as e:
            result["errors"].append(f"task_{tn}_probe_failed: {e}")

    result["total_init_states"] = total_states
    result["task_details"] = task_details
    return result


def probe_model_contract(model_path: str, suite_name: str) -> dict:
    """Verify model checkpoint supports a suite. Returns contract dict."""
    import torch
    result = {
        "model_path": model_path,
        "suite": suite_name,
        "model_loadable": False,
        "unnorm_keys": [],
        "action_dim": -1,
        "errors": [],
    }

    try:
        ckpt = None
        # Try loading as OpenVLA model or raw checkpoint
        if os.path.isdir(model_path):
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            result["model_type"] = getattr(cfg, "model_type", "unknown")
        elif model_path.endswith(".pt"):
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            result["model_type"] = "checkpoint"

        if ckpt is not None:
            result["model_loadable"] = True
            norm = ckpt.get("norm_stats", ckpt.get("normalization", {}))
            if hasattr(norm, "keys"):
                result["unnorm_keys"] = sorted(norm.keys())
            result["action_dim"] = ckpt.get("action_dim", -1)

        # Map suite to unnorm key convention
        suite_lower = suite_name.lower()
        if "spatial" in suite_lower:
            expected = "libero_spatial"
        elif "object" in suite_lower:
            expected = "libero_object"
        elif "goal" in suite_lower:
            expected = "libero_goal"
        else:
            expected = suite_lower

        result["expected_unnorm_key"] = expected
        if result["unnorm_keys"] and expected not in result["unnorm_keys"]:
            result["unnorm_available"] = False
            result["errors"].append(f"unnorm_key '{expected}' not found in {result['unnorm_keys']}")
        else:
            result["unnorm_available"] = True

    except Exception as e:
        result["errors"].append(f"model_probe_failed: {e}")

    return result
