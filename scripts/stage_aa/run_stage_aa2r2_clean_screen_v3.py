#!/usr/bin/env python3
"""AA2R2 Phase-B runner with the official M2 clip at the delivery boundary.

The prior runner and V2 validator remain immutable.  This wrapper only fixes
the caller-side path where the historical loader's 1e-6 acceptance margin
could leave a float32 value just outside [-1, 1].
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from stage_aa import action_semantics_v2 as SEMANTICS


def _load_base() -> ModuleType:
    path = ROOT / "scripts/stage_aa/run_stage_aa2r2_clean_screen.py"
    spec = importlib.util.spec_from_file_location("aa2r2_phase_b_v2_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"AA2R2_BASE_RUNNER_UNAVAILABLE:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
_BASE_MODEL_PAIRS_V2 = BASE.model_pairs_v2


def _official_final_chunk(family: str, chunk: Any) -> np.ndarray:
    values = np.asarray(chunk, dtype=np.float32)
    if family == SEMANTICS.MODEL_M2:
        # PI05's frozen executable rule is raw continuous action -> clip.
        return np.clip(values, -1.0, 1.0).astype(np.float32)
    return values


def model_pairs_v3(infer: Any, obs: dict[str, Any], language: str, family: str, counters: dict[str, int], context: dict[str, Any]):
    def official_infer(current_obs: dict[str, Any], instruction: str):
        chunk, meta = infer(current_obs, instruction)
        final = _official_final_chunk(family, chunk)
        return final, meta

    return _BASE_MODEL_PAIRS_V2(official_infer, obs, language, family, counters, context)


BASE.model_pairs_v2 = model_pairs_v3


def self_test() -> None:
    BASE.self_test()
    raw = np.asarray([[0.0] * 6 + [-1.0000007152557373]], dtype=np.float32)
    clipped = _official_final_chunk(SEMANTICS.MODEL_M2, raw)
    assert float(clipped[0, -1]) == -1.0
    assert np.array_equal(_official_final_chunk(SEMANTICS.MODEL_M0, raw), raw)
    print('{"status": "AA2R2_PHASE_B_V3_STATIC_CLIP_PASS"}')


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
        return 0
    return BASE.main()


if __name__ == "__main__":
    raise SystemExit(main())
