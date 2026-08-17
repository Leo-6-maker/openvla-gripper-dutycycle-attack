"""Exhaustive clean-only audit of the prospective native token authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from gripper_attack.stage_x_t1_native_token_authority import (  # noqa: E402
    NATIVE_ACTION_TOKEN_ALGORITHM,
    NativeActionTokenAuthorityV2,
    SuiteActionTokenBinding,
)
from scripts.stage_x.audit_stage_x1r_pgd_alignment import (  # noqa: E402
    DEFAULT_MODEL_PATHS,
    differential_cases,
    denormalize_action,
    helper_token_ids,
    load_native_suite,
    normalize_action,
)


ACTION_TOKENIZER_SOURCE = "/mnt/sdc/dty_user/openvla_attack/repos/openvla-upstream-clean-c8f03f4/prismatic/vla/action_tokenizer.py"
ACTION_TOKENIZER_SHA256 = "fdc98fcbf5b0926ef2181db71946d23ffbfa052cf8443dc933d52c42a191352c"
MODEL_DECODER_SHA256 = "2e672e75958205b05f40f4cd2467d3763b8e36eb2728289cd055c54213338e85"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authority_for(native_info: dict[str, Any]) -> NativeActionTokenAuthorityV2:
    config_path = Path(native_info["model_path"]) / "config.json"
    stats = native_info["stats"]
    binding = SuiteActionTokenBinding(
        suite=str(native_info["suite"]),
        checkpoint_path=str(native_info["model_path"]),
        checkpoint_config_sha256=sha256_file(config_path),
        tokenizer_source=ACTION_TOKENIZER_SOURCE,
        tokenizer_source_sha256=ACTION_TOKENIZER_SHA256,
        model_decoder_source_sha256=MODEL_DECODER_SHA256,
        tokenizer_files=tuple(sorted((str(k), str(v)) for k, v in native_info["tokenizer_files"].items())),
        tokenizer_vocab_size=int(native_info["tokenizer_vocab_size"]),
        n_action_bins=int(native_info["native"].n_bins),
        bins=tuple(float(x) for x in native_info["bins"]),
        bin_centers=tuple(float(x) for x in native_info["bin_centers"]),
        q01=tuple(float(x) for x in np.asarray(stats["q01"])),
        q99=tuple(float(x) for x in np.asarray(stats["q99"])),
        mask=tuple(bool(x) for x in np.asarray(stats.get("mask", np.ones(7, dtype=bool)))),
    )
    return NativeActionTokenAuthorityV2(binding)


def audit_suite(suite: str, model_path: Path) -> dict[str, Any]:
    native_info = load_native_suite(suite, model_path)
    authority = authority_for(native_info)
    bins = np.asarray(native_info["bins"], dtype=np.float64)
    cases = differential_cases(native_info)
    v2_mismatches: list[dict[str, Any]] = []
    helper_mismatches: list[dict[str, Any]] = []
    for case in cases:
        base = np.zeros(7, dtype=np.float64)
        raw = base.copy()
        if case["value_space"] == "normalized":
            normalized = base.copy()
            normalized[int(case["dim"])] = float(case["value"])
            raw = denormalize_action(normalized, native_info["stats"])
        else:
            raw[int(case["dim"])] = float(case["value"])
            normalized = normalize_action(raw, native_info["stats"])
        expected = (int(native_info["tokenizer_vocab_size"]) - np.digitize(np.clip(normalized, -1.0, 1.0), bins)).astype(np.int64)
        actual = authority.encode_normalized(normalized)
        helper = helper_token_ids(native_info, raw)
        dim = int(case["dim"])
        if not np.array_equal(expected, actual):
            v2_mismatches.append({"case": case, "expected": int(expected[dim]), "actual": int(actual[dim])})
        if not np.array_equal(expected, helper):
            helper_mismatches.append({
                "case": case,
                "native_token": int(expected[dim]),
                "legacy_helper_token": int(helper[dim]),
                "normalized_value": float(normalized[dim]),
            })
    receipt = authority.receipt()
    return {
        "suite": suite,
        "binding": receipt,
        "total_cases": len(cases),
        "v2_native_mismatch_count": len(v2_mismatches),
        "v2_native_mismatches_sample": v2_mismatches[:10],
        "legacy_helper_mismatch_count": len(helper_mismatches),
        "legacy_helper_mismatches_sample": helper_mismatches[:10],
        "legacy_helper_status": receipt["legacy_helper_status"],
        "endpoint": receipt["endpoint"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-paths", default="")
    args = parser.parse_args()
    model_paths = dict(DEFAULT_MODEL_PATHS)
    if args.model_paths:
        model_paths.update(json.loads(Path(args.model_paths).read_text(encoding="utf-8")))
    suites = {suite: audit_suite(suite, Path(model_paths[suite])) for suite in sorted(model_paths)}
    v2_mismatch_count = sum(int(row["v2_native_mismatch_count"]) for row in suites.values())
    report = {
        "schema": "STAGE_X_X1R_T1_NATIVE_ACTION_TOKEN_AUTHORITY_AUDIT_V1",
        "status": "PASS_V2_NATIVE_PARITY_LEGACY_HELPER_DIAGNOSTIC_MISMATCH" if v2_mismatch_count == 0 else "HOLD_NATIVE_AUTHORITY_MISMATCH",
        "authority_version": "STAGE_X_X1R_T1_NATIVE_ACTION_TOKEN_AUTHORITY_V2",
        "algorithm": NATIVE_ACTION_TOKEN_ALGORITHM,
        "canonical_source": ACTION_TOKENIZER_SOURCE,
        "canonical_source_sha256": ACTION_TOKENIZER_SHA256,
        "model_decoder_source_sha256": MODEL_DECODER_SHA256,
        "suites": suites,
        "protected_counters": {
            "pgd_calls": 0,
            "env_step_calls": 0,
            "physical_interventions": 0,
            "vphys_reads": 0,
            "attack_outcome_reads": 0,
            "protected_reads": 0,
            "eval160_reads": 0,
        },
        "eval160": "UNREAD",
        "protected_evaluation": "UNREAD",
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "v2_native_mismatch_count": v2_mismatch_count}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

