#!/usr/bin/env python3
"""Final reviewed entrypoint for CLEAN2000 Label V2.

The complete reviewed implementation is pinned to an immutable Git commit.
This entrypoint adds final staging cleanup and implementation provenance.
Formal execution remains separately authorized.
"""

from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path


_IMPLEMENTATION_COMMIT = "2366d47d545e21b6f7aac8a702b3de900b6d20b7"
_IMPLEMENTATION_PATH = "tools/multisuite_detector/build_clean2000_label_v2.py"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_COPY = Path(__file__).with_name(".label_v2_reviewed_runtime.py")

try:
    source = subprocess.check_output(
        ["git", "show", f"{_IMPLEMENTATION_COMMIT}:{_IMPLEMENTATION_PATH}"],
        cwd=_REPO_ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    )
    _RUNTIME_COPY.write_text(source, encoding="utf-8")
    _impl = runpy.run_path(str(_RUNTIME_COPY), run_name="_label_v2_reviewed_impl")
finally:
    _RUNTIME_COPY.unlink(missing_ok=True)

_impl["__file__"] = str(Path(__file__).resolve())
_original_postprocess = _impl["_postprocess_formal_outputs"]


def _postprocess_formal_outputs(output_root: Path, final_output_root: Path) -> None:
    _original_postprocess(output_root, final_output_root)
    manifest_path = output_root / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["implementation_source_commit"] = _IMPLEMENTATION_COMMIT
    manifest["implementation_source_path"] = _IMPLEMENTATION_PATH
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    label_path = output_root / "label_v2.csv"
    summary_path = output_root / "validation_summary.json"
    manual_path = output_root / "manual_audit_sample_manifest.csv"
    sums_path = output_root / "SHA256SUMS"
    sums = [
        f"{_impl['_sha256_file'](path)}  {path.name}"
        for path in [label_path, manifest_path, summary_path, manual_path]
    ]
    sums_path.write_text("\n".join(sums) + "\n", encoding="utf-8")


_impl["_postprocess_formal_outputs"] = _postprocess_formal_outputs


def _run_formal_build(args: list[str]) -> int:
    output_text = _impl["_arg_value"](args, "--output-root")
    if output_text is None:
        _impl["_core"].fail("formal-ledger-build requires --output-root")
    output_root = Path(output_text)
    _impl["_formal_target_preflight"](output_root)
    staging = output_root.parent / f".{output_root.name}.staging-{os.getpid()}"
    if staging.exists():
        _impl["_core"].fail(f"staging output already exists: {staging}")
    staged_args = _impl["_replace_arg"](args, "--output-root", str(staging))
    try:
        result = _impl["_core"].main(staged_args)
        if result != 0:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            return result
        _postprocess_formal_outputs(staging, output_root)
        staging.rename(output_root)
        return 0
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


_impl["_run_formal_build"] = _run_formal_build

for name, value in _impl.items():
    if not name.startswith("__"):
        globals()[name] = value

globals()["_postprocess_formal_outputs"] = _postprocess_formal_outputs
globals()["_run_formal_build"] = _run_formal_build


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return _impl["main"](args)


if __name__ == "__main__":
    raise SystemExit(main())
