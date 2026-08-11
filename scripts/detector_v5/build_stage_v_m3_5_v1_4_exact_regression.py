#!/usr/bin/env python3
"""Build a clean-source, CPU-only exact regression receipt for V1.4."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


COUNTERS = {"protected_reads": 0, "eval160_reads": 0, "attack_rollouts": 0, "vis_pgd_attack_rollouts": 0}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit("MANIFEST_OBJECT_REQUIRED")
    return value


def _expand(repo: Path, patterns: list[str]) -> list[str]:
    paths = {path.relative_to(repo).as_posix() for pattern in patterns for path in repo.glob(pattern) if path.is_file()}
    return sorted(paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-python", default=sys.executable)
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    manifest_path = args.manifest.resolve()
    manifest = _load(manifest_path)
    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"REFUSE_OVERWRITE:{output}")
    test_files = [str(path) for path in manifest.get("test_files", [])]
    binding_files = _expand(repo, [str(pattern) for pattern in manifest.get("binding_globs", [])])
    if not test_files or not binding_files or any(not (repo / path).is_file() for path in test_files + binding_files):
        raise SystemExit("REGRESSION_SCOPE_INVALID")
    if _git(repo, "status", "--porcelain"):
        raise SystemExit("SOURCE_NOT_CLEAN_BEFORE_TEST")
    output.mkdir(parents=True, exist_ok=False)
    compile_rows = []
    compile_status = "PASS"
    for relative in binding_files:
        try:
            compile((repo / relative).read_text(encoding="utf-8"), relative, "exec")
            compile_rows.append(f"PASS {relative}")
        except BaseException as exc:
            compile_status = "FAIL"
            compile_rows.append(f"FAIL {relative} {type(exc).__name__}:{exc}")
    compile_log = output / "py_compile.log"
    compile_log.write_text("\n".join(compile_rows) + "\n", encoding="utf-8")
    junit = output / "pytest.junit.xml"
    pytest_log = output / "pytest.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    with pytest_log.open("w", encoding="utf-8") as handle:
        run = subprocess.run([args.runtime_python, "-m", "pytest", "-q", *test_files, f"--junitxml={junit}"], cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    root = ET.parse(junit).getroot()
    suites = [root] if root.tag.rsplit("}", 1)[-1] == "testsuite" else list(root.findall(".//testsuite"))
    counts = {name: sum(int(node.attrib.get(name, 0)) for node in suites) for name in ("tests", "failures", "errors", "skipped")}
    collected = counts["tests"]
    failed, errors, skipped = counts["failures"], counts["errors"], counts["skipped"]
    passed = collected - failed - errors - skipped
    source_status = _git(repo, "status", "--porcelain")
    tested_bindings = {relative: _sha(repo / relative) for relative in binding_files}
    expected = int(manifest.get("expected_collected", -1))
    status = "PASS" if (run.returncode == 0 and compile_status == "PASS" and collected == expected and passed + skipped == collected and failed == errors == 0 and source_status == "" and env["CUDA_VISIBLE_DEVICES"] == "") else "FAIL"
    receipt = {
        "schema": "STAGE_V_M3_5_EXACT_A800_REGRESSION_RECEIPT_V1",
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_python": str(args.runtime_python),
        "source_commit": _git(repo, "rev-parse", "HEAD"),
        "source_tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "source_status_porcelain": source_status,
        "cuda_visible_devices": "",
        "test_files": test_files,
        "tested_bindings": tested_bindings,
        "expected_collected": expected,
        "collected": collected,
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "errors": errors,
        "deselected": 0,
        "pytest_returncode": run.returncode,
        "pytest_log_sha256": _sha(pytest_log),
        "junit_xml_sha256": _sha(junit),
        "py_compile_status": compile_status,
        "protected_counters": dict(COUNTERS),
        "binding_manifest": str(manifest_path),
        "binding_manifest_sha256": _sha(manifest_path),
        "receipt_builder": str(Path(__file__).resolve()),
        "receipt_builder_sha256": _sha(Path(__file__).resolve()),
    }
    receipt_path = output / "STAGE_V_M3_5_EXACT_A800_REGRESSION_RECEIPT_V1.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    receipt_path.with_name(receipt_path.name + ".sha256").write_text(f"{_sha(receipt_path)}  {receipt_path.name}\n", encoding="utf-8")
    print(json.dumps({"status": status, "collected": collected, "passed": passed, "skipped": skipped, "failed": failed, "errors": errors}, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
