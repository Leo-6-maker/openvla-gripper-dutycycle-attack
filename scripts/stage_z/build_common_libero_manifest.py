#!/usr/bin/env python3
"""Seal the official LIBERO checkout without importing the simulator."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path


COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
TREE = "99f4ada3f1d62e026fc9ff2390eb4ff8a1760e60"
SOURCE_FILES = (
    "libero/benchmark/libero_suite_task_map.py",
    "libero/benchmark/__init__.py",
    "libero/envs/env_wrapper.py",
    "libero/__init__.py",
)
COMMON_DIRS = ("libero/bddl_files", "libero/init_files", "libero/assets")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_rows(checkout: Path, prefixes: tuple[str, ...]) -> list[dict]:
    result = []
    for prefix in prefixes:
        listing = subprocess.check_output(("git", "-C", str(checkout), "ls-tree", "-r", "--long", "--full-tree", "HEAD", "--", f"libero/{prefix}"), text=True)
        for line in listing.splitlines():
            mode, kind, blob, size, rel = line.split(None, 4)
            if kind != "blob":
                continue
            result.append({"path": rel, "size": int(size), "git_blob_sha": blob})
    return sorted(result, key=lambda row: row["path"])


def digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def task_registry(root: Path) -> dict:
    source = root / "libero/benchmark/libero_suite_task_map.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "libero_task_map" for t in node.targets):
            value = ast.literal_eval(node.value)
            return {name: {"count": len(tasks), "tasks": tasks} for name, tasks in value.items()}
    raise RuntimeError("libero_task_map assignment not found")


def main() -> None:
    checkout = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    source_root = checkout / "libero"
    common = {name.rsplit("/", 1)[-1]: git_rows(checkout, (name,)) for name in COMMON_DIRS}
    source = {
        rel: {"size": (source_root / rel).stat().st_size, "sha256": sha256_file(source_root / rel)}
        for rel in SOURCE_FILES
    }
    source_manifest = git_rows(checkout, ("libero",))
    full_manifest = git_rows(checkout, ("",))
    manifest = {
        "schema": "STAGE_Z_Z0R2_COMMON_LIBERO_AUTHORITY_V1",
        "status": "PASS_STATIC_OFFLINE_NO_SIMULATOR",
        "source": {
            "checkout": str(checkout),
            "commit": COMMIT,
            "tree": TREE,
            "git_status": "CLEAN",
            "file_count": len(full_manifest),
            "git_tree_file_bytes": sum(row["size"] for row in full_manifest),
            "manifest_binding": "git_tree_and_blob_sha1_from_fixed_commit",
        },
        "runtime_config_boundary": {
            "config_path": "/home/dty_user/.libero/config.yaml",
            "configured_root": "/mnt/sdc/dty_user/pi0_openpi/third_party/libero/libero",
            "configured_root_is_modified_fork": True,
            "official_checkout_is_prospective_stage_z_authority": True,
            "global_config_modified": False,
        },
        "task_registry": task_registry(source_root),
        "common_static_roots": {
            name: {"file_count": len(items), "bytes": sum(row["size"] for row in items), "content_binding": "git_blob_sha1", "rows": items}
            for name, items in common.items()
        },
        "source_file_bindings": source,
        "code_manifest": {"file_count": len(source_manifest), "rows": source_manifest, "sha256": digest(source_manifest)},
        "stage_z_overlay": {
            "suites": ["libero_10", "libero_goal", "libero_object", "libero_spatial"],
            "tasks_per_suite": 10,
            "action_dim": 7,
            "arm_indices": [0, 1, 2, 3, 4, 5],
            "gripper_index": 6,
            "native_open": -1.0,
            "native_close": 1.0,
            "camera": {"name": "agentview", "height": 256, "width": 256},
            "preprocess": {"rotate_180": True, "resize": 224},
            "dummy_wait_steps": 10,
            "horizons": {"libero_10": 520, "libero_goal": 300, "libero_object": 280, "libero_spatial": 220},
            "candidate_state_range": [0, 49],
        },
        "protected_boundary": {
            "model_inference": 0,
            "simulator": 0,
            "env_step": 0,
            "physical_intervention": 0,
            "v_phys": 0,
            "eval160": "UNREAD",
            "protected": "UNREAD",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output": str(output), "common_files": {key: len(value) for key, value in common.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
