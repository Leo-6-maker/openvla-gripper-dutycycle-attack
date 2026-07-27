"""Build a FIT-only, no-action object-state index map from the official LIBERO env.

This is static/runtime schema introspection only.  It does not load OpenVLA,
read protected episode payloads, replay actions, label episodes, or train a
student.  The map is derived from the actual environment object list and
observable output, not from BDDL text alone.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import inspect
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


OBJECT_STATE_WIDTH = 14
SCHEMA = "OBJECT_STATE_INDEX_MAP_V1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def finite_vector(value: Any, width: int) -> np.ndarray | None:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != width or not np.isfinite(array).all():
        return None
    return array


def quaternion_distance(left: Any, right_wxyz: Any) -> float:
    left_xyzw = finite_vector(left, 4)
    right = finite_vector(right_wxyz, 4)
    if left_xyzw is None or right is None:
        return math.inf
    right_xyzw = np.asarray([right[1], right[2], right[3], right[0]], dtype=float)
    left_xyzw = left_xyzw / np.linalg.norm(left_xyzw)
    right_xyzw = right_xyzw / np.linalg.norm(right_xyzw)
    dot = float(abs(np.dot(left_xyzw, right_xyzw)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def read_pilot_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"pilot manifest is not a regular file: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 40:
        raise RuntimeError("pilot manifest must contain exactly 40 records")
    if manifest.get("protected_payload_read") is not False:
        raise RuntimeError("pilot manifest does not prove protected payload exclusion")
    if manifest.get("no_model") is not True or manifest.get("no_rollout") is not True:
        raise RuntimeError("pilot manifest is not no-model/no-rollout")
    keys = {(str(row.get("suite")), int(row.get("task_id"))) for row in records}
    if len(keys) != 40 or {row[0] for row in keys} != {
        "libero_10", "libero_goal", "libero_object", "libero_spatial"
    }:
        raise RuntimeError("pilot task closure is not 4 suites x 10 tasks")
    return manifest, records


def read_collector_census(records: list[dict[str, Any]]) -> dict[str, Any]:
    census: dict[str, dict[str, Any]] = {}
    for record in records:
        episode_root = Path(record["source_episode_root"])
        metadata_path = episode_root / "episode_metadata.json"
        if metadata_path.is_symlink() or not metadata_path.is_file():
            raise RuntimeError(f"missing episode metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_map = metadata.get("collector_source_sha256")
        if not isinstance(source_map, dict):
            raise RuntimeError(f"collector source map missing: {metadata_path}")
        worker_sha = str(source_map.get("official_clean_worker.py", ""))
        if len(worker_sha) != 64:
            raise RuntimeError(f"invalid collector worker SHA: {metadata_path}")
        entry = census.setdefault(
            worker_sha,
            {
                "episode_count": 0,
                "step_count": 0,
                "collector_git_heads": set(),
                "collector_source_sha256": set(),
            },
        )
        entry["episode_count"] += 1
        entry["step_count"] += int(record["observed_step_count"])
        entry["collector_git_heads"].add(str(metadata.get("collector_git_head")))
        entry["collector_source_sha256"].add(json.dumps(source_map, sort_keys=True))
    return {
        worker_sha: {
            **entry,
            "collector_git_heads": sorted(entry["collector_git_heads"]),
            "collector_source_sha256": sorted(entry["collector_source_sha256"]),
        }
        for worker_sha, entry in sorted(census.items())
    }


def task_map(suite: str, task_index: int, libero_root: Path) -> dict[str, Any]:
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark = get_benchmark(suite)(0)
    task = benchmark.get_task(task_index)
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    if bddl_path.is_symlink() or not bddl_path.is_file():
        raise RuntimeError(f"BDDL file is not regular: {bddl_path}")

    # Keep the official environment quiet in the receipt; exceptions are still
    # propagated.  No action is sent and no model is imported here.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_heights=64,
            camera_widths=64,
            render_gpu_device_id=-1,
            has_renderer=False,
            has_offscreen_renderer=False,
            horizon=2,
        )
        try:
            observation = env.reset()
            inner = env.env
            parsed_objects: list[str] = []
            for values in inner.parsed_problem.get("objects", {}).values():
                parsed_objects.extend(str(value) for value in values)
            actual_objects = [str(obj.name) for obj in inner.objects]
            state = np.asarray(observation.get("object-state", []), dtype=float).reshape(-1)
            if parsed_objects != actual_objects:
                raise RuntimeError(
                    f"object order mismatch for {suite}/task_{task_index:02d}: "
                    f"parsed={parsed_objects}, env={actual_objects}"
                )
            if state.size != len(actual_objects) * OBJECT_STATE_WIDTH:
                raise RuntimeError(
                    f"object-state width mismatch for {suite}/task_{task_index:02d}: "
                    f"{state.size} != {len(actual_objects)}*{OBJECT_STATE_WIDTH}"
                )

            objects: list[dict[str, Any]] = []
            position_errors: list[float] = []
            rotation_errors: list[float] = []
            for index, name in enumerate(actual_objects):
                start = index * OBJECT_STATE_WIDTH
                end = start + OBJECT_STATE_WIDTH
                body_id = int(inner.obj_body_id[name])
                chunk = state[start:end]
                position_error = float(
                    np.linalg.norm(chunk[:3] - np.asarray(inner.sim.data.body_xpos[body_id], dtype=float))
                )
                rotation_error = quaternion_distance(
                    chunk[3:7], inner.sim.data.body_xquat[body_id]
                )
                position_errors.append(position_error)
                rotation_errors.append(rotation_error)
                objects.append(
                    {
                        "object_name": name,
                        "object_index": index,
                        "slice_start": start,
                        "slice_end_exclusive": end,
                        "position": [start, start + 3],
                        "quaternion_xyzw": [start + 3, start + 7],
                        "object_to_eef_position": [start + 7, start + 10],
                        "object_to_eef_quaternion_xyzw": [start + 10, start + 14],
                        "body_id": body_id,
                        "body_name": str(inner.sim.model.body(body_id).name),
                    }
                )
            task_source = Path(inspect.getfile(type(task)))
            domain_source = Path(inspect.getfile(type(inner)))
            return {
                "suite": suite,
                "task_id": task_index,
                "task_name": str(getattr(task, "language", "")),
                "bddl_path": str(bddl_path.resolve()),
                "bddl_sha256": sha256_file(bddl_path),
                "task_class": f"{type(task).__module__}.{type(task).__name__}",
                "task_source_path": str(task_source.resolve()),
                "task_source_sha256": sha256_file(task_source),
                "domain_class": f"{type(inner).__module__}.{type(inner).__name__}",
                "domain_source_path": str(domain_source.resolve()),
                "domain_source_sha256": sha256_file(domain_source),
                "libero_git_head": git_head(libero_root),
                "parsed_object_order": parsed_objects,
                "environment_object_order": actual_objects,
                "object_state_width": OBJECT_STATE_WIDTH,
                "object_count": len(actual_objects),
                "objects": objects,
                "reset_body_origin_position_max_error_m": max(position_errors or [math.inf]),
                "reset_body_origin_rotation_max_error_rad": max(rotation_errors or [math.inf]),
                "mapping_status": "PASS",
            }
        finally:
            env.close()


def write_sealed_root(output_root: Path, payload: dict[str, Any]) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError(f"output root already exists: {output_root}")
    staging = output_root.parent / f".{output_root.name}.staging.{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise RuntimeError(f"staging root already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        (staging / "OBJECT_STATE_INDEX_MAP_V1.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rows = []
        for task in payload["tasks"]:
            for obj in task["objects"]:
                rows.append(
                    {
                        "suite": task["suite"],
                        "task_id": task["task_id"],
                        "object_name": obj["object_name"],
                        "object_index": obj["object_index"],
                        "slice_start": obj["slice_start"],
                        "slice_end_exclusive": obj["slice_end_exclusive"],
                        "bddl_sha256": task["bddl_sha256"],
                        "task_source_sha256": task["task_source_sha256"],
                        "mapping_status": task["mapping_status"],
                    }
                )
        with (staging / "OBJECT_STATE_INDEX_MAP_V1.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest = {
            "schema": SCHEMA,
            "status": "DERIVED_STATIC_SCHEMA_SEALED",
            "episode_payload_read": False,
            "action_replay": False,
            "model_inference": False,
            "protected_payload_read": False,
            "task_count": len(payload["tasks"]),
            "payload_files": ["OBJECT_STATE_INDEX_MAP_V1.json", "OBJECT_STATE_INDEX_MAP_V1.csv"],
        }
        (staging / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        files = sorted(path for path in staging.rglob("*") if path.is_file())
        lines = []
        for path in files:
            lines.append(f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}")
        (staging / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (staging / "SHA256SUMS.sha256").write_text(
            f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n", encoding="utf-8"
        )
        os.rename(staging, output_root)
    except Exception:
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        staging.rmdir()
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest, records = read_pilot_manifest(args.pilot_manifest)
    census = read_collector_census(records)
    tasks = []
    for suite, task_index in sorted({(str(r["suite"]), int(r["task_id"])) for r in records}):
        tasks.append(task_map(suite, task_index, args.libero_root))
    payload = {
        "schema": SCHEMA,
        "status": "DERIVED_STATIC_SCHEMA_SEALED",
        "source": "official LIBERO environment object list and object-state observable",
        "pilot_manifest_sha256": sha256_file(args.pilot_manifest),
        "pilot_manifest_schema": manifest.get("schema"),
        "collector_variant_census": census,
        "libero_root": str(args.libero_root.resolve()),
        "libero_git_head": git_head(args.libero_root),
        "task_count": len(tasks),
        "tasks": tasks,
        "episode_payload_read": False,
        "action_replay": False,
        "model_inference": False,
        "protected_payload_read": False,
    }
    write_sealed_root(args.output_root, payload)
    print(json.dumps({"output_root": str(args.output_root), "task_count": len(tasks), "status": payload["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
