"""Audit the official FIT object-state task decoder without training or replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any


OBJECT_STATE_WIDTH = 14
OBJECT_COMPONENTS = ("pos", "quat", "to_eef_pos", "to_eef_quat")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_bddl_objects(text: str) -> list[dict[str, str]]:
    """Parse the ordered :objects section; do not infer from labels or results."""

    match = re.search(r"\(:objects\s*(.*?)\n\s*\)\s*\n", text, flags=re.DOTALL)
    if not match:
        raise ValueError("BDDL objects section missing")
    objects: list[dict[str, str]] = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        item = re.fullmatch(r"([A-Za-z0-9_ ]+)\s+-\s+([A-Za-z0-9_]+)", line)
        if not item:
            raise ValueError(f"malformed BDDL object row: {line}")
        names = item.group(1).split()
        objects.extend({"name": name, "category": item.group(2)} for name in names)
    if not objects:
        raise ValueError("BDDL objects section is empty")
    return objects


def parse_bddl_interest(text: str) -> list[str]:
    match = re.search(r"\(:obj_of_interest\s*(.*?)\n\s*\)\s*\n", text, flags=re.DOTALL)
    if not match:
        return []
    return [line.strip() for line in match.group(1).splitlines() if line.strip()]


def build_object_slices(objects: list[dict[str, str]]) -> list[dict[str, Any]]:
    slices: list[dict[str, Any]] = []
    for index, obj in enumerate(objects):
        base = index * OBJECT_STATE_WIDTH
        slices.append(
            {
                "object_index": index,
                "object_name": obj["name"],
                "object_category": obj["category"],
                "offset_start": base,
                "offset_end_exclusive": base + OBJECT_STATE_WIDTH,
                "pos": [base, base + 3],
                "quat": [base + 3, base + 7],
                "to_eef_pos": [base + 7, base + 10],
                "to_eef_quat": [base + 10, base + 14],
            }
        )
    return slices


def _fit_rows(registry_csv: Path) -> list[dict[str, Any]]:
    with registry_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fit = [row for row in rows if row.get("split") == "FIT_TRAIN"]
    if len(fit) != 800:
        raise ValueError(f"expected exactly 800 FIT_TRAIN rows, got {len(fit)}")
    keys = [str(row.get("canonical_parent_key", "")) for row in fit]
    if len(set(keys)) != 800:
        raise ValueError("FIT registry contains duplicate identities")
    for row in fit:
        expected = f"{row['suite']}/task_{int(row['task_idx']):02d}/state_{int(row['state_id']):02d}"
        if row["canonical_parent_key"] != expected or int(row["state_id"]) not in range(20):
            raise ValueError(f"invalid FIT identity: {row.get('canonical_parent_key')}")
        if not Path(row["selected_artifact_root"]).is_dir():
            raise ValueError(f"artifact root missing: {row['canonical_parent_key']}")
    return sorted(fit, key=lambda row: row["canonical_parent_key"])


def _task_specs() -> list[dict[str, Any]]:
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark

    specs: list[dict[str, Any]] = []
    for suite in ("libero_object", "libero_spatial", "libero_goal", "libero_10"):
        benchmark = get_benchmark(suite)(0)
        for task_idx in range(10):
            task = benchmark.get_task(task_idx)
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            if not bddl.is_file():
                raise ValueError(f"BDDL file missing: {bddl}")
            text = bddl.read_text(encoding="utf-8")
            objects = parse_bddl_objects(text)
            specs.append(
                {
                    "suite": suite,
                    "task_idx": task_idx,
                    "task_name": task.name,
                    "task_language": task.language,
                    "problem_folder": task.problem_folder,
                    "bddl_file": task.bddl_file,
                    "bddl_path": str(bddl),
                    "bddl_sha256": sha256_file(bddl),
                    "objects": objects,
                    "obj_of_interest": parse_bddl_interest(text),
                    "object_state_width_expected": len(objects) * OBJECT_STATE_WIDTH,
                }
            )
    if len(specs) != 40:
        raise ValueError(f"expected 40 task specs, got {len(specs)}")
    return specs


def _sidecar_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "privileged_teacher_sidecar.jsonl"
    if not path.is_file():
        raise ValueError(f"privileged sidecar missing: {root}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_binding(args: argparse.Namespace) -> dict[str, Any]:
    collector = args.collector_script.resolve()
    domain = args.libero_domain_source.resolve()
    robosuite = args.robosuite_base_source.resolve()
    for path in (collector, domain, robosuite):
        if not path.is_file():
            raise ValueError(f"source file missing: {path}")
    collector_text = collector.read_text(encoding="utf-8")
    domain_text = domain.read_text(encoding="utf-8")
    robosuite_text = robosuite.read_text(encoding="utf-8")
    required = {
        "collector_reads_object_state": 'obs.get("object-state", [])' in collector_text,
        "domain_iterates_objects_in_order": "for (i, obj) in enumerate(self.objects):" in domain_text,
        "domain_appends_four_object_sensors": "sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]" in domain_text,
        "domain_disables_world_pose_observable": "active=False" in domain_text and "world_pose_in_gripper" in domain_text,
        "robosuite_concatenates_by_modality_order": "obs_by_modality[modality].append" in robosuite_text and "np.concatenate(obs, axis=-1)" in robosuite_text,
    }
    if not all(required.values()):
        raise ValueError(f"object-state source binding incomplete: {required}")
    return {
        "collector_script": str(collector),
        "collector_script_sha256": sha256_file(collector),
        "libero_domain_source": str(domain),
        "libero_domain_source_sha256": sha256_file(domain),
        "robosuite_base_source": str(robosuite),
        "robosuite_base_source_sha256": sha256_file(robosuite),
        "object_state_width_per_object": OBJECT_STATE_WIDTH,
        "object_components_in_order": list(OBJECT_COMPONENTS),
        "source_assertions": required,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    rows = _fit_rows(args.registry_csv.resolve())
    specs = {(item["suite"], item["task_idx"]): item for item in _task_specs()}
    source_binding = _source_binding(args)
    by_task: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault((row["suite"], int(row["task_idx"])), []).append(row)
    task_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for key, spec in sorted(specs.items()):
        task_rows.append(
            {
                "suite": spec["suite"],
                "task_idx": spec["task_idx"],
                "task_name": spec["task_name"],
                "task_language": spec["task_language"],
                "bddl_file": spec["bddl_file"],
                "bddl_sha256": spec["bddl_sha256"],
                "object_count": len(spec["objects"]),
                "expected_object_state_width": spec["object_state_width_expected"],
                "obj_of_interest": json.dumps(spec["obj_of_interest"], sort_keys=True),
                "status": "PASS",
                "failure_reason": "",
            }
        )
        for item in build_object_slices(spec["objects"]):
            slice_rows.append({"suite": spec["suite"], "task_idx": spec["task_idx"], **item})
        for row in by_task.get(key, []):
            root = Path(row["selected_artifact_root"])
            try:
                metadata = json.loads((root / "episode_metadata.json").read_text(encoding="utf-8"))
                sidecars = _sidecar_rows(root)
                expected_width = spec["object_state_width_expected"]
                widths = {len(item.get("object_state", [])) for item in sidecars}
                if widths != {expected_width}:
                    raise ValueError(f"object_state_widths={sorted(widths)} expected={expected_width}")
                if metadata.get("task_name") != spec["task_name"] or metadata.get("task_language") != spec["task_language"]:
                    raise ValueError("task metadata does not match official benchmark task")
                if len(sidecars) != int(metadata.get("steps", len(sidecars))):
                    raise ValueError("sidecar step count does not match metadata")
                if [int(item.get("step", -1)) for item in sidecars] != list(range(len(sidecars))):
                    raise ValueError("sidecar steps are not contiguous")
            except Exception as exc:
                task_rows[-1]["status"] = "ABSTAIN_DECODER_HOLD"
                task_rows[-1]["failure_reason"] = str(exc)
                failures.append({"canonical_parent_key": row["canonical_parent_key"], "reason": str(exc)})
    status = "PASS_TASK_CONDITIONAL_DECODER" if not failures and all(item["status"] == "PASS" for item in task_rows) else "ABSTAIN_DECODER_HOLD"
    summary = {
        "schema": "OFFICIAL_V3_PHYSICS_TASK_DECODER_V1",
        "status": status,
        "task_count": len(task_rows),
        "fit_identity_count": len(rows),
        "task_pass_count": sum(item["status"] == "PASS" for item in task_rows),
        "task_hold_count": sum(item["status"] != "PASS" for item in task_rows),
        "decoder_basis": "official BDDL object order plus sealed collector/libero/robosuite source binding",
        "object_state_width_per_object": OBJECT_STATE_WIDTH,
        "object_components_in_order": list(OBJECT_COMPONENTS),
        "failures": failures,
        "source_binding": source_binding,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    try:
        staging.mkdir(parents=True)
        with (staging / "task_decoder.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = list(task_rows[0])
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(task_rows)
        with (staging / "object_slices.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = list(slice_rows[0])
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(slice_rows)
        _atomic_text(staging / "source_bindings.json", json.dumps(source_binding, indent=2, sort_keys=True) + "\n")
        _atomic_text(staging / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
        _atomic_text(staging / "audit_report.md", "# Official V3 Physics task decoder\n\n```json\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n```\n")
        payload = sorted(path.name for path in staging.iterdir() if path.is_file())
        _atomic_text(staging / "SHA256SUMS", "".join(f"{sha256_file(staging / name)}  {name}\n" for name in payload))
        _atomic_text(staging / "SHA256SUMS.sha256", f"{sha256_file(staging / 'SHA256SUMS')}  SHA256SUMS\n")
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--collector-script", type=Path, required=True)
    parser.add_argument("--libero-domain-source", type=Path, required=True)
    parser.add_argument("--robosuite-base-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    print(json.dumps(audit(parser.parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
