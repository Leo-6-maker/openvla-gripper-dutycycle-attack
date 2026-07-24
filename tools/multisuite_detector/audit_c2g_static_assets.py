#!/usr/bin/env python3
"""Read-only census of static LIBERO/BDDL task operators and MuJoCo names.

This tool never imports or creates a LIBERO environment. It scans existing text
task files and XML assets, records content hashes, and reports gaps between the
observed task/model vocabulary and the C2g Teacher-v2 static contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

LOGICAL_FORMS = frozenset({
    "and", "or", "not", "exists", "forall", "when", "imply", "implies",
    "ordered", "preference",
})
TASK_SUFFIXES = (".bddl", ".pddl", ".lisp")
FINGER_HINT_RE = re.compile(
    r"(?:finger|gripper|jaw|leftfinger|rightfinger|l_finger|r_finger|"
    r"finger_l|finger_r|finger1|finger2)",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_operator(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def tokenize_sexpr(text: str) -> list[str]:
    """Tokenize Lisp/BDDL while removing semicolon comments."""
    without_comments = re.sub(r";[^\n]*", "", text)
    return re.findall(r'"(?:\\.|[^"\\])*"|[()]|[^\s()]+', without_comments)


def parse_sexpr(text: str) -> list[Any]:
    """Parse one or more S-expressions, failing on unbalanced parentheses."""
    tokens = tokenize_sexpr(text)
    stack: list[list[Any]] = [[]]
    for token in tokens:
        if token == "(":
            node: list[Any] = []
            stack[-1].append(node)
            stack.append(node)
        elif token == ")":
            if len(stack) == 1:
                raise ValueError("unexpected closing parenthesis")
            stack.pop()
        else:
            stack[-1].append(token)
    if len(stack) != 1:
        raise ValueError("unclosed parenthesis")
    return stack[0]


def _goal_bodies(node: Any) -> Iterable[Any]:
    if not isinstance(node, list):
        return
    if node and isinstance(node[0], str) and normalize_operator(node[0]) == "goal":
        yield from node[1:]
        return
    for child in node:
        yield from _goal_bodies(child)


def _predicate_operators(node: Any) -> Iterable[str]:
    if not isinstance(node, list) or not node:
        return
    head = normalize_operator(node[0]) if isinstance(node[0], str) else ""
    if head in LOGICAL_FORMS:
        children = node[2:] if head in {"exists", "forall"} and len(node) > 2 else node[1:]
        for child in children:
            yield from _predicate_operators(child)
        return
    if head and not head.startswith("_") and not str(node[0]).startswith(("?", ":")):
        yield head
    for child in node[1:]:
        if isinstance(child, list):
            yield from _predicate_operators(child)


def extract_goal_operators(text: str) -> tuple[str, ...]:
    parsed = parse_sexpr(text)
    operators: list[str] = []
    for form in parsed:
        for goal in _goal_bodies(form):
            operators.extend(_predicate_operators(goal))
    return tuple(sorted(operators))


def _iter_files(roots: Sequence[Path], suffixes: Sequence[str]) -> list[tuple[int, Path, Path]]:
    files: list[tuple[int, Path, Path]] = []
    normalized_suffixes = {suffix.lower() for suffix in suffixes}
    for index, root in enumerate(roots):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in normalized_suffixes:
                files.append((index, root, path))
    return files


def _manifest_entry(root_index: int, root: Path, path: Path) -> dict[str, Any]:
    return {
        "root_index": root_index,
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _default_supported_operators() -> set[str]:
    from src.gripper_attack.c2g_teacher_v2_target_resolution import _OPERATOR_ROLES
    return set(_OPERATOR_ROLES)


def _default_finger_side(name: str) -> str:
    from src.gripper_attack.c2g_teacher_v2_contact_identity import finger_side
    return finger_side(name)


def audit_static_assets(
    bddl_roots: Sequence[Path],
    xml_roots: Sequence[Path],
    *,
    supported_operators: set[str] | None = None,
    finger_side_fn: Callable[[str], str] | None = None,
    require_bddl: bool = True,
    require_xml: bool = True,
) -> dict[str, Any]:
    """Build a deterministic read-only operator/name census."""
    supported = set(supported_operators) if supported_operators is not None else _default_supported_operators()
    finger_side_fn = finger_side_fn or _default_finger_side

    task_files = _iter_files(bddl_roots, TASK_SUFFIXES)
    xml_files = _iter_files(xml_roots, (".xml",))
    manifest: list[dict[str, Any]] = []
    task_errors: list[dict[str, str]] = []
    xml_errors: list[dict[str, str]] = []
    operator_counts: Counter[str] = Counter()
    per_task: list[dict[str, Any]] = []

    for root_index, root, path in task_files:
        entry = _manifest_entry(root_index, root, path)
        entry["kind"] = "task"
        manifest.append(entry)
        try:
            operators = extract_goal_operators(path.read_text(encoding="utf-8"))
            operator_counts.update(operators)
            per_task.append({
                "root_index": root_index,
                "relative_path": entry["relative_path"],
                "operators": list(operators),
                "unsupported_operators": sorted(set(operators) - supported),
            })
        except Exception as exc:
            task_errors.append({
                "root_index": str(root_index),
                "relative_path": entry["relative_path"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })

    name_counts: dict[str, Counter[str]] = {
        "body": Counter(), "geom": Counter(), "joint": Counter(), "site": Counter(),
    }
    finger_candidates: set[str] = set()
    finger_aliases: dict[str, set[str]] = {"left": set(), "right": set()}
    unresolved_finger_candidates: set[str] = set()

    for root_index, root, path in xml_files:
        entry = _manifest_entry(root_index, root, path)
        entry["kind"] = "xml"
        manifest.append(entry)
        try:
            xml_root = ET.parse(path).getroot()
            for element in xml_root.iter():
                tag = element.tag.rsplit("}", 1)[-1].lower()
                if tag not in name_counts:
                    continue
                name = str(element.attrib.get("name", "")).strip()
                if not name:
                    continue
                name_counts[tag][name] += 1
                if FINGER_HINT_RE.search(name):
                    finger_candidates.add(name)
                    side = finger_side_fn(name)
                    if side in finger_aliases:
                        finger_aliases[side].add(name)
                    else:
                        unresolved_finger_candidates.add(name)
        except Exception as exc:
            xml_errors.append({
                "root_index": str(root_index),
                "relative_path": entry["relative_path"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })

    manifest = sorted(manifest, key=lambda row: (row["kind"], row["root_index"], row["relative_path"]))
    manifest_digest = hashlib.sha256()
    for row in manifest:
        manifest_digest.update(
            f'{row["kind"]}|{row["root_index"]}|{row["relative_path"]}|'
            f'{row["size_bytes"]}|{row["sha256"]}\n'.encode("utf-8")
        )

    observed = set(operator_counts)
    unsupported = sorted(observed - supported)
    unobserved_supported = sorted(supported - observed)

    bddl_problems: list[str] = []
    if require_bddl and not task_files:
        bddl_problems.append("NO_TASK_FILES")
    if task_errors:
        bddl_problems.append("TASK_PARSE_ERRORS")
    if unsupported:
        bddl_problems.append("UNSUPPORTED_GOAL_OPERATORS")

    xml_problems: list[str] = []
    if require_xml and not xml_files:
        xml_problems.append("NO_XML_FILES")
    if xml_errors:
        xml_problems.append("XML_PARSE_ERRORS")
    if unresolved_finger_candidates:
        xml_problems.append("UNRESOLVED_FINGER_ALIASES")
    if require_xml and not finger_aliases["left"]:
        xml_problems.append("NO_LEFT_FINGER_ALIAS")
    if require_xml and not finger_aliases["right"]:
        xml_problems.append("NO_RIGHT_FINGER_ALIAS")

    return {
        "gate": "C2G_STATIC_TASK_ASSET_INVENTORY",
        "status": "PASS" if not bddl_problems and not xml_problems else "HOLD_WITH_GAPS",
        "boundaries": {
            "read_only": True,
            "libero_environment_created": False,
            "simulator_started": False,
            "openvla_loaded": False,
            "gpu_used": False,
        },
        "task_inventory": {
            "file_count": len(task_files),
            "parse_error_count": len(task_errors),
            "parse_errors": task_errors,
            "operator_counts": dict(sorted(operator_counts.items())),
            "observed_operators": sorted(observed),
            "supported_operators": sorted(supported),
            "unsupported_operators": unsupported,
            "supported_but_unobserved_operators": unobserved_supported,
            "per_task": per_task,
            "problems": bddl_problems,
            "status": "PASS" if not bddl_problems else "HOLD",
        },
        "xml_inventory": {
            "file_count": len(xml_files),
            "parse_error_count": len(xml_errors),
            "parse_errors": xml_errors,
            "name_counts": {
                kind: dict(sorted(counter.items())) for kind, counter in sorted(name_counts.items())
            },
            "finger_candidates": sorted(finger_candidates),
            "finger_aliases": {
                side: sorted(names) for side, names in sorted(finger_aliases.items())
            },
            "unresolved_finger_candidates": sorted(unresolved_finger_candidates),
            "problems": xml_problems,
            "status": "PASS" if not xml_problems else "HOLD",
        },
        "artifact_manifest": manifest,
        "artifact_manifest_sha256": manifest_digest.hexdigest(),
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(output)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl-root", action="append", default=[])
    parser.add_argument("--xml-root", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--allow-no-bddl", action="store_true")
    parser.add_argument("--allow-no-xml", action="store_true")
    args = parser.parse_args(argv)

    report = audit_static_assets(
        [Path(value) for value in args.bddl_root],
        [Path(value) for value in args.xml_root],
        require_bddl=not args.allow_no_bddl,
        require_xml=not args.allow_no_xml,
    )
    report["exact_command"] = " ".join(shlex.quote(value) for value in sys.argv)
    write_report(report, Path(args.output_json))
    print(json.dumps({
        "status": report["status"],
        "task_files": report["task_inventory"]["file_count"],
        "xml_files": report["xml_inventory"]["file_count"],
        "unsupported_operators": report["task_inventory"]["unsupported_operators"],
        "unresolved_finger_candidates": report["xml_inventory"]["unresolved_finger_candidates"],
    }, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
