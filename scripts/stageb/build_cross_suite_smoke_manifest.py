#!/usr/bin/env python3
"""Build the static SC5 cross-suite task inventory and minimal smoke manifest.

This script is CPU/static only. It does not load OpenVLA, launch LIBERO, run
PGD, or touch GPUs. The embedded task list is copied from the official LIBERO
benchmark query performed during the 2026-06-19 server audit.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


TASKS: dict[str, list[tuple[int, str, str]]] = {
    "libero_spatial": [
        (0, "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate", "pick up the black bowl between the plate and the ramekin and place it on the plate"),
        (1, "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate", "pick up the black bowl next to the ramekin and place it on the plate"),
        (2, "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate", "pick up the black bowl from table center and place it on the plate"),
        (3, "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate", "pick up the black bowl on the cookie box and place it on the plate"),
        (4, "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate", "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"),
        (5, "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate", "pick up the black bowl on the ramekin and place it on the plate"),
        (6, "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate", "pick up the black bowl next to the cookie box and place it on the plate"),
        (7, "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate", "pick up the black bowl on the stove and place it on the plate"),
        (8, "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate", "pick up the black bowl next to the plate and place it on the plate"),
        (9, "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate", "pick up the black bowl on the wooden cabinet and place it on the plate"),
    ],
    "libero_goal": [
        (0, "open_the_middle_drawer_of_the_cabinet", "open the middle drawer of the cabinet"),
        (1, "put_the_bowl_on_the_stove", "put the bowl on the stove"),
        (2, "put_the_wine_bottle_on_top_of_the_cabinet", "put the wine bottle on top of the cabinet"),
        (3, "open_the_top_drawer_and_put_the_bowl_inside", "open the top drawer and put the bowl inside"),
        (4, "put_the_bowl_on_top_of_the_cabinet", "put the bowl on top of the cabinet"),
        (5, "push_the_plate_to_the_front_of_the_stove", "push the plate to the front of the stove"),
        (6, "put_the_cream_cheese_in_the_bowl", "put the cream cheese in the bowl"),
        (7, "turn_on_the_stove", "turn on the stove"),
        (8, "put_the_bowl_on_the_plate", "put the bowl on the plate"),
        (9, "put_the_wine_bottle_on_the_rack", "put the wine bottle on the rack"),
    ],
    "libero_10": [
        (0, "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket", "put both the alphabet soup and the tomato sauce in the basket"),
        (1, "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket", "put both the cream cheese box and the butter in the basket"),
        (2, "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it", "turn on the stove and put the moka pot on it"),
        (3, "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it", "put the black bowl in the bottom drawer of the cabinet and close it"),
        (4, "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate", "put the white mug on the left plate and put the yellow and white mug on the right plate"),
        (5, "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy", "pick up the book and place it in the back compartment of the caddy"),
        (6, "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate", "put the white mug on the plate and put the chocolate pudding to the right of the plate"),
        (7, "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket", "put both the alphabet soup and the cream cheese box in the basket"),
        (8, "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove", "put both moka pots on the stove"),
        (9, "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it", "put the yellow and white mug in the microwave and close it"),
    ],
}


def classify_task(suite: str, task_name: str, instruction: str) -> dict[str, str]:
    text = f"{task_name} {instruction}".lower()
    reason = []
    if "push" in text:
        mechanism = "planar_or_push"
        eligible = "false"
        expected = "0"
        reason.append("push_not_gripper_duty")
    elif ("pick up" in instruction and "place" in instruction and "both" not in text) or (
        instruction.startswith("put ") and "both" not in text and " and " not in instruction
    ):
        mechanism = "single_object_pick_place"
        eligible = "true"
        expected = "1"
        reason.append("single_grasp_place_or_put")
    elif any(word in text for word in ["drawer", "stove", "microwave", "turn on", "close it"]):
        mechanism = "articulated_object"
        eligible = "false"
        expected = "0"
        reason.append("articulated_or_appliance_interaction")
    elif "both" in text:
        mechanism = "multi_object_transfer"
        eligible = "event_level_audit_only"
        expected = "2"
        reason.append("multi_object_or_multi_stage")
    else:
        mechanism = "unknown_or_low_signal"
        eligible = "false"
        expected = "0"
        reason.append("static_parser_low_confidence")

    parser_confidence = "high" if mechanism in {"single_object_pick_place", "articulated_object", "planar_or_push"} else "medium"
    manual = "true" if mechanism in {"multi_object_transfer", "unknown_or_low_signal"} else "false"
    return {
        "mechanism_type": mechanism,
        "eligible_for_gripper_duty": eligible,
        "expected_event_count": expected,
        **resolve_primary_entities(instruction, mechanism),
        "object_names": infer_objects(instruction),
        "target_names": infer_targets(instruction),
        "parser_confidence": parser_confidence,
        "reason": ";".join(reason),
        "manual_review_required": manual,
    }


def resolve_primary_entities(instruction: str, mechanism: str) -> dict[str, str]:
    text = instruction.lower()
    objects = infer_objects(instruction).split("|") if infer_objects(instruction) else []
    primary_object = ""
    primary_target = ""
    secondary_objects: list[str] = []
    status = "ABSTAIN"
    source = "static_instruction_heuristic"

    if mechanism == "single_object_pick_place":
        if "black_bowl" in objects:
            primary_object = "black_bowl"
        elif "cream_cheese" in objects:
            primary_object = "cream_cheese"
        elif "wine_bottle" in objects:
            primary_object = "wine_bottle"
        elif "book" in objects:
            primary_object = "book"
        elif "bowl" in objects:
            primary_object = "bowl"
        elif objects:
            primary_object = objects[0]

        if " on the plate" in text or "place it on the plate" in text:
            primary_target = "plate"
        elif " in the bowl" in text:
            primary_target = "bowl"
        elif " on the stove" in text:
            primary_target = "stove"
        elif " on top of the cabinet" in text:
            primary_target = "cabinet_top"
        elif " on the rack" in text:
            primary_target = "rack"
        elif " caddy" in text:
            primary_target = "caddy"
        elif "basket" in text:
            primary_target = "basket"
        status = "RESOLVED" if primary_object and primary_target else "NEEDS_MANUAL_REVIEW"
    elif mechanism == "multi_object_transfer":
        primary_object = objects[0] if objects else ""
        secondary_objects = objects[1:]
        primary_target = "basket" if "basket" in text else ""
        status = "EVENT_LEVEL_ONLY"
    elif mechanism in {"articulated_object", "planar_or_push"}:
        status = "ABSTAIN_UNSUPPORTED"

    return {
        "primary_object": primary_object,
        "primary_target": primary_target,
        "secondary_objects": "|".join(secondary_objects),
        "resolver_status": status,
        "resolver_source": source,
    }


def infer_objects(instruction: str) -> str:
    objects = []
    for phrase in [
        "black bowl", "cream cheese", "alphabet soup", "tomato sauce", "butter",
        "wine bottle", "book", "white mug", "yellow and white mug",
        "chocolate pudding", "moka pot", "plate", "bowl",
    ]:
        if phrase in instruction.lower():
            objects.append(phrase.replace(" ", "_"))
    return "|".join(dict.fromkeys(objects))


def infer_targets(instruction: str) -> str:
    targets = []
    for phrase in ["plate", "basket", "stove", "cabinet", "bowl", "rack", "caddy", "microwave", "drawer"]:
        if phrase in instruction.lower():
            targets.append(phrase)
    return "|".join(dict.fromkeys(targets))


def build_task_inventory() -> list[dict[str, str]]:
    rows = []
    for suite, tasks in TASKS.items():
        for idx, name, instruction in tasks:
            row = {
                "suite": suite,
                "task_idx": str(idx),
                "task_name": name,
                "instruction": instruction,
            }
            row.update(classify_task(suite, name, instruction))
            rows.append(row)
    return rows


def build_smoke_manifest(protocol: dict) -> list[dict[str, str]]:
    inventory = {(r["suite"], int(r["task_idx"])): r for r in build_task_inventory()}
    rows = []
    for suite, spec in protocol["smoke_matrix"]["suites"].items():
        for task_idx in spec["task_indices"]:
            inv = inventory[(suite, int(task_idx))]
            for state_id in spec["states"]:
                rows.append({
                    "suite": suite,
                    "task_idx": str(task_idx),
                    "state_id": str(state_id),
                    "condition": "CLEAN",
                    "selection_role": "mechanism_coverage_preregistered",
                    "mechanism_type": inv["mechanism_type"],
                    "eligible_for_gripper_duty": inv["eligible_for_gripper_duty"],
                    "primary_object": inv["primary_object"],
                    "primary_target": inv["primary_target"],
                    "secondary_objects": inv["secondary_objects"],
                    "resolver_status": inv["resolver_status"],
                    "resolver_source": inv["resolver_source"],
                    "requires_clean_success_gate": "true",
                    "requires_invalid_feature_steps_zero": "true",
                    "attack_allowed_in_phase1": "false",
                    "instruction": inv["instruction"],
                })
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_protocol(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return parse_simple_yaml(text)


def parse_simple_yaml(text: str) -> dict:
    """Parse the limited YAML subset used by sc5_cross_suite_protocol_v1."""

    def parse_value(raw: str):
        raw = raw.strip()
        if raw in {"true", "false"}:
            return raw == "true"
        if raw.startswith("[") and raw.endswith("]"):
            body = raw[1:-1].strip()
            if not body:
                return []
            return [parse_value(part.strip()) for part in body.split(",")]
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            return raw

    raw_lines = [line.rstrip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    root: dict = {}
    stack: list[tuple[int, object]] = [(-1, root)]
    for idx, line in enumerate(raw_lines):
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"list item without list parent: {line}")
            parent.append(parse_value(content[2:]))
            continue
        key, _, rest = content.partition(":")
        if not _:
            raise ValueError(f"unsupported YAML line: {line}")
        key = key.strip()
        rest = rest.strip()
        if rest:
            value = parse_value(rest)
        else:
            next_content = ""
            for future in raw_lines[idx + 1:]:
                next_indent = len(future) - len(future.lstrip(" "))
                if next_indent > indent:
                    next_content = future.strip()
                    break
            value = [] if next_content.startswith("- ") else {}
        if not isinstance(parent, dict):
            raise ValueError(f"mapping entry without mapping parent: {line}")
        parent[key] = value
        if isinstance(value, (dict, list)):
            stack.append((indent, value))
    return root


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="configs/sc5_cross_suite_protocol_v1.yaml")
    ap.add_argument("--tables_dir", default="tables")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--manifest_only", action="store_true")
    ap.add_argument("--no_gpu", action="store_true")
    args = ap.parse_args(argv)
    if not args.no_gpu:
        raise SystemExit("--no_gpu is required for this static manifest builder")

    protocol = load_protocol(Path(args.protocol))
    tables = Path(args.tables_dir)
    smoke = build_smoke_manifest(protocol)
    if not args.manifest_only:
        write_csv(tables / "cross_suite_task_inventory.csv", build_task_inventory())
    write_csv(tables / "cross_suite_smoke_manifest.csv", smoke)
    if args.dry_run:
        print(f"task_inventory={len(build_task_inventory())} smoke_manifest={len(smoke)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
