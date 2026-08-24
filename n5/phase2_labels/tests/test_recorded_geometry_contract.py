import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import materialize_v23_recorded_geometry as geometry


class _Named:
    def __init__(self, name):
        self.name = name


class _Model:
    nbody = 24
    nsite = 0
    ngeom = 6
    body_parentid = [0] * nbody
    body_jntadr = [0] * nbody
    body_jntnum = [0] * nbody
    geom_bodyid = [0] * ngeom
    jnt_bodyid = []
    jnt_type = []

    def body(self, body_id):
        return _Named("world" if body_id == 0 else "black_book_1_main" if body_id == 23 else f"body_{body_id}")

    def geom(self, geom_id):
        return _Named("black_book_1" if geom_id == 5 else f"geom_{geom_id}")


def _valid():
    path = Path(__file__).parents[1] / "recorded_geometry_alias_ledger_v1.json"
    entry = json.loads(path.read_text(encoding="utf-8"))["entries"][0]
    index_entry = {
        "object_index": entry["index_map_object_index"],
        "body_id": entry["index_map_body_id"],
        "body_name": entry["index_map_body_name"],
        "slice_start": entry["object_state_slice_start"],
        "slice_end_exclusive": entry["object_state_slice_end_exclusive"],
    }
    resolution = {"name": entry["bddl_object"], "resolution": entry["registry_resolution"], "entity_id": entry["registry_geom_id"]}
    return entry, index_entry, resolution


def test_alias_ledger_validates():
    entry, index_entry, resolution = _valid()
    geometry.validate_alias_exception(
        entry, "libero_10", 5, entry["bddl_sha256"], entry["registry_task_sha256"],
        entry["model_inventory_sha256"], resolution, index_entry, _Model()
    )


@pytest.mark.parametrize("field", [
    "suite", "bddl_object", "registry_geom_id", "registry_geom_name",
    "index_map_object_index", "index_map_body_id", "object_state_slice_start",
    "object_state_slice_end_exclusive", "bddl_sha256", "registry_task_sha256",
    "model_inventory_sha256",
])
def test_alias_mutation_fails_closed(field):
    entry, index_entry, resolution = _valid()
    mutated = copy.deepcopy(entry)
    mutated[field] = "mutated" if isinstance(mutated[field], str) else int(mutated[field]) + 1
    with pytest.raises(geometry.GeometryHold):
        geometry.validate_alias_exception(
            mutated, "libero_10", 5, entry["bddl_sha256"], entry["registry_task_sha256"],
            entry["model_inventory_sha256"], resolution, index_entry, _Model()
        )


def test_duplicate_alias_ledger_key_rejected(tmp_path):
    path = tmp_path / "ledger.json"
    data = json.loads((Path(__file__).parents[1] / "recorded_geometry_alias_ledger_v1.json").read_text())
    data["entries"].append(copy.deepcopy(data["entries"][0]))
    data["unique_alias_mappings"] = 2
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(geometry.GeometryHold):
        geometry.load_alias_ledger(path)
