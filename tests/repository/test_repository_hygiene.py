from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_json(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_repository_ledgers_parse_and_have_consistent_counts() -> None:
    lifecycle = load_json("docs/repository/REPOSITORY_LIFECYCLE_LEDGER_V1.json")
    authority = load_json("docs/repository/IMMUTABLE_AUTHORITY_PATHS_V1.json")

    assert lifecycle["schema"] == "REPOSITORY_LIFECYCLE_LEDGER_V1"
    assert lifecycle["status"] == "CODE_R0_REPOSITORY_INVENTORY_PASS"
    assert len(lifecycle["entries"]) == len({row["path"] for row in lifecycle["entries"]})

    assert authority["schema"] == "IMMUTABLE_AUTHORITY_PATHS_V1"
    assert authority["status"] == "CODE_R1_AUTHORITY_FIREWALL_PASS"
    assert authority["entry_count"] == len(authority["entries"])
    assert len(authority["entries"]) == len({row["path"] for row in authority["entries"]})


def test_root_readme_relative_links_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme)
    missing = [
        target
        for target in targets
        if not target.startswith(("http://", "https://", "#"))
        and not (ROOT / target.split("#", 1)[0]).exists()
    ]
    assert missing == []


def test_active_surface_accounts_for_every_core_module() -> None:
    surface = (ROOT / "docs/repository/ACTIVE_CODE_SURFACE_V1.md").read_text(encoding="utf-8")
    modules = sorted(path.stem for path in (ROOT / "src/gripper_attack").glob("*.py"))
    missing = [
        module
        for module in modules
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(module)}(?![A-Za-z0-9_])", surface) is None
    ]
    assert missing == []


def test_ci_matrix_covers_every_workflow() -> None:
    matrix = (ROOT / "docs/repository/TEST_CI_MATRIX_V1.md").read_text(encoding="utf-8")
    workflows = sorted(path.name for path in (ROOT / ".github/workflows").glob("*.yml"))
    missing = [name for name in workflows if f"`{name}`" not in matrix]
    assert missing == []


def test_repository_hygiene_workflow_is_static_cpu_only() -> None:
    workflow = (ROOT / ".github/workflows/repository-hygiene.yml").read_text(
        encoding="utf-8"
    ).lower()
    for forbidden in ("cuda", "torch", "mujoco", "env.step", "model inference"):
        assert forbidden not in workflow
