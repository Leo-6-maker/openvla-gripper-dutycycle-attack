import pytest
import torch
import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "stageb" / "run_m3_gpu45_longrun_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("run_m3_gpu45_longrun_diagnostics", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)

FORBIDDEN_SEEDS = module.FORBIDDEN_SEEDS
canonical_json_sha = module.canonical_json_sha
clone_inputs = module.clone_inputs
parse_csv_ints = module.parse_csv_ints
parse_csv_strings = module.parse_csv_strings
row_summary = module.row_summary
validate_development_input = module.validate_development_input
validate_forbidden_seed = module.validate_forbidden_seed


def test_parse_gpu_binding_order():
    assert parse_csv_ints("4,5") == [4, 5]
    assert parse_csv_strings("GPU-a, GPU-b") == ["GPU-a", "GPU-b"]


def test_forbidden_seed_rejection():
    for seed in FORBIDDEN_SEEDS:
        with pytest.raises(RuntimeError):
            validate_forbidden_seed(seed)
    validate_forbidden_seed(1234)
    validate_forbidden_seed(None)


def test_authorized_input_rejection(tmp_path):
    authorized = tmp_path / "dev" / "step78"
    authorized.mkdir(parents=True)
    validate_development_input(authorized, str(authorized))
    with pytest.raises(RuntimeError):
        validate_development_input(tmp_path / "final_8" / "step70", str(authorized))


def test_clone_inputs_detaches_and_copies():
    x = torch.ones(1, 2, requires_grad=True)
    cloned = clone_inputs({"pixel_values": x})
    assert cloned["pixel_values"].data_ptr() != x.data_ptr()
    assert cloned["pixel_values"].requires_grad is False
    cloned["pixel_values"][0, 0] = 5
    assert float(x[0, 0]) == 1.0


def test_row_summary_is_repeatable_and_finite():
    row = torch.zeros(32064)
    row[31744] = 2.0
    row[31872] = 1.0
    summary1 = row_summary(row, top_k=3)
    summary2 = row_summary(row.clone(), top_k=3)
    assert summary1["finite"] is True
    assert summary1["top_token"] == 31744
    assert summary1["target_minus_close"] == 1.0
    assert summary1["score_row_sha256"] == summary2["score_row_sha256"]


def test_canonical_json_sha_sorts_keys():
    assert canonical_json_sha({"b": 1, "a": 2}) == canonical_json_sha({"a": 2, "b": 1})
