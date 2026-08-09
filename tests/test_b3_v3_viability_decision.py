import json
from pathlib import Path

from gripper_attack.b3_training_protocol import seal_directory
from gripper_attack.b3_v3_viability_decision import build_viability_decision, load_viability_decision


def test_viability_decision_is_sealed_and_separate_from_preparation_aggregate(tmp_path: Path):
    aggregate_root = tmp_path / "aggregate"
    aggregate_root.mkdir()
    runs = []
    for fold in range(4):
        for variant in ("B3_25D", "B3_25D9D"):
            for seed in (20260717, 20260718, 20260719):
                runs.append({
                    "fold_id": fold,
                    "variant": variant,
                    "seed": seed,
                    "metrics": {
                        "full_t10_event_hit_rate": 1.0,
                        "negative_episode_any_emit_rate": 0.0,
                        "release_overlap_count": 0,
                        "later_event_hit_rate": 1.0,
                        "baseline_comparison": {"close_only": {"full_t10_event_hit_rate": 0.5}},
                    },
                })
    aggregate = {
        "schema": "B3_OFFICIAL_V3_FIT_VIABILITY_AGGREGATE_V1",
        "status": "PASS_PREPARATION_ONLY",
        "run_count": 24,
        "runs": runs,
        "formal_training_authorized": False,
        "formal_attack_authorized": False,
    }
    path = aggregate_root / "viability_aggregate.json"
    path.write_text(json.dumps(aggregate, sort_keys=True) + "\n", encoding="utf-8")
    path.with_name(path.name + ".sha256").write_text("placeholder\n", encoding="utf-8")
    # seal_directory is the source of truth for the checksum files; replace the
    # deliberately temporary sidecar before sealing.
    path.with_name(path.name + ".sha256").unlink()
    seal_directory(aggregate_root)

    config = tmp_path / "decision_config.json"
    criteria = {
        name: {
            "min_mean_full_t10_event_hit_rate": 0.8,
            "max_negative_episode_any_emit_rate": 0.1,
            "max_release_overlap_count": 0,
            "min_later_event_hit_rate": 0.5,
            "require_close_baseline_not_worse": True,
            "baseline_margin": 0.0,
        }
        for name in ("B3_25D", "B3_25D9D")
    }
    config.write_text(json.dumps({"schema": "B3_OFFICIAL_V3_FIT_VIABILITY_DECISION_CONFIG_V1", "status": "PRE_REGISTERED", "criteria": criteria}, sort_keys=True) + "\n", encoding="utf-8")
    decision = build_viability_decision(aggregate_root, config, tmp_path / "decision")
    assert decision["status"] == "PASS"
    assert load_viability_decision(tmp_path / "decision")["formal_training_authorized"] is False
