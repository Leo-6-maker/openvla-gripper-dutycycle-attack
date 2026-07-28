import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import fit670_strict_contract as strict


def seal(root: Path):
    files = sorted(p for p in root.rglob("*") if p.is_file())
    (root / "SHA256SUMS").write_text(
        "\n".join(
            f"{strict.sha256_file(path)}  {path.relative_to(root).as_posix()}"
            for path in files
        ) + "\n",
        encoding="utf-8",
    )
    digest = strict.sha256_file(root / "SHA256SUMS")
    (root / "SHA256SUMS.sha256").write_text(
        f"{digest}  SHA256SUMS\n", encoding="utf-8"
    )


def identities():
    rows = []
    for index in range(670):
        suite = strict.FOUR_SUITES[index % 4]
        task = (index // 67) % 10
        state = index
        rows.append(
            {
                "suite": suite,
                "task_id": task,
                "state_id": state,
                "episode_id": f"{suite}/task_{task:02d}/state_{state:02d}",
                "collection_seed": 20260717,
                "initial_state_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                "fold": None,
            }
        )
    rows.sort(key=lambda row: row["episode_id"])
    return rows


class StrictContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rows = identities()
        self.allowlist_path = self.root / "FIT670_IDENTITY_ALLOWLIST.json"
        allowlist = {
            "gate": strict.ALLOWLIST_GATE,
            "schema": strict.ALLOWLIST_SCHEMA,
            "n_identities": 670,
            "protected_overlap": 0,
            "identity_set_digest": strict.legacy_identity_set_digest(self.rows),
            "identities": self.rows,
        }
        self.allowlist_path.write_text(json.dumps(allowlist), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def make_plan(self, n_shards=8):
        shards = []
        for sid in range(n_shards):
            assigned = self.rows[sid::n_shards]
            shards.append(
                {
                    "shard_id": sid,
                    "gpu": sid,
                    "n_identities": len(assigned),
                    "identities": assigned,
                }
            )
        plan = {
            "schema": strict.SHARD_SCHEMA,
            "status": "FROZEN",
            "n_shards": n_shards,
            "n_identities": 670,
            "input_allowlist_sha256": strict.sha256_file(self.allowlist_path),
            "shards": shards,
        }
        path = self.root / "FIT670_GPU_SHARD_PLAN.json"
        path.write_text(json.dumps(plan), encoding="utf-8")
        return path, plan

    def test_allowlist_digest_recomputed(self):
        data, by_id = strict.validate_allowlist(self.allowlist_path)
        self.assertEqual(data["n_identities"], 670)
        self.assertEqual(len(by_id), 670)

    def test_allowlist_digest_mutation_rejected(self):
        data = json.loads(self.allowlist_path.read_text())
        data["identities"][0]["collection_seed"] += 1
        self.allowlist_path.write_text(json.dumps(data))
        with self.assertRaises(strict.ContractViolation):
            strict.validate_allowlist(self.allowlist_path)

    def test_shard_exact_closure(self):
        path, _ = self.make_plan()
        _, membership = strict.validate_shard_plan(path, self.allowlist_path, 8)
        self.assertEqual(len(membership), 670)

    def test_shard_duplicate_rejected(self):
        path, plan = self.make_plan()
        plan["shards"][1]["identities"].append(plan["shards"][0]["identities"][0])
        plan["shards"][1]["n_identities"] += 1
        path.write_text(json.dumps(plan))
        with self.assertRaises(strict.ContractViolation):
            strict.validate_shard_plan(path, self.allowlist_path, 8)

    def test_shard_missing_rejected(self):
        path, plan = self.make_plan()
        plan["shards"][0]["identities"].pop()
        plan["shards"][0]["n_identities"] -= 1
        path.write_text(json.dumps(plan))
        with self.assertRaises(strict.ContractViolation):
            strict.validate_shard_plan(path, self.allowlist_path, 8)

    def test_shard_identity_mutation_rejected(self):
        path, plan = self.make_plan()
        plan["shards"][0]["identities"][0]["collection_seed"] += 1
        path.write_text(json.dumps(plan))
        with self.assertRaises(strict.ContractViolation):
            strict.validate_shard_plan(path, self.allowlist_path, 8)

    def test_seal_full_closure(self):
        root = self.root / "sealed"
        root.mkdir()
        (root / "a.json").write_text("{}")
        seal(root)
        self.assertRegex(strict.full_seal_check(root), r"^[0-9a-f]{64}$")

    def test_seal_tamper_rejected(self):
        root = self.root / "sealed"
        root.mkdir()
        (root / "a.json").write_text("{}")
        seal(root)
        (root / "a.json").write_text('{"x":1}')
        with self.assertRaises(strict.ContractViolation):
            strict.full_seal_check(root)

    def test_seal_extra_file_rejected(self):
        root = self.root / "sealed"
        root.mkdir()
        (root / "a.json").write_text("{}")
        seal(root)
        (root / "extra").write_text("x")
        with self.assertRaises(strict.ContractViolation):
            strict.full_seal_check(root)

    def test_entity_alias_identity(self):
        record = {
            "role": "object",
            "entity_type": "body",
            "entity_id": 23,
        }
        enriched = strict.enrich_entity_record(
            record,
            {
                "name": "black_book_1",
                "alias_to": "black_book_1_main",
                "resolution": "APPROVED_STRUCTURAL_ALIAS",
            },
        )
        self.assertEqual(enriched["logical_name"], "black_book_1")
        self.assertEqual(enriched["alias_to"], "black_book_1_main")
        self.assertRegex(enriched["binding_identity"], r"^[0-9a-f]{64}$")

    def test_alias_without_alias_to_rejected(self):
        with self.assertRaises(strict.ContractViolation):
            strict.enrich_entity_record(
                {"role": "object", "entity_type": "body", "entity_id": 1},
                {"name": "x", "resolution": "APPROVED_STRUCTURAL_ALIAS"},
            )

    def test_legacy_transition_schema_is_distinct(self):
        self.assertEqual(strict.TRANSITION_SCHEMA, "FIT670_INFERENCE_TRANSITION_V2")
        self.assertNotEqual(strict.TRANSITION_SCHEMA, "FIT670_INFERENCE_TRANSITION_V1")

    def test_source_structurally_fail_closed(self):
        worker = (Path(__file__).parent / "run_fit670_atomic_worker_v2.py").read_text()
        supervisor = (Path(__file__).parent / "run_fit670_supervisor_v2.py").read_text()
        finalizer = (Path(__file__).parent / "finalize_fit670_collection_v2.py").read_text()
        self.assertIn("validate_transition_v2", worker)
        self.assertIn("validate_episode_v2(target", worker)
        self.assertIn("raise SystemExit(2)", worker)
        self.assertIn("if failures:", supervisor)
        self.assertNotIn("WARNING:", finalizer)


if __name__ == "__main__":
    unittest.main()
