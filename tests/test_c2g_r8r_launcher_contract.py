import csv
import tempfile
import unittest
from pathlib import Path

from tests.test_c2g_r8r_clean2000_reuse_audit import execute


class R8RLauncherContractTests(unittest.TestCase):
    def test_source_spec_uses_secure_temp_directory_and_absent_output_path(self):
        repo = Path(__file__).resolve().parents[1]
        launcher = repo / "scripts/stageb/run_c2g_r8r_clean2000_reuse_audit.sh"
        text = launcher.read_text(encoding="utf-8")

        self.assertIn(
            'SPEC_TMPDIR="$(mktemp -d /tmp/c2g_r8r_source_spec.XXXXXX)"',
            text,
        )
        self.assertIn('SPEC="$SPEC_TMPDIR/source_spec.json"', text)
        self.assertIn("trap 'rm -rf \"$SPEC_TMPDIR\"' EXIT", text)
        self.assertNotIn("mktemp -u", text)
        self.assertNotIn(
            'SPEC="$(mktemp /tmp/c2g_r8r_source_spec.XXXXXX.json)"',
            text,
        )

    def test_field_coverage_is_a_distinct_projection(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, _ = execute(Path(temporary))
            output = Path(result["output_root"])
            episode_path = output / "clean2000_r7_episode_ledger.csv"
            coverage_path = output / "clean2000_r7_field_coverage.csv"

            self.assertNotEqual(episode_path.read_bytes(), coverage_path.read_bytes())
            with episode_path.open(newline="", encoding="utf-8") as handle:
                episode_header = next(csv.reader(handle))
            with coverage_path.open(newline="", encoding="utf-8") as handle:
                coverage_header = next(csv.reader(handle))

            self.assertLess(len(coverage_header), len(episode_header))
            self.assertTrue(set(coverage_header).issubset(set(episode_header)))
            self.assertIn("canonical_25d_complete", coverage_header)
            self.assertIn("policy_intent_9d_complete", coverage_header)
            self.assertIn("teacher_v2_raw_evidence_complete", coverage_header)
            self.assertNotIn("metadata_path", coverage_header)
            self.assertNotIn("step_records_path", coverage_header)


if __name__ == "__main__":
    unittest.main()
