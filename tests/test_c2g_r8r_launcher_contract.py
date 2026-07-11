import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
