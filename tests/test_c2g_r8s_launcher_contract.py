import unittest
from pathlib import Path


class R8SLauncherContractTests(unittest.TestCase):
    def test_launchers_use_explicit_remote_and_never_mutate_origin(self):
        repo = Path(__file__).resolve().parents[1]
        launchers = (
            repo / "scripts/stageb/run_c2g_r8r_input_for_r8s.sh",
            repo / "scripts/stageb/run_c2g_r8s_teacher_v1_semantic_replay_audit.sh",
        )
        for launcher in launchers:
            text = launcher.read_text(encoding="utf-8")
            self.assertIn(
                'REMOTE_URL="${R8S_REMOTE_URL:-https://github.com/Leo-6-maker/openvla-gripper-dutycycle-attack.git}"',
                text,
            )
            self.assertIn('git fetch "$REMOTE_URL" "$BRANCH" --quiet', text)
            self.assertNotIn("git remote set-url", text)
            self.assertNotIn("mktemp -u", text)
            self.assertIn('[[ ! -e "$OUTPUT_ROOT" ]]', text)
            self.assertIn("sha256sum -c SHA256SUMS", text)
            self.assertIn("sha256sum -c SHA256SUMS.sha256", text)

    def test_r8r_input_launcher_uses_secure_temp_directory(self):
        repo = Path(__file__).resolve().parents[1]
        launcher = repo / "scripts/stageb/run_c2g_r8r_input_for_r8s.sh"
        text = launcher.read_text(encoding="utf-8")
        self.assertIn('SPEC_TMPDIR="$(mktemp -d /tmp/c2g_r8r_source_spec.XXXXXX)"', text)
        self.assertIn('SPEC="$SPEC_TMPDIR/source_spec.json"', text)
        self.assertIn("trap 'rm -rf \"$SPEC_TMPDIR\"' EXIT", text)


if __name__ == "__main__":
    unittest.main()
