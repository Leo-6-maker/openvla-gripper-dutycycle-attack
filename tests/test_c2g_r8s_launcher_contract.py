import unittest
from pathlib import Path


class R8SLauncherContractTests(unittest.TestCase):
    def test_launcher_uses_explicit_remote_and_never_mutates_origin(self):
        repo = Path(__file__).resolve().parents[1]
        launcher = repo / "scripts/stageb/run_c2g_r8s_teacher_v1_semantic_replay_audit.sh"
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


if __name__ == "__main__":
    unittest.main()
