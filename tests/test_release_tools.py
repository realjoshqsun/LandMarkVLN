from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.audit_anonymity import audit
from scripts.build_anonymous_artifact import selected_files


ROOT = Path(__file__).resolve().parents[1]


class ReleaseToolsTest(unittest.TestCase):
    def test_allowlist_excludes_git_and_dist(self) -> None:
        files = selected_files(ROOT, ROOT / "artifact-manifest.txt")
        names = [path.as_posix() for path in files]
        self.assertFalse(any(".git" in path.parts for path in files))
        self.assertFalse(any(name.startswith("dist/") for name in names))

    def test_anonymity_scanner_detects_private_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "README.md").write_text(
                "input lives under /" + "home/example/private", encoding="utf-8"
            )
            findings = audit(root)
        self.assertTrue(findings)


if __name__ == "__main__":
    unittest.main()

