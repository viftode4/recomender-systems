"""Small filesystem examples for teammate-navigation checks."""
from pathlib import Path
import tempfile
import unittest

from operations.check_docs import ENTRY_DOCS, check_docs


class CheckDocsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ENTRY_DOCS:
            self.write(name, "# Overview\n")

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def test_relative_links_duplicate_headings_and_encoded_paths(self):
        self.write("final-project/docs/Example guide.md", "# First *result*\n# First *result*\n")
        self.write("README.md", """# Overview
[relative](final-project/docs/Example%20guide.md#first-result-1)
[spaces](<final-project/docs/Example guide.md#first-result>)
[root](/final-project/docs/Example%20guide.md#first-result)
[self](#overview)
[reference]: final-project/docs/Example%20guide.md#first-result
""")
        result = check_docs(self.root)
        self.assertEqual(result["errors"], [])
        self.assertEqual((result["local_links"], result["anchors"]), (5, 5))

    def test_missing_target_and_heading_report_source_lines(self):
        self.write("README.md", "# Overview\n[missing](gone.md)\n[heading](#gone)\n")
        result = check_docs(self.root)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["errors"], ["README.md:2: missing target gone.md",
                                            "README.md:3: missing heading #gone"])

    def test_skips_examples_remote_links_and_historical_sources(self):
        self.write("README.md", """# Overview
[remote](https://example.invalid/no-network)
[mail](mailto:example@example.invalid)
`[inline example](missing.md)`
```markdown
[example](missing.md)
```
~~~~markdown
[example](missing.md)
~~~~
""")
        for folder in ("docs/archive", "runs", "exploratory"):
            self.write(f"final-project/{folder}/old.md", "[old](missing.md)\n")
        self.assertEqual(check_docs(self.root)["local_links"], 0)
        self.assertEqual(check_docs(self.root)["errors"], [])

    def test_discovers_team_documents_and_requires_entry_files(self):
        self.write("final-project/docs/team/PLAN.md", "[broken](missing.md)\n")
        (self.root / "CONTRIBUTING.md").unlink()
        result = check_docs(self.root)
        self.assertEqual(len(result["errors"]), 2)
        self.assertIn("CONTRIBUTING.md: missing entry document", result["errors"])
        self.assertIn("final-project/docs/team/PLAN.md:1: missing target missing.md", result["errors"])


if __name__ == "__main__":
    unittest.main()
