"""Honest Verified OSS Loop onboard: unittest, not pytest; receipts bind to a SHA."""

from __future__ import annotations

import importlib.util
import stat
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_check_receipt():
    path = ROOT / ".github" / "scripts" / "check-receipt.py"
    spec = importlib.util.spec_from_file_location("check_receipt", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class VolOnboardTests(unittest.TestCase):
    def test_agents_and_prompt_pin_unittest_not_pytest(self):
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        prompt = (ROOT / "prompt.md").read_text(encoding="utf-8")
        self.assertIn("unittest discover", agents)
        self.assertIn("unittest discover", prompt)
        self.assertNotIn("pytest", agents)
        self.assertNotIn("python -m pytest", prompt)
        self.assertNotIn("mutmut run", prompt)
        self.assertIn("| Mutation | `n/a` |", agents)

    def test_verify_sh_exists_and_is_executable(self):
        path = ROOT / "scripts" / "verify.sh"
        self.assertTrue(path.is_file())
        mode = path.stat().st_mode
        self.assertTrue(mode & stat.S_IXUSR)
        text = path.read_text(encoding="utf-8")
        self.assertIn("unittest discover", text)
        self.assertIn("gym-smoke", text)
        self.assertIn("mutation n/a", text)

    def test_check_receipt_accepts_bound_yaml(self):
        check = _load_check_receipt()
        head = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        body = f"""## Evidence

```yaml
issue: "2"
base_revision: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
head_revision: {head}
tests:
  red: python -m unittest tests.test_schema
  green: python -m unittest discover -s tests -p 'test_*.py'
mutation: n/a
```
"""
        self.assertEqual(check.check(body, head), [])

    def test_check_receipt_rejects_wrong_head_and_empty(self):
        check = _load_check_receipt()
        self.assertTrue(check.check("no yaml here", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
        body = """```yaml
issue: "2"
base_revision: abcdef1
head_revision: deadbeef
tests:
  red: python -m unittest tests.test_schema
  green: python -m unittest discover -s tests -p 'test_*.py'
```
"""
        errors = check.check(body, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        self.assertTrue(any("does not match" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
