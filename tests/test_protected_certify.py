from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from evolution_lab.protected_certify import (
    CertificationError,
    certify_candidate,
    load_certification_config,
    preflight_evaluator,
    promotion_allowed,
)


class ProtectedCertificationTests(unittest.TestCase):
    def _evaluator_repo(self, root: Path) -> tuple[Path, str]:
        repo = root / "z0evals"
        (repo / "scripts").mkdir(parents=True)
        (repo / "studies" / "protected-evaluator-v0").mkdir(parents=True)
        (repo / "studies" / "protected-evaluator-v0" / "evaluator.yaml").write_text(
            "schema: z0eval.protected.v1\n", encoding="utf-8"
        )
        (repo / "scripts" / "protected_eval.py").write_text(
            """
import hashlib
import json
import os
import sys
from pathlib import Path

def arg(name):
    return sys.argv[sys.argv.index(name) + 1]

predictions = Path(arg("--predictions")).read_text(encoding="utf-8")
candidate_id = arg("--candidate-id")
if "mismatch" in predictions:
    candidate_id = "wrong-candidate"
verdict = "DISCARD" if "discard" in predictions else "KEEP"
result = {
    "schema": "z0eval.result.v1",
    "suite_id": "z0-training-protected-v0",
    "suite_version": "test-v0",
    "cohort": arg("--cohort"),
    "candidate_id": candidate_id,
    "candidate_revision": arg("--candidate-revision"),
    "metric": "exact_match",
    "score": 0.0 if verdict == "DISCARD" else 1.0,
    "verdict": verdict,
    "query_index": 1,
    "sealed_credit": True,
    "saw_hf_token": "HF_TOKEN" in os.environ,
}
result["result_sha256"] = hashlib.sha256(
    json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
print(json.dumps(result))
""".lstrip(),
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "test"], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "remote", "add", "origin", "https://github.com/kvnloo/z0evals.git"],
            check=True,
        )
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"],
            check=True,
        )
        head = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        return repo, head

    def _provider_manifest(self, root: Path, eval_revision: str) -> Path:
        path = root / "provider.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "z0.training.autoresearch.v1",
                    "provider": {
                        "repo": "kvnloo/evolutionary-training",
                        "upstream": "SouthpawIN/evolutionary-training",
                        "revision": "a" * 40,
                    },
                    "hermes_profile": "GYM-TRAINER.md",
                    "capabilities": ["sft"],
                    "commands": {
                        "train": {
                            "capability": "sft",
                            "argv": ["python3", "scripts/train.py", "--base-model", "{base_model}"],
                            "timeout_s": 10,
                        }
                    },
                    "eval_suite": {
                        "repo": "kvnloo/z0evals",
                        "revision": eval_revision,
                        "suite": "z0-training-protected-v0",
                    },
                    "protected": {
                        "training_forbidden": ["z0evals:confirm"],
                        "mutable_forbidden": ["judge"],
                    },
                    "execution": {"allowed_phases": [], "env_passthrough": ["PATH", "HOME"]},
                    "certification": {
                        "evaluator_manifest": "studies/protected-evaluator-v0/evaluator.yaml",
                        "cohort": "confirm",
                        "result_schema": "z0eval.result.v1",
                        "broker_argv": ["python3", "scripts/protected_eval.py"],
                        "env_passthrough": ["PATH", "HOME", "Z0EVAL_PROTECTED_ROOT"],
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def _predictions(self, root: Path, text: str = "keep") -> Path:
        path = root / "predictions.jsonl"
        path.write_text(text + "\n", encoding="utf-8")
        return path

    def test_checked_in_manifest_pins_merged_protected_evaluator(self) -> None:
        root = Path(__file__).resolve().parents[1]
        provider, cert = load_certification_config(root / "providers" / "evolutionary-training.json")
        self.assertEqual(provider.eval_repo, "kvnloo/z0evals")
        self.assertEqual(provider.eval_revision, "7047645132690d5947341684dcfd05d1cf160933")
        self.assertEqual(provider.eval_suite, "z0-training-protected-v0")
        self.assertEqual(cert["cohort"], "confirm")

    def test_preflight_requires_exact_evaluator_revision(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evaluator, head = self._evaluator_repo(root)
            manifest = self._provider_manifest(root, "b" * 40)
            provider, cert = load_certification_config(manifest)
            self.assertNotEqual(head, provider.eval_revision)
            with self.assertRaisesRegex(CertificationError, "revision mismatch"):
                preflight_evaluator(provider, cert, evaluator)

    def test_keep_certification_is_promotion_eligible_and_hides_trainer_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evaluator, head = self._evaluator_repo(root)
            manifest = self._provider_manifest(root, head)
            predictions = self._predictions(root)
            with patch.dict(os.environ, {"HF_TOKEN": "trainer-secret"}, clear=False):
                result = certify_candidate(
                    provider_manifest_path=manifest,
                    evaluator_root=evaluator,
                    state_dir=(root / "state").resolve(),
                    predictions_path=predictions,
                    candidate_id="candidate-1",
                    candidate_revision="c" * 40,
                )
            self.assertTrue(promotion_allowed(result))
            self.assertFalse(result["saw_hf_token"])

    def test_discard_is_valid_certification_but_not_promotion_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evaluator, head = self._evaluator_repo(root)
            manifest = self._provider_manifest(root, head)
            result = certify_candidate(
                provider_manifest_path=manifest,
                evaluator_root=evaluator,
                state_dir=(root / "state").resolve(),
                predictions_path=self._predictions(root, "discard"),
                candidate_id="candidate-2",
                candidate_revision="d" * 40,
            )
            self.assertEqual(result["verdict"], "DISCARD")
            self.assertFalse(promotion_allowed(result))

    def test_mismatched_candidate_result_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evaluator, head = self._evaluator_repo(root)
            manifest = self._provider_manifest(root, head)
            with self.assertRaisesRegex(CertificationError, "wrong candidate_id"):
                certify_candidate(
                    provider_manifest_path=manifest,
                    evaluator_root=evaluator,
                    state_dir=(root / "state").resolve(),
                    predictions_path=self._predictions(root, "mismatch"),
                    candidate_id="candidate-3",
                    candidate_revision="e" * 40,
                )

    def test_state_dir_must_be_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            evaluator, head = self._evaluator_repo(root)
            manifest = self._provider_manifest(root, head)
            with self.assertRaisesRegex(CertificationError, "state_dir must be absolute"):
                certify_candidate(
                    provider_manifest_path=manifest,
                    evaluator_root=evaluator,
                    state_dir=Path("relative-state"),
                    predictions_path=self._predictions(root),
                    candidate_id="candidate-4",
                    candidate_revision="f" * 40,
                )


if __name__ == "__main__":
    unittest.main()
