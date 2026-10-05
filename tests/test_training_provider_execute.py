from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_lab.training_provider import (
    ProviderManifest,
    ProviderProtocolError,
    build_receipt,
    execute_phase,
    render_command,
)


class TrainingProviderExecuteTests(unittest.TestCase):
    def _provider(self, root: Path) -> tuple[Path, str]:
        provider = root / "provider"
        (provider / "scripts").mkdir(parents=True)
        (provider / "GYM-TRAINER.md").write_text("# Gym Trainer\n", encoding="utf-8")
        (provider / "scripts" / "train.py").write_text(
            "import os; print(os.environ.get('TEST_SECRET', 'train'))\n", encoding="utf-8"
        )
        subprocess.run(["git", "init", "-q", str(provider)], check=True)
        subprocess.run(["git", "-C", str(provider), "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "-C", str(provider), "config", "user.name", "test"], check=True)
        subprocess.run(["git", "-C", str(provider), "remote", "add", "origin",
                        "https://github.com/kvnloo/evolutionary-training.git"], check=True)
        subprocess.run(["git", "-C", str(provider), "add", "."], check=True)
        subprocess.run(["git", "-C", str(provider), "-c", "commit.gpgsign=false",
                        "commit", "-qm", "fixture"], check=True)
        head = subprocess.check_output(
            ["git", "-C", str(provider), "rev-parse", "HEAD"], text=True
        ).strip()
        return provider, head

    def _manifest(self, revision: str, allowed: list[str] | None = None) -> ProviderManifest:
        return ProviderManifest.from_dict({
            "schema": "z0.training.autoresearch.v1",
            "provider": {
                "repo": "kvnloo/evolutionary-training",
                "upstream": "SouthpawIN/evolutionary-training",
                "revision": revision,
            },
            "hermes_profile": "GYM-TRAINER.md",
            "capabilities": ["sft"],
            "commands": {
                "train": {
                    "capability": "sft",
                    "argv": ["python3", "scripts/train.py", "--base-model", "{base_model}"],
                    "timeout_s": 30,
                }
            },
            "eval_suite": {
                "repo": "kvnloo/z0evals",
                "revision": "a" * 40,
                "suite": "contract-only/no-model-eval",
            },
            "protected": {
                "training_forbidden": ["z0evals:confirm", "z0evals:ood"],
                "mutable_forbidden": ["judge", "gates", "confirm", "ood", "z0evals"],
            },
            "execution": {
                "allowed_phases": ["train"] if allowed is None else allowed,
                "env_passthrough": ["PATH", "HOME", "TEST_SECRET"],
            },
        })

    def _request(self, approved: bool = False) -> dict:
        return {
            "candidate_id": "candidate-1",
            "experiment_id": "experiment-1",
            "phase": "train",
            "method": "sft",
            "training_inputs": ["train:hermes"],
            "mutable_surfaces": ["training_config"],
            "eval_suite": {
                "repo": "kvnloo/z0evals",
                "revision": "a" * 40,
                "suite": "contract-only/no-model-eval",
            },
            "params": {"base_model": "/models/base"},
            "execution": {"approved": approved, "budget_s": 10},
        }

    def test_render_requires_all_params(self) -> None:
        manifest = self._manifest("b" * 40)
        with self.assertRaisesRegex(ProviderProtocolError, "missing command params"):
            render_command(manifest, "train", {})

    def test_execution_requires_explicit_approval(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            provider, head = self._provider(Path(td))
            with self.assertRaisesRegex(ProviderProtocolError, "approved"):
                execute_phase(self._manifest(head), provider, self._request(), Path(td) / "run")

    def test_execution_requires_manifest_permission(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            provider, head = self._provider(Path(td))
            with self.assertRaisesRegex(ProviderProtocolError, "not allowed"):
                execute_phase(self._manifest(head, []), provider, self._request(True), Path(td) / "run")

    def test_execution_hashes_private_logs_and_receipt_omits_secret(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            provider, head = self._provider(Path(td))
            manifest = self._manifest(head)
            request = self._request(True)
            old = os.environ.get("TEST_SECRET")
            os.environ["TEST_SECRET"] = "never-put-this-in-receipt"
            try:
                result = execute_phase(manifest, provider, request, Path(td) / "run")
            finally:
                if old is None:
                    os.environ.pop("TEST_SECRET", None)
                else:
                    os.environ["TEST_SECRET"] = old
            self.assertEqual(result["status"], "success")
            log = Path(td) / "run" / "train.stdout.log"
            self.assertEqual(log.stat().st_mode & 0o777, 0o600)
            command = render_command(manifest, "train", request["params"])
            receipt = build_receipt(manifest, request, command, result)
            self.assertNotIn("never-put-this-in-receipt", str(receipt))
            self.assertEqual(receipt["execution"]["stdout"]["bytes"],
                             len("never-put-this-in-receipt\n"))


if __name__ == "__main__":
    unittest.main()
