from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

from evolution_lab.training_provider import (
    ProviderManifest,
    ProviderProtocolError,
    build_receipt,
    load_manifest,
    preflight,
    resolve_command,
    validate_request,
)


class TrainingProviderTests(unittest.TestCase):
    def _provider_repo(self, root: Path) -> tuple[Path, str]:
        provider = root / "provider"
        (provider / "scripts").mkdir(parents=True)
        (provider / "GYM-TRAINER.md").write_text("# Gym Trainer\n", encoding="utf-8")
        for script in (
            "mega_training_data.py",
            "agentic_training_loop.py",
            "continuous_evolution.py",
            "benchmark_omnisenter.py",
        ):
            (provider / "scripts" / script).write_text("print('stub')\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(provider)], check=True)
        subprocess.run(["git", "-C", str(provider), "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "-C", str(provider), "config", "user.name", "test"], check=True)
        subprocess.run(["git", "-C", str(provider), "remote", "add", "origin", "https://github.com/kvnloo/evolutionary-training.git"], check=True)
        subprocess.run(["git", "-C", str(provider), "add", "."], check=True)
        subprocess.run(["git", "-C", str(provider), "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"], check=True)
        head = subprocess.check_output(["git", "-C", str(provider), "rev-parse", "HEAD"], text=True).strip()
        return provider, head

    def _manifest_dict(self, revision: str) -> dict:
        return {
            "schema": "z0.training.autoresearch.v1",
            "provider": {
                "repo": "kvnloo/evolutionary-training",
                "upstream": "SouthpawIN/evolutionary-training",
                "revision": revision,
            },
            "hermes_profile": "GYM-TRAINER.md",
            "capabilities": ["data_prepare", "sft", "cma_es", "benchmark"],
            "commands": {
                "prepare_data": {"capability": "data_prepare", "argv": ["python3", "scripts/mega_training_data.py", "--download", "--prepare"]},
                "train": {"capability": "sft", "argv": ["python3", "scripts/agentic_training_loop.py", "--train"]},
                "evolve": {"capability": "cma_es", "argv": ["python3", "scripts/continuous_evolution.py", "--cycle"]},
                "evaluate": {"capability": "benchmark", "argv": ["python3", "scripts/benchmark_omnisenter.py", "--model", "{candidate_model}"]},
            },
            "eval_suite": {"repo": "kvnloo/z0evals", "revision": "a" * 40, "suite": "z0-training-provider-contract-v0"},
            "protected": {
                "training_forbidden": ["z0evals:confirm", "z0evals:ood"],
                "mutable_forbidden": ["judge", "gates", "confirm", "ood", "z0evals"],
            },
        }

    def _request(self) -> dict:
        return {
            "candidate_id": "dry-001",
            "experiment_id": "exp-001",
            "phase": "train",
            "method": "sft",
            "training_inputs": ["train:hermes", "validation:hermes"],
            "mutable_surfaces": ["training_config", "candidate_weights"],
            "eval_suite": {
                "repo": "kvnloo/z0evals",
                "revision": "a" * 40,
                "suite": "z0-training-provider-contract-v0",
            },
        }

    def test_checked_in_manifest_is_pinned_and_credited(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = load_manifest(root / "providers" / "evolutionary-training.json")
        self.assertEqual(manifest.repo, "kvnloo/evolutionary-training")
        self.assertEqual(manifest.upstream, "SouthpawIN/evolutionary-training")
        self.assertEqual(manifest.revision, "68ecf2057586a232bea45ad0fafcdfdfae56089f")
        self.assertEqual(manifest.eval_revision, "7047645132690d5947341684dcfd05d1cf160933")
        self.assertEqual(manifest.eval_suite, "z0-training-protected-v0")

    def test_preflight_pins_checkout_and_resolves_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            provider, head = self._provider_repo(Path(td))
            manifest = ProviderManifest.from_dict(self._manifest_dict(head))
            request = self._request()
            report = preflight(manifest, provider, request)
            command = resolve_command(manifest, "train")
            self.assertEqual(report["provider_revision"], head)
            self.assertEqual(command[:2], ("python3", "scripts/agentic_training_loop.py"))
            receipt = build_receipt(manifest, request, command)
            self.assertEqual(receipt["status"], "dry_run")
            self.assertIn("SouthpawIN", receipt["credit"])

    def test_missing_provider_revision_refuses(self) -> None:
        raw = self._manifest_dict("b" * 40)
        del raw["provider"]["revision"]
        with self.assertRaisesRegex(ProviderProtocolError, "provider.revision"):
            ProviderManifest.from_dict(raw)

    def test_undeclared_command_capability_refuses(self) -> None:
        raw = self._manifest_dict("b" * 40)
        raw["commands"]["train"]["capability"] = "shell_anything"
        with self.assertRaisesRegex(ProviderProtocolError, "undeclared capability"):
            ProviderManifest.from_dict(raw)

    def test_protected_eval_cannot_be_training_input(self) -> None:
        manifest = ProviderManifest.from_dict(self._manifest_dict("b" * 40))
        request = self._request()
        request["training_inputs"].append("z0evals:confirm")
        with self.assertRaisesRegex(ProviderProtocolError, "protected eval artifacts"):
            validate_request(manifest, request)

    def test_candidate_cannot_mutate_judge_or_gates(self) -> None:
        manifest = ProviderManifest.from_dict(self._manifest_dict("b" * 40))
        request = self._request()
        request["mutable_surfaces"].append("judge")
        with self.assertRaisesRegex(ProviderProtocolError, "protected surfaces"):
            validate_request(manifest, request)

    def test_unpinned_eval_revision_refuses(self) -> None:
        raw = self._manifest_dict("b" * 40)
        raw["eval_suite"]["revision"] = "main"
        with self.assertRaisesRegex(ProviderProtocolError, "eval_suite.revision"):
            ProviderManifest.from_dict(raw)

    def test_request_must_match_pinned_eval(self) -> None:
        manifest = ProviderManifest.from_dict(self._manifest_dict("b" * 40))
        request = self._request()
        request["eval_suite"]["revision"] = "c" * 40
        with self.assertRaisesRegex(ProviderProtocolError, "does not match pinned manifest"):
            validate_request(manifest, request)

    def test_wrong_provider_origin_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            provider, head = self._provider_repo(Path(td))
            subprocess.run(["git", "-C", str(provider), "remote", "set-url", "origin", "https://github.com/other/repo.git"], check=True)
            manifest = ProviderManifest.from_dict(self._manifest_dict(head))
            with self.assertRaisesRegex(ProviderProtocolError, "origin mismatch"):
                preflight(manifest, provider, self._request())

    def test_revision_mismatch_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            provider, _head = self._provider_repo(Path(td))
            manifest = ProviderManifest.from_dict(self._manifest_dict("b" * 40))
            with self.assertRaisesRegex(ProviderProtocolError, "revision mismatch"):
                preflight(manifest, provider, self._request())


if __name__ == "__main__":
    unittest.main()
