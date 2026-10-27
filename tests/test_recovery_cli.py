"""Hermetic unit tests for the OMP/Hermes recovery JSON CLI."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from evolution_lab import recovery_cli
from evolution_lab.schema import ACTIONS


class RecoveryCliTests(unittest.TestCase):
    @staticmethod
    def _main_json(argv: list[str] | None) -> tuple[int, dict]:
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = recovery_cli.main(argv)
        return exit_code, json.loads(output.getvalue())

    def test_main_exit_codes_and_errors(self) -> None:
        cases = (
            (["not json"], 2, "invalid JSON request:", True),
            (["[1]"], 2, "request must be a JSON object", False),
            (["{}"], 1, "advise requires fields object or event object", False),
            (['{"action":"nope"}'], 1, "unknown recovery action 'nope'", False),
            (['{"action":"observe"}'], 1, "observe requires an event object", False),
        )
        with patch.object(sys, "argv", ["recovery_cli"]):
            for argv, expected_code, expected_error, prefix in cases:
                with self.subTest(argv=argv):
                    code, payload = self._main_json(argv)
                    self.assertEqual(code, expected_code)
                    self.assertIs(payload["ok"], False)
                    if prefix:
                        self.assertTrue(
                            payload["error"].startswith(expected_error),
                            payload["error"],
                        )
                    else:
                        self.assertEqual(payload["error"], expected_error)

            # Stub status I/O here; test_status_payload covers the actual status path.
            with patch.object(
                recovery_cli,
                "status_payload",
                return_value={"ok": True, "locked_splits": False},
            ) as status:
                code, payload = self._main_json(['{"action":"status"}'])
            self.assertEqual(code, 0)
            self.assertIs(payload["ok"], True)
            self.assertIs(payload["locked_splits"], False)
            status.assert_called_once_with()

    def test_main_none_reads_sys_argv(self) -> None:
        # Both None and [] must use the patched process argv, not the test runner's.
        with patch.object(sys, "argv", ["x"]):
            for argv in (None, []):
                with self.subTest(argv=argv):
                    code, payload = self._main_json(argv)
                    self.assertEqual(code, 1)
                    self.assertIs(payload["ok"], False)
                    self.assertEqual(
                        payload["error"],
                        "advise requires fields object or event object",
                    )

    def test_status_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(recovery_cli, "repo_root", return_value=root),
                patch.object(recovery_cli, "splits_exist", return_value=False) as splits,
                patch.object(recovery_cli, "bundle_exists", return_value=False) as bundle,
                patch.object(recovery_cli, "bundle_meta", return_value={}) as meta,
            ):
                status = recovery_cli.handle_request({"action": "status"})
                self.assertIs(status["ok"], True)
                self.assertEqual(status["repo"], str(root))
                self.assertIs(status["locked_splits"], False)
                self.assertEqual(status["actions"], list(ACTIONS))
                self.assertIs(status["student_bundle"], False)
                self.assertEqual(status["student_bundle_meta"], {})
                self.assertEqual(status["default_family"], "local_plasticity")
                splits.assert_called_once_with(root / "data" / "p0")
                bundle.assert_called_once_with(root)
                meta.assert_called_once_with(root)

                with patch.dict(os.environ, {"FLYFORGE_RECOVERY_FAMILY": "x"}):
                    configured = recovery_cli.handle_request({"action": "status"})
                self.assertEqual(configured["default_family"], "x")

    def test_observe_returns_fields(self) -> None:
        observed = recovery_cli.handle_request(
            {
                "action": "observe",
                "event": {"kind": "hub_dead", "message": "hub process exited"},
            }
        )["observed"]
        self.assertTrue({"fields", "harness", "kind", "tool"}.issubset(observed))
        self.assertEqual(observed["harness"], "omp")
        self.assertEqual(observed["kind"], "hub_dead")
        self.assertIsNone(observed["tool"])
        self.assertEqual(observed["fields"]["retry"], 0)
        self.assertEqual(observed["fields"]["budget"], 1.0)
        self.assertEqual(observed["fields"]["sandbox_alive"], 0.0)

    def test_advise_from_fields_adds_guidance(self) -> None:
        fields = recovery_cli.handle_request(
            {
                "action": "observe",
                "event": {"kind": "hub_dead", "message": "hub process exited"},
            }
        )["observed"]["fields"]

        # The actual student adviser can load/train from disk: isolate CLI routing.
        def fake_advise(_fields: dict, *, family: str) -> dict:
            return {"action": "retry", "family": family}

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(recovery_cli, "advise", side_effect=fake_advise) as adviser,
        ):
            result = recovery_cli.handle_request({"action": "advise", "fields": fields})
        adviser.assert_called_once_with(fields, family="local_plasticity")
        self.assertEqual(result["action"], "retry")
        self.assertEqual(result["family"], "local_plasticity")
        self.assertIsInstance(result["guidance"], str)
        self.assertTrue(result["guidance"])

        with patch.object(
            recovery_cli, "advise", return_value={"action": "retry"}
        ) as adviser:
            explicit = recovery_cli.handle_request(
                {"action": "advise", "fields": fields, "family": "x"}
            )
        adviser.assert_called_once_with(fields, family="x")
        self.assertEqual(explicit["action"], "retry")
        self.assertTrue(explicit["guidance"])

    def test_budget_zero_is_kept(self) -> None:
        event = {"kind": "hub_dead", "message": "hub process exited"}
        with patch.object(
            recovery_cli,
            "observe_event",
            return_value={"fields": {"budget": 0.0}},
        ) as observe:
            result = recovery_cli.handle_request(
                {"action": "observe", "event": event, "budget": 0}
            )
        observe.assert_called_once_with(
            event, harness="omp", retry=0, budget=0.0, last_action=None
        )
        self.assertIsInstance(observe.call_args.kwargs["budget"], float)
        self.assertEqual(result["observed"]["fields"]["budget"], 0.0)


if __name__ == "__main__":
    unittest.main()
