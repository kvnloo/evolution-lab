#!/usr/bin/env python3
"""Bounded external-library research; synthetic fixtures, no model or training.

Two deliberately RED hypotheses. This is not a trainer, parser, evaluator,
or production adapter. Original modules are imported from external checkouts;
the existing extractor CLI runs unchanged in an isolated synthetic HOME.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

PINS = {
    "extractor": "9176c2dd81c8a101b52debd4cec88573284e2b30",
    "trainer": "68ecf2057586a232bea45ad0fafcdfdfae56089f",
}
ROOTS = {}
MODULES = {}
OBSERVATIONS = {}


def load_external(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load original external module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def endpoint(health_status):
    """Own a loopback port; POST is refused before any inference can happen."""
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append({"method": "GET", "path": self.path})
            self.send_response(health_status)
            self.end_headers()
            self.wfile.write(b"fixture-only")

        def do_POST(self):
            calls.append({"method": "POST", "path": self.path})
            self.send_response(405)
            self.end_headers()

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        if thread.is_alive():
            raise RuntimeError("owned HTTP fixture did not shut down")


class DirectDataContract(unittest.TestCase):
    def test_original_sharegpt_positive_control(self):
        data = [{"source": "synthetic-control", "conversations": [
            {"from": "human", "value": "Read synthetic.txt"},
            {"from": "gpt", "value": "read_file synthetic.txt"},
            {"from": "function", "value": "fixture result"},
        ]}]
        with contextlib.redirect_stdout(io.StringIO()):
            output = MODULES["trainer"].format_for_sft(data)
        self.assertEqual(len(output), 1)
        self.assertEqual([m["role"] for m in output[0]["messages"]],
                         ["user", "assistant", "tool"])
        OBSERVATIONS["sharegpt_control"] = {
            "input_conversations": 1, "output_conversations": len(output),
            "output_message_roles": [m["role"] for m in output[0]["messages"]],
            "kind": "synthetic schema-control; not model quality",
        }

    def test_extractor_output_directly_reaches_sft_formatter(self):
        with tempfile.TemporaryDirectory(prefix="el31-", dir=os.environ["TMPDIR"]) as tmp:
            root = Path(tmp)
            home = root / "home"
            sessions = home / ".codex" / "sessions" / "2026" / "10" / "10"
            sessions.mkdir(parents=True)
            events = [
                {"type": "session_meta", "payload": {"id": "synthetic-el31", "cwd": "synthetic-project"}},
                {"type": "event_msg", "payload": {"type": "user_message", "message": "Read synthetic.txt"}},
                {"type": "event_msg", "payload": {"type": "agent_message", "message": "Read completed"}},
                {"type": "event_msg", "payload": {"type": "tool_use", "tool": "read_file", "input": {"path": "synthetic.txt"}}},
                {"type": "event_msg", "payload": {"type": "tool_result", "tool": "read_file", "output": "fixture result"}},
            ]
            fixture = sessions / "rollout-synthetic-el31.jsonl"
            fixture.write_text("".join(json.dumps(e) + "\n" for e in events))
            env = dict(os.environ, HOME=str(home), PYTHONDONTWRITEBYTECODE="1")
            run = subprocess.run([sys.executable, str(ROOTS["extractor"] / "extract_codex.py")],
                                 cwd=root, env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
            outputs = list((root / "extracted_data").glob("codex_conversations_*.jsonl"))
            self.assertEqual(len(outputs), 1)
            data = [json.loads(line) for line in outputs[0].read_text().splitlines()]
            self.assertEqual(len(data), 1)
            self.assertEqual(len(data[0]["messages"]), 2)
            self.assertEqual(len(data[0]["tool_results"]), 2)
            self.assertEqual(data[0]["session_id"], "synthetic-el31")
            corpus, selected, total = MODULES["skills"].sample_corpus(
                outputs, char_budget=10000, max_conversations=1, seed=0)
            self.assertEqual((selected, total), (1, 1))
            self.assertIn("synthetic-el31", corpus)
            with contextlib.redirect_stdout(io.StringIO()):
                formatted = MODULES["trainer"].format_for_sft(data)
            OBSERVATIONS["direct_data_contract"] = {
                "kind": "synthetic source-native fixture; no real conversation or model run",
                "extractor_cli_exit": run.returncode,
                "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                "extracted_conversations": len(data),
                "extracted_messages": len(data[0]["messages"]),
                "extracted_tool_events": len(data[0]["tool_results"]),
                "source_session_id_retained": data[0]["session_id"] == "synthetic-el31",
                "original_skill_sampler_selected": selected,
                "original_skill_sampler_total": total,
                "corpus_chars": len(corpus),
                "trainer_formatted_conversations": len(formatted),
                "hypothesis": "existing extracted messages can directly enter provider SFT formatter",
            }
            self.assertEqual(len(formatted), 1,
                             "original extractor output is silently dropped by original SFT formatter")


class ReadOnlyPreflight(unittest.TestCase):
    def test_ready_endpoint_uses_get_only_positive_control(self):
        with endpoint(200) as (url, calls):
            self.assertTrue(MODULES["health"].check_model_health(url))
        self.assertEqual(calls, [{"method": "GET", "path": "/health"}])
        OBSERVATIONS["ready_health_control"] = {"requests": calls, "model_inference": False}

    def test_provider_dry_run_never_posts_inference(self):
        with endpoint(503) as (url, calls):
            stdout = io.StringIO()
            with patch.object(MODULES["health"], "MODELS", {"owned-fixture": url}), \
                 patch.object(sys, "argv", ["training_loop.py", "--dry-run"]), \
                 contextlib.redirect_stdout(stdout):
                MODULES["health"].main()
        OBSERVATIONS["dry_run_contract"] = {
            "kind": "owned loopback HTTP fixture; POST refused with 405 before inference",
            "provider_main_argv": ["training_loop.py", "--dry-run"],
            "fixture_health_status": 503,
            "requests": calls,
            "post_requests": sum(c["method"] == "POST" for c in calls),
            "model_inference": False,
            "stdout": stdout.getvalue(),
            "hypothesis": "advertised dry-run is a read-only/no-inference preflight",
        }
        self.assertFalse(any(c["method"] == "POST" for c in calls),
                         "original provider dry-run attempts a completions POST after failed health GET")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractor-root", type=Path, required=True)
    parser.add_argument("--trainer-root", type=Path, required=True)
    parser.add_argument("--cycle", choices=["data", "preflight", "all"], default="all")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    roots = {"extractor": args.extractor_root.resolve(), "trainer": args.trainer_root.resolve()}
    source = {}
    for name, root in roots.items():
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if revision != PINS[name]:
            parser.error(f"{name} revision mismatch: {revision}")
        tracked_changes = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True)
        if tracked_changes:
            parser.error(f"{name} tracked tree is dirty")
        source[name] = {"revision": revision}
    ROOTS.update(roots)
    paths = {
        "skills": roots["extractor"] / "corpus_to_skills.py",
        "trainer": roots["trainer"] / "scripts" / "agentic_training_loop.py",
        "health": roots["trainer"] / "scripts" / "training_loop.py",
    }
    # Hash the original source, not copied implementation.
    source["file_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in paths.items()}
    source["file_sha256"]["extract_codex_cli"] = hashlib.sha256(
        (roots["extractor"] / "extract_codex.py").read_bytes()).hexdigest()
    for name, path in paths.items():
        MODULES[name] = load_external("el31_external_" + name, path)
    suite = unittest.TestSuite()
    if args.cycle in ("data", "all"):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(DirectDataContract))
    if args.cycle in ("preflight", "all"):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ReadOnlyPreflight))
    start = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_SELF)
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = resource.getrusage(resource.RUSAGE_SELF)
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = {
        "schema": "evolution-lab.external-provider-research.v1",
        "cycle": args.cycle, "sources": source,
        "observations": OBSERVATIONS,
        "tests_run": result.testsRun,
        "failures": [{"test": str(test), "detail": detail} for test, detail in result.failures],
        "errors": [{"test": str(test), "detail": detail} for test, detail in result.errors],
        "skipped": result.skipped,
        "elapsed_seconds": time.monotonic() - start,
        "cpu_seconds": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        "child_cpu_seconds": child_after.ru_utime + child_after.ru_stime - child_before.ru_utime - child_before.ru_stime,
        "process_peak_rss_kib": after.ru_maxrss,
        "model_inference_calls": 0, "training_started": False,
        "model_quality_measured": False, "savings_measured": False,
        "scope": "original-library fixture behavior only; parent model/tool-server costs excluded",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
