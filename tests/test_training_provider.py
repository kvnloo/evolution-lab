"""Real pinned extractor -> original preparation functions; no models or services."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any
import unittest

PROVIDER = Path(os.environ.get("EL_TRAINING_PROVIDER_SOURCE", "/workspace/zer0/oss/evolutionary-training-hermes-provider-20261010"))
EXTRACTOR = Path(os.environ.get("EL_EXTRACTOR_SOURCE", "/home/kvn/.z0int/research/evolution-lab-23/ai-data-extraction"))
PROVIDER_REV = "68ecf2057586a232bea45ad0fafcdfdfae56089f"
EXTRACTOR_REV = "9176c2dd81c8a101b52debd4cec88573284e2b30"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def contract():
    return {
        "schema": "evolution_lab.training_provider.v1",
        "phases": ["prepare", "candidate"],
        "commands": ["mega_training_data.format_to_sharegpt", "agentic_training_loop.format_for_sft"],
        "provider_revision": PROVIDER_REV,
        "extractor_revision": EXTRACTOR_REV,
        "judge": None, "gates": [], "verifier": None,
    }


def make_case(root, splits=("train",), mutate_native=None) -> dict[str, Any]:
    """Invoke the unchanged extractor CLI with HOME limited to owned synthetic files."""
    for repo, rev in ((PROVIDER, PROVIDER_REV), (EXTRACTOR, EXTRACTOR_REV)):
        if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != rev:
            raise AssertionError("fixture source revision mismatch")
        if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"]):
            raise AssertionError("fixture source dirty")
    script = EXTRACTOR / "extract_claude_code.py"
    if script.read_bytes() != subprocess.check_output(["git", "-C", str(EXTRACTOR), "show", f"{EXTRACTOR_REV}:extract_claude_code.py"]):
        raise AssertionError("extractor bytes differ from pin")
    home = root / "home"
    (home / ".config").mkdir(parents=True)
    native = home / ".claude" / "projects" / "owned-synthetic"
    native.mkdir(parents=True)
    for i, split in enumerate(splits):
        events = [
            {"type": "user", "timestamp": "2026-01-01T00:00:00Z", "message": {"content": f"synthetic {split} question {i}"}},
            {"type": "assistant", "timestamp": "2026-01-01T00:00:01Z", "message": {"content": [{"type": "text", "text": f"synthetic {split} answer {i}"}]}},
        ]
        if mutate_native:
            mutate_native(events)
        (native / f"session-{i}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    command = [sys.executable, "-I", "-B", str(script)]
    start, cpu = time.perf_counter(), os.times()
    completed = subprocess.run(command, cwd=root, env=dict(os.environ, HOME=str(home)), capture_output=True, text=True, timeout=15)
    after = os.times()
    (root / "extractor.stdout").write_text(completed.stdout)
    (root / "extractor.stderr").write_text(completed.stderr)
    (root / "extractor-receipt.json").write_text(json.dumps({
        "command": command, "cwd": str(root), "HOME": str(home), "exit_code": completed.returncode,
        "wall_seconds": time.perf_counter() - start,
        "user_cpu_seconds": after.children_user - cpu.children_user,
        "system_cpu_seconds": after.children_system - cpu.children_system,
        "source_revision": EXTRACTOR_REV, "script_sha256": sha(script.read_bytes()),
    }, indent=2))
    if completed.returncode:
        raise AssertionError(completed.stderr)
    corpus, = (root / "extracted_data").glob("*.jsonl")
    records = []
    for index, line in enumerate(corpus.read_bytes().splitlines(keepends=True)):
        row = json.loads(line)
        session = row["session_id"]
        i = int(session.removeprefix("session-"))
        records.append({"record_index": index, "record_sha256": sha(line), "source_file": row["source_file"], "source_sha256": sha(Path(row["source_file"]).read_bytes()), "session_id": session, "work_item_id": f"owned-work-{i}", "split": splits[i]})
    lineage = root / "input-lineage.json"
    lineage.write_text(json.dumps({"schema": "evolution_lab.source_lineage.v1", "qualification": "producer_asserted", "corpus_sha256": sha(corpus.read_bytes()), "protected_sessions": [r["session_id"] for r in records if r["split"] == "protected"], "records": records}))
    return dict(provider_root=PROVIDER, extractor_root=EXTRACTOR, corpus_path=corpus, lineage_path=lineage, output_dir=root / "candidate", manifest=contract())


@unittest.skipUnless(PROVIDER.is_dir() and EXTRACTOR.is_dir(),
                     "pinned external checkouts required; set EL_TRAINING_PROVIDER_SOURCE and EL_EXTRACTOR_SOURCE")
class TrainingProviderTests(unittest.TestCase):
    def test_extracted_messages_must_match_pinned_native_extraction(self):
        from evolution_lab.training_provider import PreparationError, prepare_candidate
        mutations = {
            "fabricated_user": lambda r: r["messages"][0].update(content="invented training input"),
            "fabricated_assistant": lambda r: r["messages"][1].update(content="invented training target"),
            "reordered": lambda r: r["messages"].reverse(),
            "omitted_metadata": lambda r: r["messages"][0].pop("timestamp"),
            "wrong_project": lambda r: r.update(project_path="invented-project"),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                args = make_case(Path(tmp))
                row = json.loads(args["corpus_path"].read_bytes())
                mutation(row)
                data = (json.dumps(row) + "\n").encode()
                args["corpus_path"].write_bytes(data)
                lineage = json.loads(args["lineage_path"].read_bytes())
                lineage["corpus_sha256"] = lineage["records"][0]["record_sha256"] = sha(data)
                args["lineage_path"].write_text(json.dumps(lineage))
                with self.assertRaisesRegex(PreparationError, "native extraction"):
                    prepare_candidate(**args)
                self.assertFalse(args["output_dir"].exists())

    def test_content_copies_with_changed_metadata_cannot_cross_partitions(self):
        from evolution_lab.training_provider import PreparationError, prepare_candidate
        calls = 0
        def copied(events):
            nonlocal calls
            calls += 1
            events[0]["message"]["content"] = "same conversation input"
            events[1]["message"]["content"] = "same conversation output"
            events[0]["timestamp"] = f"2026-01-0{calls}T00:00:00Z"
        with tempfile.TemporaryDirectory() as tmp:
            args = make_case(Path(tmp), ("train", "validation"), mutate_native=copied)
            with self.assertRaisesRegex(PreparationError, "overlap partitions"):
                prepare_candidate(**args)
            self.assertFalse(args["output_dir"].exists())

    def test_reused_native_branch_events_cannot_cross_partitions(self):
        from evolution_lab.training_provider import PreparationError, prepare_candidate
        def shared_event(events):
            events[0]["uuid"] = "shared-native-branch-event"
        with tempfile.TemporaryDirectory() as tmp:
            args = make_case(Path(tmp), ("train", "validation"), mutate_native=shared_event)
            with self.assertRaisesRegex(PreparationError, "overlap partitions"):
                prepare_candidate(**args)
            self.assertFalse(args["output_dir"].exists())

    def test_output_replacement_cannot_redirect_candidate_writes(self):
        from unittest.mock import patch
        from evolution_lab.training_provider import PreparationError, prepare_candidate
        with tempfile.TemporaryDirectory() as tmp:
            args = make_case(Path(tmp))
            target = Path(tmp) / "unrelated-output"
            target.mkdir()
            original_open = os.open
            replaced = False
            def replace_directory(path, *values, **kwargs):
                nonlocal replaced
                if isinstance(path, (str, os.PathLike)) and Path(path).name == "train.sft.jsonl" and not replaced:
                    replaced = True
                    args["output_dir"].rename(Path(tmp) / "parked-output")
                    args["output_dir"].symlink_to(target, target_is_directory=True)
                return original_open(path, *values, **kwargs)
            with patch("os.open", side_effect=replace_directory):
                with self.assertRaises((PreparationError, OSError)):
                    prepare_candidate(**args)
            self.assertTrue(replaced)
            self.assertEqual(list(target.iterdir()), [])

    def test_equal_bytes_never_supply_origin_identity(self):
        from evolution_lab.training_provider import PreparationError, prepare_candidate
        def identical(events):
            events[0]["message"]["content"] = "same synthetic prompt"
            events[1]["message"]["content"] = "same synthetic reply"
        for splits in (("train", "train"), ("train", "validation")):
            with self.subTest(splits=splits), tempfile.TemporaryDirectory() as tmp:
                args = make_case(Path(tmp), splits, mutate_native=identical)
                supplied = json.loads(args["lineage_path"].read_bytes())
                self.assertEqual(len({r["source_sha256"] for r in supplied["records"]}), 1)
                if splits[1] == "validation":
                    with self.assertRaisesRegex(PreparationError, "overlap"):
                        prepare_candidate(**args)
                    self.assertFalse(args["output_dir"].exists())
                else:
                    result = prepare_candidate(**args)
                    retained = json.loads((args["output_dir"] / "lineage.json").read_bytes())
                    self.assertEqual(result["counts"]["train"], 2)
                    self.assertEqual([r["session_id"] for r in retained["records"]], [r["session_id"] for r in supplied["records"]])
                    self.assertFalse(retained["independently_qualified"])

    def test_empty_native_source_cannot_support_an_extracted_row(self):
        from evolution_lab.training_provider import PreparationError, prepare_candidate
        with tempfile.TemporaryDirectory() as tmp:
            args = make_case(Path(tmp))
            binding = json.loads(args["lineage_path"].read_bytes())
            Path(binding["records"][0]["source_file"]).write_bytes(b"")
            binding["records"][0]["source_sha256"] = sha(b"")
            args["lineage_path"].write_text(json.dumps(binding))
            with self.assertRaisesRegex(PreparationError, "empty native"):
                prepare_candidate(**args)
            self.assertFalse(args["output_dir"].exists())

    def test_path_alias_does_not_create_a_new_native_origin(self):
        from evolution_lab.training_provider import PreparationError, prepare_candidate
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = make_case(root)
            row = json.loads(args["corpus_path"].read_bytes())
            source = Path(row["source_file"])
            (root / "alias").symlink_to(source.parent, target_is_directory=True)
            alias = str(root / "alias" / source.name)
            line = (json.dumps({**row, "source_file": alias}) + "\n").encode()
            args["corpus_path"].write_bytes(args["corpus_path"].read_bytes() + line)
            lineage = json.loads(args["lineage_path"].read_bytes())
            lineage["records"].append({**lineage["records"][0], "source_file": alias, "record_index": 1, "record_sha256": sha(line)})
            lineage["corpus_sha256"] = sha(args["corpus_path"].read_bytes())
            args["lineage_path"].write_text(json.dumps(lineage))
            with self.assertRaisesRegex(PreparationError, "duplicate native origin"):
                prepare_candidate(**args)
            self.assertFalse(args["output_dir"].exists())

    def test_duplicate_native_origin_is_not_a_second_training_sample(self):
        from evolution_lab.training_provider import PreparationError, prepare_candidate
        with tempfile.TemporaryDirectory() as tmp:
            args = make_case(Path(tmp))
            corpus = args["corpus_path"].read_bytes()
            args["corpus_path"].write_bytes(corpus + corpus)
            lineage = json.loads(args["lineage_path"].read_bytes())
            lineage["corpus_sha256"] = sha(corpus + corpus)
            lineage["records"].append({**lineage["records"][0], "record_index": 1, "work_item_id": "conflicting-work"})
            args["lineage_path"].write_text(json.dumps(lineage))
            with self.assertRaisesRegex(PreparationError, "duplicate native origin"):
                prepare_candidate(**args)
            self.assertFalse(args["output_dir"].exists())

    def test_ambiguous_json_cannot_hide_lineage_or_tool_events(self):
        from evolution_lab.training_provider import PreparationError, prepare_candidate
        for target in ("corpus", "native", "lineage"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                args = make_case(Path(tmp))
                binding = json.loads(args["lineage_path"].read_bytes())
                if target == "corpus":
                    data = args["corpus_path"].read_bytes().replace(b'{"messages":', b'{"messages": [], "messages":', 1)
                    args["corpus_path"].write_bytes(data)
                    binding["corpus_sha256"] = binding["records"][0]["record_sha256"] = sha(data)
                    args["lineage_path"].write_text(json.dumps(binding))
                elif target == "native":
                    source = Path(binding["records"][0]["source_file"])
                    data = source.read_bytes().replace(b'"message":', b'"message": {"tool_calls": []}, "message":', 1)
                    source.write_bytes(data)
                    binding["records"][0]["source_sha256"] = sha(data)
                    args["lineage_path"].write_text(json.dumps(binding))
                else:
                    data = args["lineage_path"].read_bytes().replace(b'"protected_sessions":', b'"protected_sessions": ["unknown"], "protected_sessions":', 1)
                    args["lineage_path"].write_bytes(data)
                with self.assertRaisesRegex(PreparationError, "JSON"):
                    prepare_candidate(**args)
                self.assertFalse(args["output_dir"].exists())

    def test_pinned_sources_are_checked_before_import(self):
        from evolution_lab.training_provider import PreparationError, prepare_candidate
        for parameter, repo, relative in (("provider_root", PROVIDER, "scripts/mega_training_data.py"),
                                           ("extractor_root", EXTRACTOR, "extract_claude_code.py")):
            for mutation in ("wrong_head", "dirty", "untracked", "hidden_dirty", "hidden_symlink"):
                with self.subTest(source=parameter, mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    args = make_case(root)
                    clone = root / "source"
                    subprocess.run(["git", "clone", "--shared", "--quiet", str(repo), str(clone)], check=True, capture_output=True, timeout=15)
                    args[parameter] = clone
                    script = clone / relative
                    if mutation == "wrong_head":
                        subprocess.run(["git", "-C", str(clone), "checkout", "--quiet", "--detach", "HEAD^"], check=True, capture_output=True)
                    elif mutation == "untracked":
                        (clone / "untrusted.py").write_text("raise AssertionError('untrusted')\n")
                    else:
                        if mutation.startswith("hidden"):
                            subprocess.run(["git", "-C", str(clone), "update-index", "--skip-worktree", relative], check=True)
                        if mutation == "hidden_symlink":
                            shadow = root / "shadow.py"
                            shadow.write_bytes(script.read_bytes())
                            script.unlink()
                            script.symlink_to(shadow)
                        else:
                            script.write_bytes(script.read_bytes() + b"\nraise AssertionError('must not import dirty code')\n")
                    with self.assertRaisesRegex(PreparationError, "source"):
                        prepare_candidate(**args)
                    self.assertFalse(args["output_dir"].exists())

    def test_artifacts_are_private_and_cannot_overwrite_existing_output(self):
        from evolution_lab.training_provider import PreparationError, prepare_candidate
        with tempfile.TemporaryDirectory() as tmp:
            args = make_case(Path(tmp))
            old = os.umask(0)
            try:
                prepare_candidate(**args)
            finally:
                os.umask(old)
            self.assertEqual(args["output_dir"].stat().st_mode & 0o777, 0o700)
            files = {p.name: p.read_bytes() for p in args["output_dir"].iterdir()}
            for path in args["output_dir"].iterdir():
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaisesRegex(PreparationError, "output"):
                prepare_candidate(**args)
            self.assertEqual(files, {p.name: p.read_bytes() for p in args["output_dir"].iterdir()})
            args["output_dir"] = Path(__file__).resolve().parents[1] / "forbidden-private-output"
            with self.assertRaisesRegex(PreparationError, "output"):
                prepare_candidate(**args)
            self.assertFalse(args["output_dir"].exists())

    def test_dry_run_cli_is_model_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_case(Path(tmp), ("train", "validation", "protected"))
            manifest_path = Path(tmp) / "provider.json"
            manifest_path.write_text(json.dumps(contract()))
            # Process-local tripwires supplement the gate's real network namespace.
            guard = '''
import os, pathlib, runpy, sys
blocked = {"torch", "transformers", "datasets", "trl", "peft", "requests", "httpx", "openai", "anthropic", "pip"}
def audit(event, args):
    if event == "import" and args[0].split(".")[0] in blocked:
        raise AssertionError("model/network/install import forbidden")
    if event.startswith("socket.") or event in {"os.system", "os.exec", "os.posix_spawn"}:
        raise AssertionError("network/shell forbidden")
    if event == "subprocess.Popen" and (pathlib.Path(args[0]).name != "git" or args[1][0] != "git"):
        raise AssertionError("only read-only git source inspection may spawn")
def profile(frame, event, arg):
    if event != "call":
        return
    name = pathlib.Path(frame.f_code.co_filename).name
    if name == "training_loop.py":
        raise AssertionError("unsafe original training_loop forbidden")
    if name in {"mega_training_data.py", "agentic_training_loop.py"} and frame.f_code.co_name not in {"<module>", "format_to_sharegpt", "format_for_sft"}:
        raise AssertionError("non-preparation original function forbidden")
sys.addaudithook(audit)
sys.setprofile(profile)
sys.argv = ["evolution_lab.training_provider", *sys.argv[1:]]
runpy.run_module("evolution_lab.training_provider", run_name="__main__")
'''
            command = [sys.executable, "-B", "-c", guard, "--dry-run", "--provider-root", str(PROVIDER), "--extractor-root", str(EXTRACTOR), "--corpus", str(args["corpus_path"]), "--lineage", str(args["lineage_path"]), "--output", str(args["output_dir"]), "--manifest", str(manifest_path)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((args["output_dir"] / "candidate.json").is_file(), "CLI did not emit a candidate")
            result = json.loads(completed.stdout)
            self.assertEqual(result, json.loads((args["output_dir"] / "candidate.json").read_text()))
            self.assertEqual(result["status"], "untrained")
            self.assertEqual(result["counts"], {"train": 1, "validation": 1, "protected": 1})
            for forbidden in ("--train", "--evaluate", "--download", "--install"):
                refused = subprocess.run(command + [forbidden], capture_output=True, text=True, timeout=15)
                self.assertEqual(refused.returncode, 2)
            command.remove("--dry-run")
            self.assertEqual(subprocess.run(command, capture_output=True, timeout=15).returncode, 2)

    def test_unsupported_text_shapes_cannot_be_silently_dropped(self):
        from evolution_lab.training_provider import PreparationError, prepare_candidate
        mutations = {
            "alternate_format": lambda r: r.update(conversations=[{"from": "human", "value": "wrong"}]),
            "nontext_content": lambda r: r["messages"][0].update(content={"text": "nested"}),
            "unknown_role": lambda r: r["messages"][0].update(role="developer"),
            "dropped_field": lambda r: r["messages"][0].update(images=["owned.png"]),
            "short_record": lambda r: r.update(messages=r["messages"][:1]),
            "no_target": lambda r: r["messages"][1].update(role="user"),
            "unknown_source": lambda r: r.update(source="unqualified-source"),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                args = make_case(Path(tmp))
                row = json.loads(args["corpus_path"].read_bytes())
                mutation(row)
                data = (json.dumps(row) + "\n").encode()
                args["corpus_path"].write_bytes(data)
                lineage = json.loads(args["lineage_path"].read_bytes())
                lineage["corpus_sha256"] = lineage["records"][0]["record_sha256"] = sha(data)
                args["lineage_path"].write_text(json.dumps(lineage))
                with self.assertRaises(PreparationError):
                    prepare_candidate(**args)
                self.assertFalse(args["output_dir"].exists())
        with tempfile.TemporaryDirectory() as tmp:
            args = make_case(Path(tmp), mutate_native=lambda e: e[1]["message"]["content"].append({"type": "image", "data": "owned"}))
            with self.assertRaisesRegex(PreparationError, "unsupported"):
                prepare_candidate(**args)
            self.assertFalse(args["output_dir"].exists())

    def test_manifest_is_an_allowlist_not_a_command_runner(self):
        from evolution_lab.training_provider import PreparationError, prepare_candidate
        for key, value in (("phases", ["prepare", "train"]), ("phases", ["candidate", "evaluate"]),
                           ("commands", ["training_loop.main"]), ("commands", ["check_model_health"]),
                           ("judge", {"mutable": True}), ("gates", ["accuracy=0"]),
                           ("verifier", {"command": "python live_eval.py", "revision": "main"}),
                           ("provider_revision", "main"), ("extractor_revision", "HEAD"),
                           ("install", ["pip", "install", "trl"])):
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as tmp:
                args = make_case(Path(tmp))
                args["manifest"][key] = value
                with self.assertRaisesRegex(PreparationError, "manifest"):
                    prepare_candidate(**args)
                self.assertFalse(args["output_dir"].exists())

    def test_tool_bearing_records_are_refused_without_artifacts(self):
        from evolution_lab.training_provider import PreparationError, prepare_candidate
        mutations = {
            "tool_uses": lambda e: e[1]["message"]["content"].append({"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "owned"}}),
            "separate_results": lambda e: e.append({"type": "tool_result", "toolResult": {"id": "t1", "output": "synthetic result"}}),
            "user_tool": lambda e: e[0].update(toolUse={"name": "read_file"}),
            "orphan_result": lambda e: e.insert(0, {"type": "tool_result", "toolResult": {"output": "otherwise dropped"}}),
            "multiple_calls": lambda e: e[1]["message"].update(tool_calls=[{"function": {"name": "one"}}, {"function": {"name": "two"}}]),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                args = make_case(Path(tmp), mutate_native=mutation)
                with self.assertRaisesRegex(PreparationError, "tool.*semantics"):
                    prepare_candidate(**args)
                self.assertFalse(args["output_dir"].exists())

    def test_lineage_integrity_is_required_before_preparation(self):
        from evolution_lab.training_provider import PreparationError, prepare_candidate
        mutations = {
            "missing_work_item": lambda b: b["records"][0].pop("work_item_id"),
            "blank_session": lambda b: b["records"][0].update(session_id=" "),
            "missing_record": lambda b: b["records"].pop(),
            "duplicate_index": lambda b: b["records"][1].update(record_index=0),
            "row_hash": lambda b: b["records"][0].update(record_sha256="0" * 64),
            "source_hash": lambda b: b["records"][0].update(source_sha256="0" * 64),
            "corpus_hash": lambda b: b.update(corpus_sha256="0" * 64),
            "origin_mismatch": lambda b: b["records"][0].update(session_id="invented-origin"),
            "source_mismatch": lambda b: b["records"][0].update(source_file=b["records"][1]["source_file"]),
            "unknown_bucket": lambda b: b["records"][0].update(split="unknown"),
            "work_overlap": lambda b: b["records"][1].update(work_item_id=b["records"][0]["work_item_id"]),
            "session_overlap": lambda b: b["records"][1].update(session_id=b["records"][0]["session_id"]),
            "unknown_protected_origin": lambda b: b["protected_sessions"].append("unknown-origin"),
            "protected_as_training": lambda b: b["protected_sessions"].append(b["records"][0]["session_id"]),
            "unlisted_protected": lambda b: b["records"][1].update(split="protected"),
            "unqualified_claim": lambda b: b.update(qualification="independently_verified"),
            "empty_training": lambda b: b["records"][0].update(split="validation"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                args = make_case(Path(tmp), ("train", "validation"))
                bindings = json.loads(args["lineage_path"].read_text())
                mutate(bindings)
                args["lineage_path"].write_text(json.dumps(bindings))
                with self.assertRaises(PreparationError):
                    prepare_candidate(**args)
                self.assertFalse(args["output_dir"].exists())

    def test_split_before_preparation_preserves_lineage(self):
        from evolution_lab.training_provider import prepare_candidate
        with tempfile.TemporaryDirectory() as tmp:
            args = make_case(Path(tmp), ("train", "train", "validation", "protected"))
            supplied = json.loads(args["lineage_path"].read_text())
            supplied["records"][1]["work_item_id"] = supplied["records"][0]["work_item_id"]
            args["lineage_path"].write_text(json.dumps(supplied))
            seen = []
            def trace(frame, event, arg):
                if event == "call" and frame.f_code.co_name == "format_to_sharegpt":
                    seen.append(frame.f_locals["row"]["session_id"])
            previous = sys.getprofile()
            try:
                sys.setprofile(trace)
                result = prepare_candidate(**args)
            finally:
                sys.setprofile(previous)
            expected = [r["session_id"] for r in supplied["records"] if r["split"] == "train"]
            self.assertEqual(seen, expected, "withheld records reached provider preparation")
            self.assertEqual(result["counts"], {"train": 2, "validation": 1, "protected": 1})
            sidecar = json.loads((args["output_dir"] / "lineage.json").read_text())
            self.assertEqual(sidecar["qualification"], "producer_asserted")
            self.assertFalse(sidecar["independently_qualified"])
            for original, retained in zip(supplied["records"], sidecar["records"], strict=True):
                self.assertEqual({k: retained[k] for k in original}, original)
                self.assertEqual(retained["prepared_index"] is None, original["split"] != "train")
            for filename in ("train.sft.jsonl", "lineage.json"):
                self.assertEqual(result["artifacts"][filename]["sha256"], sha((args["output_dir"] / filename).read_bytes()))
            self.assertEqual(result["inputs"]["corpus_sha256"], sha(args["corpus_path"].read_bytes()))
            self.assertEqual(result["inputs"]["lineage_sha256"], sha(args["lineage_path"].read_bytes()))
            self.assertEqual(result["claims"], {"verified_task_success": False, "verified_model_gain": False, "verified_cost_saving": False, "promotion": False})

    def test_real_extractor_to_original_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = make_case(Path(tmp))
            self.assertIsNotNone(importlib.util.find_spec("evolution_lab.training_provider"), "preparation-only integration is missing")
            from evolution_lab.training_provider import prepare_candidate
            result = prepare_candidate(**args)
            self.assertEqual(result["status"], "untrained")
            prepared = [json.loads(line) for line in (args["output_dir"] / "train.sft.jsonl").read_text().splitlines()]
            extracted = json.loads(args["corpus_path"].read_text())
            self.assertEqual(prepared, [{"messages": [{"role": m["role"], "content": m["content"]} for m in extracted["messages"]], "source": "claude-code", "tags": []}])
            self.assertEqual(result["counts"]["train"], 1)
            self.assertEqual(result["model_artifact"], None)


if __name__ == "__main__":
    unittest.main()
