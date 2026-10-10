"""Preparation-only bridge to pinned sybil-solutions/SouthpawIN source.

Importing this module is inert. No training, evaluation or provider SDK is used.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType

from .sealed_split import SessionSplit

PROVIDER_REVISION = "68ecf2057586a232bea45ad0fafcdfdfae56089f"
EXTRACTOR_REVISION = "9176c2dd81c8a101b52debd4cec88573284e2b30"
SOURCE_FILES = {
    "scripts/mega_training_data.py": "a6be6b819919cd656f1cc6aa8ec89bc69d8c3480e76cef4aaf076ba8c59876e3",
    "scripts/agentic_training_loop.py": "c4861fe646da21d45adb6e988aa18ad2ac4191dfa74ac30bf7977234a8d0e4e5",
}
EXTRACTOR_FILE = "extract_claude_code.py"
EXTRACTOR_SHA256 = "77c8c08ce7a8a612fcb26457212ac7ba95bc098f1c47cece3ef52eb2d3884904"


class PreparationError(ValueError):
    """The entire candidate is refused; no partial qualification is permitted."""


def _json(data: bytes):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PreparationError("ambiguous JSON: duplicate object key")
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=unique)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source(root: Path, revision: str, files: dict[str, str]) -> dict[str, bytes]:
    def git(*args):
        return subprocess.check_output(["git", "--no-optional-locks", "-C", str(root), "-c", "core.fsmonitor=false", *args], stderr=subprocess.PIPE, timeout=15)

    if git("rev-parse", "HEAD").decode().strip() != revision:
        raise PreparationError("source revision mismatch")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise PreparationError("source checkout is not clean")
    blobs = {}
    for name, digest in files.items():
        path = root / name
        if path.resolve() != root.resolve() / name:
            raise PreparationError("source path must not traverse a symlink")
        data = path.read_bytes()
        if _sha(data) != digest or data != git("show", f"{revision}:{name}"):
            raise PreparationError("source bytes differ from pin")
        blobs[name] = data
    return blobs


def _module(root: Path, name: str, data: bytes) -> ModuleType:
    # Execute the verified bytes, not a cached pyc or a module found on sys.path.
    module = ModuleType(f"_el_pinned_{Path(name).stem}")
    module.__file__ = str(root / name)
    exec(compile(data, module.__file__, "exec"), module.__dict__)
    return module


def _reject_tools(value) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            token = key.lower().replace("_", "")
            if (token.startswith("tool") or token in {"functioncall", "functioncalls"}
                    or (key in {"role", "type"} and isinstance(child, str)
                        and (child.startswith("tool") or child in {"function", "function_call"}))):
                raise PreparationError("unsupported tool-bearing record: original preparation loses tool-event semantics")
            _reject_tools(child)
    elif isinstance(value, list):
        for child in value:
            _reject_tools(child)


def _supported_text(row: dict, native: bytes) -> None:
    if not native.strip():
        raise PreparationError("empty native source cannot support an extracted row")
    _reject_tools(row)
    fields = {"messages", "source", "session_id", "project_path", "project_name", "source_file", "installation"}
    if set(row) != fields or row["source"] != "claude-code":
        raise PreparationError("unsupported extractor record format or source")
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        raise PreparationError("unsupported conversation: user and assistant text required")
    for message in messages:
        if (not isinstance(message, dict) or set(message) - {"role", "content", "model", "timestamp"}
                or message.get("role") not in {"user", "assistant"}
                or not isinstance(message.get("content"), str) or not message["content"].strip()):
            raise PreparationError("unsupported message semantics; only plain user/assistant text is supported")
    if {m["role"] for m in messages} != {"user", "assistant"}:
        raise PreparationError("unsupported conversation: user and assistant text required")
    for line in native.splitlines():
        if not line.strip():
            continue
        event = _json(line)
        _reject_tools(event)
        if not isinstance(event, dict) or event.get("type") not in {"user", "assistant"}:
            raise PreparationError("unsupported native event that the extractor may drop")
        message = event.get("message")
        if not isinstance(message, dict):
            raise PreparationError("unsupported native message")
        content = message.get("content")
        if isinstance(content, list):
            if not content or any(not isinstance(p, dict) or set(p) != {"type", "text"}
                                  or p["type"] != "text" or not isinstance(p["text"], str) for p in content):
                raise PreparationError("unsupported native content that the extractor may drop")
        elif not isinstance(content, str) or not content.strip():
            raise PreparationError("unsupported native content")


def _native_extraction_matches(row: dict, native: bytes, extractor: ModuleType) -> None:
    """Use the unchanged extractor on one captured source, never ambient history."""
    source = Path(row["source_file"])
    with tempfile.TemporaryDirectory(prefix="el-native-binding-") as tmp:
        project = Path(tmp) / source.parent.name
        project.mkdir(mode=0o700)
        staged = project / source.name
        with os.fdopen(os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as handle:
            handle.write(native)
        extracted = extractor.extract_claude_project_conversations(project)
    fields = ("messages", "source", "session_id", "project_path", "project_name")
    if len(extracted) != 1 or any(row.get(key) != extracted[0].get(key) for key in fields):
        raise PreparationError("record differs from pinned native extraction")


def _validate_lineage(corpus: bytes, bindings: dict, rows: list[dict], extractor: ModuleType) -> SessionSplit:
    required = {"schema", "qualification", "corpus_sha256", "protected_sessions", "records"}
    if (not isinstance(bindings, dict) or set(bindings) != required
            or bindings["schema"] != "evolution_lab.source_lineage.v1"
            or bindings["qualification"] != "producer_asserted"
            or bindings["corpus_sha256"] != _sha(corpus)):
        raise PreparationError("missing or invalid source lineage header")
    records = bindings["records"]
    if not isinstance(records, list) or not records or len(records) != len(rows):
        raise PreparationError("every extracted record requires exactly one lineage binding")
    protected = bindings["protected_sessions"]
    if (not isinstance(protected, list) or any(not isinstance(s, str) or not s.strip() for s in protected)
            or len(set(protected)) != len(protected)):
        raise PreparationError("protected origin inventory is required")
    fields = {"record_index", "record_sha256", "source_file", "source_sha256", "session_id", "work_item_id", "split"}
    ownership: dict[tuple[str, str], str] = {}
    groups: dict[str, list[str]] = {"train": [], "validation": [], "protected": []}
    for index, (row, record, line) in enumerate(zip(rows, records, corpus.splitlines(keepends=True), strict=True)):
        if (not isinstance(record, dict) or set(record) != fields
                or type(record["record_index"]) is not int or record["record_index"] != index
                or any(not isinstance(record[k], str) or not record[k].strip() for k in fields - {"record_index"})):
            raise PreparationError("missing or invalid native session/work-item lineage")
        if record["record_sha256"] != _sha(line):
            raise PreparationError("extracted record hash mismatch")
        if not isinstance(row, dict) or any(row.get(k) != record[k] for k in ("source_file", "session_id")):
            raise PreparationError("native source identity mismatch")
        source = Path(record["source_file"])
        if (not source.is_absolute() or source.is_symlink() or not source.is_file()
                or source.stem != record["session_id"]):
            raise PreparationError("native source bytes or session identity mismatch")
        native = source.read_bytes()
        if _sha(native) != record["source_sha256"]:
            raise PreparationError("native source bytes or session identity mismatch")
        canonical_source = str(source.resolve())
        if ("source_file", canonical_source) in ownership:
            raise PreparationError("duplicate native origin; extractor emits one row per source file")
        _supported_text(row, native)
        _native_extraction_matches(row, native, extractor)
        bucket = record["split"]
        if bucket not in groups:
            raise PreparationError("unknown partition; protected origin cannot be inferred")
        # Hash equality is only an exclusion check, never an origin identifier.
        for key in ("session_id", "work_item_id", "source_file", "source_sha256"):
            identity = (key, canonical_source if key == "source_file" else record[key])
            if identity in ownership and ownership[identity] != bucket:
                raise PreparationError("native group or identical source bytes overlap partitions")
            ownership[identity] = bucket
        groups[bucket].append(record["session_id"])
    if set(protected) != set(groups["protected"]):
        raise PreparationError("unknown or incorrectly partitioned protected origin")
    if not groups["train"]:
        raise PreparationError("no supported training partition rows")
    return SessionSplit(sorted(set(groups["train"])), sorted(set(groups["validation"])),
                        sorted(set(groups["protected"])), [], len(groups["train"]),
                        len(groups["validation"]), len(groups["protected"]), 0)


def prepare_candidate(*, provider_root: Path, extractor_root: Path, corpus_path: Path,
                      lineage_path: Path, output_dir: Path, manifest: dict) -> dict:
    """Prepare an explicitly untrained artifact using the original functions."""
    allowed = {
        "schema": "evolution_lab.training_provider.v1",
        "phases": ["prepare", "candidate"],
        "commands": ["mega_training_data.format_to_sharegpt", "agentic_training_loop.format_for_sft"],
        "provider_revision": PROVIDER_REVISION, "extractor_revision": EXTRACTOR_REVISION,
        "judge": None, "gates": [], "verifier": None,
    }
    if manifest != allowed:
        raise PreparationError("provider manifest is not the pinned preparation-only allowlist")
    if output_dir.exists() or output_dir.is_symlink():
        raise PreparationError("output must be a new private directory")
    output_dir = output_dir.resolve()
    for root in (Path(__file__).resolve().parents[1], provider_root.resolve(), extractor_root.resolve()):
        if output_dir.is_relative_to(root):
            raise PreparationError("output must be outside source repositories")
    blobs = _source(provider_root, PROVIDER_REVISION, SOURCE_FILES)
    extractor_blobs = _source(extractor_root, EXTRACTOR_REVISION, {EXTRACTOR_FILE: EXTRACTOR_SHA256})
    corpus_bytes = corpus_path.read_bytes()
    lineage_bytes = lineage_path.read_bytes()
    rows = [_json(line) for line in corpus_bytes.splitlines()]
    bindings = _json(lineage_bytes)
    extractor = _module(extractor_root, EXTRACTOR_FILE, extractor_blobs[EXTRACTOR_FILE])
    split = _validate_lineage(corpus_bytes, bindings, rows, extractor)
    counts = {"train": split.n_train_events, "validation": split.n_dev_events, "protected": split.n_sealed_events}
    mega = _module(provider_root, "scripts/mega_training_data.py", blobs["scripts/mega_training_data.py"])
    agentic = _module(provider_root, "scripts/agentic_training_loop.py", blobs["scripts/agentic_training_loop.py"])
    training = [(row, binding) for row, binding in zip(rows, bindings["records"], strict=True)
                if split.bucket_of(binding["session_id"]) == "train"]
    prepared = agentic.format_for_sft([
        {"conversations": mega.format_to_sharegpt(row, row["source"]), "source": row["source"], "tags": []}
        for row, _binding in training
    ])
    train_lines = [(json.dumps(row) + "\n").encode() for row in prepared]
    retained = [{**binding, "prepared_index": None, "prepared_sha256": None} for binding in bindings["records"]]
    for index, ((_row, binding), line) in enumerate(zip(training, train_lines, strict=True)):
        retained[binding["record_index"]].update(prepared_index=index, prepared_sha256=_sha(line))
    sidecar = {**bindings, "records": retained, "independently_qualified": False}
    artifacts = {"train.sft.jsonl": b"".join(train_lines),
                 "lineage.json": (json.dumps(sidecar, indent=2) + "\n").encode()}
    result = {
        "schema": "evolution_lab.training_candidate.v1", "status": "untrained",
        "model_artifact": None, "counts": counts,
        "eligibility": "preparation_only_not_training_authorization",
        "claims": {"verified_task_success": False, "verified_model_gain": False,
                   "verified_cost_saving": False, "promotion": False},
        "inputs": {"corpus_sha256": _sha(corpus_bytes), "lineage_sha256": _sha(lineage_bytes)},
        "sources": {"provider_revision": PROVIDER_REVISION, "provider_files": SOURCE_FILES,
                    "extractor_revision": EXTRACTOR_REVISION, "extractor_sha256": EXTRACTOR_SHA256},
        "split": split.to_dict(),
        "artifacts": {name: {"sha256": _sha(data), "bytes": len(data)} for name, data in artifacts.items()},
    }
    # The manifest is the final completion marker; never overwrite any artifact.
    artifacts["candidate.json"] = (json.dumps(result, indent=2) + "\n").encode()
    output_dir.mkdir(mode=0o700)
    for name, data in artifacts.items():
        with os.fdopen(os.open(output_dir / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as handle:
            handle.write(data)
    return result


def main(argv: list[str] | None = None) -> int:
    """Explicit offline preparation CLI; no executable phase is configurable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", required=True, action="store_true", help="prepare an untrained candidate only")
    for flag in ("provider-root", "extractor-root", "corpus", "lineage", "output", "manifest"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        with redirect_stdout(sys.stderr):
            result = prepare_candidate(provider_root=args.provider_root, extractor_root=args.extractor_root,
                                       corpus_path=args.corpus, lineage_path=args.lineage,
                                       output_dir=args.output, manifest=_json(args.manifest.read_bytes()))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
