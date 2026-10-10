#!/usr/bin/env python3
"""Check private generator artifacts against a frozen run receipt, without exporting text.

This read-only producer boundary does not judge generated instructions or grant
privacy, installation, held-out efficacy or redistribution approval.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import subprocess
import sys


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_receipt_key")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError("nonstandard_json_constant")


def git(generator: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(generator), *args], capture_output=True,
                            text=True, check=False, timeout=15)
    require(result.returncode == 0, "generator_git_unavailable")
    return result.stdout.strip()


def check_lineage(lineage: dict, corpus: Path, source_hash: str) -> dict:
    require(isinstance(lineage, dict) and lineage.get("corpus_sha256") == source_hash,
            "lineage_corpus_changed")
    require(lineage.get("grouping_basis") == "source_reported_conservative_work_item",
            "unsupported_grouping_basis")
    mappings = lineage.get("records")
    if not isinstance(mappings, list):
        raise ValueError("missing_source_mapping")
    by_record = {}
    for entry in mappings:
        require(isinstance(entry, dict), "invalid_source_mapping")
        for key in ("record_sha256", "session_sha256", "work_item_sha256"):
            value = entry.get(key)
            require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
                    "invalid_source_identity")
        require(entry["record_sha256"] not in by_record, "ambiguous_source_mapping")
        by_record[entry["record_sha256"]] = entry
    sessions = {}
    messages_by_session = {}
    seen = set()
    source_digest = hashlib.sha256()
    with corpus.open("rb") as stream:
        for raw in stream:
            source_digest.update(raw)
            raw = raw.rstrip(b"\r\n")
            if not raw.strip():
                continue
            record_hash = hashlib.sha256(raw).hexdigest()
            require(record_hash not in seen and record_hash in by_record,
                    "duplicate_or_unmapped_source_record")
            seen.add(record_hash)
            record = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
            require(isinstance(record, dict), "invalid_source_record")
            session = record.get("session_id")
            require(isinstance(session, str) and bool(session.strip()), "missing_native_session")
            session_hash = hashlib.sha256(session.encode("utf-8")).hexdigest()
            entry = by_record[record_hash]
            require(session_hash == entry["session_sha256"], "source_session_changed")
            group = entry["work_item_sha256"]
            require(session_hash not in sessions or sessions[session_hash] == group,
                    "native_session_split_across_work_items")
            sessions[session_hash] = group
            messages = record.get("messages")
            require(isinstance(messages, list), "missing_source_messages")
            message_hashes = [hashlib.sha256(json.dumps(m, sort_keys=True,
                              ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
                              for m in messages]
            messages_by_session.setdefault(session_hash, []).append(message_hashes)
    require(source_digest.hexdigest() == source_hash, "source_bytes_changed_during_lineage_read")
    require(bool(seen) and seen == set(by_record), "incomplete_source_mapping")
    shared_pairs = shared_messages = 0
    for branches in messages_by_session.values():
        for left, right in combinations(branches, 2):
            prefix = 0
            for a, b in zip(left, right):
                if a != b:
                    break
                prefix += 1
            if prefix:
                shared_pairs += 1
                shared_messages += prefix
    return dict(source_records=len(seen), source_sessions=len(sessions),
                source_work_items=len(set(sessions.values())),
                source_work_item_sha256=sorted(set(sessions.values())),
                shared_prefix_pairs=shared_pairs, shared_prefix_messages=shared_messages,
                source_group_status="producer_asserted_not_independently_verified")


def check(receipt: dict, corpus: Path, candidates: Path, generator: Path) -> dict:
    require(isinstance(receipt, dict), "invalid_receipt")
    require(type(receipt.get("generator_exit")) is int and receipt["generator_exit"] == 0,
            "generator_failed_or_unknown")
    require(receipt.get("outputs") == "private_only" and receipt.get("installed") is False
            and receipt.get("efficacy_claim") is False
            and receipt.get("independent_qualification") == "not_run", "review_only_required")
    ref = receipt.get("generator_ref")
    source_hash = receipt.get("corpus_sha256")
    hashes = receipt.get("candidate_sha256")
    require(isinstance(ref, str) and re.fullmatch(r"[0-9a-f]{40}", ref) is not None,
            "invalid_generator_ref")
    require(isinstance(source_hash, str) and re.fullmatch(r"[0-9a-f]{64}", source_hash) is not None,
            "invalid_corpus_hash")
    if not isinstance(hashes, list):
        raise ValueError("invalid_candidate_hashes")
    require(all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes),
            "invalid_candidate_hashes")
    count = receipt.get("expected_skill_count")
    require(type(count) is int and count > 0 and type(receipt.get("generated_skill_files")) is int
            and receipt["generated_skill_files"] == count and len(hashes) == count,
            "declared_count_mismatch")
    require(len(set(hashes)) == count, "duplicate_candidate_identity")
    require(git(generator, "rev-parse", "HEAD") == ref, "generator_ref_changed")
    require(git(generator, "status", "--porcelain", "--untracked-files=all") == "",
            "generator_not_clean")
    require(corpus.is_file() and not corpus.is_symlink(), "missing_or_linked_corpus")
    require(digest(corpus) == source_hash, "corpus_hash_changed")
    require(candidates.is_dir() and not candidates.is_symlink(), "missing_or_linked_candidates")
    # The unchanged external generator writes exactly one directory per draft.
    folders = list(candidates.iterdir())
    require(len(folders) == count and all(p.is_dir() and not p.is_symlink() for p in folders),
            "candidate_count_or_layout_mismatch")
    files = [p / "SKILL.md" for p in folders]
    require(all(list(p.iterdir()) == [p / "SKILL.md"] for p in folders),
            "unaccounted_candidate_output")
    require(all(p.is_file() and not p.is_symlink() for p in files), "missing_or_linked_candidate")
    require(sorted(digest(p) for p in files) == sorted(hashes), "candidate_hash_changed")
    return dict(schema="evolution-lab.conversation-skills.identity.v1",
                status="review_only_identity_consistent", generator_ref=ref,
                corpus_sha256=source_hash, candidate_sha256=sorted(hashes),
                candidate_count=count, installed=False, promotion_eligible=False,
                independent_qualification="not_run", privacy_clearance="not_granted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "corpus", "candidates", "generator"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True,
                        help="Expected frozen receipt bytes; obtain from the producer handoff")
    parser.add_argument("--lineage", type=Path,
                        help="Optional private record/session/work-item hash mapping")
    parser.add_argument("--lineage-sha256")
    args = parser.parse_args()
    try:
        require(re.fullmatch(r"[0-9a-f]{64}", args.receipt_sha256) is not None,
                "invalid_receipt_pin")
        raw = args.receipt.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == args.receipt_sha256, "receipt_hash_changed")
        receipt = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
        result = check(receipt, args.corpus, args.candidates, args.generator)
        result["receipt_sha256"] = args.receipt_sha256
        require(bool(args.lineage) == bool(args.lineage_sha256), "partial_lineage_pin")
        result["source_group_status"] = "not_supplied"
        if args.lineage:
            require(re.fullmatch(r"[0-9a-f]{64}", args.lineage_sha256) is not None,
                    "invalid_lineage_pin")
            raw_lineage = args.lineage.read_bytes()
            require(hashlib.sha256(raw_lineage).hexdigest() == args.lineage_sha256,
                    "lineage_hash_changed")
            lineage = json.loads(raw_lineage, object_pairs_hook=unique_object,
                                 parse_constant=reject_constant)
            result.update(check_lineage(lineage, args.corpus, receipt["corpus_sha256"]))
            result["lineage_sha256"] = args.lineage_sha256
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        # Never include source text, draft names, caller-controlled values or private paths.
        print(json.dumps(dict(status="refused", reason="artifact_identity_inconsistent")))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
