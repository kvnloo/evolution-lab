#!/usr/bin/env python3
"""Check private generator artifacts against a frozen run receipt, without exporting text.

This read-only producer boundary does not judge generated instructions or grant
privacy, installation, held-out efficacy or redistribution approval.
"""
from __future__ import annotations

import argparse
import hashlib
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


def git(generator: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(generator), *args], capture_output=True,
                            text=True, check=False, timeout=15)
    require(result.returncode == 0, "generator_git_unavailable")
    return result.stdout.strip()


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
    args = parser.parse_args()
    try:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        result = check(receipt, args.corpus, args.candidates, args.generator)
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        # Never include source text, draft names, caller-controlled values or private paths.
        print(json.dumps(dict(status="refused", reason="artifact_identity_inconsistent")))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
