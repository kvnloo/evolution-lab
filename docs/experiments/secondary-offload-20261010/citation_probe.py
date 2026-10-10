"""Fixed offline development controls; imports the existing grounding function unchanged.

Not a citation parser, outcome judge, safety gate, or model benchmark.
Usage: python citation_probe.py Z0_SOURCE ZE_SOURCE PRIVATE_OUTPUT
Run from the Evolution Lab checkout. All staged files are public source bytes.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

Z0_REV = "5a9ce7cf6e98b486668ad91b806a9261fdb4c7e4"
ZE_REV = "6652713e25fb159a047457acfe348bd5ec98e146"
OLD = "26d92191440449eac7cd6cdd783d0d6c646e46a6"
CURRENT = "0212fee31a78cdba506f4d2684813962bfe0efb0"
MODULE = "src/z0int/verification_density.py"
WORKFLOW = ".github/workflows/test.yml"
MANIFEST = "studies/aodl-admission-v1/golden-trace-manifest.yaml"
HASHES = {
    "outcome": "e215860d2120e7b51a974306b0787a12d0e428cb81367d83c610e11fcfa2eeb9",
    "package": "3fc24a862af53f48283e26b4fa4a885142c9ebc4d8007aa050b15701c0942ca4",
    "module": "8f1d10d6e53a6e7e0f51164daa48ae7ec919ad1cde80a5a61300afa25cf131b8",
    "old": "6273f69106bb52a900e5162cc24ea84e63f23d757446e43c2e27f42b616e4bf8",
    "current": "5cb8ea19a3f038a46c61a42034a39fe6bb2cc7e48027c30fe8ee59de44b834c3",
    "manifest": "f88d7f5f9c1f6bcccda10d4b23f16acd672d6548f9dd8d990d770fb4e1695a82",
}


def main():
    z0, ze, output = (Path(p).resolve() for p in sys.argv[1:])
    os.umask(0o077)
    sys.dont_write_bytecode = True
    sources = {"module": (z0 / MODULE).read_bytes(), "manifest": (ze / MANIFEST).read_bytes()}
    sources["outcome"] = (z0 / "src/z0int/outcome_verifier.py").read_bytes()
    sources["package"] = (z0 / "src/z0int/__init__.py").read_bytes()
    for key, revision in (("old", OLD), ("current", CURRENT)):
        sources[key] = subprocess.check_output(["git", "show", f"{revision}:{WORKFLOW}"])
    for name, data in sources.items():
        if hashlib.sha256(data).hexdigest() != HASHES[name]:
            raise ValueError(f"Wrong frozen source: {name}")
    sys.path.insert(0, str(z0 / "src"))
    module = importlib.import_module("z0int.verification_density")
    assert module.PROMOTED == frozenset()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, path in (("old", WORKFLOW), ("current", WORKFLOW), ("manifest", MANIFEST)):
        dest = output / name / path
        dest.parent.mkdir(mode=0o700, parents=True)
        dest.write_bytes(sources[name])
        os.utime(dest, (100, 100))  # Known pre-answer file; not a historical timestamp.
    n = len(sources["old"].splitlines())
    # Synthetic statements on real source files; none are observed model responses.
    cases = [
        ("existing_path", "old", f"See {WORKFLOW}:13.", {}, "answer_grounded"),
        ("false_statement", "old", f"{WORKFLOW}:13 installs the jax extra.", {}, "answer_grounded"),
        ("missing_path", "old", "missing-source.py:1", {}, "answer_ungrounded"),
        ("beyond_eof", "old", f"{WORKFLOW}:999", {}, "answer_ungrounded"),
        ("zero_line", "old", f"{WORKFLOW}:0", {}, "answer_grounded"),
        ("one_past_eof", "old", f"{WORKFLOW}:{n + 1}", {}, "answer_grounded"),
        ("range_beyond_eof", "old", f"{WORKFLOW}:13-999", {}, "answer_grounded"),
        ("stale_revision_statement", "current", f"At {OLD}, {WORKFLOW}:13 installs the jax extra.", {}, "answer_grounded"),
        ("manifest_path", "manifest", f"See {MANIFEST}:24.", {}, "answer_grounded"),
        ("false_manifest_statement", "manifest", f"{MANIFEST}:24 proves all real tasks succeed.", {}, "answer_grounded"),
        ("url_not_scored", "old", "https://example.invalid/missing.py:99", {}, None),
        ("no_citation", "old", "Insufficient evidence; abstain.", {}, None),
        ("missing_base", "absent", f"{WORKFLOW}:13", {}, None),
        ("edited_turn", "old", f"{WORKFLOW}:13", {"edit_ops": ["synthetic-edit-marker"]}, None),
        ("post_answer_file", "old", f"{WORKFLOW}:999", {"ended_at": 50}, "answer_grounded"),
    ]
    rows = []
    for name, root, text, extra, expected in cases:
        turn = {"cwd": str(output / root), "_final_text": text, "ended_at": 200, **extra}
        # No Git suffix fallback needed for these direct staged paths; no peer tree read.
        signal = module.grounding(turn, lambda _: None, {})
        actual = signal["kind"] if signal else None
        assert actual == expected, (name, actual, expected)
        if signal:
            assert signal["confidence"] == "low" and signal["label_class"] == "soft"
        rows.append({"case": name, "source": root, "signal": signal})
    # Stronger existing no-model primitive for this exact source fact, not a new judge.
    old_line = sources["old"].decode().splitlines()[12]
    current_line = sources["current"].decode().splitlines()[14]
    assert old_line.strip() == "- run: pip install -e ."
    assert current_line.strip() == '- run: pip install -e ".[jax]"'
    result = {
        "classification": "PARTIAL", "performance": "NOT_COMPARABLE",
        "independent_review": "NOT_RUN", "input_kind": "synthetic controls on public natural-task source files",
        "revisions": {"grounding_feature_branch": Z0_REV, "z0evals": ZE_REV, "EL28": OLD, "EL_nightly": CURRENT},
        "source_sha256": HASHES, "cases": rows,
        "source_fact_control": {"old_install_line": old_line, "current_install_line": current_line},
        "provider_calls": 0, "model_efficacy": "NOT_RUN", "full_workflow_cost": None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
