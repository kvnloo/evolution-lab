"""Offline UNIT characterization, not a judge, provider client, or promotion gate.

Imports unchanged public owning tests/collectors from exact source exports.
Usage: python trace_seam_probe.py SOURCE_DIRECTORY PRIVATE_OUTPUT_DIRECTORY
SOURCE_DIRECTORY contains z0evals/ and z0intelligence/ at the revisions below.
No natural work-item outcomes or model-performance measurements are generated.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

EVAL_REV = "6652713e25fb159a047457acfe348bd5ec98e146"
Z0_REV = "5873723eb33bd5c09e5220b3619e25e01efacd64"
FROZEN = {
    "z0evals/studies/aodl-admission-v1/collect_golden.py": "f9848e75b6f39fdbd3575fafa817787a5f0bbdc51344cf438e2fe5031ce4d741",
    "z0evals/studies/aodl-admission-v1/import_golden.py": "04f22200f3e3aef971d41ad94cc79fa5a6bf61ca2bef012bc3fdaf2d16c3d6dd",
    "z0evals/studies/aodl-admission-v1/golden-trace-manifest.yaml": "f88d7f5f9c1f6bcccda10d4b23f16acd672d6548f9dd8d990d770fb4e1695a82",
    "z0evals/studies/aodl-admission-v1/golden-trace.schema.json": "24d5be7a2cafe0b80bfd5d7fd527b83ceed0a5cab4137306fda75ffa95d2b7d0",
    "z0evals/studies/aodl-admission-v1/golden-canary-bundle.schema.json": "77479a20ef16492051e14904cc4765c05c66dba55ba0b0a354274bf7823fed27",
    "z0evals/tests/test_golden_trace.py": "8780e274b4f720b5c8bd52be4b60661085c9c5243b87856a4cf16f21dab8815f",
    "z0intelligence/src/z0int/hermes_decisions.py": "9fb18d826684c44729d0ba66fff3d5c795a53ae7540fcba1b8facd5c9874c67b",
    "z0intelligence/src/z0int/decision_opportunity.py": "6dc904d6089abe5ffb061c240296974718d4cd53fb992adbd2d245ac6d27ae8f",
    "z0intelligence/src/z0int/paths.py": "fe2cca470385cf4ecb6e8d6909f87f8f8d9901fd8473e325024648eccc10bf16",
    "z0intelligence/src/z0int/receipt.py": "1b1442418fa2d39ebd0f6192a8c777f69b84f25df8263063780b4aeb5206f32c",
    "z0intelligence/src/z0int/__init__.py": "3fc24a862af53f48283e26b4fa4a885142c9ebc4d8007aa050b15701c0942ca4",
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    source, output = (Path(p).resolve() for p in sys.argv[1:])
    os.umask(0o077)
    sys.dont_write_bytecode = True
    for name, digest in FROZEN.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Wrong frozen source: {name}")
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    owning = load(source / "z0evals/tests/test_golden_trace.py", "owning_tests")
    sys.path.insert(0, str(source / "z0intelligence/src"))
    hermes = importlib.import_module("z0int.hermes_decisions")
    opportunity = importlib.import_module("z0int.decision_opportunity")
    receipt = importlib.import_module("z0int.receipt")

    # Public owning fixture, never a historical call or a protected label.
    cases = {
        "no_gold": None,
        "positive_gold": {"verified_success": True},
        "negative_gold": {"verified_success": False},
        "contradictory_gold": {"verified_success": True, "test_pass": False},
    }
    results = {}
    normalized = {}
    for name, outcome in cases.items():
        receipts, _, tokens = owning.base_rows()
        outcomes = [] if outcome is None else [{
            "trace_id": owning.PHYSICAL, "outcome_tier": "gold", "outcome": outcome,
        }]
        proof = owning.G.collect(receipts, outcomes, tokens, owning.ROOT_TRACE)
        case = output / name
        case.mkdir()
        for filename, rows in (("receipts", receipts), ("outcomes", outcomes), ("tokens", tokens)):
            (case / (filename + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
        run = subprocess.run([
            sys.executable, str(source / "z0evals/studies/aodl-admission-v1/collect_golden.py"),
            "--receipts", str(case / "receipts.jsonl"), "--outcomes", str(case / "outcomes.jsonl"),
            "--tokenomics", str(case / "tokens.jsonl"), "--trace-id", owning.ROOT_TRACE,
            "--out", str(case / "proof.json"), "--require-verified",
        ], capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        assert run.returncode == (1 if outcome is None else 0), run.stderr
        results[name] = {
            "structural_complete": proof["structural_execution_complete"],
            "outcome_complete": proof["verified_outcome_complete"],
            "require_verified_exit": run.returncode,
            "outcome": outcome,
        }
        if outcome is not None:
            owner_outcome = receipt.Outcome(**outcome)
            normalized_rows = [{"trace_id": owning.PHYSICAL,
                                "outcome_tier": owner_outcome.tier(), "outcome": outcome}]
            normalized_proof = owning.G.collect(receipts, normalized_rows, tokens, owning.ROOT_TRACE)
            assert normalized_proof["verified_outcome_complete"] == (name == "positive_gold")
            normalized[name] = {"owner_tier": owner_outcome.tier(),
                                "owner_is_verified": owner_outcome.is_verified(),
                                "collector_outcome_complete": normalized_proof["verified_outcome_complete"]}

    # Existing importer checks completeness flags, not positivity of joined outcome.
    bundle_dir = output / "negative-bundle"
    owning.make_bundle(bundle_dir)
    golden_path = bundle_dir / "golden-trace.json"
    golden = json.loads(golden_path.read_text())
    golden["stages"]["verified_outcome"]["outcome"] = {"verified_success": False}
    golden_path.write_text(json.dumps(golden) + "\n")
    owning.I.validate_bundle(bundle_dir)  # validation only: no import into results

    receipts, _, tokens = owning.base_rows()
    unknown_usage = copy.deepcopy(receipts)
    unknown_usage[-1].pop("output_tokens")
    assert not owning.G.collect(unknown_usage, [], tokens, owning.ROOT_TRACE)["structural_execution_complete"]
    wrong_link = copy.deepcopy(receipts)
    wrong_link[-2]["extra"]["permit_dispatch_id"] = "different-dispatch"
    assert not owning.G.collect(wrong_link, [], tokens, owning.ROOT_TRACE)["structural_execution_complete"]

    observed_failed = hermes.observed({"ended": "completed", "verified_success": False})
    assert observed_failed == "answered"  # behaviour, explicitly not correctness
    simple = opportunity.build_decision_opportunity(".", "what branch am I on?", packet={})
    assert opportunity.deterministic_gate(simple) == "ASK"
    unsafe = opportunity.build_decision_opportunity(".", "make a change", packet={}, effects=["write"])
    assert opportunity.deterministic_gate(unsafe) == "ASK"
    assert simple["expected_outcome"]["verifier"] is None
    result = {
        "evidence_kind": "synthetic_unit_characterization_only",
        "z0evals_revision": EVAL_REV,
        "z0intelligence_revision": Z0_REV,
        "source_sha256": FROZEN,
        "cases": results,
        "existing_owner_normalized_control": normalized,
        "negative_outcome_bundle_validation": "ACCEPTED",
        "missing_usage_control": "REJECTED",
        "wrong_dispatch_control": "REJECTED",
        "shadow_observed_completed_with_false_success": observed_failed,
        "missing_fact_gate": opportunity.deterministic_gate(simple),
        "unauthorized_write_gate": opportunity.deterministic_gate(unsafe),
        "natural_work_items": 0,
        "provider_calls": 0,
        "independent_review": "NOT_RUN",
        "offload_performance": "NOT_COMPARABLE",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
