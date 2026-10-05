"""Consume score-only protected certification from a pinned z0evals checkout."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .training_provider import ProviderProtocolError, load_manifest

PIN_RE = re.compile(r"^[0-9a-f]{40}$")
RESULT_SCHEMA = "z0eval.result.v1"


class CertificationError(ValueError):
    pass


def _git(root: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CertificationError(f"git inspection failed: {exc}") from exc
    if proc.returncode:
        raise CertificationError(proc.stderr.strip() or proc.stdout.strip() or "git failed")
    return proc.stdout.strip()


def _repo_from_remote(remote: str) -> str | None:
    value = remote.strip().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    if value.startswith("git@github.com:"):
        return value.split(":", 1)[1]
    return value.split("github.com/", 1)[1] if "github.com/" in value else None


def _safe_child(root: Path, rel: str, field: str) -> Path:
    part = Path(rel)
    if part.is_absolute() or ".." in part.parts:
        raise CertificationError(f"{field} must stay inside the evaluator checkout")
    target = (root / part).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise CertificationError(f"{field} escapes the evaluator checkout") from exc
    return target


def load_certification_config(provider_manifest_path: Path) -> tuple[Any, dict[str, Any]]:
    provider = load_manifest(provider_manifest_path)
    raw = json.loads(provider_manifest_path.read_text(encoding="utf-8"))
    cert = raw.get("certification")
    if not isinstance(cert, dict):
        raise CertificationError("provider manifest must declare certification")
    evaluator_manifest = str(cert.get("evaluator_manifest") or "")
    cohort = str(cert.get("cohort") or "")
    required_schema = str(cert.get("result_schema") or "")
    if not evaluator_manifest or not cohort or required_schema != RESULT_SCHEMA:
        raise CertificationError("certification manifest/cohort/result_schema are required")
    return provider, cert


def preflight_evaluator(provider: Any, cert: dict[str, Any], evaluator_root: Path) -> dict[str, str]:
    root = evaluator_root.resolve()
    if not root.is_dir():
        raise CertificationError(f"z0evals checkout does not exist: {root}")
    head = _git(root, "rev-parse", "HEAD")
    if not PIN_RE.fullmatch(head):
        raise CertificationError("z0evals checkout HEAD is not a full git SHA")
    if head != provider.eval_revision:
        raise CertificationError(
            f"z0evals revision mismatch: expected {provider.eval_revision}, got {head}"
        )
    origin = _repo_from_remote(_git(root, "remote", "get-url", "origin"))
    if origin != provider.eval_repo:
        raise CertificationError(
            f"z0evals origin mismatch: expected {provider.eval_repo}, got {origin or 'unknown'}"
        )

    script = _safe_child(root, "scripts/protected_eval.py", "protected evaluator script")
    manifest = _safe_child(root, str(cert["evaluator_manifest"]), "certification.evaluator_manifest")
    if not script.is_file():
        raise CertificationError("pinned z0evals checkout has no protected evaluator script")
    if not manifest.is_file():
        raise CertificationError("pinned z0evals checkout has no declared evaluator manifest")
    return {
        "repo": provider.eval_repo,
        "revision": head,
        "suite": provider.eval_suite,
        "evaluator_manifest": str(cert["evaluator_manifest"]),
        "cohort": str(cert["cohort"]),
    }


def _validate_result(
    result: dict[str, Any],
    *,
    provider: Any,
    cert: dict[str, Any],
    candidate_id: str,
    candidate_revision: str,
) -> None:
    if result.get("schema") != RESULT_SCHEMA:
        raise CertificationError("protected evaluator returned the wrong result schema")
    if result.get("suite_id") != provider.eval_suite:
        raise CertificationError("protected evaluator returned the wrong suite")
    if result.get("cohort") != cert["cohort"]:
        raise CertificationError("protected evaluator returned the wrong cohort")
    if result.get("candidate_id") != candidate_id:
        raise CertificationError("protected evaluator returned the wrong candidate_id")
    if result.get("candidate_revision") != candidate_revision:
        raise CertificationError("protected evaluator returned the wrong candidate_revision")
    if result.get("sealed_credit") is not True:
        raise CertificationError("protected evaluator result does not carry sealed credit")
    if result.get("verdict") not in {"KEEP", "DISCARD", "PARTIAL", "NOT_COMPARABLE"}:
        raise CertificationError("protected evaluator returned an invalid verdict")
    digest = str(result.get("result_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise CertificationError("protected evaluator result is missing result_sha256")


def certify_candidate(
    *,
    provider_manifest_path: Path,
    evaluator_root: Path,
    state_dir: Path,
    predictions_path: Path,
    candidate_id: str,
    candidate_revision: str,
    timeout_s: int = 300,
) -> dict[str, Any]:
    if not PIN_RE.fullmatch(candidate_revision):
        raise CertificationError("candidate_revision must be a full 40-character git SHA")
    if not state_dir.is_absolute():
        raise CertificationError("state_dir must be absolute")
    if not predictions_path.is_file():
        raise CertificationError("predictions file does not exist")

    try:
        provider, cert = load_certification_config(provider_manifest_path)
    except ProviderProtocolError as exc:
        raise CertificationError(str(exc)) from exc
    preflight_evaluator(provider, cert, evaluator_root)

    root = evaluator_root.resolve()
    command = [
        "python3",
        "scripts/protected_eval.py",
        "--manifest",
        str(cert["evaluator_manifest"]),
        "--state-dir",
        str(state_dir),
        "score",
        "--cohort",
        str(cert["cohort"]),
        "--predictions",
        str(predictions_path.resolve()),
        "--candidate-id",
        candidate_id,
        "--candidate-revision",
        candidate_revision,
    ]
    try:
        proc = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CertificationError(f"protected evaluator invocation failed: {exc}") from exc
    if proc.returncode:
        message = proc.stderr.strip() or proc.stdout.strip() or "protected evaluator failed"
        raise CertificationError(message)

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise CertificationError("protected evaluator returned non-JSON output") from exc
    if not isinstance(result, dict):
        raise CertificationError("protected evaluator result must be an object")
    _validate_result(
        result,
        provider=provider,
        cert=cert,
        candidate_id=candidate_id,
        candidate_revision=candidate_revision,
    )
    return result


def promotion_allowed(result: dict[str, Any]) -> bool:
    return (
        result.get("schema") == RESULT_SCHEMA
        and result.get("sealed_credit") is True
        and result.get("verdict") == "KEEP"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Request protected z0eval certification")
    parser.add_argument("--provider-manifest", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        result = certify_candidate(
            provider_manifest_path=args.provider_manifest,
            evaluator_root=args.evaluator_root,
            state_dir=args.state_dir,
            predictions_path=args.predictions,
            candidate_id=args.candidate_id,
            candidate_revision=args.candidate_revision,
        )
    except CertificationError as exc:
        raise SystemExit(f"certification refused: {exc}") from exc

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": result, "promotion_allowed": promotion_allowed(result)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
