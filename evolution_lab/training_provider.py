"""Fail-closed adapter for external training providers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import string
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL_SCHEMA = "z0.training.autoresearch.v1"
RECEIPT_SCHEMA = "z0.training.receipt.v1"
PIN_RE = re.compile(r"^[0-9a-f]{40}$")
FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PHASES = frozenset({"prepare_data", "train", "evolve", "evaluate"})


class ProviderProtocolError(ValueError):
    pass


def _pin(value: Any, field: str) -> str:
    value = str(value or "")
    if not PIN_RE.fullmatch(value):
        raise ProviderProtocolError(f"{field} must be a full 40-character git SHA")
    return value


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderProtocolError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProviderProtocolError(f"{path} must contain a JSON object")
    return value


@dataclass(frozen=True)
class CommandSpec:
    capability: str
    argv: tuple[str, ...]
    timeout_s: int


@dataclass(frozen=True)
class ProviderManifest:
    repo: str
    upstream: str
    revision: str
    hermes_profile: str
    capabilities: frozenset[str]
    commands: dict[str, CommandSpec]
    eval_repo: str
    eval_revision: str
    eval_suite: str
    training_forbidden: frozenset[str]
    mutable_forbidden: frozenset[str]
    execution_allowed: frozenset[str]
    env_passthrough: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ProviderManifest":
        if raw.get("schema") != PROTOCOL_SCHEMA:
            raise ProviderProtocolError(f"schema must be {PROTOCOL_SCHEMA}")
        provider = raw.get("provider") or {}
        repo, upstream = str(provider.get("repo") or ""), str(provider.get("upstream") or "")
        if "/" not in repo or "/" not in upstream:
            raise ProviderProtocolError("provider repo and upstream must be owner/repo")
        revision = _pin(provider.get("revision"), "provider.revision")

        profile = str(raw.get("hermes_profile") or "")
        if not profile or Path(profile).is_absolute() or ".." in Path(profile).parts:
            raise ProviderProtocolError("hermes_profile must be a safe relative path")

        capabilities = frozenset(map(str, raw.get("capabilities") or []))
        if not capabilities:
            raise ProviderProtocolError("capabilities must not be empty")
        commands: dict[str, CommandSpec] = {}
        for phase, spec in (raw.get("commands") or {}).items():
            if phase not in PHASES or not isinstance(spec, dict):
                raise ProviderProtocolError(f"invalid provider phase: {phase}")
            capability = str(spec.get("capability") or "")
            if capability not in capabilities:
                raise ProviderProtocolError(f"commands.{phase} requires undeclared capability {capability!r}")
            argv = spec.get("argv")
            if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
                raise ProviderProtocolError(f"commands.{phase}.argv must be a non-empty string list")
            timeout_s = int(spec.get("timeout_s", 3600))
            if timeout_s < 1 or timeout_s > 86400:
                raise ProviderProtocolError(f"commands.{phase}.timeout_s must be in [1, 86400]")
            commands[phase] = CommandSpec(capability, tuple(argv), timeout_s)
        if not commands:
            raise ProviderProtocolError("commands must not be empty")

        ev = raw.get("eval_suite") or {}
        eval_repo = str(ev.get("repo") or "")
        if "/" not in eval_repo:
            raise ProviderProtocolError("eval_suite.repo must be owner/repo")
        eval_revision = _pin(ev.get("revision"), "eval_suite.revision")
        eval_suite = str(ev.get("suite") or "")
        if not eval_suite:
            raise ProviderProtocolError("eval_suite.suite must not be empty")

        protected = raw.get("protected") or {}
        train_forbidden = frozenset(map(str, protected.get("training_forbidden") or []))
        mutable_forbidden = frozenset(map(str, protected.get("mutable_forbidden") or []))
        if not train_forbidden or not mutable_forbidden:
            raise ProviderProtocolError("protected training and mutable sets must not be empty")

        execution = raw.get("execution") or {}
        allowed = frozenset(map(str, execution.get("allowed_phases") or []))
        unknown = allowed - commands.keys()
        if unknown:
            raise ProviderProtocolError("execution allows undeclared phases: " + ", ".join(sorted(unknown)))
        env_passthrough = tuple(map(str, execution.get("env_passthrough") or ["PATH", "HOME"]))
        return cls(repo, upstream, revision, profile, capabilities, commands, eval_repo,
                   eval_revision, eval_suite, train_forbidden, mutable_forbidden,
                   allowed, env_passthrough)


def load_manifest(path: Path) -> ProviderManifest:
    return ProviderManifest.from_dict(_json(path))


def _git(root: Path, *args: str) -> str:
    try:
        p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                           check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProviderProtocolError(f"git inspection failed: {exc}") from exc
    if p.returncode:
        raise ProviderProtocolError(p.stderr.strip() or p.stdout.strip() or "git failed")
    return p.stdout.strip()


def _repo_from_remote(remote: str) -> str | None:
    value = remote.strip().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    if value.startswith("git@github.com:"):
        return value.split(":", 1)[1]
    return value.split("github.com/", 1)[1] if "github.com/" in value else None


def _child(root: Path, rel: str, field: str) -> Path:
    part = Path(rel)
    if part.is_absolute() or ".." in part.parts:
        raise ProviderProtocolError(f"{field} must stay inside the provider checkout")
    target = (root / part).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ProviderProtocolError(f"{field} escapes the provider checkout") from exc
    return target


def validate_request(m: ProviderManifest, request: dict[str, Any]) -> None:
    phase = str(request.get("phase") or "")
    if phase not in m.commands:
        raise ProviderProtocolError(f"phase {phase!r} is not declared by provider")
    leaked = set(map(str, request.get("training_inputs") or [])) & m.training_forbidden
    if leaked:
        raise ProviderProtocolError("protected eval artifacts cannot be training inputs: " + ", ".join(sorted(leaked)))
    mutated = set(map(str, request.get("mutable_surfaces") or [])) & m.mutable_forbidden
    if mutated:
        raise ProviderProtocolError("candidate cannot mutate protected surfaces: " + ", ".join(sorted(mutated)))
    ev = request.get("eval_suite") or {}
    if str(ev.get("repo") or "") != m.eval_repo:
        raise ProviderProtocolError("request eval suite repo does not match pinned manifest")
    if _pin(ev.get("revision"), "request.eval_suite.revision") != m.eval_revision:
        raise ProviderProtocolError("request eval suite revision does not match pinned manifest")
    if str(ev.get("suite") or "") != m.eval_suite:
        raise ProviderProtocolError("request eval suite id does not match pinned manifest")
    if not isinstance(request.get("params") or {}, dict):
        raise ProviderProtocolError("request.params must be an object")


def preflight(m: ProviderManifest, provider_root: Path, request: dict[str, Any] | None = None) -> dict[str, Any]:
    root = provider_root.resolve()
    if not root.is_dir():
        raise ProviderProtocolError(f"provider checkout does not exist: {root}")
    head = _pin(_git(root, "rev-parse", "HEAD"), "provider checkout HEAD")
    origin = _repo_from_remote(_git(root, "remote", "get-url", "origin"))
    if origin != m.repo:
        raise ProviderProtocolError(f"provider checkout origin mismatch: expected {m.repo}, got {origin or 'unknown'}")
    if head != m.revision:
        raise ProviderProtocolError(f"provider checkout revision mismatch: expected {m.revision}, got {head}")
    if not _child(root, m.hermes_profile, "hermes_profile").is_file():
        raise ProviderProtocolError(f"Hermes profile missing: {m.hermes_profile}")
    for phase, spec in m.commands.items():
        script = next((x for x in spec.argv[1:] if x.endswith(".py") and not x.startswith("{")), None)
        if script and not _child(root, script, f"commands.{phase}.argv").is_file():
            raise ProviderProtocolError(f"provider command script missing: {script}")
    if request is not None:
        validate_request(m, request)
    return {"schema": PROTOCOL_SCHEMA, "provider_repo": m.repo, "provider_upstream": m.upstream,
            "provider_revision": head, "provider_origin": origin, "hermes_profile": m.hermes_profile,
            "eval_repo": m.eval_repo, "eval_revision": m.eval_revision, "eval_suite": m.eval_suite}


def resolve_command(m: ProviderManifest, phase: str) -> tuple[str, ...]:
    if phase not in m.commands:
        raise ProviderProtocolError(f"phase {phase!r} is not declared by provider")
    return m.commands[phase].argv


def render_command(m: ProviderManifest, phase: str, params: dict[str, Any] | None = None) -> tuple[str, ...]:
    command = resolve_command(m, phase)
    params = params or {}
    fields: set[str] = set()
    fmt = string.Formatter()
    for token in command:
        for _literal, field, _spec, _conv in fmt.parse(token):
            if field:
                if not FIELD_RE.fullmatch(field):
                    raise ProviderProtocolError(f"unsafe command placeholder: {field}")
                fields.add(field)
    missing = fields - params.keys()
    if missing:
        raise ProviderProtocolError("missing command params: " + ", ".join(sorted(missing)))
    safe: dict[str, str] = {}
    for key in fields:
        value = str(params[key])
        if "\x00" in value:
            raise ProviderProtocolError(f"command param {key} contains NUL")
        safe[key] = value
    return tuple(token.format_map(safe) for token in command)


def _write_private(path: Path, data: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(0o600)
    return {"file": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def execute_phase(m: ProviderManifest, provider_root: Path, request: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    validate_request(m, request)
    phase = str(request["phase"])
    if phase not in m.execution_allowed:
        raise ProviderProtocolError(f"phase {phase!r} is not allowed for execution")
    execution = request.get("execution") or {}
    if execution.get("approved") is not True:
        raise ProviderProtocolError("request.execution.approved must be true for execution")
    command = render_command(m, phase, request.get("params") or {})
    spec = m.commands[phase]
    requested_budget = int(execution.get("budget_s") or spec.timeout_s)
    if requested_budget < 1:
        raise ProviderProtocolError("request.execution.budget_s must be positive")
    timeout_s = min(spec.timeout_s, requested_budget)
    env = {key: os.environ[key] for key in m.env_passthrough if key in os.environ}
    env["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    try:
        p = subprocess.run(command, cwd=provider_root.resolve(), env=env, capture_output=True,
                           check=False, timeout=timeout_s)
        status = "success" if p.returncode == 0 else "failed"
        returncode: int | None = p.returncode
        stdout, stderr = p.stdout, p.stderr
    except subprocess.TimeoutExpired as exc:
        status, returncode = "timeout", None
        stdout, stderr = exc.stdout or b"", exc.stderr or b""
    elapsed = time.monotonic() - started
    rd = run_dir.resolve()
    out = _write_private(rd / f"{phase}.stdout.log", stdout)
    err = _write_private(rd / f"{phase}.stderr.log", stderr)
    return {"status": status, "returncode": returncode, "elapsed_s": round(elapsed, 6),
            "timeout_s": timeout_s, "stdout": out, "stderr": err, "command": list(command)}


def build_receipt(m: ProviderManifest, request: dict[str, Any], command: tuple[str, ...],
                  execution_result: dict[str, Any] | None = None) -> dict[str, Any]:
    packed = json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    receipt = {
        "schema": RECEIPT_SCHEMA, "status": execution_result["status"] if execution_result else "dry_run",
        "candidate_id": str(request.get("candidate_id") or ""),
        "experiment_id": str(request.get("experiment_id") or ""),
        "phase": str(request.get("phase") or ""), "method": str(request.get("method") or ""),
        "provider": {"repo": m.repo, "upstream": m.upstream, "revision": m.revision},
        "hermes_profile": m.hermes_profile,
        "eval_suite": {"repo": m.eval_repo, "revision": m.eval_revision, "suite": m.eval_suite},
        "command": list(command), "request_sha256": hashlib.sha256(packed).hexdigest(),
        "promotion_authority": "evolution-lab", "runtime_target": "shadow-or-canary-only",
        "credit": "SouthpawIN: Gym Trainer and evolutionary-training implementation",
    }
    if execution_result:
        receipt["execution"] = execution_result
    return receipt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate or execute an external z0 training provider phase")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--provider-root", type=Path, required=True)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--receipt", type=Path)
    p.add_argument("--run-dir", type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    try:
        m, request = load_manifest(args.manifest), _json(args.request)
        report = preflight(m, args.provider_root, request)
        command = render_command(m, str(request.get("phase") or ""), request.get("params") or {})
        result = None
        if args.execute:
            if args.run_dir is None:
                raise ProviderProtocolError("--run-dir is required with --execute")
            result = execute_phase(m, args.provider_root, request, args.run_dir)
        receipt = build_receipt(m, request, command, result)
    except ProviderProtocolError as exc:
        raise SystemExit(f"provider preflight refused: {exc}") from exc
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"preflight": report, "resolved_command": list(command), "receipt": receipt}, indent=2, sort_keys=True))
    return 0 if result is None or result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
