"""Verified OSS Loop evidence receipt. Stdlib + git. No invented mutation scores."""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

MUTATION_NA = "n/a"
DEFAULT_ISSUE = "2"
DEFAULT_TESTS_RED = (
    "python -m unittest tests.test_splits tests.test_receipt"
    "  # expected fail before locked splits/receipt modules existed"
)
DEFAULT_TESTS_GREEN = "python -m unittest discover -s tests -p 'test_*.py'"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    return text or None


def git_head_revision(cwd: Path | None = None) -> str | None:
    return _git(["rev-parse", "HEAD"], cwd or repo_root())


def git_base_revision(cwd: Path | None = None) -> str | None:
    root = cwd or repo_root()
    for ref in ("origin/main", "main"):
        base = _git(["merge-base", "HEAD", ref], root)
        if base:
            return base
    return git_head_revision(root)


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    text = str(value)
    if text == "" or text.lower() in {"null", "true", "false", "yes", "no", "n/a"}:
        return json_quote(text)
    if any(ch in text for ch in ":#[]{},&*?!'\"\n"):
        return json_quote(text)
    return text


def json_quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def dump_yaml(obj: Any, *, indent: int = 0) -> str:
    """Minimal YAML for receipt dicts (mappings, lists, scalars). Stdlib only."""
    lines: list[str] = []
    _dump(obj, lines, indent)
    return "\n".join(lines) + "\n"


def _dump(obj: Any, lines: list[str], indent: int) -> None:
    prefix = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            lines.append(prefix + "{}")
            return
        for key, value in obj.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                _dump(value, lines, indent + 1)
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                if not value:
                    lines[-1] = f"{prefix}{key}: []"
                    continue
                for item in value:
                    if isinstance(item, (dict, list)):
                        lines.append(f"{prefix}-")
                        _dump(item, lines, indent + 1)
                    else:
                        lines.append(f"{prefix}- {_scalar(item)}")
            else:
                lines.append(f"{prefix}{key}: {_scalar(value)}")
        return
    if isinstance(obj, list):
        for item in obj:
            lines.append(f"{prefix}- {_scalar(item)}")
        return
    lines.append(prefix + _scalar(obj))


def evidence_receipt(
    *,
    issue: str = DEFAULT_ISSUE,
    cwd: Path | None = None,
    tests_red: str = DEFAULT_TESTS_RED,
    tests_green: str = DEFAULT_TESTS_GREEN,
    extra_limitations: list[str] | None = None,
) -> dict[str, Any]:
    """Return a Verified OSS Loop evidence dict. mutation is always n/a here."""
    root = cwd or repo_root()
    head = git_head_revision(root)
    base = git_base_revision(root)
    limitations = ["joules_unknown"]
    if extra_limitations:
        for item in extra_limitations:
            if item not in limitations:
                limitations.append(item)
    if not head:
        limitations.append("head_revision unavailable (git failed)")
    return {
        "issue": str(issue),
        "base_revision": base or "",
        "head_revision": head or "",
        "tests": {
            "red": tests_red,
            "green": tests_green,
        },
        "mutation": MUTATION_NA,
        "limitations": limitations,
    }


def receipt_yaml(
    receipt: dict[str, Any] | None = None,
    **kwargs: Any,
) -> str:
    data = receipt if receipt is not None else evidence_receipt(**kwargs)
    return dump_yaml(data)
