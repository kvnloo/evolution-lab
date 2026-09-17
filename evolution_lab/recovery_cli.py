"""JSON CLI for OMP/Hermes recovery controller extensions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .advise import advise
from .observe import action_guidance, observe_event
from .schema import ACTIONS, GenomeError
from .splits import splits_exist
from .student_bundle import bundle_exists, bundle_meta


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def status_payload() -> dict[str, Any]:
    root = repo_root()
    data_dir = root / "data" / "p0"
    return {
        "ok": True,
        "repo": str(root),
        "locked_splits": splits_exist(data_dir),
        "actions": list(ACTIONS),
        "default_family": os.environ.get("FLYFORGE_RECOVERY_FAMILY", "local_plasticity"),
        "student_bundle": bundle_exists(root),
        "student_bundle_meta": bundle_meta(root),
        "python": sys.executable,
    }


def handle_request(payload: dict[str, Any]) -> dict[str, Any]:
    action = str(payload.get("action") or "advise")
    family = str(payload.get("family") or os.environ.get("FLYFORGE_RECOVERY_FAMILY", "local_plasticity"))

    if action == "status":
        return status_payload()

    if action == "observe":
        event = payload.get("event")
        if not isinstance(event, dict):
            raise GenomeError("observe requires an event object")
        observed = observe_event(
            event,
            harness=str(payload.get("harness") or "omp"),
            retry=int(payload.get("retry") or 0),
            budget=float(payload.get("budget") if payload.get("budget") is not None else 1.0),
            last_action=payload.get("last_action"),
        )
        return {"observed": observed}

    if action in {"advise", "plan"}:
        fields = payload.get("fields")
        if fields is None and isinstance(payload.get("event"), dict):
            observed = observe_event(
                payload["event"],
                harness=str(payload.get("harness") or "omp"),
                retry=int(payload.get("retry") or 0),
                budget=float(payload.get("budget") if payload.get("budget") is not None else 1.0),
                last_action=payload.get("last_action"),
            )
            fields = observed["fields"]
        if not isinstance(fields, dict):
            raise GenomeError("advise requires fields object or event object")
        result = advise(fields, family=family)
        result["guidance"] = action_guidance(result["action"])["guidance"]
        return result

    raise GenomeError(f"unknown recovery action {action!r}")


def main(argv: list[str] | None = None) -> int:
    raw = (argv or sys.argv[1:])[0] if (argv or sys.argv[1:]) else "{}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"ok": False, "error": f"invalid JSON request: {exc}"}))
        return 2
    if not isinstance(payload, dict):
        print(json.dumps({"ok": False, "error": "request must be a JSON object"}))
        return 2
    try:
        result = handle_request(payload)
        print(json.dumps({"ok": True, **result}, indent=2))
        return 0
    except (GenomeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
