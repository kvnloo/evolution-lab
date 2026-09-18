"""Export immutable z0int.candidate_artifact.v1 bundles (always unpromoted)."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any


SCHEMA = "z0int.candidate_artifact.v1"
DEFAULT_RUNTIME = "z0int.backend.mushroom.v1"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_revision(root: Path) -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def export_candidate(
    *,
    run_dir: Path | None = None,
    candidate: str | Path,
    output: Path,
    capability_id: str = "coding.recovery_action",
    family: str = "local_plasticity",
    runtime: str = DEFAULT_RUNTIME,
    genome_id: str | None = None,
    open_metrics: dict[str, Any] | None = None,
    dev_metrics: dict[str, Any] | None = None,
    open_split_digest: str | None = None,
    dev_split_digest: str | None = None,
    extra_files: list[Path] | None = None,
) -> dict[str, Any]:
    """Write a new directory with hashed files + manifest. Never promotes."""
    cand = Path(candidate)
    if not cand.is_file():
        root = Path(__file__).resolve().parents[1]
        alt = root / "data" / str(candidate)
        if alt.is_file():
            cand = alt
        else:
            raise FileNotFoundError(str(candidate))

    out = Path(output)
    if out.exists():
        raise FileExistsError(f"output exists: {out}")
    out.mkdir(parents=True)

    files_meta = []
    for src in [cand, *(extra_files or [])]:
        src = Path(src)
        dest = out / src.name
        shutil.copy2(src, dest)
        files_meta.append({"path": src.name, "sha256": _sha256_file(dest)})

    file_hash_blob = "|".join(sorted(f["sha256"] for f in files_meta)).encode()
    artifact_id = "sha256:" + hashlib.sha256(file_hash_blob).hexdigest()

    root = Path(__file__).resolve().parents[1]
    open_m = dict(open_metrics or {})
    dev_m = dict(dev_metrics or {})
    for side in (open_m, dev_m):
        for bad in list(side):
            if any(k in bad.lower() for k in ("prompt", "secret", "token", "password")):
                side.pop(bad, None)

    manifest = {
        "schema": SCHEMA,
        "artifact_id": artifact_id,
        "producer": {
            "repo": "kvnloo/evolution-lab",
            "revision": _git_revision(root),
        },
        "capability_id": capability_id,
        "family": family,
        "runtime": runtime,
        "files": files_meta,
        "training": {
            "genome_id": genome_id or cand.stem,
            "run_dir": str(run_dir) if run_dir else None,
            "open_split_digest": open_split_digest,
            "dev_split_digest": dev_split_digest,
        },
        "claims": {
            "open_metrics": open_m,
            "dev_metrics": dev_m,
        },
        "promotion": {"status": "candidate", "exported_at": time.time()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, "artifact_id": artifact_id, "output": str(out), "manifest": manifest}
