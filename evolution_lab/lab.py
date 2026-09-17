"""One-shot experiment pipeline: seed → run → evolve → table → DAgger → summary."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .archive import Archive
from .dagger import default_dagger_smoke
from .engine import summarize
from .splits import splits_exist


def git_head(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run_lab(
    *,
    run_dir: Path,
    level: int = 1,
    evolve_generations: int = 0,
    evolve_children: int = 4,
    dagger_rounds: int = 2,
    fresh: bool = False,
    skip_dagger: bool = False,
    seed_if_missing: bool = True,
) -> dict[str, Any]:
    """Execute the standard FlyForge experiment loop and write experiment_summary.json."""
    from .cli import cmd_evolve, cmd_run, cmd_seed, cmd_table

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]

    if not splits_exist():
        raise RuntimeError(
            "locked splits missing under data/p0/ — run: python -m evolution_lab lock-splits"
        )

    if fresh:
        for name in ("archive.jsonl", "control_table.json", "experiment_summary.json", "dashboard.html"):
            path = run_dir / name
            if path.is_file():
                path.unlink()

    genomes_dir = run_dir / "genomes"
    if seed_if_missing and (not genomes_dir.is_dir() or not any(genomes_dir.glob("*.json"))):
        cmd_seed(run_dir)

    cmd_run(run_dir, level)
    if evolve_generations > 0:
        cmd_evolve(run_dir, evolve_generations, level, evolve_children)
    cmd_table(run_dir, level)

    dagger_report: dict[str, Any] | None = None
    if not skip_dagger:
        dagger_report = default_dagger_smoke(rounds=dagger_rounds)

    archive_stats = summarize(Archive(run_dir / "archive.jsonl").read())
    control_table = json.loads((run_dir / "control_table.json").read_text(encoding="utf-8"))

    summary: dict[str, Any] = {
        "schema": "evolution-lab.experiment-summary.v1",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(root),
        "run_dir": str(run_dir),
        "level": level,
        "evolve_generations": evolve_generations,
        "dagger_rounds": 0 if skip_dagger else dagger_rounds,
        "archive": archive_stats,
        "control_table": {
            "kill_criterion": control_table.get("kill_criterion"),
            "vs_teacher": control_table.get("vs_teacher"),
            "rows": control_table.get("rows"),
        },
        "dagger": dagger_report,
        "artifacts": {
            "archive": str(run_dir / "archive.jsonl"),
            "control_table": str(run_dir / "control_table.json"),
            "dashboard": str(run_dir / "dashboard.html"),
        },
    }

    summary_path = run_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
