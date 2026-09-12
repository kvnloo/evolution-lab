"""CLI: seed, run, evolve, dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np

from .archive import Archive
from .dashboard import write_dashboard
from .elites import MapElites
from .engine import run_one, seed_genomes, summarize
from .gym import make_env, rollout_teacher
from .mutate import crossover, mutate
from .schema import ExperimentGenome, genome_from_dict, load_genome


def default_run_dir(root: Path) -> Path:
    return root / "runs" / "p0"


def _root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def cmd_seed(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    gdir = run_dir / "genomes"
    gdir.mkdir(parents=True, exist_ok=True)
    for g in seed_genomes():
        (gdir / f"{g.id}.json").write_text(json.dumps(g.to_dict(), indent=2) + "\n")
    print(f"wrote {len(list(gdir.glob('*.json')))} genomes to {gdir}")


def _load_all(run_dir: Path) -> list[ExperimentGenome]:
    gdir = run_dir / "genomes"
    files = sorted(gdir.glob("*.json"))
    if not files:
        return seed_genomes()
    return [load_genome(p) for p in files]


def cmd_run(run_dir: Path, level: int) -> None:
    archive = Archive(run_dir / "archive.jsonl")
    elites = MapElites()
    prior = archive.read()
    for g in _load_all(run_dir):
        rec = run_one(g, archive, elites, level=level, prior=prior)
        prior.append(rec)
        print(f"{rec['status']:7} {g.id:28} success={rec.get('metrics', {}).get('success_rate', '—')} ΔHV={rec.get('delta_hv', 0):.5f}")
    stats = summarize(archive.read())
    print(json.dumps(stats, indent=2))
    write_dashboard(archive.read(), run_dir / "dashboard.html")
    print(f"dashboard {run_dir / 'dashboard.html'}")


def cmd_evolve(run_dir: Path, generations: int, level: int, n_children: int) -> None:
    archive = Archive(run_dir / "archive.jsonl")
    elites = MapElites()
    prior = archive.read()
    population = _load_all(run_dir)
    rng = np.random.default_rng(42)
    for gen in range(generations):
        ok_ids = {r["experiment_id"] for r in prior if r.get("status") == "ok"}
        parents = [g for g in population if g.id in ok_ids] or population
        children = []
        for i in range(n_children):
            p = parents[int(rng.integers(0, len(parents)))]
            if len(parents) > 1 and rng.random() < 0.25:
                q = parents[int(rng.integers(0, len(parents)))]
                child = crossover(p, q, rng, generation=gen + 1)
            else:
                child = mutate(p, rng, generation=gen + 1)
            children.append(child)
            rec = run_one(child, archive, elites, level=level, prior=prior)
            prior.append(rec)
            population.append(child)
            print(f"g{gen+1} {rec['status']:7} {child.id:36} {rec.get('metrics', {}).get('success_rate', '—')}")
        (run_dir / "genomes").mkdir(parents=True, exist_ok=True)
        for c in children:
            (run_dir / "genomes" / f"{c.id}.json").write_text(json.dumps(c.to_dict(), indent=2) + "\n")
    write_dashboard(archive.read(), run_dir / "dashboard.html")
    print(json.dumps(summarize(archive.read()), indent=2))


def cmd_gym_smoke(n: int, delayed_cue: bool) -> None:
    """Closed-loop teacher on the P0 gym. Not FlyGym, not OpenEvolve."""
    env = make_env("hermes_recovery", delayed_cue=delayed_cue)
    rewards = []
    for seed in range(n):
        _ep, mean = rollout_teacher(env, seed=seed)
        rewards.append(mean)
    report = {
        "env": "hermes_recovery",
        "n": n,
        "teacher_mean_reward": float(np.mean(rewards)),
        "teacher_perfect": all(r == 1.0 for r in rewards),
        "note": "FlyGym/OpenEvolve are not this env; make_env refuses those names.",
    }
    print(json.dumps(report, indent=2))
    if not report["teacher_perfect"]:
        raise SystemExit(1)


def cmd_quest_falsification(run_dir: Path, level: int) -> None:
    """Does the best reservoir still beat a refit rewired graph?"""
    from .schema import Architecture, Training

    archive = Archive(run_dir / "archive.jsonl")
    elites = MapElites()
    prior = archive.read()
    best = None
    best_s = -1.0
    for r in prior:
        if r.get("status") != "ok":
            continue
        fam = (r.get("genome") or {}).get("architecture", {}).get("family")
        if fam not in {"fixed_reservoir", "fly_connectome"}:
            continue
        s = float((r.get("metrics") or {}).get("success_rate") or 0)
        if s > best_s:
            best_s, best = s, r
    if best is None:
        print("no reservoir champion yet; run `run` first")
        return
    champ = genome_from_dict(best["genome"])
    ctrl = ExperimentGenome(
        id=f"falsify-rewire-{champ.id}",
        lineage="falsification",
        hypothesis=f"Degree-style rewire of {champ.id} with refit readout.",
        parents=(champ.id,),
        role="skeptic",
        architecture=Architecture(
            family="rewired_reservoir",
            hidden=champ.architecture.hidden,
            history=champ.architecture.history,
            sparsity=champ.architecture.sparsity,
        ),
        training=Training(seed=champ.training.seed + 123),
        curriculum=champ.curriculum,
    )
    rec = run_one(ctrl, archive, elites, level=level, prior=prior)
    print(json.dumps({"champion": champ.id, "champion_success": best_s, "rewire": rec["metrics"]}, indent=2))
    write_dashboard(archive.read(), run_dir / "dashboard.html")


def main(argv: list[str] | None = None) -> int:
    root = _root_from_here()
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--run-dir", type=Path, default=default_run_dir(root))
    p = argparse.ArgumentParser(prog="evolution_lab")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seed", parents=[parent])
    pr = sub.add_parser("run", parents=[parent])
    pr.add_argument("--level", type=int, default=1)
    pe = sub.add_parser("evolve", parents=[parent])
    pe.add_argument("--generations", type=int, default=2)
    pe.add_argument("--level", type=int, default=1)
    pe.add_argument("--children", type=int, default=4)
    pq = sub.add_parser("quest", parents=[parent])
    pq.add_argument("name", choices=["falsification"])
    pq.add_argument("--level", type=int, default=1)
    sub.add_parser("dashboard", parents=[parent])
    pg = sub.add_parser("gym-smoke", parents=[parent])
    pg.add_argument("--n", type=int, default=24)
    pg.add_argument("--no-delayed-cue", action="store_true")
    args = p.parse_args(argv)
    run_dir = args.run_dir
    if args.cmd == "seed":
        cmd_seed(run_dir)
    elif args.cmd == "run":
        cmd_run(run_dir, args.level)
    elif args.cmd == "evolve":
        cmd_evolve(run_dir, args.generations, args.level, args.children)
    elif args.cmd == "quest":
        cmd_quest_falsification(run_dir, args.level)
    elif args.cmd == "dashboard":
        archive = Archive(run_dir / "archive.jsonl")
        dest = write_dashboard(archive.read(), run_dir / "dashboard.html")
        print(dest)
    elif args.cmd == "gym-smoke":
        cmd_gym_smoke(args.n, delayed_cue=not args.no_delayed_cue)
    return 0
