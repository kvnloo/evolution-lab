"""CLI: seed, run, evolve, dashboard, lock-splits, receipt, table."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np

from .archive import Archive
from .dashboard import write_dashboard
from .elites import MapElites
from .engine import run_one, seed_genomes, seed_jev_genomes, summarize
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


def cmd_lock_splits(data_dir: Path) -> None:
    from .splits import lock_splits

    dest = lock_splits(data_dir)
    print(f"locked splits {dest}")


def cmd_seed_jev(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    gdir = run_dir / "genomes"
    gdir.mkdir(parents=True, exist_ok=True)
    genomes = seed_jev_genomes()
    for g in genomes:
        (gdir / f"{g.id}.json").write_text(json.dumps(g.to_dict(), indent=2) + "\n")
    print(f"wrote {len(genomes)} Jev genomes to {gdir}")


def cmd_lock_jev_splits(data_dir: Path) -> None:
    from .jev_splits import lock_splits

    dest = lock_splits(data_dir)
    print(f"locked Jev splits {dest}")


def cmd_jev_smoke(run_dir: Path, level: int) -> None:
    from .jev_splits import lock_splits, splits_exist
    from .openjev_runner import openjev_available

    if not openjev_available():
        raise SystemExit("openjev-phase1 not installed; pip install -e '.[openjev]'")
    if not splits_exist():
        lock_splits()
    cmd_seed_jev(run_dir)
    cmd_run(run_dir, level)


def cmd_receipt(*, issue: str) -> None:
    from .receipt import receipt_yaml

    print(receipt_yaml(issue=issue), end="")


def cmd_table(run_dir: Path, level: int) -> None:
    """P1 Track A control table + kill criterion. Writes control_table.json."""
    from .table import write_control_table

    genomes = _load_all(run_dir)
    path = write_control_table(level=level, run_dir=run_dir, genomes=genomes)
    print(path)


def cmd_advise(payload: str, family: str) -> None:
    from .advise import advise_json

    print(advise_json(payload, family=family), end="")


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


def cmd_dagger_smoke(*, rounds: int) -> None:
    """P3 closed-loop DAgger smoke on locked splits."""
    from .dagger import default_dagger_smoke

    report = default_dagger_smoke(rounds=rounds)
    print(json.dumps(report, indent=2))



def cmd_fly_bench(
    *,
    config: Path | None,
    candidate: Path | None,
    run_dir: Path | None,
    tag: str | None,
    skip_unit_tests: bool,
) -> None:
    from .bench import load_candidate, run_fly_bench
    from .select import format_summary_block, load_config

    cfg = load_config(config)
    root = _root_from_here()
    rd = run_dir
    if rd is None and tag:
        rd = root / str((cfg.get("paths") or {}).get("run_dir", "runs/autoresearch")) / tag
    cand = load_candidate(candidate) if candidate else None
    result = run_fly_bench(cand, config=cfg, run_dir=rd, skip_unit_tests=skip_unit_tests)
    print(format_summary_block(result))
    if not result.gates_pass:
        raise SystemExit(1)


def cmd_autoresearch(
    *,
    config: Path | None,
    run_dir: Path | None,
    tag: str | None,
    baseline_only: bool,
    max_experiments: int | None,
    patience: int | None,
    promote_bundle: bool,
    skip_unit_tests: bool,
    seed: int,
) -> None:
    from .autoresearch import run_autoresearch_loop, run_baseline

    if baseline_only:
        run_baseline(config_path=config, run_dir=run_dir, tag=tag, skip_unit_tests=skip_unit_tests)
        return
    summary = run_autoresearch_loop(
        config_path=config,
        run_dir=run_dir,
        tag=tag,
        max_experiments=max_experiments,
        patience=patience,
        promote_bundle=promote_bundle,
        skip_unit_tests=skip_unit_tests,
        seed=seed,
    )
    print(json.dumps(summary, indent=2))

def cmd_cycle(
    *,
    forever: bool,
    max_sessions: int | None,
    max_hours: float | None,
    max_experiments: int | None,
    patience: int | None,
    skip_unit_tests: bool,
    promote_bundle: bool,
    seed: int,
) -> None:
    from .cycle import run_cycle

    summary = run_cycle(
        forever=forever,
        max_sessions=max_sessions,
        max_hours=max_hours,
        max_experiments=max_experiments,
        patience=patience,
        skip_unit_tests=skip_unit_tests,
        promote_bundle=promote_bundle,
        seed=seed,
    )
    print(json.dumps(summary, indent=2))


def cmd_sweep(*, workers: int | None) -> None:
    from .sweep import run_sweep

    print(json.dumps(run_sweep(workers=workers), indent=2, default=str))


def cmd_jev_distill() -> None:
    from .jev_distill import run_jev_distill

    print(json.dumps(run_jev_distill(), indent=2))


def cmd_jev_predict(prompt: str) -> None:
    from .jev_distill import predict_prompt

    print(json.dumps(predict_prompt(prompt), indent=2))


def cmd_lab(
    run_dir: Path,
    *,
    level: int,
    evolve_generations: int,
    evolve_children: int,
    dagger_rounds: int,
    fresh: bool,
    skip_dagger: bool,
) -> None:
    """Full experiment loop: seed → run → evolve → table → DAgger → summary."""
    from .lab import run_lab

    summary = run_lab(
        run_dir=run_dir,
        level=level,
        evolve_generations=evolve_generations,
        evolve_children=evolve_children,
        dagger_rounds=dagger_rounds,
        fresh=fresh,
        skip_dagger=skip_dagger,
    )
    print(json.dumps(summary, indent=2))



def cmd_bundle_student(*, dagger_rounds: int) -> None:
    """Train (optional DAgger) and write data/p0/recovery_student.npz for OMP."""
    from .dagger import aggregate_dagger_episodes, mean_closed_loop_reward, make_env, run_dagger
    from .engine import seed_genomes
    from .models import fit_student
    from .splits import load_splits
    from .student_bundle import save_bundle

    genome = next(g for g in seed_genomes() if g.architecture.family == "local_plasticity")
    data = load_splits()
    env = make_env("hermes_recovery", delayed_cue=True, history=genome.architecture.history)
    metrics: dict = {}
    if dagger_rounds > 0:
        metrics["dagger"] = run_dagger(
            genome,
            base_train=data.train,
            dagger_seeds=list(range(32, 56)),
            eval_seeds=list(range(24)),
            rounds=dagger_rounds,
        )
        train_pool = list(data.train)
        student = fit_student(genome, data.train)
        for _ in range(dagger_rounds):
            train_pool = train_pool + aggregate_dagger_episodes(env, student, list(range(32, 56)))
            student = fit_student(genome, train_pool, init_extras=student.extras)
    else:
        student = fit_student(genome, data.train)
    metrics["closed_loop"] = mean_closed_loop_reward(env, student, list(range(24)))
    path = save_bundle(genome, student, metrics=metrics)
    print(json.dumps({"bundle": str(path), **metrics}, indent=2))

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
    pl = sub.add_parser("lock-splits", parents=[parent])
    pl.add_argument("--data-dir", type=Path, default=None)
    prc = sub.add_parser("receipt", parents=[parent])
    prc.add_argument("--issue", default="2")
    sub.add_parser("seed-jev", parents=[parent])
    pjl = sub.add_parser("lock-jev-splits", parents=[parent])
    pjl.add_argument("--data-dir", type=Path, default=None)
    pjs = sub.add_parser("jev-smoke", parents=[parent])
    pjs.add_argument("--level", type=int, default=0)
    pt = sub.add_parser("table", parents=[parent])
    pt.add_argument("--level", type=int, default=1)
    pd = sub.add_parser("dagger-smoke", parents=[parent])
    pd.add_argument("--rounds", type=int, default=2)
    pbs = sub.add_parser("bundle-student", parents=[parent])
    pbs.add_argument("--dagger-rounds", type=int, default=0)
    plab = sub.add_parser("lab", parents=[parent], help="full experiment loop")
    plab.add_argument("--level", type=int, default=1)
    plab.add_argument("--evolve-generations", type=int, default=0)
    plab.add_argument("--evolve-children", type=int, default=4)
    plab.add_argument("--dagger-rounds", type=int, default=2)
    plab.add_argument("--fresh", action="store_true", help="drop archive/table before run")
    plab.add_argument("--skip-dagger", action="store_true")

    pfb = sub.add_parser("fly-bench", parents=[parent], help="benchmark one recovery kernel candidate")
    pfb.add_argument("--config", type=Path, default=None)
    pfb.add_argument("--candidate", type=Path, default=None)
    pfb.add_argument("--tag", type=str, default=None)
    pfb.add_argument("--skip-unit-tests", action="store_true")
    par = sub.add_parser("autoresearch", parents=[parent], help="continuous kernel autoresearch loop")
    par.add_argument("--config", type=Path, default=None)
    par.add_argument("--tag", type=str, default=None)
    par.add_argument("--baseline-only", action="store_true")
    par.add_argument("--max-experiments", type=int, default=None)
    par.add_argument("--patience", type=int, default=None)
    par.add_argument("--promote-bundle", action="store_true")
    par.add_argument("--skip-unit-tests", action="store_true")
    par.add_argument("--seed", type=int, default=42)

    pcy = sub.add_parser("cycle", parents=[parent], help="outer loop around autoresearch (evolve the fly)")
    pcy.add_argument("--forever", action="store_true")
    pcy.add_argument("--max-sessions", type=int, default=None)
    pcy.add_argument("--max-hours", type=float, default=None)
    pcy.add_argument("--max-experiments", type=int, default=None)
    pcy.add_argument("--patience", type=int, default=None)
    pcy.add_argument("--skip-unit-tests", action="store_true")
    pcy.add_argument("--promote-bundle", action="store_true", default=True)
    pcy.add_argument("--no-promote-bundle", action="store_false", dest="promote_bundle")
    pcy.add_argument("--seed", type=int, default=42)

    psw = sub.add_parser("sweep", parents=[parent], help="parallel fly experiment pack (75 pct CPUs)")
    psw.add_argument("--workers", type=int, default=None)
    sub.add_parser("jev-distill", parents=[parent], help="distill TypeSafe Jev skill routing into the fly")
    jpp = sub.add_parser("jev-predict", parents=[parent], help="fly pre-predict Jev skill from a prompt")
    jpp.add_argument("prompt", nargs="+")
    prc_ctrl = sub.add_parser("recovery", parents=[parent])
    prc_ctrl.add_argument("json", nargs="?", default="{}", help="recovery controller JSON request")

    pa = sub.add_parser("advise", parents=[parent])
    pa.add_argument("json", help="sanitized Hermes observation object")
    pa.add_argument(
        "--family",
        default="rule",
        choices=("rule", "local_plasticity"),
        help="teacher_rule (default) or mushroom-body analogue",
    )
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
    elif args.cmd == "lock-splits":
        data_dir = args.data_dir or (root / "data" / "p0")
        cmd_lock_splits(data_dir)
    elif args.cmd == "seed-jev":
        cmd_seed_jev(run_dir)
    elif args.cmd == "lock-jev-splits":
        data_dir = args.data_dir or (root / "data" / "jev" / "synthetic")
        cmd_lock_jev_splits(data_dir)
    elif args.cmd == "jev-smoke":
        cmd_jev_smoke(run_dir, args.level)
    elif args.cmd == "receipt":
        cmd_receipt(issue=args.issue)
    elif args.cmd == "table":
        cmd_table(run_dir, args.level)
    elif args.cmd == "recovery":
        from .recovery_cli import main as recovery_main

        raise SystemExit(recovery_main([args.json]))
    elif args.cmd == "advise":
        cmd_advise(args.json, args.family)
    elif args.cmd == "dagger-smoke":
        cmd_dagger_smoke(rounds=args.rounds)
    elif args.cmd == "bundle-student":
        cmd_bundle_student(dagger_rounds=args.dagger_rounds)
    elif args.cmd == "fly-bench":
        cmd_fly_bench(
            config=args.config,
            candidate=getattr(args, "candidate", None),
            run_dir=run_dir if args.tag is None else None,
            tag=args.tag,
            skip_unit_tests=getattr(args, "skip_unit_tests", False),
        )
    elif args.cmd == "autoresearch":
        cmd_autoresearch(
            config=getattr(args, "config", None),
            run_dir=run_dir if getattr(args, "tag", None) is None else None,
            tag=getattr(args, "tag", None),
            baseline_only=getattr(args, "baseline_only", False),
            max_experiments=getattr(args, "max_experiments", None),
            patience=getattr(args, "patience", None),
            promote_bundle=getattr(args, "promote_bundle", False),
            skip_unit_tests=getattr(args, "skip_unit_tests", False),
            seed=getattr(args, "seed", 42),
        )

    elif args.cmd == "sweep":
        cmd_sweep(workers=getattr(args, "workers", None))
    elif args.cmd == "jev-distill":
        cmd_jev_distill()
    elif args.cmd == "jev-predict":
        cmd_jev_predict(" ".join(args.prompt))
    elif args.cmd == "cycle":
        cmd_cycle(
            forever=getattr(args, "forever", False),
            max_sessions=getattr(args, "max_sessions", None),
            max_hours=getattr(args, "max_hours", None),
            skip_unit_tests=getattr(args, "skip_unit_tests", False),
            max_experiments=getattr(args, "max_experiments", None),
            patience=getattr(args, "patience", None),
            promote_bundle=getattr(args, "promote_bundle", True),
            seed=getattr(args, "seed", 42),
        )

    elif args.cmd == "lab":
        cmd_lab(
            run_dir,
            level=args.level,
            evolve_generations=args.evolve_generations,
            evolve_children=args.evolve_children,
            dagger_rounds=args.dagger_rounds,
            fresh=args.fresh,
            skip_dagger=args.skip_dagger,
        )
    return 0
