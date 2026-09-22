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


def cmd_next_action_predict(*, text: str) -> None:
    from .next_action_gpu import predict_next_action

    print(json.dumps(predict_next_action(text), indent=2))


def cmd_next_action_confirm() -> None:
    from .next_action_gpu import shadow_confirm

    print(json.dumps(shadow_confirm(), indent=2))


def cmd_next_action_shadow(*, text: str) -> None:
    from .next_action_gpu import live_shadow

    print(json.dumps(live_shadow(text), indent=2))

def cmd_next_action_decide(*, text: str) -> None:
    from .next_action_gpu import decide_next_action

    print(json.dumps(decide_next_action(text), indent=2))


def cmd_next_action_coverage() -> None:
    from .next_action_gpu import evaluate_coverage_policy

    print(json.dumps(evaluate_coverage_policy(), indent=2))

def cmd_next_action_weak(*, boosts: str, n_pop: int, epochs: int) -> None:
    from .next_action_gpu import run_weak_family_search

    bs = tuple(int(x) for x in boosts.split(",") if x.strip())
    print(json.dumps(run_weak_family_search(boosts=bs or (1, 2, 4, 8), n_pop=n_pop, epochs=epochs), indent=2))


def cmd_next_action_2x2(*, n_pop: int, epochs: int, promote: bool) -> None:
    from .experiment_2x2 import run_2x2

    print(json.dumps(run_2x2(n_pop=n_pop, epochs=epochs, promote=promote), indent=2, default=str))

def cmd_capability_mine() -> None:
    from .capability_miner import run_capability_mine

    print(json.dumps(run_capability_mine(), indent=2, default=str))


def cmd_export_z0int(
    *,
    run: str | None,
    candidate: str,
    output: str,
    capability_id: str,
    family: str,
    runtime: str,
    genome_id: str | None,
) -> None:
    """Export unpromoted z0int.candidate_artifact.v1 bundle (offline)."""
    import json
    from pathlib import Path

    from .export_z0int import export_candidate

    result = export_candidate(
        run_dir=Path(run) if run else None,
        candidate=candidate,
        output=Path(output),
        capability_id=capability_id,
        family=family,
        runtime=runtime,
        genome_id=genome_id,
    )
    print(json.dumps(result, indent=2, default=str))


def cmd_preflight(*, text: str, capability: str | None, baseline_in: int | None, baseline_out: int | None) -> None:
    from .preflight import preflight

    print(
        json.dumps(
            preflight(
                text,
                capability_hint=capability,
                baseline_input_tokens=baseline_in,
                baseline_output_tokens=baseline_out,
            ),
            indent=2,
            default=str,
        )
    )

def cmd_l2_recovery(*, closed_loop_seeds: int, no_write: bool, no_promote: bool) -> None:
    from .l2_benchmark import run_l2_recovery_benchmark

    print(
        json.dumps(
            run_l2_recovery_benchmark(
                closed_loop_seeds=closed_loop_seeds,
                write=not no_write,
                promote=not no_promote,
            ),
            indent=2,
            default=str,
        )
    )









def cmd_outer_loop(*, skip_mine: bool, skip_l2: bool, closed_loop_seeds: int) -> None:
    from .outer_loop import run_outer_loop

    print(
        json.dumps(
            run_outer_loop(
                mine=not skip_mine,
                l2=not skip_l2,
                closed_loop_seeds=closed_loop_seeds,
            ),
            indent=2,
            default=str,
        )
    )


def cmd_gpu_abab(*, max_waves: int, n_pop: int, epochs: int) -> None:
    from .gpu_abab import run_gpu_abab

    print(json.dumps(run_gpu_abab(max_waves=max_waves, n_pop=n_pop, epochs=epochs), indent=2))


def cmd_gpu_evolve(*, n_pop: int, top_k: int, epochs: int, source: str, n_train: int, gold_only: bool) -> None:
    from .gpu_evolve import run_gpu_evolve
    from .next_action_gpu import run_next_action_gpu
    from .schema import DataRecipe

    recipe = DataRecipe(n_train=n_train, gold_only=gold_only, source=source)
    if source == "next_action":
        print(json.dumps(run_next_action_gpu(recipe=recipe, n_pop=n_pop, epochs=epochs), indent=2))
    else:
        print(json.dumps(run_gpu_evolve(n_pop=n_pop, top_k=top_k, epochs=epochs, recipe=recipe), indent=2))


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


def cmd_tool_tournament(
    *,
    fixtures: Path | None,
    out: Path,
    compositions: str | None,
    backend: str,
    seed: int,
) -> None:
    """Frozen grouped evaluation of system compositions (issue #23)."""
    from .tool_tournament import run_tournament

    names = None
    if compositions:
        names = [part for part in compositions.split(",") if part.strip()]
    try:
        run = run_tournament(
            fixtures_path=fixtures,
            composition_names=names,
            backend=backend,
            seed=seed,
            out_dir=out,
        )
    except RuntimeError as exc:
        raise SystemExit(f"tool-tournament: {exc}") from exc
    winners = {role: w["composition"] for role, w in run.report["role_winners"].items()}
    print(
        json.dumps(
            {
                "schema": run.report["schema"],
                "run_dir": str(run.run_dir),
                "compositions": list(run.report["compositions"]),
                "skipped_compositions": run.report["skipped_compositions"],
                "role_winners": winners,
                "pareto_frontier": [p["composition"] for p in run.pareto["frontier"]],
            },
            indent=2,
        )
    )


def cmd_phase2(
    *,
    action: str,
    observations: Path,
    fixtures: Path | None,
    out: Path,
    sealed_size: int,
    holdout_families: list[str],
    arms: list[str] | None = None,
) -> None:
    """Phase 2 preparation.  Compiles episodes and freezes splits; trains nothing."""
    from .episode import (
        build_split_plan,
        compile_episodes,
        label_census,
        selective_teacher_candidates,
        write_episodes,
    )

    out.mkdir(parents=True, exist_ok=True)
    episodes = compile_episodes(observations, fixtures_path=fixtures, arms=arms)
    write_episodes(episodes, out / "episodes.jsonl")
    summary: dict[str, Any] = {
        "schema": "qroute.phase2.plan.v1",
        "observations": str(observations),
        "out": str(out),
        "n_episodes": len(episodes),
        "arms": sorted(arms) if arms else None,
        "label_census": label_census(episodes),
        "trained": False,
    }
    if action in ("select", "plan"):
        selection = selective_teacher_candidates(episodes)
        (out / "teacher_selection.json").write_text(
            json.dumps(selection, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        summary["teacher_selection"] = {
            "n_selected": selection["n_selected"],
            "selection_rate": selection["selection_rate"],
            "reason_counts": selection["reason_counts"],
        }
    if action in ("split", "plan"):
        plan = build_split_plan(
            episodes,
            sealed_size=sealed_size,
            holdout_families=holdout_families or None,
        )
        (out / "splits.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        summary["splits"] = {
            "bucket_counts": plan["bucket_counts"],
            "split_digest": plan["split_digest"],
            "ood_families": plan["ood"]["holdout_families"],
            "sealed_size": len(plan["sealed"]["selected"]),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_q_route(
    *,
    action: str,
    out: Path | None,
    run: Path | None,
    sources: str | None,
    seed: int,
    default_out: Path,
    observations: Path | None = None,
) -> None:
    """Q-Route: learned per-region routing with an earned-complexity gate."""
    from .q_route import run as qr

    if action in ("build", "analyze"):
        dest = Path(out) if out is not None else (Path(run) if run is not None else default_out)
    else:
        dest = Path(run) if run is not None else (Path(out) if out is not None else default_out)
    try:
        if action == "build":
            summary = qr.build_run(dest, sources_value=sources)
        elif action == "utility":
            summary = qr.build_utility(dest)
        elif action == "q":
            summary = qr.build_q(dest)
        elif action == "gate":
            summary = qr.build_gate(dest)
        elif action == "distill":
            summary = qr.build_distill(dest, seed=seed)
        elif action == "report":
            path = qr.build_report(dest)
            summary = {"schema": qr.RUN_SCHEMA, "run_dir": str(dest), "report": str(path)}
        elif action == "analyze":
            summary = qr.build_analyze(dest, observations_path=observations)
        else:  # pragma: no cover - argparse constrains the choices
            raise RuntimeError(f"unknown q-route action: {action}")
    except RuntimeError as exc:
        raise SystemExit(f"q-route: {exc}") from exc
    print(json.dumps(summary, indent=2, sort_keys=True))


def cmd_capacity_queue(
    *,
    action: str,
    queue: Path | None,
    provider: str,
    model: str | None,
    max_tokens: int | None,
    max_cost_usd: float | None,
    include_protected: bool,
) -> None:
    """What useful compatible work is queued, and what would consume it?"""
    from .capacity_queue import QUEUE_SCHEMA, load_queue, plan_capacity

    items = load_queue(queue)
    if action == "list":
        ordered = sorted(items, key=lambda item: (item.priority, item.item_id))
        print(
            json.dumps(
                {
                    "schema": QUEUE_SCHEMA,
                    "count": len(ordered),
                    "items": [item.to_dict() for item in ordered],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    plan = plan_capacity(
        items,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        max_cost_usd=max_cost_usd,
        include_protected=include_protected,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))


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

    pnap = sub.add_parser("next-action-predict", parents=[parent])
    pnap.add_argument("text")
    sub.add_parser("next-action-confirm", parents=[parent], help="offline confirm readout of wave-5 champion")
    pnas = sub.add_parser("next-action-shadow", parents=[parent], help="append one live shadow row")
    pnas.add_argument("text")
    pnad = sub.add_parser(
        "next-action-decide",
        parents=[parent],
        help="coverage cascade: local high-conf EXEC/DEL else escalate to Jev",
    )
    pnad.add_argument("text")
    sub.add_parser(
        "next-action-coverage",
        parents=[parent],
        help="confirm-split risk/coverage for locked cascade policy",
    )
    pweak = sub.add_parser(
        "next-action-weak",
        parents=[parent],
        help="oversample EDIT/WEB/VERIFY/ABSTAIN on frozen arch; promote-only",
    )
    pweak.add_argument("--boosts", default="1,2,4,8", help="comma-separated oversample factors")
    pweak.add_argument("--pop", type=int, default=256)
    pweak.add_argument("--epochs", type=int, default=6)
    p2x2 = sub.add_parser(
        "next-action-2x2",
        parents=[parent],
        help="2x2 labels x features; metric coverage at 95 pct precision; promote-only",
    )
    p2x2.add_argument("--pop", type=int, default=256)
    p2x2.add_argument("--epochs", type=int, default=6)
    p2x2.add_argument("--no-promote", action="store_true")
    sub.add_parser(
        "capability-mine",
        parents=[parent],
        help="mine micro-process CapabilityCards + sealed L0 atlas under ~/.z0int/research",
    )
    pez = sub.add_parser(
        "export-z0int",
        parents=[parent],
        help="Export unpromoted z0int.candidate_artifact.v1 (offline; never promotes)",
    )
    pez.add_argument("--run", default=None)
    pez.add_argument("--candidate", required=True)
    pez.add_argument("--output", required=True)
    pez.add_argument("--capability-id", default="coding.recovery_action", dest="capability_id")
    pez.add_argument("--family", default="local_plasticity")
    pez.add_argument("--runtime", default="z0int.backend.mushroom.v1")
    pez.add_argument("--genome-id", default=None, dest="genome_id")
    ppre = sub.add_parser(
        "preflight",
        parents=[parent],
        help="RESEARCH/SHADOW only — production preflight is z0intelligence; never mints verified_success",
    )
    ppre.add_argument("text")
    ppre.add_argument("--capability", default=None, help="capability_id hint (e.g. blender.scene_reasoning)")
    ppre.add_argument("--baseline-in", type=int, default=None, dest="baseline_in")
    ppre.add_argument("--baseline-out", type=int, default=None, dest="baseline_out")
    pl2 = sub.add_parser(
        "l2-recovery",
        parents=[parent],
        help="L2 sealed outcome battery for recovery_action (+ shadow→canary gate)",
    )
    pl2.add_argument("--closed-loop-seeds", type=int, default=24)
    pl2.add_argument("--no-write", action="store_true", help="Do not write ~/.z0int/benchmarks")
    pl2.add_argument("--no-promote", action="store_true", help="Eval only; skip canary marker")





    po = sub.add_parser(
        "outer-loop",
        parents=[parent],
        help="mine → L2 recovery sealed battery → canary (thin continuous loop)",
    )
    po.add_argument("--skip-mine", action="store_true")
    po.add_argument("--skip-l2", action="store_true")
    po.add_argument("--closed-loop-seeds", type=int, default=24)

    pabab = sub.add_parser("gpu-abab", parents=[parent], help="ABAB around GPU next-action recipes")
    pabab.add_argument("--max-waves", type=int, default=8)
    pabab.add_argument("--pop", type=int, default=256)
    pabab.add_argument("--epochs", type=int, default=6)
    pgpu = sub.add_parser("gpu-evolve", parents=[parent], help="GPU population filter then CPU judge")
    pgpu.add_argument("--pop", type=int, default=256)
    pgpu.add_argument("--top", type=int, default=5)
    pgpu.add_argument("--epochs", type=int, default=8)
    pgpu.add_argument("--source", default="hermes_recovery", choices=["hermes_recovery", "next_action"])
    pgpu.add_argument("--n-train", type=int, default=64)
    pgpu.add_argument("--gold-only", action="store_true")
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
    ptt = sub.add_parser(
        "tool-tournament",
        parents=[parent],
        help="frozen grouped evaluation of tool-calling system compositions (issue #23)",
    )
    ptt.add_argument("--fixtures", type=Path, default=None, help="frozen fixtures.jsonl path")
    ptt.add_argument("--out", type=Path, default=root / "runs" / "tool_tournament")
    ptt.add_argument("--compositions", type=str, default=None, help="comma list of names or A–F aliases")
    ptt.add_argument(
        "--backend",
        type=str,
        default="deterministic",
        choices=("deterministic", "scripted", "local-slm"),
        help="backend source; deterministic and scripted need no GPU/network",
    )
    ptt.add_argument("--seed", type=int, default=42)
    pqr = sub.add_parser(
        "q-route",
        parents=[parent],
        help="learned per-region routing over the mechanism ladder (earned complexity)",
    )
    pqr.add_argument(
        "action",
        choices=("build", "utility", "q", "gate", "distill", "report", "analyze"),
    )
    pqr.add_argument(
        "--out",
        type=Path,
        default=None,
        help="run directory for `build` (default: runs/q_route)",
    )
    pqr.add_argument(
        "--run",
        type=Path,
        default=None,
        help="run directory for the other stages (default: runs/q_route)",
    )
    pqr.add_argument(
        "--sources",
        type=str,
        default=None,
        help="override inputs as comma list of kind:path (composition|bounded|orchestration)",
    )
    pqr.add_argument(
        "--observations",
        type=Path,
        default=None,
        help="densified Phase 1B observations.jsonl for analyze (section G)",
    )
    pqr.add_argument("--seed", type=int, default=42)
    p2 = sub.add_parser(
        "phase2",
        parents=[parent],
        help="Phase 2 preparation: compile episodes and freeze splits (trains nothing)",
    )
    p2.add_argument(
        "action",
        choices=("compile", "select", "split", "plan"),
        help="compile episodes | nominate teacher candidates | freeze splits | all three",
    )
    p2.add_argument("--observations", type=Path, required=True,
                    help="Phase 1B observations.jsonl written by densify_measurements.py")
    p2.add_argument("--fixtures", type=Path, default=None,
                    help="fixture file joining state text onto episodes")
    p2.add_argument("--out", type=Path, default=None,
                    help="output directory (default: runs/phase2)")
    p2.add_argument("--sealed-size", type=int, default=12)
    p2.add_argument("--holdout-family", action="append", default=[])
    p2.add_argument("--arm", action="append", default=[],
                    help="restrict the corpus to this arm (repeatable); omit for every receipt")
    pcq = sub.add_parser(
        "capacity-queue",
        parents=[parent],
        help="explicitly queued useful work for otherwise-expiring free capacity",
    )
    pcq.add_argument("action", choices=("list", "plan"))
    pcq.add_argument(
        "--queue",
        type=Path,
        default=None,
        help="queue JSON (default: data/capacity_queue/queue.json)",
    )
    pcq.add_argument("--provider", type=str, default="local", help="capacity owner provider id")
    pcq.add_argument("--model", type=str, default=None)
    pcq.add_argument("--max-tokens", type=int, default=None)
    pcq.add_argument("--max-cost-usd", type=float, default=None)
    pcq.add_argument(
        "--include-protected",
        action="store_true",
        help="include certification-only splits (confirm/OOD/held-out); off by default",
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
    elif args.cmd == "tool-tournament":
        cmd_tool_tournament(
            fixtures=getattr(args, "fixtures", None),
            out=args.out,
            compositions=getattr(args, "compositions", None),
            backend=getattr(args, "backend", "deterministic"),
            seed=getattr(args, "seed", 42),
        )
    elif args.cmd == "phase2":
        cmd_phase2(
            action=args.action,
            observations=args.observations,
            fixtures=args.fixtures,
            out=args.out or (root / "runs" / "phase2"),
            sealed_size=args.sealed_size,
            holdout_families=args.holdout_family,
            arms=args.arm or None,
        )
    elif args.cmd == "capacity-queue":
        cmd_capacity_queue(
            action=args.action,
            queue=args.queue,
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
            max_cost_usd=args.max_cost_usd,
            include_protected=args.include_protected,
        )
    elif args.cmd == "q-route":
        cmd_q_route(
            action=args.action,
            out=getattr(args, "out", None),
            run=getattr(args, "run", None),
            sources=getattr(args, "sources", None),
            seed=getattr(args, "seed", 42),
            default_out=root / "runs" / "q_route",
            observations=getattr(args, "observations", None),
        )
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

    elif args.cmd == "next-action-predict":
        cmd_next_action_predict(text=args.text)
    elif args.cmd == "next-action-confirm":
        cmd_next_action_confirm()
    elif args.cmd == "next-action-shadow":
        cmd_next_action_shadow(text=args.text)
    elif args.cmd == "next-action-decide":
        cmd_next_action_decide(text=args.text)
    elif args.cmd == "next-action-coverage":
        cmd_next_action_coverage()
    elif args.cmd == "next-action-weak":
        cmd_next_action_weak(boosts=args.boosts, n_pop=args.pop, epochs=args.epochs)
    elif args.cmd == "next-action-2x2":
        cmd_next_action_2x2(n_pop=args.pop, epochs=args.epochs, promote=not args.no_promote)
    elif args.cmd == "capability-mine":
        cmd_capability_mine()
    elif args.cmd == "export-z0int":
        cmd_export_z0int(
            run=getattr(args, "run", None),
            candidate=args.candidate,
            output=args.output,
            capability_id=getattr(args, "capability_id", "coding.recovery_action"),
            family=getattr(args, "family", "local_plasticity"),
            runtime=getattr(args, "runtime", "z0int.backend.mushroom.v1"),
            genome_id=getattr(args, "genome_id", None),
        )
    elif args.cmd == "preflight":
        cmd_preflight(
            text=args.text,
            capability=getattr(args, "capability", None),
            baseline_in=getattr(args, "baseline_in", None),
            baseline_out=getattr(args, "baseline_out", None),
        )
    elif args.cmd == "l2-recovery":
        cmd_l2_recovery(
            closed_loop_seeds=getattr(args, "closed_loop_seeds", 24),
            no_write=getattr(args, "no_write", False),
            no_promote=getattr(args, "no_promote", False),
        )





    elif args.cmd == "outer-loop":
        cmd_outer_loop(
            skip_mine=getattr(args, "skip_mine", False),
            skip_l2=getattr(args, "skip_l2", False),
            closed_loop_seeds=getattr(args, "closed_loop_seeds", 24),
        )

    elif args.cmd == "gpu-abab":
        cmd_gpu_abab(max_waves=args.max_waves, n_pop=args.pop, epochs=args.epochs)
    elif args.cmd == "gpu-evolve":
        cmd_gpu_evolve(n_pop=args.pop, top_k=args.top, epochs=args.epochs, source=args.source, n_train=args.n_train, gold_only=args.gold_only)
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
