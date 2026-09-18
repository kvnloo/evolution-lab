# FlyForge autoresearch program

Optimize the **Hermes recovery kernel** (`local_plasticity`, ~640 weights) like a systems profiler.
Ground truth: frontier-kb critical path — P0 contract locked, P1 control table, P3 closed-loop.

## Frozen judge (never edit during search)

- `evolution_lab/bench.py`
- `evolution_lab/select.py`
- `autoresearch/config.json` gates and objective weights
- `data/p0/*` locked splits

## What you MAY change (phase 1)

- Propose candidates via `autoresearch_propose.propose_candidate` knobs only:
  - `hidden`, `dagger_rounds`, `plasticity_lr`, `plasticity_epochs`, `k_winners`, `seed` (`history` is locked at 8)
- Do **not** edit bench gates, observe.py secret rules, or locked splits.

## Objective

1. **Hard gates** (discard if any fail):
   - closed_loop_reward >= 1.0
   - confirm_success >= 0.95
   - val_success >= 0.95
   - ood_success >= 0.85
   - violations == 0
2. **Primary score** (minimize): `0.7 * advise_warm_p99_ms + 0.3 * closed_loop_ms`
3. **Keep** when score improves by >= 5% vs champion with gates pass.

Hosted joules stay `unknown`; latency proxy stands in until measured.

## Commands

```bash
# Record baseline champion
python3 -m evolution_lab fly-bench

# Autoresearch session (bounded)
python3 -m evolution_lab autoresearch --tag 20260917 --max-experiments 20 --patience 10

# Promote winning bundle to OMP hot path
python3 -m evolution_lab autoresearch --tag 20260917 --max-experiments 5 --promote-bundle
```

## Output

Each experiment prints a grep-friendly block (`fly_bench_v1`). Audit log: `runs/autoresearch/<tag>/results.tsv`.
Champion: `runs/autoresearch/<tag>/champion.json`.

## Stop rules

- Patience: N discards without keep
- max_experiments / max_hours from config
- Champion re-bench regression → halt

## Two timescales (ABAB)

- **A** (`cycle` + `abab_meta`): chooses the next *investigation* — 1-knob `autoresearch` session, or on plateau a parallel `sweep`. World: `runs/autoresearch/league/abab.json`.
- **B**: executes that choice. Frozen judge is still `bench.py` / `select.py`.
- **C**: a single discriminating bench (width floor, k_winners) when A says the experiment is cheaper than more search.

Do not edit the judge during A or B. `NO_UPDATE` on idle does **not** stop `--forever`; it flips the world to `slow` and the next session sweeps.

## Relation to `lab`

- `lab` / MAP-Elites: discover **families** across the control table
- `autoresearch`: tune the **production kernel** (bundled OMP student) for latency under gates
