# Goal — FlyForge continuous kernel evolution

## Objective
Run a continuous outer cycle around `evolution_lab autoresearch` that evolves the best
Hermes recovery fly (`local_plasticity`) without stopping. SearchDriver proposes;
`bench.py` / `select.py` stay the frozen judge (frontier-kb / OpenTuner split).
Hosted joules stay unknown; minimize latency proxy under PRODUCT_TARGET gates.

## Success criteria
1. Outer loop process `fly-cycle` is running and writes `runs/autoresearch/league/status.json` after every session.
2. Every session starts from the current league champion (not a fresh baseline).
3. Hard gates remain: closed_loop_reward >= 1.0, confirm/val >= 0.95, ood >= 0.85, violations == 0.
4. A keep promotes `data/p0/recovery_student.npz` and updates `league/champion.json`.
5. `history` is not a search knob (locked splits use history=8).
6. League `latency_score` is <= the 20260917 champion (6.319) and never regresses.

## Verification
```
python3 -m unittest tests.test_autoresearch_select tests.test_cycle tests.test_autoresearch_propose -v
python3 -m evolution_lab recovery '{"action":"status"}'
test -f runs/autoresearch/league/status.json
```

## Boundaries
- Allowed: `evolution_lab/cycle.py`, `autoresearch_propose.py`, `cli.py`, `autoresearch/`, `scripts/fly-cycle.sh`, tests, `runs/autoresearch/**` (gitignored), promote into `data/p0/recovery_student.npz`.
- Frozen: `bench.py`, `select.py` gates, `data/p0/*.npz` splits, `observe.py` secret rules, `ROADMAP.md`.
- Forbidden: MaleCNS SGD, FlyGym/OpenEvolve fork, editing judge during search.

## Stop conditions
- User says stop.
- League champion fails gates on re-bench → halt and page.
- Do not stop for patience, token budget, or a single session plateau — shrink epsilon and continue.
