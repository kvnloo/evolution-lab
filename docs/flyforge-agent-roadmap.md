# FlyForge agent research roadmap

Companion to maintainer-owned [ROADMAP.md](../ROADMAP.md). Agents may update **this** file; do not rewrite `ROADMAP.md` or `roadmap.yml`.

**Goal (unchanged):** continuously discover, verify, and explain systems that expand the frontier between capability and resources — for Kevin’s stack that means a **cheap Hermes recovery specialist**, AODL **ports** (existing kinds only), and mushroom-body **motifs**, not MaleCNS SGD, not DEX memes, not a Qwen replacement.

## Status snapshot (2026-09-17)

| Track | State | Evidence |
| --- | --- | --- |
| P0 Hermes contract | Done on `main` | GitHub [#2](https://github.com/kvnloo/evolution-lab/issues/2), Linear PER-1524 |
| P1 control table + MB motif | Code on `cursor/p1-control-table-9425` @ `20555e7` | GitHub [#3](https://github.com/kvnloo/evolution-lab/issues/3), PER-1525 |
| Vault numbers + patch | Pushed `cursor/p1-control-table-kb-9425` | [frontier-kb#16](https://github.com/kvnloo/frontier-kb/pull/16) |
| TMNF / SHERWOOD | Separate branches; fail-closed in lab gym | Not P0; do not vendor binaries |
| fly-wirehead | Pinout demo only | [kvnloo/fly-wirehead](https://github.com/kvnloo/fly-wirehead) |

## What “useful fly” means here

1. **Hermes recovery** on typed events → `{retry, restart_sandbox, escalate, noop, page_human}` with secrets fail-closed.
2. **`local_plasticity`**: frozen sparse PN→KC on **engineered Hermes history** (flatten + last + max-over-time; not compound eye, not MaleCNS); **only** KC→MBON plastic (~640 weights on L1).
3. **CLI:** `python -m evolution_lab table --level 1`, `python -m evolution_lab advise … --family local_plasticity`.
4. **Kill criterion (P1):** if GRU **and** direct-input both ≥ reservoir on **val**, do not spend a cycle on MaleCNS SGD; pivot to motif students. **Measured:** criterion **does not fire**; MaleCNS SGD still **refused** regardless.

### Locked L1 table (delayed cue, `data/p0/`, seed 20260912)

| Family | Confirm | Val | Params |
| --- | ---: | ---: | ---: |
| teacher | 1.00 | 1.00 | 0 |
| direct_input | 0.79 | 0.63 | 75 |
| mlp | 1.00 | 1.00 | 4032 |
| gru | 0.92 | 0.90 | 3000 |
| fixed_reservoir | 1.00 | 0.96 | 3264 |
| local_plasticity | 1.00 | 0.98 | 640 |

Best perfect learned student on confirm (tie-break cost): **local_plasticity**. `cost_vs_mlp ≈ 0.31`. Teacher cost ≈ 0 params — do not read `cost_vs_teacher` as joules.

### Honest gap: closed-loop ≠ last-step

Last-step training hits confirm 1.00; **closed-loop** rollout of that student in `hermes_recovery` is ~**0.56** mean reward vs teacher **1.00**. That is **P3 / DAgger** (train on visited states), not a reason to fake SFT or score MaleCNS on SWE-bench.

## Next engineering (priority)

1. **Open / land draft PR** `cursor/p1-control-table-9425` → **`nightly`** (overnight target); receipt issue `"3"`; do not merge `main`/`dev`.
2. **P3 closed-loop:** aggregate student trajectories on locked splits; bounded RL only when `tinker_rl` is wired (fail-closed today).
3. **AODL port:** HOTL 0.2 executor-shaped specialist JSON (existing kinds only) — see frontier-kb `lit-20260912-aodl-fly-specialist-port-sketch`.
4. **Cloud environment:** add `kvnloo/evolution-lab` to the agent environment + GitHub App; keep `GH_TOKEN` for push. OpenCode key: prefer secret name **`OPENCODE_API_KEY`** (dashboard cannot rename; duplicate then delete misnamed `OPENROUTER_API_KEY` if present).
5. **P2 distill:** teacher trajectories + outcomes after P1 table is on `nightly`.

## Explicit non-goals

- MaleCNS global SGD, FlyWire-as-trader (SHERWOOD clip), TMNF racing as P0, replacing Qwen3.8-27B, hosted joules without measurement, merging `main` from workers, self-applying `claimable` on GitHub #3.

## Sister repos

| Repo | Role |
| --- | --- |
| [frontier-kb](https://github.com/kvnloo/frontier-kb) | Claims vault; [[literature/lit-20260914-p1-control-table-kill-criterion]] in git |
| [aodl](https://github.com/kvnloo/aodl) | HOTL 0.2; specialist is a port |
| [fly-wirehead](https://github.com/kvnloo/fly-wirehead) | MaleCNS Shorts pinout demo |

## Commands (proof)

```sh
python -m unittest discover -s tests -p 'test_*.py'
python -m evolution_lab gym-smoke --n 8
python -m evolution_lab table --level 1
python -m evolution_lab advise '{"sandbox_alive": 0, "retry": 0, "budget": 1}' --family local_plasticity
```
