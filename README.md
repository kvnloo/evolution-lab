# Evolution Lab

Experiment engine for evolving fly-style specialists on the FlyForge stack.

**Start with [kvnloo/openjev](https://github.com/kvnloo/openjev).** OpenJev is what people clone
first — it sets up typed JEV decisions, SLM scorers, and the runtime to *run* models. Evolution
Lab is what you add to *evolve* them: genomes, locked splits, promotion ladder, DAgger, and
control tables. Track B genomes call OpenJev as their training backend; Track A recovery fly runs
native NumPy here.

```bash
# From openjev repo root (recommended entry point):
bash scripts/setup-flyforge.sh
```

Or install this repo directly after OpenJev is already editable:

```bash
pip install -e /path/to/openjev   # runtime substrate first
pip install -e .                  # evolution engine
```

Integration contract: [openjev/docs/evolution-lab.md](https://github.com/kvnloo/openjev/blob/master/docs/evolution-lab.md).

This is **not** a fork of FlyGym, OpenEvolve, Stonkfly, or [kvnloo/fly-wirehead](https://github.com/kvnloo/fly-wirehead). The gym is `hermes_recovery` (Gymnasium `reset`/`step`). The knowledge vault stays in [kvnloo/frontier-kb](https://github.com/kvnloo/frontier-kb).

The object under optimization is **the frontier of measured systems**, not “make a fly beat Qwen.” Every candidate is an **experiment genome**. Selection is Pareto hypervolume + MAP-Elites niches, with a promotion ladder so most ideas die at a seconds-scale sanity test.

Tinker SFT/RL and Cursor Cloud Agent fan-out are declared backends, not fake training runs.

Contribution contract: [Verified OSS Loop](https://github.com/kvnloo/verified-oss-loop). Maintainer roadmap: [ROADMAP.md](ROADMAP.md). Process dump (GitHub Pages): https://kvnloo.github.io/z0int/ Workers claim `claimable` issues (P0 is [#2](https://github.com/kvnloo/z0int/issues/2)); they never merge `main`. Rollout scheme `rolling`: day-pass PRs target `preview`; overnight PRs target `nightly`.

## Goal

> Continuously discover, verify, and explain computational systems that expand the achievable frontier between capability and resources.

## P0 task

Typed Hermes recovery events → bounded action `{retry, restart_sandbox, escalate, noop, page_human}` → external verifier. Secret-bearing observations fail closed at level 0.

Control table (must always be in the archive together):

| Genome | Question |
| --- | --- |
| `teacher_rule` | reference policy |
| `direct_input` | last-step linear control (FLM lesson) |
| `mlp` | flattened history, no recurrence |
| `gru` | recurrent compression |
| `fixed_reservoir` | frozen sparse recurrent + readout |
| `rewired_reservoir` | degree-preserving-ish random graph, **refit** |
| `local_plasticity` | mushroom-body analogue: frozen PN→KC (flattened Hermes history, not the compound eye) + local KC→MBON |

## Run

**One command (recommended):**

```sh
pip install -e .
bash scripts/lab-run.sh
# or: python -m evolution_lab lab --evolve-generations 2 --dagger-rounds 2
```

Writes `runs/p0/experiment_summary.json` (archive stats, P1 control table, P3 DAgger closed-loop).

**Step-by-step:**

```sh
pip install -e .
python -m evolution_lab seed
python -m evolution_lab run --level 1
python -m evolution_lab evolve --generations 2
python -m evolution_lab dashboard
python -m evolution_lab gym-smoke
python -m evolution_lab table --level 1
python -m evolution_lab dagger-smoke --rounds 2
python -m evolution_lab advise '{"sandbox_alive": 0, "retry": 0, "budget": 1}'
python -m evolution_lab advise '{"sandbox_alive": 1, "transient": 1, "retry": 0}' --family local_plasticity
python -m unittest discover -s tests -p 'test_*.py'
bash scripts/verify.sh
python -m evolution_lab receipt --issue 3
```

Scheduled CI: `.github/workflows/experiment.yml` (daily + manual dispatch). Cursor Cloud: `.cursor/environment.json`.

`table` writes `runs/p0/control_table.json` (or `--run-dir`): Track A families together, `vs_teacher` vs the registered product target, and the P1 kill criterion (GRU **and** direct-input dominating the reservoir on val → do not MaleCNS SGD; pivot to motif students). Hosted joules stay unknown. `fly_connectome` remains an alias of `fixed_reservoir`.

Logs: `runs/<run_id>/archive.jsonl`  
Dashboard: `runs/<run_id>/dashboard.html`

On the delayed-cue recovery task (L1, locked `data/p0/`): teacher **1.00**; mlp / rewired-reservoir / **local_plasticity** confirm **1.00**; fixed-reservoir confirm **1.00** val **~0.96**; gru **~0.92 / 0.90**; direct-input **~0.79 / 0.63**. The mushroom-body analogue uses **640** plastic KC→MBON weights (flatten+last+maxpool Hermes PNs, not the compound eye). Kill criterion does **not** fire: GRU and direct-input do not both dominate the reservoir on val. Still do **not** SGD MaleCNS. Hosted joules stay unknown. Teacher cost is ~0 params, so `cost_vs_teacher` is not an energy number; `cost_vs_mlp` compares learned students.

## Gym

| Name | Role here |
| --- | --- |
| `hermes_recovery` | P0 gym (`evolution_lab.gym`) |
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | API we duck-type |
| [FlyGym 2.x](https://github.com/NeLy-EPFL/flygym) | Later visuo-motor env (consume 2.x; do not fork `flygym-gymnasium`) |
| [OpenEnv](https://github.com/huggingface/OpenEnv) | Later HTTP/Docker Hermes sandbox |
| [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) | Later mutation backend. `make_env("openevolve")` fails closed |

## Sister repos

| Repo | Owns |
| --- | --- |
| [kvnloo/openjev](https://github.com/kvnloo/openjev) | **Runtime infra** — clone first; JEV lane, SLM scorers, setup script |
| [kvnloo/frontier-kb](https://github.com/kvnloo/frontier-kb) | Claims, protocol, Pareto interpretation |
| [kvnloo/aodl](https://github.com/kvnloo/aodl) | HOTL 0.2; specialist is a **port**, not a new kind |
| [kvnloo/fly-wirehead](https://github.com/kvnloo/fly-wirehead) | MaleCNS Shorts pinout demo (fork of mattyhempstead/fly-wirehead) |

## What this is not

- Not MaleCNS LIF playback
- Not a Tinker client until `tinker_sft` is wired (it refuses today)
- Not an AODL runtime
- Not a FlyGym or OpenEvolve rewrite
