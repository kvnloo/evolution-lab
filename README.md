# Evolution Lab

Original experiment engine for the FlyForge / Hermes-recovery research program.

This is **not** a fork of FlyGym, OpenEvolve, Stonkfly, or [kvnloo/fly-wirehead](https://github.com/kvnloo/fly-wirehead). The gym is `hermes_recovery` (Gymnasium `reset`/`step`). The knowledge vault stays in [kvnloo/frontier-kb](https://github.com/kvnloo/frontier-kb).

The object under optimization is **the frontier of measured systems**, not “make a fly beat Qwen.” Every candidate is an **experiment genome**. Selection is Pareto hypervolume + MAP-Elites niches, with a promotion ladder so most ideas die at a seconds-scale sanity test.

Tinker SFT/RL and Cursor Cloud Agent fan-out are declared backends, not fake training runs.

Contribution contract: [Verified OSS Loop](https://github.com/kvnloo/verified-oss-loop). Maintainer roadmap: [ROADMAP.md](ROADMAP.md). Workers claim `claimable` issues (P0 is [#2](https://github.com/kvnloo/evolution-lab/issues/2)); they never merge `main`. Rollout scheme `rolling`: day-pass PRs target `preview`; overnight PRs target `nightly`.

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

```sh
pip install -e .
python -m evolution_lab seed
python -m evolution_lab run --level 1
python -m evolution_lab evolve --generations 2
python -m evolution_lab dashboard
python -m evolution_lab gym-smoke
python -m evolution_lab table --level 1
python -m evolution_lab advise '{"sandbox_alive": 0, "retry": 0, "budget": 1}'
python -m evolution_lab advise '{"sandbox_alive": 1, "transient": 1, "retry": 0}' --family local_plasticity
python -m unittest discover -s tests -p 'test_*.py'
bash scripts/verify.sh
python -m evolution_lab receipt --issue 3
```

`table` writes `runs/p0/control_table.json` (or `--run-dir`): Track A families together, `vs_teacher` vs the registered product target, and the P1 kill criterion (GRU **and** direct-input dominating the reservoir on val → do not MaleCNS SGD; pivot to motif students). Hosted joules stay unknown. `fly_connectome` remains an alias of `fixed_reservoir`.

Logs: `runs/<run_id>/archive.jsonl`  
Dashboard: `runs/<run_id>/dashboard.html`

On the delayed-cue recovery task (L1): teacher **1.00**, mlp/fixed-reservoir/rewire **1.00**, gru **~0.92**, direct-input **~0.79**. The all-systems 2-D front is the free teacher; learned students are compared on `front_learned`. Hosted joules stay unknown.

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
| [kvnloo/frontier-kb](https://github.com/kvnloo/frontier-kb) | Claims, protocol, Pareto interpretation |
| [kvnloo/aodl](https://github.com/kvnloo/aodl) | HOTL 0.2; specialist is a **port**, not a new kind |
| [kvnloo/fly-wirehead](https://github.com/kvnloo/fly-wirehead) | MaleCNS Shorts pinout demo (fork of mattyhempstead/fly-wirehead) |

## What this is not

- Not MaleCNS LIF playback
- Not a Tinker client until `tinker_sft` is wired (it refuses today)
- Not an AODL runtime
- Not a FlyGym or OpenEvolve rewrite
