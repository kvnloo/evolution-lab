# Phase 2 corpus slice — executed

Run: `p1b-20260921T1430Z` · schema `qroute.phase2.plan.v1` · **no training performed**

## What this is

The first Phase 2 slice, exactly as proposed in
`openjev/docs/phase1b-decision-artifact.md` § 15: teach Q-Route the
bounded-choice ownership it is currently guessing at, without contaminating
anything with model-generated labels.

```bash
evolution-lab phase2 plan \
  --observations results/phase1b/p1b-20260921T1430Z/observations.jsonl \
  --fixtures     benchmarks/fixtures/local-cognition-v1/examples.jsonl \
  --arm compiler+hammer2.1_3b --arm compiler+qwen3.5_4b --arm compiler+qwen3.5_9b \
  --arm compiler+functiongemma_270m --arm compiler+nemotron_orchestrator_8b \
  --holdout-family abstention --holdout-family dependencies \
  --holdout-family parallelism --holdout-family uncertainty \
  --sealed-size 12 --out runs/phase2/p1b-20260921T1430Z
```

## Result

| | |
|---|---|
| episodes | **420** (28 states x 5 arms x 3 reps) |
| label census | **gold 420** (100%) |
| `verified_success` null | 0 |
| `budget_units` present | 420 / 420 |
| arms | the 5 compiler-first arms only |
| distinct tasks | 28 |
| **split_digest** | **`c9ef295816bba61b`** |
| episodes.jsonl sha256[:16] | `e5ac110980a9040a` |

Buckets (episode level, from `splits.json`):

| bucket | episodes | tasks |
|---|---|---|
| train | 195 | 13 |
| validation | 45 | 3 |
| confirm | 75 | 5 |
| ood | 90 | 6 |
| sealed_human_audited | 15 | 1 |

Chronological stage before holdouts: train 285 / validation 60 / confirm 75
(70/15/15 by task). OOD and sealed are then carved out of those.

## Exit criteria

| criterion | result |
|---|---|
| episodes validate against the schema | pass — canonical `Episode` records |
| no bucket overlaps | pass — 0 episode-level and 0 **task-level** overlaps |
| `split_digest` stable across two runs | pass — `c9ef295816bba61b` on both |
| label census 100% gold | pass — 420/420 |

## Carried caveats

- The sealed set requested 12 episodes and sealed **15**, because the smallest
  sealable unit is one task and the first task carries 15 episodes. The reason
  is recorded in `splits.json.sealed.overshoot_reason` rather than quietly
  sealing a partial task and leaking its siblings into training.
- `runs/` is gitignored, so the corpus itself is local. This file records the
  digest and the exact command that reproduces it.

## Not done here

No router was trained, no threshold tuned, no gate touched, and neither
`mushroom` nor `fly` was activated. Teacher inference spend was $0 — every label
is a deterministic fixture gold label.
