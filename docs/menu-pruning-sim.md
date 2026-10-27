# Menu-pruning simulator (GJ-2)

This is an **offline simulation only**. It neither schedules real workers nor
estimates actual model quality, fleet throughput, or compute savings.

## Model and budget

A menu has **K** numbered arms. Arm i has a fixed but unobserved Bernoulli
success probability **p_i**. A run can use explicit probabilities or sample
each p_i independently from a seeded **Beta(a, b)** distribution (default
Beta(1, 1)). Every completed trial reveals one Bernoulli outcome; policies
never receive the true p_i.

There are **L** parallel lanes and a total **B lane-minute** budget. A trial
on arm i occupies one lane for **cost_i** simulated minutes (default 1).
Idle lanes schedule trials independently. The simulator uses deterministic
completion events, not real time, threads, processes, or device access.
Trial outcomes are hidden until completion. Cost is **reserved at dispatch**,
so in-flight work counts against B; no policy can oversubscribe the budget.
After a policy commits to one arm, already-dispatched trials finish but no new
trials begin.

The environment, Bernoulli outcomes, and policy decisions have independently
seeded random streams derived from the same input seed. Thus different
policies see the same latent menu for a given seed, without using global RNG.

## Policies

Each policy implements the observable interface:
`choose(state, rng) -> arm | None` and `prune(state) -> set[arm]`.
`state` exposes observed counts, successes, in-flight counts, live arms,
costs, budget, and a seeded policy RNG, **not** latent success probabilities.

| Policy | Allocation | Pruning |
| --- | --- | --- |
| `uniform` | Round-robin across affordable live arms | Never |
| `successive_halving` | Equal completed-plus-pending trial quotas for each live arm | At each completed rung, keep the top `ceil(n / eta)` by observed mean (ties: lower arm ID) |
| `ucb1` | Explore unobserved arms, then choose the largest empirical-mean UCB1 bonus | Remove i when Hoeffding upper(i) is strictly below the largest lower bound |
| `thompson` | Sample Beta posterior after an exploration floor | Monte Carlo estimate of P(i is best); remove i when estimated probability is below epsilon |

Defaults: halving `eta=2` and `rung_trials=8` per arm in rung 0, multiplied by
eta for each subsequent rung; UCB1 `delta=0.001`; Thompson
`epsilon=0.005`, 512 posterior draws per check, and 12 trials per live arm
before any elimination. Thompson starts with independent Beta(1, 1) priors;
its pruning check is performed after roughly one fresh observation per live
arm to avoid excessive Monte Carlo work. If all estimates fall below epsilon,
the highest estimated probability arm is retained.

UCB1 uses `mean + sqrt(2 log(t)/n)` for selection (accounting for pending
work when distributing lanes). Its pruning bounds are clipped to [0, 1],
with radius `sqrt(log(2K/delta)/(2n))`; unobserved arms have [0, 1].
These are **pointwise Hoeffding bounds**, not a time-uniform or
multiple-look guarantee. Thompson probabilities are finite Monte Carlo
estimates, not exact posterior probabilities. Neither policy is guaranteed
never to prune the best arm.

## Metrics

A run returns:

- `best_arm_found`: recommended arm equals the first argmax of the true p_i.
  Ties use the lowest arm ID.
- `simple_regret`: `max(p_i) - p_recommendation` (zero for any equally good arm).
- `lane_minutes_to_commit`: reserved lane-minutes at the **first** point only
  one arm remains; `null` if the policy never reaches one live arm. This is
  resource expenditure, **not** elapsed wall-clock minutes.
- `wrongly_pruned_best`: whether the first true argmax was ever eliminated.
- `trials_per_arm`: number of finished trials per arm (includes in-flight
  trials that finish after commitment).
- `successes_per_arm`: completed Bernoulli successes (seeded outcome audit).
- `elapsed_simulated_minutes`: final simulated completion time; unlike total
  lane-minutes spent, this accounts for overlap between occupied lanes.
- `spent_lane_minutes`, `recommendation`, `true_best_arm`, `live_arms`,
  and `latent_probabilities` for audit and budget reconciliation.

`run_experiment(policy_factory, K, L, B, seeds)` creates a fresh policy for
each seed. It reports mean regret, fraction best found, fraction wrongly
pruned, fraction committed, and median lane-minutes-to-commit **among runs
that committed**; the median is `null` when none committed. The committed
fraction must be read alongside the conditional median.

## Run

From the repository root:

    python -m evolution_lab.menu_sim --K 16 --L 4 --B 400 --seeds 200 --json
    python -m unittest discover -s tests -p 'test_menu_sim.py' -v

`--json` emits four newline-delimited JSON objects, one per policy; other
invocations print one text summary per policy. CLI seeds are `0..N-1`.
For custom menus and lane costs, use `run_simulation` or `run_experiment` with
`probabilities=[...]` and `costs=[...]`. These pure Python APIs do not
read/write files or spawn workers.

## Mapping fleet experiments

Map K to the candidate experiment arms; L to the number of independent
worker slots able to run one trial each; B to the **total pooled lane-minute
allocation**, not wall-clock time. Set `cost_i` to a representative trial
duration times occupied lanes (one in this model). Validate empirically
whether trials are independent Bernoulli draws and whether costs are stable.
Real tasks with asynchronous failures, nonstationarity, correlated outcomes,
unequal lane capacity, setup overhead, or preemption require a richer model.

**Do not use simulated rankings as a promotion/KEEP decision.** Validate
policies and parameters on held-out real measurements with an independent
review gate before altering fleet allocations.
