# Pre-registration: verified loop v0 (learned ACT/ASK/OBSERVE gate vs deterministic gate)

Status: **registered before any learner code or result exists on this branch.** This file is
committed first; the learner, the run and the result are later commits. Any change to the rules
below after the first result is a new pre-registration (v1) with its own commit, and the v0
result stays reported against v0.

Context: kvnloo/z0#15 steps 7-9, z0intelligence#54/#55/#56, evolution-lab#24 ("learn state
selection, retrieval routing, and observe-vs-act from verified trajectories").

## Question

Given only what the deterministic gate saw at prompt time, can a learned policy choose **when to
ACT** (vs fall back to ASK / OBSERVE / ABSTAIN) so that, **at the same ACT coverage**, fewer of the
turns it lets through end in a verified non-success?

A positive answer only earns **shadow**: the learned policy is logged next to the deterministic
gate and controls nothing. Promotion past shadow needs a separate, later registration with
online (shadow) evidence.

## Data

- Input: `z0int.loop.training_row.v0`, `table_version` 0.1.0, produced by
  `z0int outcomes export` (z0intelligence branch `feat/loop-export-v0`). The run records the
  manifest's `feature_schema_sha` and source sha256 prefixes.
- **Features**: the `features` object only (prompt-time). `observed` is post-decision and is never
  a model input. `gate=*` columns are inputs (the learner may condition on the baseline's choice).
- **Label**: `y = label.y_success` (1 = verified_success; 0 = verified_failure or contested).
- **Analysis set**: rows with `has_opportunity`, `label.resolved`, and `observed.action == "ACT"`.
  Only on those turns is the verified outcome the outcome of acting. Rows where the agent asked,
  has no observed row, or is unverified are excluded and counted.
- **Groups**: `group` (hashed session). All splitting is by group.

## Policies compared

All policies share one hard constraint: **ACT only if `legal.ACT == 1`.** Legality (authority,
blocking unknowns, contradictions) is authored and is not learnable. A learned policy can only
withhold ACT; it can never grant it.

- **G** (baseline): the deterministic gate, ACT iff `gate == ACT`.
- **Model**: L2-regularised logistic regression, pure Python, deterministic: features
  standardised on the training fold, lambda = 1.0, full-batch gradient descent, lr 0.1, 2000
  steps, zero init. No hyper-parameter search (one model, so nothing is selected on outcome).
- **L_cov** (matched coverage): ACT iff legal and p_hat >= tau_cov, where tau_cov is set on the
  training fold so that the training-fold ACT coverage of L_cov equals that of G (ties broken by
  lowering tau, i.e. never below G's coverage).
- **L_crc** (risk-controlled abstention): ACT iff legal and p_hat >= tau_crc, where tau_crc is
  chosen by conformal risk control on a calibration split (the last quarter of the training
  fold's groups, in hash order; the model is refit on the remaining three quarters): the smallest
  tau on the grid {0.00, 0.01, ..., 1.00} with `(n * R_hat(tau) + 1) / (n + 1) <= alpha`, where
  `R_hat(tau)` is the calibration mean of the loss `1[ACT(tau) and y = 0]` and alpha = 0.02. If no
  tau qualifies, L_crc never ACTs.
- **Fallback** when a policy does not ACT: ASK if `legal.ASK`, else OBSERVE if `legal.OBSERVE`,
  else ABSTAIN. Reported as a distribution; no outcome is claimed for it (v0 has no outcome for
  non-ACT actions; z0int#55).

## Evaluation

- Grouped K-fold cross-validation, K = 5. Groups sorted by hash, assigned round-robin. With fewer
  than K groups, K = number of groups (reported; such a run cannot pass the sufficiency gate).
- Out-of-fold (OOF) decisions are pooled.
- **Selective not-success rate** of a policy = mean(1 - y) over analysis rows it ACTs on.
  **Coverage** = fraction of analysis rows it ACTs on.
- **Primary endpoint**: `delta = risk(L_cov) - risk(G)` on pooled OOF rows.
- **CI**: 95% percentile cluster bootstrap over groups, B = 2000, seed 0.
- Secondary, never decisional: OOF AUROC and Brier vs the training-prevalence predictor;
  L_crc coverage and realised joint risk `mean(1[ACT and y = 0])`; risk-coverage curve;
  sensitivity of every number with unverified rows counted as not-success.

## Sufficiency gate (checked first)

The run is **INSUFFICIENT_DATA** unless all of these hold on the analysis set:

1. at least 300 rows;
2. at least 30 not-success rows (y = 0);
3. at least 10 groups;
4. K = 5, and every test fold has at least one y = 0 and one y = 1 row.

When the gate fails, the full analysis still runs and is reported as **descriptive only**; no
promotion claim is made from it, in either direction.

## Decision rule

If the sufficiency gate passes:

- **SHADOW_CANDIDATE** if all hold:
  1. coverage matched: `|coverage(L_cov) - coverage(G)| <= 0.02` on pooled OOF rows;
  2. `delta < 0` and the CI upper bound `< 0`;
  3. `delta < 0` in at least 3 of the 5 folds;
  4. L_crc's pooled OOF joint risk `<= alpha + 0.01` (the abstention is calibrated);
  5. zero rows where any learned policy ACTs while `legal.ACT == 0` (structural check).
- **NO_IMPROVEMENT** otherwise (including a coverage match failure).

Outcomes are exactly one of `INSUFFICIENT_DATA`, `NO_IMPROVEMENT`, `SHADOW_CANDIDATE`.

## Data-volume projection (reported with every run)

From a counts-only 7-day transcript sweep (`z0int outcomes verify --since 7d --all-turns`) and the
table manifest:

- verified turns per week, resolved (y not null) per week, not-success per week;
- the fraction of turns that carry an opportunity record (feature coverage);
- weeks until the sufficiency gate passes at current utilization, as the max over its count
  criteria;
- the analysis-set size needed to detect a halving of the base not-success rate at matched
  coverage with 80% power, one-sided alpha 0.05 (two-proportion normal approximation, which is
  conservative for this paired design), and weeks until that size arrives.

## Known threats (accepted for v0)

- Verification is **missing not at random**: turns with tests, commits or PRs get labels; chat
  turns rarely do. The analysis set over-represents coding turns.
- `verified_success` is observational: no counterfactual action was executed.
- The gate ACTs on most turns, so matched coverage leaves the learned policy only a small set of
  turns to withhold.
- One heavy session can dominate; grouped CV and the cluster bootstrap are the mitigation.
