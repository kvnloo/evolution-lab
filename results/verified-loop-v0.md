# Verified loop v0 -- result

Pre-registration: `docs/prereg/verified-loop-v0.md`. Table `0.1.0`, feature schema `69d7eef44ba1707e`. Metrics and counts only.

## Decision: **INSUFFICIENT_DATA** (analysis is descriptive only)

| sufficiency check | observed | passed |
|---|---|---|
| rows>=300 | 6 | False |
| not_success>=30 | 1 | False |
| groups>=10 | 1 | False |
| k==5_and_every_test_fold_has_both_classes | k=None | False |

Input rows 23; analysis rows 6; excluded {'unverified_or_no_label': 16, 'observed_not_act': 1}.

Grouped CV not possible (not possible with fewer than 2 groups). Gate only, descriptive: n=6, not-success=1, G ACT coverage 1.000, selective not-success 0.167.

Sensitivity (unverified counted as not-success, 19 rows, not decisional): G 0.722 vs L_cov n/a, AUROC 0.464.

## Data-volume projection

- sweep span actually covered: 1.00 days (rates scaled to a week); sessions 331 by cohort {'agent': 5, 'harness': 325, 'interactive': 1}

| population | turns/wk | resolved/wk | not-success/wk (observed n) | sessions/wk | not-success rate (planning, source) | feature coverage since emission began |
|---|---:|---:|---:|---:|---|---:|
| live (interactive+agent) | 217 | 28 | 7.0 (1) | 42 | 0.250 (0.043, pooled_all_cohorts) | 1.000 |
| all cohorts (incl. eval/probe harness) | 2492 | 651 | 28.0 (4) | 2317 | 0.043 (0.043, pooled_all_cohorts) | 0.052 |

| population | scenario | weeks to sufficiency gate | binding | rows for 80% power (halving) | weeks to power |
|---|---|---:|---|---:|---:|
| live (interactive+agent) | current_feature_coverage (1.00) | 24.1 | not_success | 834 | 29.6 |
| live (interactive+agent) | full_feature_coverage (1.00) | 24.1 | not_success | 834 | 29.6 |
| all cohorts (incl. eval/probe harness) | current_feature_coverage (0.05) | 19.8 | not_success | 834 | 24.3 |
| all cohorts (incl. eval/probe harness) | full_feature_coverage (1.00) | 1.0 | not_success | 834 | 1.3 |

- linear extrapolation of a short window at current utilization; tiny not-success counts make the base rate (and so the power target) very uncertain; labels mature over the fix window (up to 7 days), so resolved counts lag
